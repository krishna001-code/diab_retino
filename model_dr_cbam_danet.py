import os
import glob
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
warnings.filterwarnings("ignore")


# ============================================================
#  CBAM — Convolutional Block Attention Module
#  Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
#  Adds channel-wise + spatial attention on top of any feature map.
# ============================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Compresses spatial dims via both avg-pool and max-pool,
    then learns channel importance weights via a shared MLP.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        mid = max(1, in_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # squeeze H×W → 1×1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (implemented as 1×1 convolutions for efficiency)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out)   # (B, C, 1, 1)
        return x * scale                             # broadcast over H×W


class SpatialAttention(nn.Module):
    """
    Learns a 2-D spatial attention map by looking at
    channel-wise avg & max descriptors across the spatial plane.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat  = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        scale   = self.sigmoid(self.conv(concat))       # (B, 1, H, W)
        return x * scale                                 # broadcast over C


class CBAM(nn.Module):
    """
    Full CBAM block:  channel attention  →  spatial attention.
    Drop-in replacement/wrapper for any feature tensor.
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================
#  DANet — Dual Attention Network modules
#  Paper: "Dual Attention Network for Scene Segmentation" (CVPR 2019)
#  Two complementary attention heads:
#    1. PAM – Position Attention Module  (self-attention over spatial positions)
#    2. CAM – Channel Attention Module   (self-attention over channel pairs)
# ============================================================

class PositionAttentionModule(nn.Module):
    """
    Self-attention over spatial positions (H×W nodes).
    Captures long-range pixel-level dependencies.
    Uses 1×1 convolutions to project to query/key/value spaces.
    """
    def __init__(self, in_channels, reduction=8):
        super(PositionAttentionModule, self).__init__()
        mid = max(1, in_channels // reduction)

        self.query_conv = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma  = nn.Parameter(torch.zeros(1))   # learnable scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # Project to query, key, value
        Q = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C')
        K = self.key_conv(x).view(B, -1, N)                      # (B, C', N)
        V = self.value_conv(x).view(B, -1, N)                    # (B, C,  N)

        # Attention map: each position attends to all positions
        attn = self.softmax(torch.bmm(Q, K))          # (B, N, N)

        # Weighted aggregation
        out = torch.bmm(V, attn.permute(0, 2, 1))     # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x   # residual


class ChannelAttentionModule(nn.Module):
    """
    Self-attention over channel pairs (C nodes).
    Each channel attends to every other channel to capture
    semantic channel-interdependencies.
    """
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma   = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        # Flatten spatial dims
        Q = x.view(B, C, N)                  # (B, C, N)
        K = x.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        V = x.view(B, C, N)                  # (B, C, N)

        # Channel-to-channel attention
        energy = torch.bmm(Q, K)             # (B, C, C)
        attn   = self.softmax(energy)         # (B, C, C)

        out = torch.bmm(attn, V)             # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x   # residual


class DualAttentionModule(nn.Module):
    """
    Combines PAM + CAM outputs via a learned fusion head.
    The two complementary attention streams are summed, then
    projected back to `out_channels` through a lightweight head.
    """
    def __init__(self, in_channels, out_channels, pam_reduction=8):
        super(DualAttentionModule, self).__init__()

        # Lightweight projection before attention (saves memory on large feature maps)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pam = PositionAttentionModule(out_channels, reduction=pam_reduction)
        self.cam = ChannelAttentionModule()

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        feat  = self.proj(x)
        pam_out = self.pam(feat)
        cam_out = self.cam(feat)
        out   = self.fusion(pam_out + cam_out)
        return out


# ============================================================
#  Loss Functions (unchanged from model_dr.py)
# ============================================================

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, samples_per_cls, beta=0.999, gamma=2.0,
                 label_smoothing=0.05, device='cuda'):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes

        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth  = (1 - self.label_smoothing) * targets_one_hot + \
                          self.label_smoothing / self.num_classes

        probs    = F.softmax(logits, dim=1)
        pt       = torch.sum(targets_smooth * probs, dim=1)
        focal_wt = (1 - pt) ** self.gamma

        log_probs   = F.log_softmax(logits, dim=1)
        ce_loss     = -torch.sum(targets_smooth * log_probs, dim=1)
        class_wts   = self.weights[targets]

        return (focal_wt * ce_loss * class_wts).mean()


class OrdinalEMDLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalEMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs      = F.softmax(logits, dim=1)
        pred_cdf   = torch.cumsum(probs, dim=1)
        tgt_one_hot = F.one_hot(targets, self.num_classes).float()
        tgt_cdf    = torch.cumsum(tgt_one_hot, dim=1)
        return torch.mean(torch.sum(torch.abs(pred_cdf - tgt_cdf), dim=1))


class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix, device='cuda'):
        super(CostSensitiveLoss, self).__init__()
        self.cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32).to(device)

    def forward(self, logits, targets):
        ce_loss    = F.cross_entropy(logits, targets, reduction='none')
        costs      = self.cost_matrix[targets, :]
        probs      = F.softmax(logits, dim=1)
        cost_wts   = torch.sum(costs * probs, dim=1)
        return torch.mean(ce_loss * cost_wts)


class HybridDRLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes=4, beta=0.999, gamma=2.0,
                 label_smoothing=0.05, ordinal_weight=0.3, use_cost_sensitive=True,
                 device='cuda'):
        super(HybridDRLoss, self).__init__()

        self.focal_loss   = ClassBalancedFocalLoss(
            num_classes, samples_per_cls, beta, gamma, label_smoothing, device)
        self.ordinal_loss  = OrdinalEMDLoss(num_classes)
        self.ordinal_weight = ordinal_weight

        self.use_cost_sensitive = use_cost_sensitive
        if use_cost_sensitive:
            cost_matrix = [
                [1.0, 1.0, 1.2, 1.3],
                [1.0, 1.0, 1.1, 1.2],
                [1.5, 1.3, 1.0, 1.0],
                [1.5, 1.4, 1.0, 1.0]
            ]
            self.cost_loss  = CostSensitiveLoss(cost_matrix, device)
            self.cost_weight = 0.1

    def forward(self, logits, targets):
        focal   = self.focal_loss(logits, targets)
        ordinal = self.ordinal_loss(logits, targets)
        total   = focal + self.ordinal_weight * ordinal

        if self.use_cost_sensitive:
            cost  = self.cost_loss(logits, targets)
            total = total + self.cost_weight * cost

        return total, focal, ordinal


# ============================================================
#  Utility Classes
# ============================================================

class QWKEarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.001):
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_qwk   = -1.0
        self.early_stop = False
        self.delta      = delta

    def __call__(self, qwk_score):
        if qwk_score > self.best_qwk + self.delta:
            self.best_qwk = qwk_score
            self.counter  = 0
            if self.verbose:
                print(f'QWK improved to {qwk_score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'QWK EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================
#  DRmodel — DenseNet121 + CBAM + DANet
#
#  Architecture:
#    DenseNet121 (pretrained)
#      └─ Backbone features (frozen except denseblock4 + norm5)
#      └─ After denseblock4: CBAM refinement  ← NEW
#      └─ After global pool:  DANet          ← NEW
#      └─ Custom classifier head
#
#  Feature flow:
#    Image (3×224×224)
#      → DenseNet features → denseblock4 output (1024×7×7)
#      → CBAM  (channel + spatial attention)       → refined (1024×7×7)
#      → Global Avg Pool                           → (1024×1×1)
#      → DANet (position + channel self-attention) → (512×1×1)
#      → Flatten → Dropout → Linear(512→4)
# ============================================================

class DRmodel(nn.Module):
    """
    DenseNet121 backbone enhanced with:
      - CBAM after the last dense block (spatial + channel refinement)
      - DANet dual attention for global feature enrichment
    """
    def __init__(self, num_classes=4, cbam_reduction=16, dam_out=512):
        super(DRmodel, self).__init__()

        # ── Backbone ──────────────────────────────────────────────
        backbone = models.densenet121(pretrained=True)

        # Freeze all layers
        for p in backbone.parameters():
            p.requires_grad = False

        # Unfreeze only denseblock4 + norm5 for fine-tuning
        for p in backbone.features.denseblock4.parameters():
            p.requires_grad = True
        for p in backbone.features.norm5.parameters():
            p.requires_grad = True

        # Keep backbone.features; discard the original pooling + FC classifier
        self.features = backbone.features        # outputs (B, 1024, 7, 7) for 224 input
        num_backbone_ch = backbone.classifier.in_features  # 1024

        # ── CBAM after denseblock4 ────────────────────────────────
        # Refines which channels and spatial regions matter most
        # before global pooling collapses spatial information.
        self.cbam = CBAM(
            in_channels    = num_backbone_ch,   # 1024
            reduction_ratio = cbam_reduction,   # 16  → mid = 64
            spatial_kernel  = 7                 # 7×7 conv matches feature map size
        )

        # ── Global Average Pooling ────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)       # (B, 1024, 7, 7) → (B, 1024, 1, 1)

        # ── DANet after pooling ───────────────────────────────────
        # Projects 1024 → dam_out (512) channels, then applies
        # dual (position + channel) self-attention.
        # Even though spatial dims are 1×1 after GAP, the channel
        # attention branch still captures inter-channel dependencies.
        # For richer spatial context, we apply DANet BEFORE GAP
        # on a downsampled feature map.
        self.dam = DualAttentionModule(
            in_channels  = num_backbone_ch,  # 1024
            out_channels = dam_out,          # 512
            pam_reduction = 8
        )
        # Second GAP after DANet (since DANet works on spatial features)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        # ── Classifier Head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dam_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. DenseNet feature extraction
        feat = self.features(x)               # (B, 1024, 7, 7)
        feat = F.relu(feat, inplace=True)

        # 2. CBAM attention refinement (on spatial feature map)
        feat = self.cbam(feat)                # (B, 1024, 7, 7)

        # 3. Dual Attention (still on spatial feature map before GAP)
        feat = self.dam(feat)                 # (B, 512, 7, 7)

        # 4. Global Average Pooling → flatten
        feat = self.gap2(feat)                # (B, 512, 1, 1)
        feat = feat.view(feat.size(0), -1)   # (B, 512)

        # 5. Classification head
        out = self.classifier(feat)           # (B, num_classes)
        return out


# ============================================================
#  Helper Functions
# ============================================================

def calculate_class_distribution(train_dataset):
    class_counts = [0] * 4
    for _, label in train_dataset:
        class_counts[label] += 1
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples")
    return class_counts


def calculate_metrics(y_true, y_pred, class_names):
    qwk         = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1_macro    = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    report      = classification_report(y_true, y_pred,
                                        target_names=class_names, output_dict=True)
    return qwk, f1_macro, f1_weighted, report


def evaluate_model_qwk(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    qwk, f1_macro, f1_weighted, report = calculate_metrics(
        all_targets, all_preds, class_names)
    return qwk, f1_macro, f1_weighted, report, all_targets, all_preds


def get_mean_std(loader, device):
    nimages = 0
    mean = torch.zeros(3, device=device)
    std  = torch.zeros(3, device=device)
    for images, _ in loader:
        images = images.to(device)
        bs     = images.size(0)
        images = images.view(bs, 3, -1)
        mean  += images.mean(2).sum(0)
        std   += images.std(2).sum(0)
        nimages += bs
    return mean / nimages, std / nimages


def copy_images_from_folder(source_split_dir, dest_split_dir, dataset_name):
    if not os.path.exists(source_split_dir):
        print(f"Warning: {source_split_dir} does not exist. Skipping {dataset_name}.")
        return
    for stage in ["0", "2", "3", "4"]:
        source_stage_dir = os.path.join(source_split_dir, stage)
        dest_stage_dir   = os.path.join(dest_split_dir, stage)
        if not os.path.exists(source_stage_dir):
            print(f"Warning: Stage {stage} not found in {source_split_dir}")
            continue
        image_files = [f for f in os.listdir(source_stage_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png',
                                              '.bmp', '.tiff', '.tif'))]
        print(f"Copying {len(image_files)} images from {dataset_name} - {stage}")
        for img_file in image_files:
            src  = os.path.join(source_stage_dir, img_file)
            dest = os.path.join(dest_stage_dir, f"{dataset_name}_{img_file}")
            try:
                shutil.copy2(src, dest)
            except Exception as e:
                print(f"Error copying {src}: {e}")


# ============================================================
#  Main Training Script
# ============================================================

if __name__ == '__main__':
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    output_dir = r'C:\Users\hamsa\OneDrive\Pictures\ML\DR\combined_output_dir'

    # Dataset summary
    print("\nDataset summary:")
    for split in ["train", "test"]:
        print(f"\n{split.upper()} set:")
        split_dir = os.path.join(output_dir, split)
        for stage in ["0", "2", "3", "4"]:
            stage_dir = os.path.join(split_dir, stage)
            if os.path.exists(stage_dir):
                count = len([f for f in os.listdir(stage_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png',
                                                    '.bmp', '.tiff', '.tif'))])
                print(f"  Stage {stage}: {count} images")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training.")

    device = torch.device("cuda")

    # ── Normalization stats ────────────────────────────────────
    print("\nCalculating normalization statistics …")
    temp_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    stat_ds  = datasets.ImageFolder(os.path.join(output_dir, 'train'), transform=temp_tf)
    stat_loader = DataLoader(stat_ds, batch_size=64, num_workers=0,
                             shuffle=False, pin_memory=False)
    mean, std = get_mean_std(stat_loader, device)
    mean_list = [round(m.item(), 4) for m in mean]
    std_list  = [round(s.item(), 4) for s in std]
    print(f"Normalize  mean={mean_list}  std={std_list}")

    # ── Transforms ────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list)
    ])

    train_ds = datasets.ImageFolder(os.path.join(output_dir, 'train'), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(output_dir, 'test'),  transform=val_tf)

    print(f"Classes found : {train_ds.classes}")
    print(f"Training samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    samples_per_cls = calculate_class_distribution(train_ds)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    model = DRmodel(num_classes=len(train_ds.classes),
                    cbam_reduction=16,
                    dam_out=512).to(device)

    print("\nModel architecture:")
    print(f"  Backbone  : DenseNet121 (pretrained, denseblock4+norm5 unfrozen)")
    print(f"  CBAM      : in=1024ch  reduction=16  spatial_kernel=7")
    print(f"  DANet     : in=1024ch  out=512ch  PAM+CAM dual attention")
    print(f"  Classifier: 512 → 256 → {len(train_ds.classes)}")

    # ── Loss ──────────────────────────────────────────────────
    criterion = HybridDRLoss(
        samples_per_cls=samples_per_cls,
        num_classes=4, beta=0.999, gamma=2.0,
        label_smoothing=0.05, ordinal_weight=0.3,
        use_cost_sensitive=True, device=device
    )

    # ── Optimizer (attention params get a slightly higher LR) ─
    attention_params = (list(model.cbam.parameters()) +
                        list(model.dam.parameters()))
    attention_ids    = {id(p) for p in attention_params}
    base_params      = [p for p in model.parameters()
                        if id(p) not in attention_ids and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': base_params,      'lr': 1e-4,  'weight_decay': 1e-4},
        {'params': attention_params, 'lr': 5e-4,  'weight_decay': 1e-5},  # faster LR for attention
    ], betas=(0.9, 0.999), eps=1e-8)

    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    early_stopper = QWKEarlyStopping(patience=15, verbose=True, delta=0.001)

    # ── Training state ────────────────────────────────────────
    best_qwk    = 0.0
    best_acc    = 0.0
    best_epoch  = 0
    train_losses, val_losses   = [], []
    train_accs,  val_accs     = [], []
    qwk_scores                = []
    focal_losses, ordinal_losses = [], []

    class_names = ['No_DR', 'Moderate', 'Severe', 'PDR']

    print(f"\n=== TRAINING — DenseNet121 + CBAM + DANet + HybridDRLoss ===")
    print(f"Primary Metric : QWK  |  Early Stopping patience=15")

    for epoch in range(75):
        print(f"\nEpoch {epoch+1}/75  |  LR={optimizer.param_groups[0]['lr']:.6f}")

        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else test_loader

            running_loss     = 0.0
            running_focal    = 0.0
            running_ordinal  = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss, focal, ordinal = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                n = inputs.size(0)
                running_loss    += loss.item() * n
                running_focal   += focal.item() * n
                running_ordinal += ordinal.item() * n
                running_corrects += torch.sum(preds == labels.data)

            ds_size    = len(train_ds) if phase == 'train' else len(test_ds)
            epoch_loss = running_loss    / ds_size
            epoch_fl   = running_focal   / ds_size
            epoch_ol   = running_ordinal / ds_size
            epoch_acc  = running_corrects.double() / ds_size

            print(f'{phase}  Loss={epoch_loss:.4f}  (Focal={epoch_fl:.4f}  '
                  f'Ordinal={epoch_ol:.4f})  Acc={epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                focal_losses.append(epoch_fl)
                ordinal_losses.append(epoch_ol)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

                qwk, f1_macro, f1_weighted, report = evaluate_model_qwk(
                    model, test_loader, device, class_names)
                qwk_scores.append(qwk)
                print(f'*** QWK={qwk:.4f}  F1-Macro={f1_macro:.4f}  '
                      f'F1-Weighted={f1_weighted:.4f} ***')

                if qwk > best_qwk:
                    best_qwk   = qwk
                    best_acc   = epoch_acc
                    best_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'qwk': qwk, 'accuracy': epoch_acc.item(),
                        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
                        'class_names': class_names,
                        'mean': mean_list, 'std': std_list,
                        'samples_per_cls': samples_per_cls
                    }, 'best_qwk_dr_model_cbam_danet.pth')
                    print(f'*** NEW BEST — QWK={best_qwk:.4f} — model saved ***')

                early_stopper(qwk)
                if early_stopper.early_stop:
                    print(f"Early stopping at epoch {epoch+1}. Best QWK={best_qwk:.4f}")
                    break

        if early_stopper.early_stop:
            break

        scheduler.step()

    # ── Final Evaluation ──────────────────────────────────────
    print(f"\n=== FINAL EVALUATION (CBAM + DANet model) ===")
    ckpt = torch.load('best_qwk_dr_model_cbam_danet.pth')
    model.load_state_dict(ckpt['model_state_dict'])

    final_qwk, final_f1_macro, final_f1_weighted, final_report, y_true, y_pred = \
        evaluate_model_qwk(model, test_loader, device, class_names)

    print(f"Best QWK      : {final_qwk:.4f}  (Epoch {best_epoch+1})")
    print(f"F1-Macro      : {final_f1_macro:.4f}")
    print(f"F1-Weighted   : {final_f1_weighted:.4f}")
    print(f"Accuracy      : {best_acc:.4f}")

    print("\n=== Per-Class Performance ===")
    for cn in class_names:
        if cn in final_report:
            r = final_report[cn]
            print(f"  {cn:<12}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    interp = ("Almost Perfect (≥0.9)" if final_qwk >= 0.9 else
              "Strong (≥0.8)"         if final_qwk >= 0.8 else
              "Moderate (≥0.6)"       if final_qwk >= 0.6 else
              "Fair (≥0.4)"           if final_qwk >= 0.4 else
              "Poor (<0.4)")
    print(f"\nQWK Interpretation: {interp}")

    # ── Save complete model info ───────────────────────────────
    torch.save({
        'state_dict':    model.state_dict(),
        'class_names':   class_names,
        'num_classes':   len(class_names),
        'mean': mean_list, 'std': std_list,
        'samples_per_cls': samples_per_cls,
        'best_qwk':      final_qwk,
        'best_epoch':    best_epoch,
        'best_accuracy': best_acc.item(),
        'best_f1_macro': final_f1_macro,
        'train_losses':  train_losses,
        'val_losses':    val_losses,
        'train_accs':    train_accs,
        'val_accs':      val_accs,
        'qwk_scores':    qwk_scores,
        'focal_losses':  focal_losses,
        'ordinal_losses':ordinal_losses,
        'final_report':  final_report,
        'model_config': {
            'backbone':      'DenseNet121',
            'attention':     'CBAM + DANet (PAM+CAM)',
            'cbam_reduction': 16,
            'dam_out':        512,
        },
        'hybrid_loss_config': {
            'beta': 0.999, 'gamma': 2.0,
            'label_smoothing': 0.05,
            'ordinal_weight':  0.3,
            'use_cost_sensitive': True,
            'cost_weight': 0.1
        }
    }, 'cbam_danet_dr_model_complete.pth')

    print('\nModel saved → cbam_danet_dr_model_complete.pth')
    print(f'TRAINING COMPLETE — BEST QWK: {final_qwk:.4f}')
