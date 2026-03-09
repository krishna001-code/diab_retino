# AI-Powered Diabetic Retinopathy Screening for Eye Camps
## Complete Pipeline, Architecture & Solution Document

---

## Table of Contents

1. [Problem Statement Breakdown](#1-problem-statement-breakdown)
2. [General Solution Overview](#2-general-solution-overview)
3. [General Pipeline — Non-Technical](#3-general-pipeline--non-technical)
4. [System Architecture — High Level](#4-system-architecture--high-level)
5. [Detailed Technical Pipeline](#5-detailed-technical-pipeline)
6. [Model Architecture — Deep Dive](#6-model-architecture--deep-dive)
7. [Explainability Layer](#7-explainability-layer--grad-cam--cbam-heatmaps)
8. [Triage & Referral Engine](#8-triage--referral-engine)
9. [Safety Protocol & Interface Design](#9-safety-protocol--interface-design)
10. [Loss Function — Clinical Knowledge Encoded](#10-loss-function--hybriddrlosspython)
11. [Training Strategy](#11-training-strategy)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Field Robustness](#13-field-robustness--handling-real-world-noise)
14. [Full End-to-End Flow Summary](#14-full-end-to-end-flow-summary)

---

## 1. Problem Statement Breakdown

Before building anything, the problem statement must be decoded into precise engineering requirements.

| Problem Statement Requirement | Engineering Translation |
|---|---|
| Classify images into standard clinical DR stages | 4-class ordinal classification: No DR / Moderate / Severe / PDR |
| Identify patients requiring urgent referral | Rule-based triage engine on top of model output |
| Safely filter low-risk cases | High-confidence Grade 0 predictions routed to "monitor" queue |
| Explainable reasoning (heatmaps, lesion localization) | Grad-CAM + CBAM spatial attention maps overlaid on fundus image |
| Handle real-world noise: blur, uneven lighting, low contrast | Augmentation strategy + image quality gate at inference |
| Predictions must render quickly | Optimized inference — no TTA, single forward pass at deployment |
| Interface must state "Screening Support – Non-Diagnostic" | Hardcoded disclaimer on every prediction output |
| Trust & Explainability for doctors | Visual attention overlay + per-class confidence scores + lesion region highlight |

**The four non-negotiable deliverables from the problem statement:**

```
1. Core Classification     →  DR Grade (0 / 2 / 3 / 4)
2. Actionable Triage       →  "Monitor / Follow-up / Refer / Urgent"
3. Explainability          →  Visual heatmap + textual rationale
4. Safety Protocol         →  "Screening Support – Non-Diagnostic" on every output
```

---

## 2. General Solution Overview

### What We Are Building

An AI-powered **triage assistant** — not a replacement for an ophthalmologist, but a first-line filter that:

- **Automatically grades** every fundus image the moment it is captured
- **Prioritizes the queue** — Grade 3/4 patients are flagged and moved to the front
- **Generates a visual explanation** so the camp doctor can verify the AI's reasoning at a glance
- **Documents the safety boundary** on every output — the system never presents itself as a diagnostic tool

### The Core Philosophy

> The AI handles the **volume problem** (screening hundreds of healthy patients).  
> The doctor handles the **judgment problem** (making the final clinical decision).  
> The interface enforces the **trust problem** (the AI never overstates its authority).

### Why This Approach

Most DR screening tools fail in the field for one of three reasons:
1. They are accurate in the lab but brittle under variable camera quality
2. They give a prediction with no explanation — doctors cannot trust a black box
3. They present themselves as diagnostic, creating medico-legal risk

This solution is designed to avoid all three failure modes from the ground up.

---

## 3. General Pipeline — Non-Technical

This section describes the full patient journey through the system in plain language.

```
┌─────────────────────────────────────────────────────────────────┐
│                         EYE CAMP                                │
│                                                                 │
│  Patient arrives → Fundus camera captures retinal photograph   │
│                              ↓                                  │
│              Image transferred to screening device              │
│                              ↓                                  │
│         ┌────────────────────────────────────────┐              │
│         │         AI SCREENING ASSISTANT         │              │
│         │                                        │              │
│         │  Step 1: Image Quality Check           │              │
│         │  Step 2: DR Severity Grading           │              │
│         │  Step 3: Triage Recommendation         │              │
│         │  Step 4: Explainability Overlay        │              │
│         │  Step 5: Safety Disclaimer Output      │              │
│         └────────────────────────────────────────┘              │
│                              ↓                                  │
│              Camp Nurse / Technician sees result:               │
│                                                                 │
│    ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐    │
│    │ Grade 0  │  │ Grade 2   │  │ Grade 3  │  │ Grade 4  │    │
│    │ MONITOR  │  │ FOLLOW-UP │  │  REFER   │  │ URGENT   │    │
│    └──────────┘  └───────────┘  └──────────┘  └──────────┘    │
│                              ↓                                  │
│         Doctor reviews flagged cases (Grade 3 & 4 only)        │
│         with heatmap overlay to verify AI reasoning            │
│                                                                 │
│   Disclaimer on every output: "Screening Support – Non-Diagnostic" │
└─────────────────────────────────────────────────────────────────┘
```

### The Triage Benefit

Without AI:
- Ophthalmologist reviews **all 200 patients** → 3–5 minutes each → impossible

With AI:
- AI filters **180 Grade 0/2 patients** → routed automatically
- Ophthalmologist reviews **only 20 Grade 3/4 patients** → focused attention where it matters
- **10× throughput improvement** without compromising safety

---

## 4. System Architecture — High Level

The full system has four layers. Each layer has a clear responsibility.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — INPUT LAYER                                               │
│  Fundus image ingestion → Quality gate → Preprocessing               │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — AI INFERENCE LAYER                                        │
│  DenseNet121 Backbone → CBAM Attention → DANet Dual Attention        │
│  → Classifier → Grade (0/2/3/4) + Confidence Scores                 │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — EXPLAINABILITY LAYER                                      │
│  Grad-CAM → Attention heatmap overlaid on fundus image               │
│  CBAM spatial map → Lesion region localization                       │
│  Textual rationale → Per-class confidence breakdown                  │
└──────────────────────────────────────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — OUTPUT LAYER                                              │
│  Triage recommendation → Queue prioritization                        │
│  Structured report → Safety disclaimer                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Detailed Technical Pipeline

### Stage 1 — Image Ingestion & Quality Gate

Before the model ever sees the image, a quality check runs to flag poor captures.

```python
Quality Checks:
  ├── Brightness check   → reject if mean pixel value < 20 or > 235
  ├── Blur detection     → Laplacian variance < threshold → flag as blurry
  ├── FOV check          → circular mask coverage > 70% of image area
  └── Resolution check   → minimum 512×512 before resize
```

If the image **fails quality check** → output: `"Image quality insufficient. Retake required."`  
If the image **passes** → proceed to preprocessing.

---

### Stage 2 — Preprocessing

```python
Preprocessing Pipeline:
  ├── Resize to (256, 256)
  ├── CenterCrop to (224, 224)
  ├── ToTensor
  └── Normalize(mean=[computed from train set], std=[computed from train set])
```

**Why dataset-computed normalization (not ImageNet)?**  
Fundus images have a fundamentally different colour profile from natural images — yellowish-orange with high contrast vascular structures. Using the training set's own mean/std ensures the model sees inputs consistent with what it was trained on.

---

### Stage 3 — Model Inference

Single forward pass through the full model:

```
Input tensor (1 × 3 × 224 × 224)
  ↓
DenseNet121 features → (1 × 1024 × 7 × 7)
  ↓
CBAM Channel Attention → (1 × 1024 × 7 × 7)     # channel re-weighting
  ↓
CBAM Spatial Attention → (1 × 1024 × 7 × 7)     # spatial heatmap
  ↓
DANet projection conv  → (1 × 512 × 7 × 7)      # dimensionality reduction
  ↓
PAM (Position Attention) → (1 × 512 × 7 × 7)   # spatial self-attention
  ↓
CAM (Channel Attention)  → (1 × 512 × 7 × 7)   # channel self-attention
  ↓
Dual fusion conv         → (1 × 512 × 7 × 7)
  ↓
Global Average Pool      → (1 × 512)
  ↓
Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→4)
  ↓
Logits                   → (1 × 4)
  ↓
Softmax                  → Confidence scores [p0, p2, p3, p4]
  ↓
Argmax                   → Predicted DR Grade
```

**Output of Stage 3:**
```python
{
  "grade": 3,
  "grade_label": "Severe DR",
  "confidence": 0.87,
  "scores": {
    "No_DR":   0.04,
    "Moderate": 0.07,
    "Severe":  0.87,
    "PDR":     0.02
  }
}
```

---

### Stage 4 — Explainability Generation

Two complementary explanations are generated simultaneously:

**4a. Grad-CAM Heatmap**
```
Compute gradients of the predicted class score
  with respect to the final convolutional feature map (DANet output)
  ↓
Global average pool the gradients → class activation weights
  ↓
Weight the feature map channels by these activation weights
  ↓
ReLU → Upsample to original image size (224×224)
  ↓
Normalize to [0, 1] → apply colormap (jet/plasma)
  ↓
Overlay on original fundus image at 40% opacity
```

**4b. CBAM Spatial Attention Map**
```
Extract spatial attention output from CBAM module (7×7 map)
  ↓
Upsample to 224×224 using bilinear interpolation
  ↓
Threshold at 0.5 → highlight high-attention regions
  ↓
Overlay as contour outline on fundus image
```

**4c. Textual Rationale**
```python
Rationale templates by grade:

Grade 0: "No significant lesions detected. Retinal vasculature appears normal.
          Optic disc and macula show no signs of pathological change."

Grade 2: "Microaneurysms and/or haemorrhages detected. Attention concentrated
          in [region]. Early diabetic changes present — closer monitoring advised."

Grade 3: "Extensive retinal haemorrhages and venous changes detected.
          High-attention regions indicate significant vascular compromise.
          Urgent ophthalmologist review recommended."

Grade 4: "Neovascularisation pattern detected. Proliferative changes present.
          Immediate specialist intervention required to prevent vision loss."
```

---

### Stage 5 — Triage Engine

The triage engine maps model output to a strict clinical action using a confidence-gated rule system:

```python
def triage(grade, confidence):

    if grade == 0 and confidence >= 0.85:
        return {
            "action":   "MONITOR",
            "urgency":  "Low",
            "message":  "Annual screening recommended.",
            "queue":    "Standard"
        }

    elif grade == 0 and confidence < 0.85:
        return {
            "action":   "REVIEW",
            "urgency":  "Low",
            "message":  "Low confidence — manual verification advised.",
            "queue":    "Doctor Review"
        }

    elif grade == 2:
        return {
            "action":   "FOLLOW-UP",
            "urgency":  "Moderate",
            "message":  "Schedule follow-up within 3–6 months.",
            "queue":    "Follow-up Queue"
        }

    elif grade == 3:
        return {
            "action":   "REFER",
            "urgency":  "High",
            "message":  "Refer to ophthalmologist within 4 weeks.",
            "queue":    "Priority Queue"
        }

    elif grade == 4:
        return {
            "action":   "URGENT",
            "urgency":  "Critical",
            "message":  "Immediate specialist intervention required.",
            "queue":    "URGENT — Front of Queue"
        }
```

**Key design decision:** Low-confidence Grade 0 predictions are NOT automatically cleared. They go to a doctor review queue. The system errs on the side of caution.

---

### Stage 6 — Structured Output Report

Every patient gets a structured output record:

```
╔══════════════════════════════════════════════════════════════╗
║           DIABETIC RETINOPATHY SCREENING REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║  Patient ID   : [ID]              Date: [DATE]               ║
║  Eye          : Left / Right                                  ║
╠══════════════════════════════════════════════════════════════╣
║  DR GRADE     : SEVERE (Grade 3)                             ║
║  CONFIDENCE   : 87%                                          ║
╠══════════════════════════════════════════════════════════════╣
║  TRIAGE ACTION: ⚠️  REFER — Priority Queue                   ║
║  Refer to ophthalmologist within 4 weeks.                    ║
╠══════════════════════════════════════════════════════════════╣
║  CONFIDENCE BREAKDOWN                                        ║
║  No DR     ████░░░░░░░░░░░░  4%                              ║
║  Moderate  ███████░░░░░░░░░  7%                              ║
║  Severe    ██████████████░░  87%  ← PREDICTED               ║
║  PDR       ██░░░░░░░░░░░░░░  2%                              ║
╠══════════════════════════════════════════════════════════════╣
║  VISUAL EXPLANATION: [Grad-CAM Heatmap Overlay]             ║
║  Extensive haemorrhages detected — high-attention regions    ║
║  indicate significant vascular compromise.                   ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠️  SCREENING SUPPORT ONLY — NON-DIAGNOSTIC                 ║
║  This result must be reviewed and confirmed by a qualified   ║
║  ophthalmologist before any clinical decision is made.       ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 6. Model Architecture — Deep Dive

### 6.1 Why DenseNet121?

| Property | Benefit for DR Screening |
|---|---|
| Dense connectivity | Low-level features (microaneurysms, vascular patterns) persist at every depth — not lost in deep layers |
| Parameter efficiency | Strong accuracy with fewer weights — faster inference on field devices |
| Pretrained ImageNet weights | Feature detectors transfer well to retinal texture patterns |
| Established medical imaging baseline | Used in CheXNet (chest X-ray), shown effective for fundus analysis |

**Selective Fine-Tuning:**
```
Frozen layers:        conv0, denseblock1, denseblock2, denseblock3
Trainable layers:     denseblock4, norm5, CBAM, DANet, Classifier
```
Freezing the early layers preserves general feature detectors (edges, textures). Unfreezing only the final block allows the model to adapt high-level representations to DR-specific pathology without overfitting on a limited dataset.

---

### 6.2 CBAM — Full Implementation Logic

```
Input Feature Map:  (B × 1024 × 7 × 7)

── Channel Attention ──────────────────────────────────────
  AvgPool(spatial) → (B × 1024 × 1 × 1)
  MaxPool(spatial) → (B × 1024 × 1 × 1)
  Both fed through shared MLP:
    Linear(1024 → 64) → ReLU → Linear(64 → 1024)
  Avg_out + Max_out → Sigmoid → scale vector (B × 1024 × 1 × 1)
  Output = Input × scale              → (B × 1024 × 7 × 7)

── Spatial Attention ───────────────────────────────────────
  ChannelAvgPool(across channels) → (B × 1 × 7 × 7)
  ChannelMaxPool(across channels) → (B × 1 × 7 × 7)
  Concat → (B × 2 × 7 × 7)
  Conv2d(2→1, kernel=7, padding=3) → (B × 1 × 7 × 7)
  Sigmoid → spatial heatmap
  Output = Input × heatmap           → (B × 1024 × 7 × 7)
```

**Clinical significance of CBAM spatial map:** The 7×7 spatial attention map, when upsampled to 224×224, directly highlights which retinal regions drove the classification decision. This is the first layer of explainability.

---

### 6.3 DANet — Full Implementation Logic

```
Input Feature Map:  (B × 1024 × 7 × 7)

── Projection ──────────────────────────────────────────────
  Conv2d(1024→512, kernel=1) → BN → ReLU → (B × 512 × 7 × 7)

── Position Attention Module (PAM) ─────────────────────────
  N = H × W = 7 × 7 = 49
  Q = Conv2d(512→64)(feat) → reshape → (B × N × 64)
  K = Conv2d(512→64)(feat) → reshape → (B × 64 × N)
  V = Conv2d(512→512)(feat)→ reshape → (B × 512 × N)

  Attention = Softmax(Q × K)          → (B × N × N)
  Output    = V × Attention^T         → (B × 512 × N)
  Reshape   → (B × 512 × H × W)
  PAM_out   = γ_pam × Output + feat   (learnable γ, init=0)

── Channel Attention Module (CAM) ──────────────────────────
  Q = feat → reshape → (B × 512 × N)
  K = feat → reshape → (B × N × 512)
  Energy = Q × K                      → (B × 512 × 512)
  Attention = Softmax(Energy)
  V = feat → reshape → (B × 512 × N)
  Output = Attention × V              → (B × 512 × N)
  Reshape → (B × 512 × H × W)
  CAM_out = γ_cam × Output + feat     (learnable γ, init=0)

── Fusion ──────────────────────────────────────────────────
  Combined = PAM_out + CAM_out
  Output   = Conv2d → BN → ReLU → Dropout2d(0.1)
           → (B × 512 × 7 × 7)
```

**Why learnable γ initialized to 0?** At the start of training, the attention modules have no useful signal. γ=0 means the residual connection dominates — the model behaves like standard DenseNet. As training progresses, γ grows and the attention contribution increases progressively. This is a stable training technique from the original DANet paper.

---

## 7. Explainability Layer — Grad-CAM + CBAM Heatmaps

This section directly addresses the problem statement requirement: *"Doctors must understand why a decision was made."*

### 7.1 Grad-CAM Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        # target_layer = model.dam.fusion (DANet fusion output)
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def generate(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Weight activations by gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP over spatial
        cam = (weights * self.activations).sum(dim=1)            # weighted sum over channels
        cam = F.relu(cam)                                         # ReLU — positive contributions only

        # Normalize and upsample to image size
        cam = cam / cam.max()
        cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode='bilinear')
        return cam.squeeze().detach().numpy()
```

### 7.2 Two-Level Explanation

| Level | Source | What It Shows |
|---|---|---|
| **Grad-CAM** | Gradients from final DANet layer | Which broad regions drove the prediction |
| **CBAM Spatial Map** | Attention weights from spatial attention | Which fine-grained lesion areas were focused on |
| **Overlay** | Both combined on original image | Complete visual explanation for the doctor |

### 7.3 What the Doctor Sees

```
Original Fundus Image     Grad-CAM Overlay          CBAM Attention Map
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │       │  ██             │       │     ○○          │
│   retina img    │  →    │ ████  model     │  +    │   ○○○○  lesion  │
│                 │       │  ██   focused   │       │     ○○  zones   │
└─────────────────┘       └─────────────────┘       └─────────────────┘

"Model's attention concentrated in superior temporal quadrant —
 consistent with haemorrhagic changes in that region."
```

---

## 8. Triage & Referral Engine

### 8.1 The Clinical Action Map

| DR Grade | Triage Label | Clinical Action | Queue Priority |
|---|---|---|---|
| 0 — No DR (conf ≥ 0.85) | ✅ MONITOR | Annual screening | Standard |
| 0 — No DR (conf < 0.85) | 🔍 REVIEW | Doctor verification | Review Queue |
| 2 — Moderate | 🟡 FOLLOW-UP | Re-screen in 3–6 months | Follow-up Queue |
| 3 — Severe | 🟠 REFER | Ophthalmologist within 4 weeks | Priority Queue |
| 4 — PDR | 🔴 URGENT | Immediate intervention | Front of Queue |

### 8.2 Confidence Thresholding

Triage is not purely grade-based — confidence matters:

```
High confidence (≥ 0.85) + Grade 0  →  Safe to clear without doctor review
Low confidence  (< 0.85) + Grade 0  →  Escalate to doctor review
Any Grade 3 or 4               →  Always escalated, regardless of confidence
```

This prevents the system from auto-clearing borderline cases just because they were technically classified as Grade 0.

### 8.3 Bilateral Eye Logic

Each patient has two eyes. The final triage action takes the **worse of the two eyes**:

```python
final_triage = max(left_eye_grade, right_eye_grade)
```

A patient with Grade 0 (left) and Grade 3 (right) is triaged as **REFER** — not MONITOR.

---

## 9. Safety Protocol & Interface Design

### 9.1 The Non-Diagnostic Boundary

The problem statement explicitly requires: *"Interface must clearly state: Screening Support – Non-Diagnostic."*

This is not just a disclaimer — it is a design constraint that shapes the entire output interface:

**What the system IS:**
- A triage assistant that prioritizes the ophthalmologist's review queue
- A screening support tool that reduces time-to-detection for severe cases
- An explainability aid that helps doctors verify AI reasoning

**What the system IS NOT:**
- A diagnostic tool
- A replacement for ophthalmologist judgment
- A system that can independently authorize treatment decisions

### 9.2 Hardcoded Safety Elements

The following elements appear on **every single output**, regardless of grade:

```
⚠️  SCREENING SUPPORT ONLY — NON-DIAGNOSTIC

This AI-generated result is intended to assist trained medical
personnel in prioritizing patient review. It does not constitute
a clinical diagnosis. All Grade 2, 3, and 4 results must be
reviewed and confirmed by a qualified ophthalmologist before any
clinical or treatment decision is made.
```

### 9.3 Low-Confidence Escalation

When confidence is below threshold, the output changes tone entirely:

```
CAUTION: Low-confidence prediction (63%)

The model was unable to make a high-confidence determination for
this image. This may be due to image quality, ambiguous clinical
presentation, or a borderline case between severity stages.

Recommendation: Manual ophthalmologist review required.
DO NOT use this result to make a triage decision.
```

---

## 10. Loss Function — HybridDRLoss

### 10.1 Why Standard Cross-Entropy Fails Here

| Problem | Cross-Entropy Behaviour | Clinical Consequence |
|---|---|---|
| Class imbalance | Biases toward majority class (No DR) | Misses all severe cases |
| Ignores ordinal structure | Treats Grade 0→4 confusion = Grade 2→3 confusion | No clinical safety distinction |
| Symmetric error cost | Under-grading and over-grading penalized equally | Missing severe DR tolerated |

### 10.2 The Hybrid Loss Formula

```
HybridDRLoss = ClassBalancedFocalLoss
             + 0.3 × OrdinalEMDLoss
             + 0.1 × CostSensitiveLoss
```

### 10.3 Component Deep Dive

**Class-Balanced Focal Loss**
```
Effective number: E_n = 1 - β^n  (β = 0.999)
Class weight:     w_c = (1 - β) / E_n
Focal weight:     (1 - p_t)^γ    (γ = 2.0)
Label smoothing:  ε = 0.05

Loss = Σ w_c × (1 - p_t)² × CE(smoothed_target, logits)
```
- Rare classes (Severe, PDR) receive significantly higher gradient weight
- Hard examples (low confidence) receive higher gradient weight
- Label smoothing prevents the model from becoming overconfident

**Ordinal EMD Loss**
```
pred_CDF = cumsum(softmax(logits))    # [p0, p0+p2, p0+p2+p3, 1.0]
true_CDF = cumsum(one_hot(target))    # [0, 0, 1, 1] for Grade 3

EMD = mean( Σ |pred_CDF - true_CDF| )
```
A model predicting Grade 0 for a true Grade 4 has CDF difference of [1,1,1,0] → EMD = 3.0  
A model predicting Grade 2 for a true Grade 3 has CDF difference of [0,1,0,0] → EMD = 1.0  
This directly encodes that distant grade errors are worse.

**Cost-Sensitive Loss**
```
Cost Matrix (under-grading rows have higher costs):

              → Predicted
True ↓    No_DR  Moderate  Severe   PDR
No_DR     [ 1.0,   1.0,    1.2,    1.3 ]
Moderate  [ 1.0,   1.0,    1.1,    1.2 ]
Severe    [ 1.5,   1.3,    1.0,    1.0 ]  ← missing severe DR: 1.5× penalty
PDR       [ 1.5,   1.4,    1.0,    1.0 ]  ← missing PDR: 1.5× penalty

cost_weight = Σ (cost_row × predicted_probs)
final_loss  = mean(CE_loss × cost_weight)
```

---

## 11. Training Strategy

### 11.1 Data Preparation

```
Stage 1 — Dataset Merging
  copy_images_from_folder() merges multiple source datasets
  Prefix each file with source name → no filename collision
  Organise into: train/  0/  2/  3/  4/
                 test/   0/  2/  3/  4/

Stage 2 — Normalization Statistics
  Compute channel-wise mean and std over the training set
  (NOT ImageNet stats — fundus colour profile is fundamentally different)

Stage 3 — Augmentation
  Train: Resize(256) → CenterCrop(224) → RandomRotation(±15°)
         → HFlip(0.5) → VFlip(0.2) → ColorJitter → Normalize
  Test:  Resize(256) → CenterCrop(224) → Normalize
```

### 11.2 Optimizer Configuration

```python
# Separate learning rates for pretrained vs newly initialized modules
optimizer = AdamW([
    {'params': backbone_params,   'lr': 1e-4,  'weight_decay': 1e-4},
    {'params': attention_params,  'lr': 5e-4,  'weight_decay': 1e-5},
])
```

**Why higher LR for attention modules?**  
CBAM and DANet are randomly initialized — they need to converge faster than the pretrained DenseNet backbone. A 5× higher learning rate allows attention modules to learn meaningful patterns while the backbone adjusts gradually.

### 11.3 Training Loop

```
For each epoch (max 75):

  TRAIN phase:
    Forward pass → HybridDRLoss
    Backward pass → clip_grad_norm_(max_norm=1.0)
    AdamW step

  VALIDATION phase:
    Forward pass (no gradient)
    Compute QWK, F1-macro, F1-weighted, accuracy
    If QWK improved → save best model checkpoint
    QWK early stopping check (patience=15, delta=0.001)

  After each epoch:
    CosineAnnealingLR scheduler step
```

---

## 12. Evaluation Framework

### 12.1 Primary Metric — Quadratic Weighted Kappa (QWK)

QWK is the **official metric for the Kaggle DR Detection competition** and the standard used by ophthalmologists for grading consistency evaluation.

```
QWK = 1 - (Σ W_ij × O_ij) / (Σ W_ij × E_ij)

Where:
  W_ij = (i - j)² / (N - 1)²   (quadratic weight matrix)
  O_ij = observed confusion matrix
  E_ij = expected confusion matrix under independence
```

| QWK Score | Clinical Interpretation |
|---|---|
| ≥ 0.90 | Almost Perfect Agreement |
| ≥ 0.80 | Strong Agreement ← **Clinical reliability threshold** |
| ≥ 0.60 | Moderate Agreement |
| ≥ 0.40 | Fair Agreement |
| < 0.40 | Poor Agreement — not suitable for clinical use |

### 12.2 Full Evaluation Suite

| Metric | What It Catches |
|---|---|
| **QWK** | Ordinal agreement — primary clinical benchmark |
| **F1-Macro** | Balanced per-grade performance — catches poor recall on rare grades |
| **F1-Weighted** | Real-world population accuracy |
| **Per-Class Precision** | False positive rate per grade |
| **Per-Class Recall** | Miss rate per grade — especially critical for Grade 3/4 |
| **Confusion Matrix** | Whether errors are adjacent (acceptable) or distant (dangerous) |

### 12.3 The Confusion Matrix That Matters

```
Acceptable errors:   Grade 2 ↔ Grade 3   (adjacent, same management tier)
Dangerous errors:    Grade 0 ↔ Grade 3   (patient sent home when should be referred)
                     Grade 0 ↔ Grade 4   (patient sent home when sight-threatening)
```

The off-diagonal corners of the confusion matrix are the ones that cause blindness. HybridDRLoss directly penalizes these.

---

## 13. Field Robustness — Handling Real-World Noise

The problem statement specifically calls out: *"blur, uneven lighting, low contrast."*

### 13.1 Image-Level Robustness (Preprocessing)

| Real-World Issue | Handling Strategy |
|---|---|
| **Uneven lighting** | CLAHE (Contrast Limited Adaptive Histogram Equalization) optional preprocessing |
| **Low contrast** | ColorJitter augmentation during training builds tolerance |
| **Motion blur** | Blur detection gate rejects images below sharpness threshold |
| **Poor FOV framing** | CenterCrop + circular mask coverage check |
| **Different camera models** | Dataset includes multiple sources + ColorJitter normalizes variation |

### 13.2 Model-Level Robustness

| Architecture Choice | Robustness Benefit |
|---|---|
| CBAM attention | Suppresses noisy background channels — focuses on signal |
| DenseNet dense connections | Feature reuse means partial information still flows through |
| Dropout(0.5) in classifier | Prevents over-reliance on any single feature dimension |
| Label smoothing (ε=0.05) | Prevents overconfident predictions on noisy inputs |

---

## 14. Full End-to-End Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA                                                     │
│  Multiple DR datasets → copy_images_from_folder()                  │
│  → combined_output_dir / train / test / 0 / 2 / 3 / 4             │
│  → get_mean_std() → dataset-calibrated normalization               │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — TRAINING                                                 │
│  ImageFolder → DataLoader (batch=32, augmentation)                 │
│  → DRmodel (DenseNet121 + CBAM + DANet)                            │
│  → HybridDRLoss (Focal + Ordinal + Cost-Sensitive)                 │
│  → AdamW (dual LR) + CosineAnnealingLR + grad clipping            │
│  → QWK early stopping (patience=15)                                │
│  → best_qwk_dr_model_cbam_danet.pth                                │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — INFERENCE (at eye camp)                                 │
│  Fundus image → Quality Gate → Preprocessing                       │
│  → Single forward pass → Grade + Confidence scores                 │
│  → Grad-CAM + CBAM spatial map → Heatmap overlay                  │
│  → Textual rationale generation                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — TRIAGE OUTPUT                                           │
│  Grade + Confidence → Triage engine                                │
│  → Action: MONITOR / FOLLOW-UP / REFER / URGENT                   │
│  → Structured report with heatmap + confidence breakdown           │
│  → Hardcoded: "Screening Support – Non-Diagnostic"                 │
│  → Bilateral eye logic → final patient-level triage                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Addressing Every Problem Statement Requirement

| Requirement | Implementation | Location |
|---|---|---|
| Classify into clinical DR stages | 4-class ordinal model (0/2/3/4) | `class DRmodel` |
| Identify urgent referral cases | Confidence-gated triage engine | `def triage()` |
| Safely filter low-risk cases | High-confidence Grade 0 auto-cleared | Triage engine threshold |
| Explainable reasoning — heatmaps | Grad-CAM on DANet output | `class GradCAM` |
| Lesion localization | CBAM spatial attention map overlay | `class SpatialAttention` |
| Field robustness — blur | Laplacian variance quality gate | Quality gate Stage 1 |
| Field robustness — lighting | ColorJitter augmentation + CLAHE | Training transforms |
| Fast throughput | Single forward pass, no TTA | Inference pipeline |
| Screening Support – Non-Diagnostic | Hardcoded on every output | Output report template |
| Trust for doctors | Grade + confidence + heatmap + text rationale | Explainability layer |

---

*DenseNet121 + CBAM + DANet | HybridDRLoss | QWK-Optimized | Explainability via Grad-CAM*  
*⚠️ Screening Support Only — Non-Diagnostic*
