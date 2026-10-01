---
title: "Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)"
type: paper
tags: [self-supervised-learning, video-representation, jepa, feature-prediction, vision-transformer, masked-modeling]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/video-ssl.md, papers/assran_2025_vjepa2.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/V-JEPA.pdf]
---

# Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)

**Authors:** Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran†, Nicolas Ballas†
**Affiliations:** FAIR at Meta; Inria; École normale supérieure / CNRS / PSL; Univ. Gustave Eiffel / CNRS / LIGM; Courant Institute / NYU; Center for Data Science / NYU
**Date:** February 14, 2024
**Code:** https://github.com/facebookresearch/jepa

---

## Abstract

This paper explores feature prediction as a stand-alone objective for unsupervised learning from video, and introduces V-JEPA — a collection of vision models trained solely using a feature prediction objective, without the use of pretrained image encoders, text, negative examples, reconstruction, or other sources of supervision. The models are trained on 2 million videos collected from public datasets and evaluated on downstream image and video tasks. Results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adapting the model's parameters (frozen backbone). The largest model, a ViT-H/16 trained only on videos, obtains 81.9% on Kinetics-400, 72.2% on Something-Something-v2, and 77.9% on ImageNet1K.

---

## Key Contributions

- **Feature prediction as a stand-alone video SSL objective:** Demonstrates that predicting in learned representation space (rather than pixel space) is sufficient for strong video representations, without relying on pretrained encoders, text supervision, or negative samples.
- **V-JEPA model family:** Three encoders — ViT-L/16₂₂₄, ViT-H/16₂₂₄, ViT-H/16₃₈₄ — trained on VideoMix2M (2M videos), establishing the original V-JEPA recipe subsequently scaled in V-JEPA 2.
- **3D Multi-block spatiotemporal masking:** A masking strategy using unions of spatially continuous blocks spanning the full temporal dimension, yielding a ~90% masking ratio that forces genuine spatiotemporal reasoning (vs. random-tube or causal masking strategies which perform worse).
- **Attentive probing protocol:** A non-linear cross-attention pooling head that significantly outperforms linear probing (+17 pts K400, +16 pts SSv2) and is adopted as the standard frozen evaluation protocol.
- **Feature prediction > pixel prediction:** Consistent improvement over pixel-reconstruction baselines under both frozen evaluation and fine-tuning, while training ~2× faster (270M samples seen vs. 400K–2400M for pixel methods).
- **Label efficiency:** V-JEPA representations are more label-efficient than pixel-prediction models; the performance gap widens when the number of labeled examples decreases.

---

## Methodology

### Training Objective

V-JEPA uses a Joint Embedding Predictive Architecture (JEPA). Given a video clip, two views are sampled by masking:
- **Context view** $x$: the complement of the masked region (unmasked patches)
- **Target view** $y$: the masked region

The objective is to predict the target encoder's representation of $y$ from the context encoder's representation of $x$:

$$\text{minimize}_{\theta, \phi} \; \| P_\phi(E_\theta(x), \Delta_y) - \text{sg}(\overline{E}_\theta(y)) \|_1 \tag{1}$$

where:
- $E_\theta(\cdot)$ = x-encoder (ViT, trained via gradient descent)
- $\overline{E}_\theta(\cdot)$ = y-encoder (EMA teacher, stop-gradient target)
- $P_\phi(\cdot)$ = predictor network (narrow ViT)
- $\Delta_y$ = learnable mask tokens encoding spatio-temporal position of masked patches
- $\text{sg}(\cdot)$ = stop-gradient operator

**Theoretical motivation:** Under this $\ell_1$ loss, the optimal predictor satisfies $P^*(E_\theta(x)) = \text{median}(Y | E_\theta(x))$, which pushes the encoder gradient toward minimizing the median absolute deviation (MAD) of the target. The EMA teacher evolves faster than the encoder, keeping it near the optimum and preventing collapse without needing negative pairs or data augmentation invariance.

**Full loss (Appendix B):**

$$\text{Loss} = \frac{1}{M} \sum_{k \in \{i_1, \ldots, i_M\}} \|\hat{s}_k - s_k\|_1 \tag{2}$$

where $\hat{s}_k = P_\phi(z_N, m_M)$ are predictor outputs and $s_k = \overline{E}_\theta(x_L)$ are EMA encoder targets.

---

### Architecture

**Encoder $E_\theta$:** Vision Transformer (ViT), three variants:

| Model | Params | Resolution | Notes |
|---|---|---|---|
| ViT-L/16₂₂₄ | 300M | 224 | Batch 3072 |
| ViT-H/16₂₂₄ | 630M | 224 | Batch 3072 |
| ViT-H/16₃₈₄ | 630M | 384 | Batch 2400 |

Video tokenization: 3D convolution with filters $2 \times 16 \times 16$, temporal stride 2, spatial stride 16 → tubelets of shape $8 \times 14 \times 14 \times d$ for 16 frames at 224px. Flattened to 1568 tokens. Absolute 3D sin-cos positional embeddings added.

**Predictor $P_\phi$:** Narrow ViT — 12 transformer blocks, embedding dimension 384. Shared across all encoder scales. Takes as input the x-encoder's output tokens concatenated with learnable mask tokens (sum of a shared learnable vector and a 3D sin-cos positional embedding for each masked position).

**No [CLS] token** is used in V-JEPA pretraining.

---

### 3D Multi-Block Masking

Two types of masks are sampled per clip and their union taken:

| Type | Num. Blocks | Spatial Scale | Coverage |
|---|---|---|---|
| Short-range | 8 | 15% of frame | ~15% × 8 blocks |
| Long-range | 2 | 70% of frame | ~70% × 2 blocks |

- **Block aspect ratio:** uniformly sampled in [0.75, 1.5] for each block
- **Temporal extent:** blocks span the **full** temporal dimension of the clip
- **Combined masking ratio:** ~90% of all spatio-temporal patches

The context view $x$ is the complement (unmasked ~10% of patches). This high masking ratio with full temporal coverage forces the predictor to reason about temporal dynamics and object persistence, rather than copying visible frame information.

**Why multi-block beats alternatives (Table 4 ablation):**

| Masking strategy | K400 | SSv2 | IN1K |
|---|---|---|---|
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| **multi-block** | **72.9** | **67.4** | **72.8** |

Random-tube masking degrades to low-quality representations; restricting context to the first frames (causal) also hurts, since the model cannot access spatial context from later frames.

---

### Pretraining Dataset — VideoMix2M

| Source | Samples | Notes |
|---|---|---|
| HowTo100M (HT) | ~1.1M | Instructional YouTube videos |
| Kinetics-400/600/700 (K710) | ~700K | Action recognition videos |
| Something-Something v2 (SSv2) | ~168K | Hand-object interaction |
| **Total (VideoMix2M)** | **~2M** | Validation splits excluded |

Training: 90,000 iterations, batch size 3072 (L/16 and H/16₂₂₄) or 2400 (H/16₃₈₄). Input: 16 frames, temporal stride 4 (≈3 seconds). Multi-mask strategy: two masks sampled per clip, sharing one y-encoder forward pass.

**Training schedule:** Learning rate linearly warmed up from $2 \times 10^{-4}$ to $6.25 \times 10^{-4}$ over 12,000 iterations, then cosine decay to $10^{-6}$. EMA momentum from 0.998 → 1.0 (linearly). Weight decay from 0.04 → 0.4 (linearly).

---

### Attentive Probing

Standard linear probing fails for unnormalized JEPA representations (no reason the encoder produces a linearly separable subspace). V-JEPA uses a learned cross-attention pooling layer as the probe:

$$\text{pool}(s_{1:L}) = \text{softmax}\left(\frac{q^\top \mathbf{W}_k s_i}{\sum_j \exp(q^\top \mathbf{W}_k s_j)}\right) \mathbf{W}_v s_i$$

The output is added back to query token $q$ (residual), then fed to a 2-layer MLP with GeLU, LayerNorm, and a linear classifier. Probe architecture: 12 heads × 12 dim. Parameters trained jointly with the linear classifier; encoder frozen. This yields +17.3 pts K400 and +16.1 pts SSv2 over average pooling (Table 3).

---

## Experimental Results

### Feature vs. Pixel Prediction (Table 1 — ViT-L/16, 90K iterations)

| Target | K400 (frozen) | SSv2 (frozen) | IN1K (frozen) | K400 (fine-tuned) |
|---|---|---|---|---|
| Pixels (MSE) | 68.6 | 66.0 | 73.3 | 85.4 |
| **Features (V-JEPA)** | **73.7** | **66.2** | **74.8** | **85.6** |

Feature prediction consistently improves frozen performance (+5.1 K400, +0.2 SSv2, +1.5 IN1K) while matching fine-tuning performance.

### Comparison vs. Pixel Prediction Methods (Table 5 — ViT-L/16 family, frozen + fine-tuning)

| Method | #Samples Seen | K400 (fz) | SSv2 (fz) | IN1K (fz) | K400-ft | SSv2-ft |
|---|---|---|---|---|---|---|
| OmniMAE | 2400M | 65.6 | 60.5 | **75.1** | 84.0 | 74.2 |
| VideoMAE | 410M | 77.8 | 65.4 | 71.1 | 85.4 | 74.3 |
| Hiera | 1500M | 75.5 | 64.2 | 68.9 | **87.3** | **75.1** |
| **V-JEPA** | **270M** | **80.8** | **69.5** | 74.8 | 85.6 | 75.1 |

V-JEPA outperforms all pixel prediction baselines in frozen evaluation using 6–9× fewer training samples. Fine-tuning is competitive with Hiera (larger model).

### Comparison vs. State-of-the-Art (Table 6 — frozen, attentive probe)

Best V-JEPA results (ViT-H/16):

| Benchmark | V-JEPA | Best Video Baseline | Best Image Baseline |
|---|---|---|---|
| K400 | 82.0 | 79.8 (VideoMAEv2, 1100M params) | 83.4 (DINOv2, 1100M) |
| SSv2 | 71.4 | 66.2 (VideoMAEv2) | 50.6 (DINOv2) |
| AVA | 25.8 | 20.7 (VideoMAEv2) | 23.2 (OpenCLIP) |
| IN1K | 75.9 | 72.3 (VideoMAEv2) | 86.2 (DINOv2) |

V-JEPA achieves **+5 pts SSv2, +2 pts K400, +5 pts AVA** over best video baseline (VideoMAEv2). Gap vs. large image models (DINOv2) on appearance tasks reflects lack of internet-scale image pretraining data.

### Label Efficiency (Table 7 — frozen, varying labeled examples)

At 5% labeled data (≈29 samples/class for K400, ≈48 for SSv2):

| Method | K400@5% | SSv2@5% |
|---|---|---|
| MVD | 62.6 ± 0.2 | 42.9 ± 0.8 |
| VideoMAE | 62.3 ± 0.3 | 41.4 ± 0.8 |
| VideoMAEv2 | 37.0 ± 0.3 | 28.0 ± 1.0 |
| **V-JEPA (ViT-H/16)** | **67.0 ± 0.2** | **51.9 ± 0.3** |
| **V-JEPA (ViT-H/16₃₈₄)** | **68.2 ± 0.2** | **54.0 ± 0.2** |

Reducing available labels by 10× causes only 12–14% drop for V-JEPA, vs. 14–30% for pixel prediction baselines.

### Sample Efficiency (Table 16)

V-JEPA processes 210M samples vs. 1600M–1900M for DINOv2/VideoMAEv2, and 39,000M for OpenCLIP — achieving competitive performance with orders-of-magnitude fewer samples seen.

---

## Limitations & Open Questions

### Limitations
- **Small pretraining dataset (2M videos):** Much smaller than internet-scale image datasets used by DINOv2 (142M images) or OpenCLIP (2B images). The authors hypothesize that existing public video datasets lack the diversity of web-scale image data, limiting performance on static appearance tasks (e.g., ImageNet gap vs. DINOv2).
- **No language supervision:** Models are unaware of semantic category labels during pretraining; image classification performance is lower than language-aligned models.
- **Short temporal window (≈3 seconds):** Pretraining clips are short; long-range temporal dependencies (minutes-scale) are not captured.
- **Narrow predictor may bottleneck quality:** The small 22M-param predictor may limit the richness of what V-JEPA can predict about complex scenes with many independently moving objects.
- **Absolute sincos positional embeddings:** Later replaced by 3D-RoPE in V-JEPA 2, which improves training stability at larger scale.

### Open Questions
- Does V-JEPA benefit from **much larger video datasets** (e.g., 1M+ hours as in V-JEPA 2)? *(Answered affirmatively by Assran et al. 2025)*
- Can the approach be extended to **longer temporal contexts** (minutes, hours)?
- How does V-JEPA's representation compare to language-supervised models on **semantic reasoning tasks** (VQA, captioning)?
- Is the small predictor a bottleneck, and does **scaling the predictor** help? *(V-JEPA 2 keeps predictor fixed at ~22M despite scaling encoder to 1B)*
- Can V-JEPA representations be used for **robot planning** without further training? *(V-JEPA 2-AC shows this requires action-conditioned post-training)*

---

## See Also

- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md)
- [Video Self-Supervised Learning](../concepts/video-ssl.md)
- [V-JEPA 2 (Assran et al. 2025)](assran_2025_vjepa2.md) — scales this recipe to 1B params, 1M+ hours, robot adaptation
