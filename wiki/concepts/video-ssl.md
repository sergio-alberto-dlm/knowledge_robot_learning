---
title: Video Self-Supervised Learning
type: concept
tags: [self-supervised-learning, video-representation, vision-transformer, pretraining]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/latent-world-models.md, papers/bardes_2024_vjepa.md, papers/assran_2025_vjepa2.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/V-JEPA.pdf, raw/papers/pdf/V-JEPA2.pdf]
---

# Video Self-Supervised Learning

## Definition

Video Self-Supervised Learning (Video SSL) refers to methods that learn visual representations from unlabeled video data using pretext tasks derived from the video signal itself (temporal ordering, masking, contrastive objectives, etc.), without requiring human annotations.

## Key Pretext Tasks

| Task | Method | Predicts |
|---|---|---|
| Masked frame reconstruction | VideoMAE, MAE | Pixel values of masked patches |
| Masked representation prediction | V-JEPA, V-JEPA 2 | Latent representations of masked patches |
| Contrastive temporal | MoCo-v3 (video) | Agreement across augmented clips |
| Future frame prediction | SVP, FitVid | Next frame pixels |

## Masking Strategies for Video

**Spatial masking:** Randomly mask patches within each frame independently. Simple but allows temporal leakage (unmasked frames give away masked content).

**Spatiotemporal masking (Multiblock):** Mask contiguous spatiotemporal blocks across multiple time steps. Used in V-JEPA and V-JEPA 2 — forces the model to reason about motion and temporal dynamics.

- Spatial mask scale: [0.15, 0.7] of frame area
- Temporal mask scale: [1.0, 1.0] (full temporal extent)
- Tubelet size: 2×16×16 (T×H×W)

## Scaling Laws

V-JEPA 2 demonstrates consistent scaling in video SSL:
- Data: 2M → 22M videos: +1.0 pt average accuracy
- Model: 300M → 1B params: +1.5 pt average accuracy
- Training duration: 90K → 252K iterations: +0.8 pt
- Resolution: 256px/16fr → 384px/64fr: +1.0 pt

## Position Embeddings for Video

**Absolute sincos (V-JEPA 1):** Fixed position embeddings concatenated to patch features. Simple but may not generalize to different resolutions.

**3D-RoPE (V-JEPA 2):** Rotary Position Embedding applied separately to temporal, height, and width feature dimensions (feature dim split into three equal segments). Benefits:
- Improved training stability at scale (ViT-g+)
- Better generalization across resolutions and durations
- Encodes relative positions rather than absolute positions

## Benchmark Tasks

**Motion understanding (requires temporal reasoning across frames):**
- Something-Something v2 (SSv2): hand-object interaction videos
- Diving-48: diving action classification
- Jester: hand gesture recognition

**Appearance understanding (solvable from single frame):**
- Kinetics-400/600/700
- COIN (instructional videos)
- ImageNet (static images)

**Action anticipation:**
- Epic-Kitchens-100 (EK100): predict action 1 second before it starts (egocentric cooking)

## Evaluation Protocol

**Attentive probe (V-JEPA standard):** A single cross-attention layer with a learnable query token pools the frozen encoder's token sequence into one vector; this is fed through a 2-layer MLP (GeLU + LayerNorm) and linear classifier. V-JEPA showed this outperforms average linear pooling by +17 pts K400 and +16 pts SSv2, and was adopted as the default frozen evaluation for V-JEPA and V-JEPA 2. V-JEPA 2 uses a deeper 4-layer version.

**Linear probe:** Trains a linear layer on average-pooled features. Simple but understimates JEPA representations since the encoder output is not constrained to be linearly separable.

## Sample Efficiency

V-JEPA processes ~210M samples (90K iterations × batch 3072 × 1 clip) vs. 1600M–1900M for DINOv2/VideoMAEv2 and 39,000M for OpenCLIP — competitive performance with orders-of-magnitude fewer training samples.

## See Also

- [Joint Embedding Predictive Architecture](joint-embedding-predictive-architecture.md)
- [Latent World Models](latent-world-models.md)
- [V-JEPA (Bardes et al. 2024)](../papers/bardes_2024_vjepa.md) — original video feature prediction paper
- [V-JEPA 2 (Assran et al. 2025)](../papers/assran_2025_vjepa2.md) — scaled version with 1M+ hours and robot adaptation
