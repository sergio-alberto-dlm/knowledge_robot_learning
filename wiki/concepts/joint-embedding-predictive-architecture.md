---
title: Joint Embedding Predictive Architecture (JEPA)
type: concept
tags: [self-supervised-learning, representation-learning, world-models, yann-lecun]
related: [concepts/video-ssl.md, concepts/latent-world-models.md, concepts/sigreg.md, concepts/successor-features.md, concepts/zero-shot-unsupervised-rl.md, papers/bardes_2024_vjepa.md, papers/assran_2025_vjepa2.md, papers/balestriero_2025_lejepa.md, papers/bagatella_2025_tdjepa.md, open_questions/jepa-rl-humanoid-grasping.md, papers/zhou_2024_dino_wm.md, concepts/pretrained-visual-representations.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/V-JEPA.pdf, raw/papers/pdf/V-JEPA2.pdf, raw/papers/pdf/LeJEPA.pdf, raw/papers/pdf/TD-JEPA.pdf]
---

# Joint Embedding Predictive Architecture (JEPA)

## Definition

JEPA is a self-supervised learning framework proposed by LeCun (2022) in which a model learns by making predictions in a **learned latent representation space** rather than in pixel space. The core idea: given two views $x$ and $y$ of the same signal (e.g., unmasked and masked video), an encoder $E_\theta$ processes the context view, and a predictor $P_\phi$ predicts the representation of the target view $y$:

$$\min_{\theta, \phi} \| P_\phi(E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|$$

The stop-gradient $\text{sg}(\cdot)$ and EMA teacher $\bar{\theta}$ prevent representation collapse (the trivial solution where all representations collapse to a constant).

## Key Properties

- **Predicts in representation space, not pixel space:** Unlike generative models (e.g., diffusion, VAE), JEPA does not require reconstructing every detail. It focuses on predictable, semantically meaningful structure while ignoring unpredictable details (e.g., exact position of every leaf on a tree).
- **Action-free pretraining:** The model learns from raw observations without requiring action labels or reward signals.
- **Scalable:** SSL from observation data scales to internet-scale data (billions of images, millions of hours of video).

## Variants

| Model | Modality | Masking Strategy | Anti-collapse mechanism |
|---|---|---|---|
| I-JEPA (Assran et al. 2023) | Images | Spatial block masking | EMA teacher + stop-gradient |
| V-JEPA (Bardes et al. 2024) | Video | Spatiotemporal multiblock masking | EMA teacher + stop-gradient |
| V-JEPA 2 (Assran et al. 2025) | Video + Robot data | Same + 3D-RoPE, VM22M scale | EMA teacher + stop-gradient |
| **LeJEPA** (Balestriero & LeCun, 2025) | Images (any arch.) | Multi-view global/local crops | **SIGReg** (no EMA, no stop-grad) |
| **TD-JEPA** (Bagatella et al., 2025) | RL states (pixels or prop.) | Policy-conditioned, multi-step TD | Target networks + orthonormality reg |

## Distinction from Other SSL Approaches

- vs. **Contrastive learning (SimCLR, MoCo):** JEPA does not require data augmentation pairs or negative samples.
- vs. **Generative SSL (MAE, VideoMAE):** JEPA predicts in representation space, not pixel space — avoids wasting capacity on unpredictable details.
- vs. **VICReg / DINO:** JEPA uses a directional prediction objective rather than a symmetric variance-covariance regularization or knowledge distillation on full views.

## Cognitive Motivation

LeCun motivates JEPA from predictive coding theories of brain function (Rao & Ballard, 1999; Friston, 2010): the brain maintains an internal world model that predicts future sensory states and uses prediction error to learn. JEPA operationalizes this in deep learning.

## See Also

- [Video Self-Supervised Learning](video-ssl.md)
- [Latent World Models](latent-world-models.md)
- [SIGReg](sigreg.md) — the collapse-free regularizer introduced in LeJEPA
- [V-JEPA (Bardes et al. 2024)](../papers/bardes_2024_vjepa.md) — original video JEPA instantiation
- [V-JEPA 2 (Assran et al. 2025)](../papers/assran_2025_vjepa2.md) — scaled version with robot adaptation
- [LeJEPA (Balestriero & LeCun, 2025)](../papers/balestriero_2025_lejepa.md) — theoretically grounded, heuristic-free JEPA
- [TD-JEPA (Bagatella et al., 2025)](../papers/bagatella_2025_tdjepa.md) — JEPA with TD bootstrapping for zero-shot unsupervised RL; predictor recovers successor features
- [Successor Features & Successor Measures](successor-features.md) — theoretical foundation that TD-JEPA's predictor recovers
- [Zero-Shot / Unsupervised RL](zero-shot-unsupervised-rl.md) — the problem setting TD-JEPA solves
- [JEPA World Models as Physics Priors for RL](../open_questions/jepa-rl-humanoid-grasping.md) — open research hypothesis: use a JEPA world model, not a VLA, as the prior for RL on humanoid grasping
- [Zhou et al. (2024) — DINO-WM](../papers/zhou_2024_dino_wm.md) — action-conditioned predictor on frozen DINOv2 (a JEPA-style self-distilled encoder), predicting in latent rather than pixel space
- [Pretrained Visual Representations for Robotics](pretrained-visual-representations.md)
