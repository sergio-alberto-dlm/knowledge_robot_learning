---
title: Pretrained Visual Representations for Robotics
type: concept
tags: [pretrained-visual-representations, dinov2, self-supervised-learning, patch-features, world-models, manipulation, representation-learning]
related: [papers/zhou_2024_dino_wm.md, papers/terver_2026_jepa_wm.md, papers/agarwal_2023_dex_func_grasp.md, papers/hansen_2025_newt.md, concepts/latent-world-models.md, concepts/joint-embedding-predictive-architecture.md, concepts/video-ssl.md, codebase/rl_lab/research-roadmap.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/DINO_WM.pdf]
---

# Pretrained Visual Representations for Robotics

## What It Is

Instead of learning perception from scratch on scarce robot data, a robot system can reuse a **frozen encoder pretrained on large image or video corpora** and train only the parts that are robot-specific: dynamics, policy, or value. The argument (Zhou et al. 2024) is that perception is a general task that transfers across environments, while robot data should be spent on what is task- and embodiment-specific.

## Common Encoders

| Encoder | Pretraining | Output used |
|---|---|---|
| ImageNet ResNet-18 | Supervised classification | Global vector |
| R3M | Time-contrastive + language on human video (Ego4D) | Global vector |
| MVP | Masked autoencoding on robot/egocentric images | Global vector |
| DINO / **DINOv2** / DINOv3 | Image self-distillation (JEPA-like) | **CLS vector *or* patch tokens** |
| I-JEPA / V-JEPA / V-JEPA 2 | Latent masked prediction (image/video) | Patch/tube tokens |

## Key Findings in the Wiki

### 1. Patch features ≫ global vectors for spatial control (DINO-WM)

The same world-model recipe was run on different frozen encoders (Zhou et al. 2024, Table 2). All of them solve easy navigation (PointMaze ~0.94–0.98 SR). On tasks that need precise spatial reasoning, however, **global-vector encoders collapse**:

| Task | ResNet-18 | R3M | DINOv2 CLS | **DINOv2 patches** |
|---|---|---|---|---|
| Wall SR | 0.12 | 0.34 | 0.58 | **0.96** |
| Push-T SR | 0.20 | 0.42 | 0.44 | **0.90** |
| Granular CD ↓ | 0.90 | 0.95 | 0.79 | **0.26** |

Even within one model (DINOv2), switching from CLS to patch tokens roughly doubles Push-T success. Patch grids keep **where things are**, which contact-rich and multi-object manipulation needs. They also generalised better to changes in particle count, because per-patch statistics stay in distribution.

### 2. Image SSL can beat video SSL for manipulation (JEPA-WM)

[Terver et al. (2026)](../papers/terver_2026_jepa_wm.md) find that DINOv2/DINOv3 encoders outperform V-JEPA and V-JEPA 2 as frozen world-model encoders on manipulation benchmarks. Internet-scale *video* pretraining does not automatically give better control features than *image* SSL.

### 3. Dense DINOv2 features for correspondence (functional grasping)

[Agarwal et al. (2023)](../papers/agarwal_2023_dex_func_grasp.md) use pixel-level DINOv2 feature matching to transfer a single annotated affordance region to new object instances. This is the same property of dense, part-level semantics, applied to perception rather than dynamics.

### 4. Frozen features inside reward-driven models (Newt)

[Newt](../papers/hansen_2025_newt.md) feeds frozen DINOv2 features into a TD-MPC2-style model trained with rewards. DINO-WM's reward-free TD-MPC2 baseline scored 0 on every task, which suggests self-predictive latents *without* reward shaping are weak. That makes a strong frozen encoder most valuable when no task signal is available.

## Design Considerations

- **Frozen vs. fine-tuned:** freezing keeps the prior intact and makes training cheap (DINO-WM's predictor is ~19M parameters). The cost is that task-irrelevant visual variation (lighting, distractors) stays in the latent and can mislead latent-distance costs.
- **Reconstruction coupling hurts:** backpropagating a pixel-reconstruction loss into the DINO-WM predictor lowered Push-T success from 0.92 to 0.80.
- **Token count vs. speed:** patch grids (e.g. 196 tokens × 384 dims) make predictors and CEM planning much heavier than global vectors. DINO-WM needs ~53 s per CEM plan.

> **Verify:** The R3M and MVP pretraining descriptions come from DINO-WM's related-work section and general knowledge. Neither paper is compiled in the wiki.

## See Also

- [Zhou et al. (2024) — DINO-WM](../papers/zhou_2024_dino_wm.md)
- [Terver et al. (2026) — JEPA World Models](../papers/terver_2026_jepa_wm.md)
- [Latent World Models](latent-world-models.md)
- [Joint Embedding Predictive Architecture (JEPA)](joint-embedding-predictive-architecture.md)
- [Video Self-Supervised Learning](video-ssl.md)
- [Agarwal et al. (2023) — Dexterous Functional Grasping](../papers/agarwal_2023_dex_func_grasp.md)
- [Hansen et al. (2025) — Newt](../papers/hansen_2025_newt.md)
- [rl_lab — Research Roadmap](../codebase/rl_lab/research-roadmap.md) — planned frozen DINOv2 ViT-S/14 patch features pooled 3×3 (3,456-d) + 9×9 depth; measured encoder cost (bf16 ViT-S at 126²: 4.84 s per 51,200 frames)
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md) — DINO repurposed as a per-object identity and semantics embedding (ROI crop, computed at reset)
