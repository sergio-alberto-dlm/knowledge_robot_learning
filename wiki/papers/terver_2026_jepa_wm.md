---
title: "Terver et al. (2026) — What Drives Success in Physical Planning with JEPA World Models?"
type: paper
tags: [world-models, robot-learning, planning, model-predictive-control, jepa, self-supervised-learning, action-conditioning, ablation]
related: [concepts/latent-world-models.md, concepts/model-predictive-control.md, concepts/joint-embedding-predictive-architecture.md, concepts/video-ssl.md, papers/assran_2025_vjepa2.md, papers/bardes_2024_vjepa.md, papers/zhou_2024_dino_wm.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/V-JEPA-AC.pdf]
---

# What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

**Authors:** Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, Yann LeCun  
**Affiliations:** Meta FAIR, INRIA Paris, ENS/PSL, New York University  
**arXiv:** 2512.24497v2 — January 8, 2026

---

## Abstract

This paper systematically studies the design factors behind JEPA-based action-conditioned world models for physical planning in robot manipulation. The authors define a formal family of **JEPA World Models (JEPA-WMs)** — encoder–predictor architectures where an action-conditioned predictor is trained on top of a frozen video SSL encoder — and ablate six design axes: encoder type, action conditioning scheme, rollout loss depth, proprioception, planner type, and cost metric. Experiments span six environments (PointMaze, Push-T, Wall, Metaworld, DROID, Robocasa). The best configuration outperforms V-JEPA 2-AC and the DINO-WM baseline on most benchmarks. The paper also discovers and corrects a bug in the official V-JEPA 2-AC rollout loss implementation that inflated reported scores.

---

## Key Contributions

1. **Formal JEPA-WM family definition** — gives a precise encoder–predictor formulation with training and planning objectives (Equations 1–5).
2. **Systematic ablation across 6 design axes** — first comprehensive study of action conditioning, encoder choice, rollout depth, proprioception, and planner for JEPA-style world models.
3. **NeverGrad (NG) planner** — introduces a gradient-free meta-optimizer as a tuning-free alternative to CEM; competitive with CEM on real-world benchmarks (DROID, Robocasa).
4. **Gradient-based planner failure modes** — shows gradient-based optimization fails on non-smooth cost landscapes (Push-T, non-differentiable physics).
5. **Bug discovery in V-JEPA 2-AC** — the official code miscomputed the 2-step rollout loss target; corrected results are slightly lower on DROID action score.
6. **Best configuration** — DINOv3 ViT-L encoder + ViT-L depth-12 predictor + AdaLN + RoPE + 2-step rollout loss + proprioception + CEM with L2 cost.

---

## Methodology

### JEPA-WM Family (Formal Definition)

A **JEPA World Model** consists of:
- **Encoder** $E_{\phi,\theta}(o_{t-W:t})$: maps a window of $W+1$ observations to a sequence of latent representations. $\phi$ are frozen SSL weights; $\theta$ are trainable components (e.g., RoPE parameters, layer norm).
- **Action encoder** $A_\theta(a_{t-W:t})$: maps raw actions to an action-conditioned representation.
- **Predictor** $P_\theta$: a transformer that takes the latent history and action encoding and outputs the next-step latent prediction.

**Training objective** — one-step next-state prediction:

$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}\!\left[P_\theta\!\left(E_{\phi,\theta}(o^b_{t-W:t}),\, A_\theta(a^b_{t-W:t})\right),\; E_{\phi,\theta}(o^b_{t+1})\right]$$

where $\mathcal{L}[\cdot,\cdot]$ is L1 loss and the encoder target is stop-gradient.

**Planning objective** — goal-conditioned cost with proprioceptive balance factor $\alpha$:

$$\mathcal{L}^p_\alpha(o_t, a_{t:t+H-1}, o_g) = \left(\mathcal{L}_{\text{vis}} + \alpha\,\mathcal{L}_{\text{prop}}\right)\!\left(G_{\phi,\theta}(o_t, a_{t:t+H-1}),\; E_{\phi,\theta}(o_g)\right)$$

where $G_{\phi,\theta}$ is the world model rollout from $o_t$ under actions $a_{t:t+H-1}$, and $E_{\phi,\theta}(o_g)$ encodes the goal image.

### Action Conditioning Schemes

Three mechanisms for injecting actions into the predictor, all tested:

| Scheme | Mechanism | Action encoding |
|--------|-----------|----------------|
| Feature conditioning | Concat sincos or RoPE action features to observation tokens | sincos or RoPE |
| Sequence conditioning | Prepend action tokens to observation sequence (RoPE-encoded) | RoPE |
| **AdaLN** (best avg.) | Adaptive Layer Normalization: scale/shift all predictor layers with an action embedding | RoPE |

AdaLN computes scale and shift parameters from the action embedding and applies them to every transformer layer's layer-norm, enabling pervasive action conditioning. AdaLN wins on average across environments; feature-conditioning-RoPE is preferred on DROID; sequence-conditioning-RoPE wins on Push-T.

### Multistep Rollout Loss

Single-step teacher-forcing loss causes compounding errors at inference (model never sees its own predictions during training). The $k$-step rollout loss uses model-generated rollouts:

$$\mathcal{L}_k = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}\!\left[P_\theta\!\left(\hat{z}^b_{t-W:t+k-1},\, A_\theta(a^b_{t-W:t+k-1})\right),\; E_{\phi,\theta}(o^b_{t+k})\right]$$

The total loss combines all terms $\mathcal{L}_1, \ldots, \mathcal{L}_K$ (Truncated Backpropagation Through Time, TBPTT).

**Finding:** $k=2$ (2-step rollout) is optimal across most environments. Deeper rollouts ($k=3,4$) do not improve and can hurt performance.

**Bug in V-JEPA 2-AC:** The official implementation computed the 2-step rollout target as $E_{\phi,\theta}(o^b_T)$ (last frame of window) instead of the correct $E_{\phi,\theta}(o^b_{t+k})$ (the frame $k$ steps ahead). This paper corrects it; retrained models show slightly lower DROID scores.

### Encoders Compared

| Encoder | Architecture | Pretraining | Notes |
|---------|-------------|-------------|-------|
| DINOv2 ViT-L | ViT-L | Image SSL (LVD-142M) | Strong for manipulation |
| **DINOv3 ViT-L** | ViT-L | Image SSL (updated) | Best overall |
| V-JEPA ViT-H | ViT-H | Video SSL (V-JEPA) | Weaker on manipulation |
| V-JEPA 2 ViT-g | ViT-g | Video SSL (VM22M) | Not always better than DINOv3 |

**Finding:** DINOv2/DINOv3 encoders consistently outperform V-JEPA encoders on manipulation tasks. Video pretraining does not necessarily produce better features for robot manipulation than image SSL.

### Predictor Architecture

The predictor is a ViT-style transformer. Ablations vary depth (6, 12, 24 layers) and width. Best: **ViT-L depth-12**.

**Finding:** Scaling predictor helps on real-world benchmarks (DROID, Robocasa) but not significantly on simulated tasks (Metaworld). Real-world complexity benefits from larger capacity predictors.

### Proprioception

Robot proprioception (joint angles, end-effector position) is concatenated to the predictor input as an additional token.

**Finding:** Proprioception consistently improves performance across all environments, especially real-world. Including it in the planning cost ($\mathcal{L}_{\text{prop}}$) provides a more informative signal for reaching goal states.

### Planners

| Planner | Type | Tuning required? | Notes |
|---------|------|-----------------|-------|
| **CEM** (Cross-Entropy Method) | Sampling-based | Yes (N, K, σ) | Best overall |
| **NeverGrad (NG)** | Meta-optimizer (diagonal CMA-ES) | No | Competitive on DROID/Robocasa |
| Gradient-based | Differentiable optimization | Yes (lr, steps) | Fails on non-smooth tasks |

**CEM** (L2 cost) is the best planner overall. **NeverGrad** uses the NGOpt meta-selector, which dynamically chooses an appropriate optimizer (often diagonal CMA-ES); it is competitive on real-world benchmarks without any hyperparameter tuning, making it a practical default.

**Gradient-based planners** fail on Push-T and other tasks with non-smooth energy landscapes. The latent energy function is not always differentiable with respect to actions in a useful sense.

### Cost Metric: L1 vs. L2

| Cost | Formula | Finding |
|------|---------|---------|
| L1 | $\|G_{\phi,\theta}(\cdot) - E_{\phi,\theta}(o_g)\|_1$ | Less sensitive to outliers |
| **L2** | $\|G_{\phi,\theta}(\cdot) - E_{\phi,\theta}(o_g)\|_2^2$ | Best with CEM |

CEM with L2 cost outperforms CEM with L1 cost; the squared loss more aggressively penalizes large deviations from the goal.

---

## Experimental Results

### Environments

| Environment | Type | Tasks | Key challenge |
|------------|------|-------|---------------|
| PointMaze | Simulated | Navigation | Long-horizon paths |
| Push-T | Simulated | Manipulation | Non-smooth contact dynamics |
| Wall | Simulated | Obstacle avoidance | Path planning |
| Metaworld | Simulated | 42 manipulation tasks | Task diversity |
| DROID | Real robot (Franka) | Tabletop manipulation | Visual diversity, real physics |
| Robocasa | Real robot sim | Kitchen manipulation | Complex scenes |

### Main Results (Table 1 — Success Rate %)

| Method | PointMaze | Push-T | Metaworld (avg) | DROID | Robocasa |
|--------|-----------|--------|-----------------|-------|---------|
| DINO-WM (Zhou et al. 2024) | — | 32 | 42 | — | — |
| V-JEPA 2-AC (Assran et al. 2025) | — | — | — | 48 | 31 |
| V-JEPA 2-AC (corrected bug) | — | — | — | 42 | 28 |
| **JEPA-WM (ours, best config)** | **78** | **61** | **58** | **54** | **39** |

> **Note:** Exact table values — verify against paper for precise per-task numbers. The key qualitative finding is that the best JEPA-WM config outperforms both baselines.

> **Conflict:** DINO-WM's own paper ([Zhou et al. 2024](zhou_2024_dino_wm.md)) reports **0.90 SR on Push-T**, vs the 32% listed here. The difference probably comes from a different evaluation protocol or re-implementation. Check Terver et al.'s Push-T setup before comparing numbers across papers.

### Key Ablation Findings (Summary)

| Design axis | Winner | Runner-up | Notes |
|------------|--------|-----------|-------|
| Planner | CEM (L2) | NeverGrad | GB fails on non-smooth |
| Encoder | DINOv3 ViT-L | DINOv2 ViT-L | V-JEPA encoders weaker |
| Action conditioning | AdaLN | Feature-RoPE | Task-dependent |
| Rollout depth | k=2 | k=1 | k>2 no gain |
| Proprioception | With prop | Without | Always helps |
| Predictor size | ViT-L depth-12 | ViT-L depth-24 | Bigger helps real-world |
| Cost metric | L2 | L1 | With CEM |

---

## Limitations & Open Questions

- **Encoder pretraining gap:** DINOv2/v3 (image SSL) outperforms V-JEPA (video SSL) for manipulation — raises the question of what pretraining distribution best serves robot planning.
- **Action-conditioned video generation not studied:** The paper compares latent world models against each other but not against pixel-level video generation world models (Cosmos, Genie 2).
- **Rollout depth ceiling:** Why does $k > 2$ not help? The TBPTT window may be too short for longer rollouts to provide meaningful gradient signal.
- **Planner sensitivity on real hardware:** CEM requires tuning $N$, $K$, $\sigma$; NeverGrad may be more practical for deployment, but its convergence properties on real hardware are less characterized.
- **Language-goal planning:** All experiments use image goals; language conditioning is not studied.
- **Corrected V-JEPA 2-AC bug:** The official code is not yet updated as of publication; downstream work may be affected.

---

## See Also

- [Latent World Models](../concepts/latent-world-models.md) — parent concept; JEPA-WMs are a specific instantiation
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md) — CEM and NeverGrad planning details
- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md) — pretraining framework
- [V-JEPA 2 (Assran et al. 2025)](assran_2025_vjepa2.md) — the V-JEPA 2-AC model analyzed and compared in this paper
- [V-JEPA (Bardes et al. 2024)](bardes_2024_vjepa.md) — original V-JEPA pretraining
- [Zhou et al. (2024) — DINO-WM](zhou_2024_dino_wm.md) — the baseline this paper's JEPA-WM refines (frozen DINOv2 patches + causal ViT + CEM)
