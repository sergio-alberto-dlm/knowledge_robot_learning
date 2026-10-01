---
title: "Zhou et al. (2024) — DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
type: paper
tags: [latent-world-models, world-models, dinov2, pretrained-visual-representations, model-predictive-control, cem, zero-shot-planning, offline-learning, goal-conditioned, manipulation, deformable-objects]
related: [concepts/latent-world-models.md, concepts/model-predictive-control.md, concepts/pretrained-visual-representations.md, concepts/joint-embedding-predictive-architecture.md, papers/terver_2026_jepa_wm.md, papers/assran_2025_vjepa2.md, papers/hansen_2025_newt.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/DINO_WM.pdf]
---

# Zhou et al. (2024) — DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

**arXiv:** 2411.04983v2 (v1 Nov 2024; v2 1 Feb 2025, cs.RO)
**Authors:** Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto (NYU Courant; Meta AI)
**Project page:** dino-wm.github.io

## Abstract

DINO-WM is a **task-agnostic world model** trained on **offline trajectories** that models visual dynamics in the space of **frozen DINOv2 patch features**, without reconstructing pixels. A causal ViT predictor forecasts the next frame's patch embeddings from past embeddings and actions. At test time, tasks are posed as **visual goal reaching**: model predictive control with the Cross-Entropy Method (CEM) finds action sequences whose predicted final latent matches the goal image's latent. Nothing else is needed: no expert demonstrations, reward model, or pre-learned inverse model. Across six suites (maze navigation, wall navigation, reaching, Push-T, rope and granular manipulation) it matches or beats DreamerV3, TD-MPC2 and IRIS trained on the same reward-free data, with the largest gains on contact-rich and deformable manipulation.

## Motivation

The authors argue a useful world model should:
1. be **trainable offline** from pre-collected trajectories;
2. support **test-time behaviour optimisation** rather than a fixed feed-forward policy;
3. be **task-agnostic**, with no rewards, discounts or termination signals baked into its latents.

Existing latent world models (Dreamer, TD-MPC, IRIS) learn their encoder jointly with reconstruction or reward objectives, which ties them to a task. Pixel-space video models (diffusion) are expensive and physically unreliable. DINO-WM instead **borrows perception from a large pretrained encoder** and learns only dynamics.

## Key Contributions

1. **Pretrained patch features as the world-model state.** The frozen DINOv2 patch embeddings $z_t \in \mathbb{R}^{N\times E}$ (not a single CLS vector) serve as a spatial, object-centric prior.
2. **A frame-level causal ViT transition model** that predicts all patches of the next frame at once, conditioned on actions and proprioception.
3. **Decoder-free training.** A pixel decoder is trained separately *only for visualisation*. Backpropagating its loss into the predictor hurts planning.
4. **Zero-shot goal-reaching via latent MPC/CEM**, with an evaluation on six environments plus three generalisation families (unseen wall layouts, object shapes, particle counts).

## Methodology

### Components

| Component | Form | Trained? |
|---|---|---|
| Observation model | $z_t = \mathrm{enc}_\theta(o_t)$, DINOv2 patch features: 196×196 input → 14×14 patches × 384 dims (the dimensions match ViT-S/14; the paper does not name the variant) | **Frozen** |
| Transition model | $z_{t+1} \sim p_\theta(z_{t+1}\mid z_{t-H:t}, a_{t-H:t})$, a ViT with no tokenisation layer (decoder-only), depth 6, 16 heads, MLP 2048, **~19M params** | Yes |
| Action / proprio encoder | MLP $\phi$ maps a $K$-dim action (emb. dim 10) and proprioception, **concatenated to every patch vector** | Yes |
| Decoder (optional) | Transposed-conv stack (VQ-VAE-2 style), $\mathcal{L}_{rec} = \|q_\theta(z_t) - o_t\|^2$ | Separately, visualisation only |

### Frame-level causal attention

Each patch $z_t^i$ attends to **all patches of all previous frames** $\{z^i_{t-H:t-1}\}_{i=1}^N$. The whole next frame is predicted at once, unlike IRIS, which predicts token by token within a frame. The authors argue this captures global structure and temporal dynamics better.

### Training objective (teacher forcing, latent only)

$$\mathcal{L}_{pred} = \big\|p_\theta\big(\mathrm{enc}_\theta(o_{t-H:t}),\,\phi(a_{t-H:t})\big) - \mathrm{enc}_\theta(o_{t+1})\big\|^2$$

- Trajectories are sliced into segments of length $H+1$, with a loss on each of the $H$ predicted frames.
- History $H=3$ for PointMaze, Reacher, Push-T and PushObj, and $H=1$ for Wall, Rope and Granular. **Frameskip** is 5 (1 for Rope and Granular) so that consecutive targets differ meaningfully.
- Shared hyperparameters: AdamW, predictor LR 5e-5, action-encoder LR 5e-4, decoder LR 3e-4, 100 epochs, batch 32.
- There is **no multi-step rollout loss** (contrast with JEPA-WM's 2-step rollout; see [Latent World Models](../concepts/latent-world-models.md)).

### Planning

- The goal $o_g$ and current observation $o_0$ are encoded, the actions are rolled out in latent space, and the cost is **terminal latent MSE**, $\mathcal{C} = \|\hat z_T - z_g\|^2$.
- **CEM:** sample $N$ action sequences from a Gaussian, keep the top-$K$, refit the mean and covariance, and iterate. The first $k$ actions are executed, then the loop replans in receding horizon (**MPC**).
- **Gradient descent** through the differentiable model is also tried: $a_t \leftarrow a_t - \eta\,\partial\mathcal{C}/\partial a_t$.

## Environments and Data

| Env | Task | Offline data |
|---|---|---|
| **PointMaze** (D4RL) | 2-DoF force-actuated ball reaches a goal | 2,000 random trajectories |
| **Wall** (custom) | Navigate through a door between two rooms | 1,920 random trajectories × 50 steps |
| **Reacher** (DMC) | Match a full 2-joint arm pose, not just the end-effector | 3,000 × 100 steps |
| **Push-T** | Push agent and T-block to a target pose within 25 steps | 18,500 **replayed expert trajectories with noise** |
| **Rope** (Nvidia Flex, XArm) | Deform a rope to a goal configuration | 1,000 × 20 random steps |
| **Granular** (Flex, XArm) | Gather ~100 particles into a square shape | 1,000 × 20 random steps |

All observations are 224×224 RGB. **Generalisation families:**
- **WallRandom:** unseen wall and door positions; 10,240 training trajectories.
- **PushObj:** trained on 4 block shapes, tested on 2 unseen ones (Tetris-like and "+"); 20,000 trajectories.
- **GranularRandom:** different particle counts, evaluated with the model trained on fixed counts.

> **Nuance:** The paper claims solutions "without expert demonstrations", but the Push-T offline dataset is built by replaying **expert** trajectories with noise. The *planner* uses no demonstrations, but data coverage on Push-T is expert-derived.

## Results

### Planning with offline world models (Table 1)

Success rate (SR ↑) is over 50 goal/start pairs. Rope and Granular report Chamfer distance (CD ↓) over 10 instances.

| Model | Maze SR | Wall SR | Reach SR | PushT SR | Rope CD | Granular CD |
|---|---|---|---|---|---|---|
| IRIS | 0.74 | 0.04 | 0.18 | 0.32 | 1.11 | 0.37 |
| DreamerV3 | **1.00** | **1.00** | 0.64 | 0.30 | 2.49 | 1.05 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |
| **DINO-WM** | 0.98 | 0.96 | **0.92** | **0.90** | **0.41** | **0.26** |

- DINO-WM is on par with DreamerV3 on the simple navigation tasks and far ahead on manipulation (Push-T 0.90 vs 0.32; Granular CD 0.26 vs 0.37).
- TD-MPC2 collapses because it depends on reward signals to shape its latents, and none were provided. This is a fairness caveat: every baseline was run reward-free with MPC on top of its world model, which is not how those methods are normally used.

### Encoder ablation: does the pretrained representation matter? (Table 2)

| Encoder | Maze | Wall | Reach | PushT | Rope CD | Granular CD |
|---|---|---|---|---|---|---|
| R3M (ResNet-18, human video) | 0.94 | 0.34 | 0.40 | 0.42 | 1.13 | 0.95 |
| ImageNet ResNet-18 | **0.98** | 0.12 | 0.06 | 0.20 | 1.08 | 0.90 |
| DINOv2 CLS (global) | 0.96 | 0.58 | 0.60 | 0.44 | 0.84 | 0.79 |
| **DINOv2 patches** | **0.98** | **0.96** | **0.92** | **0.90** | **0.41** | **0.26** |

All encoders solve PointMaze. As tasks need more precise spatial understanding, **global-vector encoders collapse**, and **patch features** are what make the difference. See [Pretrained Visual Representations for Robotics](../concepts/pretrained-visual-representations.md).

### Generalisation to unseen configurations (Table 3)

| Model | WallRandom SR | PushObj SR | GranularRandom CD |
|---|---|---|---|
| IRIS | 0.06 | 0.14 | 0.86 |
| DreamerV3 | 0.76 | 0.18 | 1.53 |
| R3M | 0.40 | 0.16 | 1.12 |
| ResNet | 0.40 | 0.14 | 0.98 |
| DINO CLS | 0.64 | 0.18 | 1.36 |
| **DINO-WM** | **0.82** | **0.34** | **0.63** |

- WallRandom suggests DINO-WM learns general "wall and door" concepts. PushObj stays hard for every method: with only 4 training shapes, physical parameters of new shapes are hard to infer.
- The authors attribute the GranularRandom result to patch features keeping per-patch statistics in distribution even when the global particle count changes.

> The text says "From Table 5" for these generalisation results, but they are in **Table 3** (Table 5 is the data-scaling ablation).

### Prediction quality (Table 4 / App. Table 9)

LPIPS ↓ on decoded open-loop predictions:

| Method | PushT | Wall | Rope | Granular |
|---|---|---|---|---|
| R3M | 0.045 | 0.008 | 0.023 | 0.080 |
| ResNet | 0.063 | 0.002 | 0.025 | 0.080 |
| DINO CLS | 0.039 | 0.004 | 0.029 | 0.086 |
| AVDC (diffusion) | 0.046 | 0.030 | 0.060 | 0.106 |
| **DINO-WM** | **0.007** | **0.0016** | **0.009** | **0.035** |

- SSIM follows the same ranking; for example, Granular 0.940 vs ≤0.917.
- The claimed **56% LPIPS improvement** on the hardest task is Granular, 0.035 vs 0.080.
- DINO-WM's rollouts look better even than those of models whose encoders were trained with reconstruction objectives, although DINO-WM's own predictor never sees pixels.
- **AVDC** (a text-conditioned diffusion video model) produces realistic but physically implausible frames and cannot reach exact goals. Its action-conditioned variant drifts from ground truth over long open-loop rollouts (App. A.6).

### Ablations (Appendix)

| Ablation | Finding |
|---|---|
| **Data scaling on Push-T** (Table 5) | SR 0.08 → 0.48 → 0.72 → 0.88 → 0.92 for n = 200 / 1k / 5k / 10k / 18.5k trajectories; LPIPS 0.056 → 0.005 |
| **Causal mask** (Table 6, Push-T SR) | Without mask: 0.76 / 0.36 / 0.08 for H = 1/2/3, because the model "cheats" by attending to future frames. With mask: 0.76 / 0.88 / 0.92, so longer history helps (velocity, momentum) |
| **Decoder loss into predictor** (Table 7) | 0.92 without vs **0.80 with** reconstruction gradients, so decoupling feature learning from reconstruction helps |
| **Planner** (Table 8) | MPC (CEM + replanning) ≫ open-loop CEM ≫ open-loop GD: PointMaze 0.98 / 0.80 / 0.22; Push-T 0.90 / 0.86 / 0.28; Wall 0.96 / 0.74 / – |

### Compute (App. A.8, NVIDIA A6000)

- One predictor forward pass: **0.014 s** for batch 32. The deformable-object simulator takes **3.0 s per step**.
- **A full CEM plan (100 samples × 10 iterations) takes 53 s.** This is fast compared with simulating, but far from real-time control.

## Limitations

- **Needs offline data with enough state-action coverage.** The authors suggest pairing it with exploration and online model updates.
- **Needs ground-truth actions**, so it cannot learn from action-free internet video.
- **Plans only in raw action space** over short horizons. Hierarchical planning with low-level policies is left as future work.
- **Goals must be images** of the exact target configuration, and the cost is a plain latent MSE over all patches, so irrelevant visual differences count against a plan.
- **Simulation only:** there are no real-robot experiments.
- **Slow planning** (~53 s per CEM plan), and gradient-based planning performs poorly.
- **Baseline protocol:** DreamerV3, TD-MPC2 and IRIS were used without rewards and with an external MPC planner, which disadvantages methods designed around reward learning.

## Conflicts with Other Wiki Sources

> **Conflict:** [Terver et al. (2026) — JEPA World Models](terver_2026_jepa_wm.md) lists DINO-WM at **32% on Push-T** (and 42% on Metaworld), while DINO-WM reports **90%** on its own Push-T setup. The gap most likely reflects a different evaluation protocol in Terver et al. (goal sampling, horizon, planning budget, or re-implementation), not an error in either paper. Terver's table is already marked "verify against paper" in the wiki. Compare the two protocols before quoting either number.

> **Consistent:** Terver et al. also find that gradient-based planners fail on Push-T and that CEM works best, which matches DINO-WM's Table 8. Both papers find DINOv2-family image encoders are strong for manipulation world models.

## Relation to Other Work

- **Terver et al. (2026)** treat DINO-WM as a member of the **JEPA-WM** family (frozen SSL encoder + action-conditioned predictor). Their best configuration refines it: DINOv3 ViT-L, AdaLN action conditioning, a 2-step rollout loss, and proprioception.
- **V-JEPA 2-AC** (Assran et al. 2025) follows the same recipe with a *video*-SSL encoder and a real Franka robot.
- **Newt** (Hansen et al. 2025) also uses frozen DINOv2 features, but inside a reward-driven, TD-MPC2-style self-predictive model. DINO-WM's TD-MPC2 result shows what happens when that reward signal is removed.

## See Also

- [Latent World Models](../concepts/latent-world-models.md)
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md)
- [Pretrained Visual Representations for Robotics](../concepts/pretrained-visual-representations.md)
- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md)
- [Terver et al. (2026) — JEPA World Models](terver_2026_jepa_wm.md)
- [Assran et al. (2025) — V-JEPA 2](assran_2025_vjepa2.md)
- [Hansen et al. (2025) — Newt](hansen_2025_newt.md)
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md)
