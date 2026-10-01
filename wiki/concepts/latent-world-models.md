---
title: Latent World Models
type: concept
tags: [world-models, representation-learning, planning, model-based-rl, robot-learning, multitask-rl, self-predictive, td-mpc2]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/model-predictive-control.md, concepts/video-ssl.md, papers/assran_2025_vjepa2.md, papers/terver_2026_jepa_wm.md, papers/hansen_2025_newt.md, open_questions/jepa-rl-humanoid-grasping.md, papers/zhou_2024_dino_wm.md, concepts/pretrained-visual-representations.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/V-JEPA-AC.pdf, raw/papers/pdf/newt.pdf, raw/papers/pdf/DINO_WM.pdf]
---

# Latent World Models

## Definition

A latent world model learns a dynamics function that predicts future **representations** (rather than future pixels) conditioned on current state and actions. Given encoder $E$, a world model predictor $P_\phi$ predicts:

$$\hat{z}_{t+1} = P_\phi(a_t, s_t, z_t), \quad z_t = E(x_t)$$

where $z_t$ is the latent state, $a_t$ the action, and $s_t$ proprioceptive state. The encoder is often frozen after pretraining, and only $P_\phi$ is trained on interaction data.

## Why Latent Space?

- **Computational efficiency:** Planning in latent space is orders of magnitude cheaper than planning in pixel space (no pixel-level video generation needed).
- **Abstraction:** Latent representations capture semantically meaningful structure, avoiding wasted computation on unpredictable low-level details.
- **Data efficiency:** A pretrained encoder (e.g., from internet-scale video SSL) provides rich representations; only a small amount of interaction data is needed to learn the dynamics predictor.

## Key Design Choices

### Action Conditioning
The predictor takes actions as additional inputs. In V-JEPA 2-AC, actions are 7D end-effector delta commands; in other systems (Dreamer, TDMPC), actions are discrete or continuous control signals.

### Autoregressive vs. Single-Step Prediction
Autoregressive models roll out $T$ steps by feeding predictions back as inputs. This enables long-horizon planning but accumulates errors. Rollout loss during training mitigates this.

### Loss Functions
- **Teacher-forcing loss:** Predict $z_{t+1}$ from ground-truth $z_t$.
- **Rollout loss:** Predict $z_{T+1}$ from model-generated $\hat{z}_{1:T}$.
- Both are needed: teacher-forcing for stable training, rollout loss to reduce compounding errors at inference.

## Self-Predictive / Control-Centric World Models (TD-MPC2 Family)

A **self-predictive world model** jointly trains the encoder and all predictors end-to-end, supervised entirely by the control signal — no reconstruction decoder, no SSL pretraining phase. The encoder is trained to produce representations that are useful for predicting rewards, values, and dynamics, not for reconstructing pixels.

**TD-MPC2** (Hansen et al. 2022/2024) is the canonical example:

| Component | Function |
|-----------|----------|
| Encoder h | Maps (state, image, language) → latent z |
| Dynamics d | Predicts z' from (z, a, g) |
| Reward R | Predicts r from (z, a, g) — cross-entropy on log-bin space |
| Value Q | Predicts discounted return from (z, a, g) |
| Policy prior p | Predicts optimal action — trained with BC + Q-value + entropy |

**Training signal**: self-prediction (sg consistency loss), reward cross-entropy, value cross-entropy. No decoder required — stop-gradient prevents collapse.

**Key distinction from JEPA-WMs**: the encoder is trained *jointly* with dynamics/reward/value using the control objective, not frozen from a separate SSL stage. This makes TD-MPC2 models *control-centric*: their representations are optimized for planning success, not representation quality per se.

**Newt** (Hansen et al. 2025) extends TD-MPC2 to **massively multitask online RL** (200 tasks, 10 domains) by adding:
- Language conditioning via frozen CLIP-ViT/B embeddings
- Optional image conditioning via frozen DINOv2/B embeddings
- Discrete reward/value regression (cross-entropy on log-transformed bins) to handle heterogeneous reward scales across tasks
- Four-pronged demonstration strategy: pretraining, constrained planning, oversampling, action supervision in policy updates

**Scaling law finding**: unlike single-task RL where model/batch scaling is marginal, multitask RL benefits clearly from larger models (up to 80M params) and batch sizes (up to 1024). There exists a compute-optimal (model, batch) size for any fixed number of training tasks.

## DINO-WM: Frozen Patch Features + Causal ViT Predictor

DINO-WM (Zhou et al. 2024) is the simplest instance of the frozen-encoder recipe and a precursor of the JEPA-WM family. It works as follows:
- **State:** frozen **DINOv2 patch tokens** (14×14×384).
- **Predictor:** a ~19M-parameter ViT with a **frame-level causal mask**, where every patch attends to all patches of previous frames and whole frames are predicted at once.
- **Conditioning:** action and proprioception embeddings are concatenated to every patch.
- **Loss:** teacher-forced latent MSE only, trained on **reward-free offline** trajectories.
- **Planning:** CEM-MPC on terminal latent distance to a goal image.

Findings:
- **Patch tokens ≫ CLS/global vectors:** Push-T 0.90 vs 0.44 SR.
- **The causal mask is essential:** without it, longer history collapses Push-T SR from 0.76 to 0.08.
- **Backpropagating a pixel-decoder loss hurts:** 0.92 → 0.80.
- **Reward-free TD-MPC2 latents fail entirely**, which shows why reward shaping matters for self-predictive models.

See [Pretrained Visual Representations](pretrained-visual-representations.md).

## JEPA World Models (JEPA-WMs)

Terver et al. (2026) formalize the **JEPA-WM** family: encoder–predictor architectures that post-train an action-conditioned predictor on top of a frozen video/image SSL encoder. The formal training objective is:

$$\mathcal{L} = \frac{1}{B}\sum_{b=1}^B \mathcal{L}\!\left[P_\theta\!\left(E_{\phi,\theta}(o^b_{t-W:t}),\,A_\theta(a^b_{t-W:t})\right),\;E_{\phi,\theta}(o^b_{t+1})\right]$$

Key design findings from their systematic ablation:

| Design axis | Best choice | Finding |
|-------------|-------------|---------|
| Encoder | DINOv3 ViT-L | Image SSL outperforms video SSL for manipulation |
| Action conditioning | AdaLN (on average) | Modulates all predictor layers; task-dependent |
| Rollout depth | $k=2$ | 2-step optimal; deeper rollouts do not help |
| Proprioception | Always include | Consistently improves all environments |
| Predictor scale | ViT-L depth-12 | Scaling helps real-world but not simulated |

**Encoder pretraining gap:** DINOv2/v3 (image SSL) beats V-JEPA and V-JEPA 2 (video SSL) on robot manipulation benchmarks. Internet-scale video pretraining is not necessarily superior to image pretraining for downstream task success in manipulation.

## Representative Systems

| System | Encoder | Dynamics | Planning | Notes |
|---|---|---|---|---|
| Dreamer (Hafner et al. 2019) | CNN/RSSM | Recurrent latent | Actor-critic in latent | Simulated envs |
| TD-MPC2 (Hansen et al. 2022/2024) | MLP (jointly trained) | MLP predictor | CEM | Single-task continuous control |
| **Newt (Hansen et al. 2025)** | MLP + CLIP + DINOv2 (jointly trained) | MLP | CEM | **200 tasks, massively multitask online RL** |
| V-JEPA 2-AC (Assran et al. 2025) | ViT (frozen, 1B) | Causal Transformer | CEM (image goals) | Real robot, zero-shot |
| [DINO-WM](../papers/zhou_2024_dino_wm.md) (Zhou et al. 2024) | DINOv2 patches (frozen) | Frame-causal ViT (~19M) | CEM-MPC (image goals) | Reward-free offline data; sim only |
| JEPA-WM (Terver et al. 2026) | DINOv3 ViT-L (frozen) | ViT-L depth-12 + AdaLN | CEM L2 / NeverGrad | Best config across sim+real |

**Self-predictive vs. JEPA-WM**: self-predictive models (TD-MPC2 / Newt) jointly train encoder + dynamics + reward + value — representations emerge from control objectives. JEPA-WMs freeze a large SSL-pretrained encoder and only train an action-conditioned predictor on top — representations come from internet-scale video/image pretraining. The right choice depends on whether a large-scale SSL encoder is available for the domain.

## Relationship to Action-Conditioned Video Generation

Video generation world models (e.g., Cosmos, Genie) predict future **pixels** rather than latent representations. This enables visually evaluating model fidelity, but:
- Planning via pixel generation is ~15-250× slower than latent planning
- Generation objective wastes capacity on visually realistic but semantically redundant details
- Latent models tend to achieve higher task success rates for the same compute budget

> **Verify:** Comparative planning efficiency numbers from V-JEPA 2 paper: V-JEPA 2-AC requires 16s/action vs. Cosmos 4min/action on same GPU.

## See Also

- [Joint Embedding Predictive Architecture](joint-embedding-predictive-architecture.md)
- [Model Predictive Control for Robot Learning](model-predictive-control.md)
- [V-JEPA 2 (Assran et al. 2025)](../papers/assran_2025_vjepa2.md)
- [Newt / MMBench (Hansen et al. 2025)](../papers/hansen_2025_newt.md) — massively multitask self-predictive world model; 200 tasks, online RL
- [JEPA World Models (Terver et al. 2026)](../papers/terver_2026_jepa_wm.md) — frozen-encoder JEPA-WM ablation study
- [JEPA World Models as Physics Priors for RL](../open_questions/jepa-rl-humanoid-grasping.md) — open research hypothesis building on the JEPA-WM family as an RL state encoder rather than an MPC planner
- [Zhou et al. (2024) — DINO-WM](../papers/zhou_2024_dino_wm.md) — frozen DINOv2 patch-feature world model; reward-free offline training, CEM-MPC planning
- [Pretrained Visual Representations for Robotics](pretrained-visual-representations.md)
