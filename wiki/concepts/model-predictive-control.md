---
title: Model Predictive Control (MPC) for Robot Learning
type: concept
tags: [planning, control, model-based-rl, robot-learning, optimization]
related: [concepts/latent-world-models.md, concepts/joint-embedding-predictive-architecture.md, papers/assran_2025_vjepa2.md, papers/terver_2026_jepa_wm.md, papers/zhou_2024_dino_wm.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/V-JEPA-AC.pdf, raw/papers/pdf/DINO_WM.pdf]
---

# Model Predictive Control (MPC) for Robot Learning

## Definition

Model Predictive Control (MPC) is a planning paradigm where, at each time step, an agent:
1. Uses a world model to simulate future states for candidate action sequences
2. Selects the action sequence that minimizes a cost/energy function
3. Executes only the **first action** of the optimal sequence (receding horizon)
4. Re-observes the new state and re-plans

This closed-loop replanning corrects for model errors and unmodeled disturbances.

## Goal-Conditioned Energy Function

In robot manipulation with image goals, the cost function is often defined in the latent space of a world model:

$$\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1$$

where $z_g = E(x_g)$ is the encoding of the goal image. Planning finds:

$$a^*_{1:T} = \arg\min_{\hat{a}_{1:T}} \mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g)$$

## Action Optimization: Cross-Entropy Method (CEM)

CEM (Rubinstein 1997) is a derivative-free sampling-based optimizer commonly used with world models:

1. Initialize action distributions: $a_t \sim \mathcal{N}(0, I)$ for $t = 1, \ldots, T$
2. Sample $N$ trajectories from current distributions
3. Evaluate energy for each trajectory
4. Select top-$k$ trajectories; update distribution mean and variance from their statistics
5. Repeat for $K$ refinement iterations
6. Return the mean of the final distributions as the action sequence

**V-JEPA 2-AC settings:** $N=800$ samples, $K=10$ refinement steps, $T=1$ horizon. Actions constrained to L1-ball radius 0.075 (≈13 cm max). Planning time: 16 seconds/action on single RTX 4090.

**DINO-WM settings:** planning cost is terminal latent MSE $\|\hat z_T - z_g\|^2$ over DINOv2 patch tokens. A full plan with 100 samples × 10 iterations takes **~53 s on an A6000**. Receding-horizon replanning matters: on Push-T, MPC reaches 0.90 SR vs 0.86 for open-loop CEM (PointMaze 0.98 vs 0.80; Wall 0.96 vs 0.74).

## NeverGrad (NG) Planner

Terver et al. (2026) introduce **NeverGrad** (Rapin & Teytaud 2018) as a tuning-free alternative to CEM for JEPA-WM planning. NeverGrad's `NGOpt` meta-selector automatically chooses an optimization algorithm (typically diagonal CMA-ES) based on problem dimensionality and budget:

- **No hyperparameter tuning required** — unlike CEM, which requires setting $N$, $K$, $\sigma_0$.
- **Competitive on real-world benchmarks** (DROID, Robocasa) — matches or approaches CEM performance without any task-specific tuning.
- **Slightly weaker on simulated tasks** — CEM L2 retains an advantage where careful tuning is feasible.

NeverGrad is a practical default for deployment scenarios where hyperparameter search is expensive.

## Gradient-Based Planning

For differentiable world models, actions can be optimized directly via backpropagation through the world model rollout:

$$a^*_{1:T} = \arg\min_{\hat{a}_{1:T}} \mathcal{E}(\hat{a}_{1:T};\, z_k, s_k, z_g)$$

**Limitation:** Gradient-based optimization fails on tasks with non-smooth energy landscapes (e.g., Push-T with contact dynamics). The cost function is not always differentiable with respect to actions in a useful sense. Sampling-based methods (CEM, NeverGrad) are more robust. DINO-WM reports the same pattern: open-loop gradient descent reaches only 0.22 (PointMaze) and 0.28 (Push-T) SR, vs 0.80 / 0.86 for open-loop CEM, on the same model.

## Advantages for Zero-Shot Transfer

- **No task-specific training:** Planning with a world model requires only a goal specification (image or language), not task-specific demonstrations or rewards.
- **Generalization to new objects and environments:** As long as the world model's representation generalizes (e.g., from internet-scale video pretraining), planning can work in unseen settings.
- **Gradient-based planning possible:** For differentiable world models, gradient-based optimization of $\mathcal{E}$ with respect to actions is an alternative to CEM.

## Limitations

- **Computational cost:** Sampling-based MPC (CEM) requires many world-model rollouts per step. Latent world models are faster than pixel-space models but still slow for real-time control.
- **Short horizons only:** Error accumulation in autoregressive models limits reliable planning to short horizons (~16 seconds for V-JEPA 2-AC).
- **Locally convex energy landscapes:** V-JEPA 2-AC energy functions appear smooth and locally convex (as visualized), which facilitates CEM convergence but may miss globally optimal solutions.
- **Image goal assumption:** Goals must be specified as images; language-based goal specification requires additional alignment work.

## See Also

- [Latent World Models](latent-world-models.md)
- [Joint Embedding Predictive Architecture](joint-embedding-predictive-architecture.md)
- [V-JEPA 2 (Assran et al. 2025)](../papers/assran_2025_vjepa2.md)
- [Zhou et al. (2024) — DINO-WM](../papers/zhou_2024_dino_wm.md) — CEM vs GD vs MPC comparison on a frozen DINOv2 latent world model
