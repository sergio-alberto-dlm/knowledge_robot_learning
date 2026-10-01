---
title: rl_lab — PPO Agent (networks, update, buffers)
type: codebase
tags: [rl_lab, ppo, gae, actor-critic, gaussian-policy, entropy, value-clipping, rollout-buffer]
related: [codebase/rl_lab/_overview.md, codebase/rl_lab/maniskill-training-pipeline.md, codebase/rl_lab/evaluation-protocol.md, concepts/policy-gradient-methods.md, papers/schulman_2017_ppo.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/src/ex3_ppo.py, raw/repos/rl_lab/rl/networks.py, raw/repos/rl_lab/rl/buffers.py, raw/repos/rl_lab/rl/common.py]
---

# rl_lab — PPO Agent

A hand-written PPO (originally a course exercise; the `# TODO` scaffolding comments are still in the file) used by both the legacy MuJoCo scripts and the ManiSkill pipeline. See [Policy Gradient Methods](../../concepts/policy-gradient-methods.md) for the algorithm.

## Files

| File | Contents |
|---|---|
| `src/ex3_ppo.py` | `PPOAgent`, `PPOUpdateStats`, `_grad_norm` |
| `rl/networks.py` | `build_mlp`, `GaussianActor` (PPO), `ValueNet`, `SquashedGaussianActor` + `QNet`/`DoubleQNet` (SAC), `PPO_LOG_STD_FLOOR` |
| `rl/buffers.py` | `RolloutBuffer` (single env), **`VecRolloutBuffer`** (vectorised; used for ManiSkill), `ReplayBuffer` (SAC) |
| `rl/common.py` | `set_seed` (python/numpy/torch/CUDA), `ensure_dir` |

## Networks

- **`GaussianActor`:** MLP mean head (`hidden_sizes = [256, 256, 128]`, ReLU) plus a **state-independent** `log_std` parameter vector, initialised at −1.0.
  - `act()` samples an action.
  - `act_inference()` returns the mean; `PPOAgent.predict_action` wraps it and clips to [−1, 1]. It is the deterministic policy used for headline evaluation.
  - `clamp_log_std()` sets `log_std ← max(log_std, −1.2)` in place after every optimiser step.
- **`ValueNet`:** an MLP with the same hidden sizes that outputs a scalar V(s). This is the component Peldaño 2 plans to replace with a SwiftTD linear critic ([Research Roadmap](research-roadmap.md)).
- **Separate actor and critic networks** (no shared trunk) but a **single Adam optimiser** over both parameter sets.

### The log-std floor (−1.2, σ = 0.301)

Documented in `rl/networks.py`, the PPO update code and `PROTOCOL.md`:
- **Without it:** with `entropy_coeff = 0`, every run on `PickCubeSO100-v1` collapsed σ to 0.09–0.17 within ~150 iterations. Under `pd_joint_delta_pos` (±0.05 rad per step) that leaves about ±0.005 rad of exploration, and training stalled at grasp-and-hold with 0% success.
- **Why an entropy bonus cannot fix it:** with a state-independent σ, the entropy is $H = \sum_i \log\sigma_i + \tfrac{d}{2}\log(2\pi e)$, so $\partial H/\partial\log\sigma_i = 1$ is **constant**. The surrogate's push on σ scales with advantage magnitude, which grows as the policy improves, so a constant force eventually loses. Tuning `entropy_coeff` only delays that. The logged entropy 1.31363 matches $6(-1.2) + 8.514 = 1.3137$, which confirms σ sits on the floor.
- **Consequence:** σ is at the floor from iteration ~100 onward, so the sampled policy and its mean are materially different controllers. The evaluation protocol therefore scores both ([Evaluation Protocol](evaluation-protocol.md)).
- The floor is **part of the policy definition** and is frozen in the protocol, not tuned.

## `PPOAgent.update()`

For `n_epochs` passes over randomly permuted minibatches of the flattened rollout (`n_steps = num_steps × num_envs = 51,200`, minibatch 1,600, so 32 minibatches per epoch and 8 epochs):

1. Rebuild the action distribution, then compute the new log-probability, mean, std, value and entropy.
2. **KL** between the old and new diagonal Gaussians, analytically: $\sum_d \log\frac{\sigma}{\sigma_{old}} + \frac{\sigma_{old}^2 + (\mu_{old}-\mu)^2}{2\sigma^2} - \tfrac12$, averaged over the batch.
3. **KL-adaptive LR** (`adjust_learning_rate`): divide by 1.5 if KL > 2·target, multiply by 1.5 if KL < target/1.5, bounded by `[min_lr, max_lr]`. **In the ManiSkill config it is inert**, because `min_lr = max_lr = 3e-4` deliberately pins the LR constant to match the official ManiSkill recipe.
4. **Clipped surrogate:** $-\mathbb{E}[\min(r A, \mathrm{clip}(r, 1\pm\epsilon)A)]$ with ε = 0.2.
5. **Clipped value loss:** $\max\big((V-R)^2,\ (V_{old} + \mathrm{clip}(V - V_{old}, \pm\epsilon) - R)^2\big)$, which reuses the same ε = 0.2 in *absolute value units*.
6. **Entropy loss:** $-\mathbb{E}[H]$, weighted by `entropy_coeff` (0 in the ManiSkill config).
7. **Joint gradient clipping** (`max_grad_norm = 0.5`) over actor and critic parameters together. Pre-clip gradient norms are measured **separately** for actor and critic and logged. The shared clip couples them: a large critic gradient scales the actor step down. This matters because raising γ inflates value targets.
8. After the Adam step, apply `clamp_log_std()`.

**Performance note:** the gradient-norm diagnostics are accumulated as on-device tensors and synced to the CPU once per update. Calling `.item()` per minibatch would add 512 GPU→CPU syncs per iteration (actor and critic × 256 minibatches), and wall-clock is a reported axis of the project.

> **Verify (potential issue, not flagged in the repo):** value clipping uses an absolute ε = 0.2, but at γ = 0.95 the value targets reach $V(\text{success}) \approx 1/(1-\gamma) = 20$. The pessimistic `max` zeroes the gradient for samples whose clipped loss is larger, so the critic may track large target changes slowly. The seed-44 high-KL episode and the planned SwiftTD critic comparison would both be sensitive to how well the MLP critic tracks. Worth an ablation (value clipping off, or ε scaled to the return scale) before Peldaño 2 uses the MLP critic as the comparison point.

## `VecRolloutBuffer` and GAE

- Buffers are shaped `(num_steps, num_envs, …)` and flattened by `get()`.
- **Truncation bootstrapping:** when an env auto-resets, the next stored value belongs to the *new* episode. The training loop therefore calls `set_final_values(done_mask, V(final_observation))`, and `compute_returns` uses

$$\tilde V_{t+1} = (1 - d_t)\,V(s_{t+1}) + V_{final,t}, \qquad \delta_t = r_t + \gamma \tilde V_{t+1} - V(s_t)$$

  with advantage recursion $A_t = \delta_t + \gamma\lambda(1-d_t)A_{t+1}$ and returns $R_t = A_t + V(s_t)$. For genuine terminations, $V_{final}$ stays 0.
- Advantages are **normalised over the whole rollout** (zero mean, unit std) inside `compute_returns`.
- `RolloutBuffer` (single env) implements the same GAE without the final-value term. It is used by the legacy MuJoCo script, where episodes only truncate.

## Checkpoints

`save()` stores the actor, critic, optimiser, current LR and iteration. `load()` restores them and returns the iteration, so `--resume` can continue a run. Older checkpoints without `learning_rate` fall back to the optimiser's LR.

## See Also

- [rl_lab Overview](_overview.md)
- [ManiSkill Training Pipeline](maniskill-training-pipeline.md): how the agent is driven
- [Evaluation Protocol](evaluation-protocol.md): why deterministic and stochastic modes are both scored
- [Policy Gradient Methods](../../concepts/policy-gradient-methods.md)
- [Schulman et al. (2017) — PPO](../../papers/schulman_2017_ppo.md)
