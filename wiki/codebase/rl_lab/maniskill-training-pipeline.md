---
title: rl_lab — ManiSkill Training Pipeline (PickCubeSO100-v1)
type: codebase
tags: [rl_lab, maniskill, ppo, gpu-parallel-simulation, reward-design, terminations, discount-factor, wandb, so100]
related: [codebase/rl_lab/_overview.md, codebase/rl_lab/ppo-agent.md, codebase/rl_lab/evaluation-protocol.md, concepts/policy-gradient-methods.md, concepts/td-lambda-and-eligibility-traces.md, concepts/sim-to-real.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/scripts/train_ppo_maniskill.py, raw/repos/rl_lab/src/ex3_ppo_config.py, raw/repos/rl_lab/scripts/run_gamma_pilot.sh, raw/repos/rl_lab/scripts/eval_ppo_maniskill.py, raw/repos/rl_lab/PROTOCOL.md]
---

# rl_lab — ManiSkill Training Pipeline

`scripts/train_ppo_maniskill.py` trains the [PPO agent](ppo-agent.md) on ManiSkill 3's `PickCubeSO100-v1`: an SO-100 arm picks a cube and moves it to a goal position. Rollouts, observations and updates all stay on the GPU.

## Task and Environment

| Setting | Value |
|---|---|
| Task | `PickCubeSO100-v1`, 50-step episodes |
| Observation | `obs_mode="state"`, **36-d privileged**: qpos 6, qvel 6, tcp_pose 7, goal_pos 3, is_grasped 1, obj_pose 7, tcp_to_obj 3, obj_to_goal 3 |
| Action | `pd_joint_delta_pos`, 6-d in [−1, 1] (±0.05 rad joint deltas) |
| Reward | `normalized_dense` (≤ 1 per step) |
| Backend | `physx_cuda`, falling back to `physx_cpu` without a GPU |
| Train env | 1,024 envs, `auto_reset=True`, **`ignore_terminations=True`** |
| Eval env | 128 envs, `ignore_terminations=False` (terminate on success), `record_metrics=True` |

### Why training ignores terminations (the `make_vec_env` docstring)

ManiSkill sets `terminated = success`. Honouring that, a successful episode is worth only its terminal reward (1.0), while **grasping the cube and stalling pays 0.53 per step forever**, i.e. 0.53/(1−γ) = 2.65 at γ = 0.8. **Finishing was worth 2.65× *less* than never finishing**, and PPO correctly learned to hover near the goal without settling.

With terminations ignored, the agent keeps collecting 1.0 per step after success, so $V(\text{success}) \approx 1/(1-\gamma)$ is learned from realised returns and dominates stalling at any γ, since the ratio is 1.0/0.53. The official ManiSkill approach (bootstrapping $V(s_{final})$ at termination) only gets part of the way there: success states are terminal, so the critic never sees a return target for them. Measured on this task, that approach reached ~1% success versus ~30% for this setting.

The trade-off is that raising γ inflates value targets (5.0 at γ = 0.8, 20 at γ = 0.95) and therefore the value loss. This is why actor and critic gradient norms are logged separately ([PPO Agent](ppo-agent.md)).

## Configuration (`PPO_MANISKILL_PARAMETERS`)

The "v4" config is aligned with the official ManiSkill state-based PPO recipe. Configs v1–v3 all plateaued at 0% success (grasp-and-hold), and an entropy bonus kept inflating σ.

| Param | Value | Param | Value |
|---|---|---|---|
| `num_envs` | 1024 | `num_steps` | 50 (one full episode) |
| `n_epochs` | 8 | `mini_batch_size` | 1600 (32 minibatches) |
| **`gamma`** | **0.95** (official 0.8) | `gae_lambda` | 0.9 |
| `clip_ratio` | 0.2 | `entropy_coeff` | 0.0 |
| `learning_rate` = `min_lr` = `max_lr` | 3e-4 (constant) | `max_grad_norm` | 0.5 |
| `value_loss_coeff` | 0.5 | `hidden_sizes` | [256, 256, 128] |
| `total_iterations` | 2500 (128M env-steps) | `eval_interval` = `save_interval` | 100 |
| `num_eval_envs` | 128 | `seed` | 44 in the file (last edited for the seed-44 run) |

> **Stale comment:** the config annotates `gamma: 0.95` as the "official default for short 50-step episodes". The header comment and `PROTOCOL.md` say the official value is **0.8**, and 0.95 is the project's own decision from the γ pilot (below).

## Training Loop

For each iteration:
1. **Rollout:** 50 steps × 1,024 envs under `torch.inference_mode()`.
   - `select_action` returns the raw sample, the clipped action, V(s), the log-probability, μ and σ.
   - The **unclipped** sample is stored, so log-probabilities stay consistent.
   - When `final_observation` appears in `infos`, $V(s_{final})$ is computed for the finished envs and registered with the buffer.
2. **GAE:** `buffer.compute_returns(V(last_obs))` under `no_grad`. It runs outside inference mode because inference tensors cannot be saved for backward.
3. **Update:** `agent.update(rollout_batch)`.
4. **Evaluate:** every 100 iterations with the frozen `evaluate(agent, eval_env, n_episodes=128)`, logging both deterministic and stochastic metrics ([Evaluation Protocol](evaluation-protocol.md)).
5. **Log:** every scalar goes into one dict, emitted to TensorBoard keyed by **`global_step` (env-steps)**. `time/wall_clock_s` holds the cumulative wall time (rollout + update + eval), and `--track` sends the same dict to W&B with `env_step` as the default x-axis and `wall_time_s` selectable.
6. **Checkpoint** every 100 iterations: `iter_<N>.pt` in the run directory.

**Logging robustness:** logging can never kill training. On 2026-08-31 an external process deleted a run directory mid-run, TensorBoard's async writer raised, and 2,362 iterations of GPU time were lost. Writes are now wrapped: on failure the writer is rebuilt once and otherwise disabled, and checkpoint saves recreate the directory.

**CLI:** `--seed`, `--gamma`, `--total_iterations`, `--resume [ckpt|latest]`, `--track`, `--wandb_project` (default `swifttd-percepcion`), `--wandb_group`, `--wandb_entity`. Run directories are named `YY_MM_DD_HH_MM_SS_model_seed{S}_gamma{γ}`. On resume, the reset seed is offset by the start iteration so already-consumed rollout randomness is not replayed.

**Throughput:** ~19.8k env-steps/s on one RTX A6000 (~2.9 s/iteration; 2,500 iterations ≈ 2.07 h).

## The γ Pilot (`scripts/run_gamma_pilot.sh`)

**Motivation.** 512 episodes were collected under the trained γ = 0.8 policy, with held-out R² for predicting the Monte-Carlo return-to-go:

| γ | Effective horizon 1/(1−γ) | Trace half-life for γλ (λ = 0.9) | R², reward only | R², linear on 36-d state |
|---|---|---|---|---|
| 0.80 | 5 | 2.11 steps | **0.554** | 0.518 |
| 0.90 | 10 | 3.29 | 0.223 | 0.454 |
| 0.95 | 20 | 4.42 | **0.018** | 0.450 |
| 0.99 | 100 | 6.01 | 0.033 | 0.512 |

At γ = 0.8, **V is essentially the current shaped reward**: reading $r_t$ alone explains 55% of the return-to-go, better than a linear fit on the whole state. Success lands at ~32 steps, where $\gamma^{32} \approx 8\times10^{-4}$, and the eligibility trace is ~2 steps long. SwiftTD's credit-assignment machinery would be nearly inert. (The trace half-lives are $\ln 0.5 / \ln(\gamma\lambda)$. See [TD(λ) & Eligibility Traces](../../concepts/td-lambda-and-eligibility-traces.md).)

**Result** (seed 42, both arms run sequentially so GPU contention cannot distort wall-clock):

| | γ = 0.80 | γ = 0.95 |
|---|---|---|
| `train/success_once` plateau | 0.611 | **0.986** |
| Eval success, deterministic / stochastic | 0.45 / 0.52 | **0.91 / 0.95** |
| Steps-to-success | 30.8 | **28.7** |
| Env-steps to 0.50 stochastic eval success | 92.2M | **30.7M** |

γ = 0.95 gives ~2× the success rate and ~3× the sample efficiency. It shows no extra instability (21 iterations with KL > 0.2 in each arm) and makes the value-learning problem non-trivial. γ = 0.99 was not pursued. The γ = 0.8 arm's checkpoints were lost to the directory deletion, but its scalars survived in stdout logs and W&B.

## Evaluation Script (`scripts/eval_ppo_maniskill.py`)

- **Default:** score a checkpoint (the latest one if `--model_path` is omitted) with the **frozen protocol**. It prints a deterministic vs stochastic table of success, return statistics, episode length and steps-to-success.
- `--play` opens the SAPIEN viewer, and `--record_video` records the first N episodes. Both use a separate single-env sequential path (seeds from 1000), so they are **for inspection only, not for reported numbers**.

## See Also

- [rl_lab Overview](_overview.md)
- [PPO Agent](ppo-agent.md)
- [Evaluation Protocol](evaluation-protocol.md)
- [Policy Gradient Methods](../../concepts/policy-gradient-methods.md)
- [TD(λ) & Eligibility Traces](../../concepts/td-lambda-and-eligibility-traces.md)
- [Sim-to-Real Transfer](../../concepts/sim-to-real.md)
