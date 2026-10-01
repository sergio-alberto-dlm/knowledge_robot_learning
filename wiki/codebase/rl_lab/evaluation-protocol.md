---
title: rl_lab — Frozen Evaluation Protocol & Privileged Baseline
type: codebase
tags: [rl_lab, evaluation-protocol, reproducibility, seeds, success-rate, baseline, maniskill]
related: [codebase/rl_lab/_overview.md, codebase/rl_lab/ppo-agent.md, codebase/rl_lab/maniskill-training-pipeline.md, codebase/rl_lab/research-roadmap.md, concepts/rl-evaluation-methodology.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/rl/evaluation.py, raw/repos/rl_lab/PROTOCOL.md, raw/repos/rl_lab/guia_peldano0.md]
---

# rl_lab — Frozen Evaluation Protocol & Privileged Baseline

`rl/evaluation.py` is the project's **measuring instrument**. It is written and frozen in Peldaño 0 (stage 0), before any perception or critic change, and every later stage must import `evaluate()` unchanged. **Rule:** any change to it invalidates every number measured before it. `PROTOCOL.md` is the written companion.

## API

```python
evaluate(agent, env, n_episodes=128, seed_base=90_000, action_seed=90_017) -> EvalResult
```

- **`env`** must be a ManiSkill vector env with `auto_reset=True` and `record_metrics=True`.
- **`agent`** must expose `predict_action(obs)` (the policy mean), `actor.log_std`, `device`, and `eval_mode()`/`train_mode()`. The agent's train/eval mode is restored afterwards.
- **`EvalResult`** holds `deterministic` and `stochastic` `EvalMetrics`. The bare attributes (`success_rate`, …) forward to the deterministic pass.
- **`EvalMetrics`** keeps **per-episode arrays** (returns, lengths, successes, steps_to_success) so confidence intervals can be computed later.

## Frozen Constants

| Constant | Value | Notes |
|---|---|---|
| `EVAL_SEED_BASE` | 90,000 | Reserved range; training must never use it |
| `EVAL_ACTION_SEED` | 90,017 | A separate stream for the stochastic pass's action noise |
| `DEFAULT_N_EVAL_EPISODES` | 128 | Must be a multiple of `num_eval_envs` (128) |
| `PPO_LOG_STD_FLOOR` | −1.2 (σ = 0.301) | Part of the policy definition |

## Metric Definitions

- **success_rate:** the fraction of episodes with `success_once`, over exactly `n_episodes`, with each env contributing an equal quota.
- **mean_return:** ManiSkill's **undiscounted** episode return. Because the eval env terminates on success, **better policies score *lower* returns** (seed 42: return fell from 20.7 to 11.1 while success rose from 0 to 0.89). It is a sanity check, not a quality axis.
- **steps_to_success:** mean steps to first success **over successful episodes only**. Otherwise it would measure how long the policy takes to give up. Always read it next to success rate.
- **mean_length:** mean episode length over all episodes.
- **"final":** the mean of the **last 5 eval points** (iterations 2100–2500), not the last single point. Consecutive checkpoints swing ±3 points of success, which is real checkpoint variation, not instrument noise.
- **"steps to 80% success":** the first evaluated env-step at which success is ≥ 0.80 **and stays ≥ 0.80 for every later eval**. Resolution is the 5.12M-step eval interval.

## Three Load-Bearing Properties

Each was established by measurement on the seed-42 `iter_1500.pt` checkpoint.

1. **Equal per-env quota, which removes a sampling bias.** Successes end at ~32 steps and failures at 50, so "the first N episodes to finish" over-samples successes. The same checkpoint scored 0.470 (N = 100, truncated from a 128-env wave), 0.375 (N = 128) and 0.441 (N = 256). `evaluate` now gives each env exactly `n_episodes // num_envs` episodes and **rejects** non-multiples.
2. **`reconfigure=True` on reset, for reproducibility.** `env.reset(seed=…)` does not clear residual GPU-sim state. Four calls on a reused env gave 0.383 / 0.469 / 0.453 / 0.414. Rebuilding the scene (~2 s per pass) makes it repeatable, and the **measured noise floor is ≈ 0**: one deviation of one episode (0.8 points) was observed; ≤ 1 episode is the floor.
3. **Dual mode.** Because σ sits on the floor, the policy mean and the sampled policy are different controllers. The **deterministic** pass (the mean) is the headline, being what would be deployed. The **stochastic** pass (mean + σ·ε with seeded noise) is the policy as trained, and is comparable to `train/success_once`. Both are reported, and a deterministic number is never compared with a stochastic one.

**Implementation details:**
- With `auto_reset`, the top-level `info` at a terminal step already belongs to the *next* episode. The finished episode's truth is read from `info["final_info"]`.
- The first-success step is tracked per step from top-level info. `final_info` catches a success on the very last step.
- A 10,000-step safety cap prevents infinite loops.

> **Internal inconsistency:** the `rl/evaluation.py` docstring and `PROTOCOL.md` quote different calibration numbers for the same checkpoint. The docstring gives "6/6 identical" repeats and deterministic vs stochastic success of 0.414 vs 0.477. `PROTOCOL.md` gives "12 of 13 bit-exact" and **0.367 vs 0.438**. The docstring appears to predate the final equal-quota fix; `PROTOCOL.md` is the authoritative record.

## Privileged Baseline: the Project's Upper Bound (FROZEN)

Setup: 3 seeds, γ = 0.95, 2,500 iterations = 128M env-steps, RTX A6000, eval every 100 iterations with 128 episodes per mode. The headline is deterministic, with stochastic in parentheses.

| Metric | Seed 42 | Seed 43 | Seed 44 | **Mean ± std** |
|---|---|---|---|---|
| Success rate (final) | 0.891 (0.944) | 0.894 (0.939) | 0.944 (0.948) | **0.909 ± 0.030** (0.944 ± 0.005) |
| Mean return (final) | 11.07 (11.40) | 11.06 (11.45) | 10.46 (11.22) | 10.86 ± 0.35 (11.35 ± 0.12) |
| Steps-to-success | 28.04 (32.94) | 28.16 (32.90) | 28.34 (32.72) | **28.18 ± 0.15** (32.85 ± 0.11) |
| Env-steps to 80% | 61.4M (41.0M) | 46.1M (41.0M) | 61.4M (30.7M) | 56.3 ± 8.9M (37.5 ± 5.9M) |
| Wall-clock to 80% | 59.8 (39.5) min | 45.3 (40.3) min | 57.6 (28.4) min | 54.2 ± 7.8 min (36.1 ± 6.7) |

### What the seeds show

1. **Detection threshold:** ±0.030 means later stages need differences of more than **~0.06** (about 2× the seed spread) at n = 3 to count. The Peldaño 1 exit criterion (70–80% of baseline) is therefore **0.64–0.73 deterministic success**.
2. **Deterministic mode is ~6× more seed-variable than stochastic** (std 0.030 vs 0.005). The action noise smooths over seed-dependent brittleness in the mean action.
3. **Steps-to-success is the tightest metric** (0.5% spread vs 3.3% for success and 16% for env-steps-to-80%). It is the most sensitive quality axis, but only meaningful next to success rate.
4. **Sample-efficiency claims from 3 seeds are blunt:** seed 43 crossed 80% 15M steps earlier and still finished level with the others.

### Seed-44 anomalies, recorded rather than smoothed over

- **The "sustained" rule mattered:** seed 44 first touched 0.80 at 46.1M steps, dipped to 0.797, and held only from 61.4M.
- **~1,000 iterations in a high-KL regime:** 494 of 2,500 iterations had KL > 0.2 (vs 41 and 61 for the other seeds), concentrated in iterations 1000–2000. The KL-adaptive LR that would damp this is inert (LR pinned). Value loss and gradient norms stayed in family, and the run produced the *best* final success.
- **Still climbing at cutoff:** +0.048 over the last window, ≈ 2.9σ. The 0.909 mean may slightly underestimate the plateau.

**Supporting numbers:** `train/success_once` plateaus at ~0.985 for all seeds, and throughput is 19.6–19.9k steps/s. All three logged W&B configs are identical except `seed`, verified field by field.

## See Also

- [rl_lab Overview](_overview.md)
- [ManiSkill Training Pipeline](maniskill-training-pipeline.md): the γ decision that defines this baseline
- [PPO Agent](ppo-agent.md): the σ floor behind the dual mode
- [Research Roadmap](research-roadmap.md): how later stages are judged against this table
- [RL Evaluation Methodology](../../concepts/rl-evaluation-methodology.md): the general lessons
