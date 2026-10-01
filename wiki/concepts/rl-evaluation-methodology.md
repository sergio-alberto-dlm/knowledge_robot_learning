---
title: RL Evaluation Methodology (Protocols, Seeds, and Measurement Pitfalls)
type: concept
tags: [evaluation-methodology, reproducibility, seeds, statistical-significance, success-rate, gpu-simulation, rl-benchmarking]
related: [codebase/rl_lab/evaluation-protocol.md, concepts/continual-learning-and-tracking.md, concepts/policy-gradient-methods.md, papers/zhou_2024_dino_wm.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/PROTOCOL.md, raw/repos/rl_lab/rl/evaluation.py]
---

# RL Evaluation Methodology

## Why It Matters

RL results are noisy, and small, undocumented choices in *how* a policy is scored can move its numbers by more than the effect being studied. A staged research project (change the perception, then the critic, …) needs its measuring instrument **frozen before the first comparison**. Otherwise every comparison mixes the intended change with a silent change in measurement. The principles below come mainly from the [rl_lab evaluation protocol](../codebase/rl_lab/evaluation-protocol.md), where each one was established by measurement on GPU-parallel ManiSkill.

## Pitfalls and Remedies

| Pitfall | What happened (rl_lab, PickCubeSO100) | Remedy |
|---|---|---|
| **Duration-biased episode sampling** in vectorised eval | Successes end early (~32 steps) and failures run to 50, so "the first N to finish" over-samples successes. One checkpoint scored 0.470 / 0.375 / 0.441 for N = 100 / 128 / 256 | Give every env an **equal quota** of episodes; reject N that is not a multiple of the env count |
| **Residual simulator state** | GPU-sim `reset(seed)` did not clear physics state; the same checkpoint gave 0.383–0.469 across four calls | Rebuild the scene (`reconfigure=True`) and **measure the noise floor** (≈ 0 here) |
| **Stochastic vs deterministic policy** | With σ at its floor, the policy mean and the sampled policy differ by ~7 points | **Score both**, declare one the headline, and never compare across modes |
| **Eval seeds leaking into training** | — | Reserve a disjoint seed range for evaluation (e.g. 90,000+), with a separate stream for action noise |
| **Last-checkpoint noise** | Consecutive checkpoints swing ±3 points even with a zero-noise instrument | Report **"final" as the mean of the last k evals** (k = 5) |
| **First-crossing sample efficiency** | A run touched 80% at 46M steps, dipped, and held only from 61M | Define crossings as **first eval that is ≥ threshold *and stays* there** |
| **Return as a quality metric** | With terminate-on-success, better policies earn *lower* undiscounted return | Use success rate and steps-to-success; keep return only as a sanity check |
| **Steps-to-success diluted by failures** | Including failures measures time-to-give-up | Average over **successful episodes only**, and read it next to success rate |
| **Wall-clock contaminated by contention** | Throughput jumped 5× mid-run from GPU sharing | One run per GPU, arms run sequentially; wall-clock is a first-class axis when compute is the claim |

## Seeds and Detection Thresholds

- **Measure the seed spread and derive the smallest detectable effect from it.** With 3 seeds at 0.909 ± 0.030, differences under ~0.06 (about 2× std) are declared ties.
- **Different metrics have very different spreads:** success ±3.3%, steps-to-success ±0.5%, env-steps-to-80% ±16%. Pick the *tight* metric as the sensitive axis, and treat sample-efficiency claims from few seeds with suspicion.
- **Deterministic mode can be much more seed-variable than stochastic mode** (6× in rl_lab), because action noise masks brittleness in the mean action.
- **Pre-commit decision rules** (e.g. "switch to Plan B if success < 0.30 at iteration 1,500") before looking at the curve. A threshold chosen while watching the curve is not a threshold.

## Relation to Other Evaluation Settings

- **Lifetime error** in online and continual learning ([Continual Learning & Tracking](continual-learning-and-tracking.md)) removes the train/test split entirely. The protocol here is the episodic, train-then-evaluate counterpart.
- **Simulation benchmarks in the compiled papers** often report a single number without these controls. When comparing numbers *across* papers, the protocol differences can dominate. See the DINO-WM vs JEPA-WM Push-T conflict in [Zhou et al. (2024)](../papers/zhou_2024_dino_wm.md).

## See Also

- [rl_lab — Evaluation Protocol](../codebase/rl_lab/evaluation-protocol.md)
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md)
- [Policy Gradient Methods](policy-gradient-methods.md)
- [Zhou et al. (2024) — DINO-WM](../papers/zhou_2024_dino_wm.md)
