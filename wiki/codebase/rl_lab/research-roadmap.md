---
title: rl_lab — Research Roadmap (SwiftTD Critic on Frozen DINOv2 + Depth Features)
type: codebase
tags: [rl_lab, research-plan, swifttd, dinov2, depth, linear-critic, teacher-student, distillation, observation-contract, compute-budget]
related: [codebase/rl_lab/_overview.md, codebase/rl_lab/evaluation-protocol.md, codebase/rl_lab/ppo-agent.md, papers/javed_2024_swifttd.md, concepts/step-size-adaptation.md, concepts/pretrained-visual-representations.md, concepts/td-lambda-and-eligibility-traces.md, concepts/legged-locomotion.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/plan_swifttd_percepcion.md, raw/repos/rl_lab/guia_peldano0.md, raw/repos/rl_lab/guia_peldano1.md, raw/repos/rl_lab/PROTOCOL.md]
---

# rl_lab — Research Roadmap

Summarised from the Spanish planning documents: `plan_swifttd_percepcion.md` (the master plan) and `guia_peldano0.md` / `guia_peldano1.md` (the operational guides). "Peldaño" means "rung": the stages are strictly sequential.

## Research Questions

- **Primary:** can a **linear critic trained with SwiftTD** on **frozen foundation-model features (DINOv2 + depth)** match or beat a standard deep (MLP) critic for value learning in manipulation, with less compute, no replay, and online learning?
- **Secondary (credit analysis):** which features receive the most credit, i.e. the highest learned step sizes, under SwiftTD? Does that match what a human considers task-relevant? The plan is to map per-feature step sizes back onto DINOv2 patches as an image heatmap (an adaptation of SwiftTD's Appendix C), called "THE figure of the project" if it works.

Background: [Javed et al. (2024) — SwiftTD](../../papers/javed_2024_swifttd.md), [Step-Size Adaptation](../../concepts/step-size-adaptation.md), [Pretrained Visual Representations](../../concepts/pretrained-visual-representations.md).

## Locked Infrastructure Decisions (not to be reopened before Peldaño 4)

| Decision | Choice | Reason |
|---|---|---|
| Simulator | **ManiSkill** | Already works, parallel, renders RGB-D. Switching mid-project adds an uncontrolled variable |
| Isaac Sim | Not now | Too heavy for 8 GB GPUs |
| Genesis | Watchlist only | Maturing, but the ecosystem is young |
| SwiftTD | Official package (`pip install SwiftTD`), **`SwiftTDNonSparse`** for dense DINO features | C++ core |
| Control algorithm | **The existing PPO, untouched** | SwiftTD enters **only the critic** in stages 1–3 |

## Stages

### Peldaño 0: Evaluation infrastructure. **CLOSED**
- **Built:** the frozen `evaluate()`, a `--seed` CLI, dual-axis logging (env-steps and wall-clock) via W&B, the 3-seed privileged baseline, and `PROTOCOL.md`.
- **Result:** **0.909 ± 0.030** deterministic success (see [Evaluation Protocol](evaluation-protocol.md)). γ = 0.95 decided (see [Training Pipeline](maniskill-training-pipeline.md)).
- **Postponed:** the multi-GPU cluster pipeline (a target of ≥8× throughput).

### Peldaño 1: Realistic perception. **PLANNING** (HEAD commit "WIP: planning stage 1")

**Goal:** the same task and PPO, but observations change from `[proprio, object pose]` to `[proprio, DINOv2(rgb) features, depth features]`.

**The goal is invisible, so `goal_pos` stays.** `PickCubeSO100-v1` puts `goal_site` in `_hidden_objects` and hides it before the cameras capture. The goal is re-randomised every episode. `goal_pos` is therefore the **task specification** (like a language instruction), not privileged perception. The honest claim is **"no privileged *object* information"**, not "no privileged information".

**Observation contract** (36 → 23 kept dimensions):

| Field | Dims | Peldaño 1 |
|---|---|---|
| `agent.qpos`, `agent.qvel` | 6 + 6 | keep (proprioception) |
| `extra.tcp_pose` | 7 | keep (forward kinematics) |
| `extra.goal_pos` | 3 | keep (task specification; invisible) |
| `extra.is_grasped` | 1 | keep: **documented concession**. It is a physics-engine check, but ManiSkill's visual baselines include it; ablating it is the cheapest experiment in the project |
| `extra.obj_pose`, `tcp_to_obj_pos`, `obj_to_goal_pos` | 7 + 3 + 3 | **removed** (these require knowing where the cube is) |

The 23 kept dimensions are exactly what ManiSkill's own `obs_mode="rgbd"` provides, so the setup matches the house standard.

**Architecture decision: the encoder is a vec-env wrapper, not part of the agent.** The env emits a flat tensor `[23 proprio | DINO features | depth features]`, so:
- `rl/evaluation.py` and `PPOAgent` stay byte-identical; only `obs_dim` changes.
- Peldaño 2's SwiftTD critic sees exactly the actor's input.

Three details to plan for:
- `infos["final_observation"]` must also be encoded before bootstrapping.
- The wrapper needs a **dual mode** that emits the 36-d privileged state on a side channel, for teacher–student distillation.
- Bit-exact determinism must be **re-measured** because of bf16, TF32 and cuDNN.

**Feature design** (planned for `rl/perception.py`):
- **Resolution:** render directly at **126×126** (= 9×14), giving **9×9 = 81 DINOv2 ViT-S/14 patches** with no resize artefact.
- **DINOv2 aggregation:** **mean-pool into 3×3 regions** of 3×3 patches, giving 9 × 384 = **3,456 dims**. CLS-only is rejected because it destroys spatial information; unpooled (31,104 dims) is rejected because it needs a 6.4 GB rollout buffer. This follows DINO-WM's finding that patch features ≫ CLS for manipulation (see [DINO-WM](../../papers/zhou_2024_dino_wm.md)).
- **Depth:** converted from int16 mm to metres, then average-pooled to 9×9 = **81 dims**, aligned one-to-one with the DINO patch grid so the modality ablation compares two views of the same spatial layout.
- **Total observation: 23 + 3,456 + 81 = 3,560 dims.** Rollout buffer ≈ 0.73 GB.
- **Frozen normaliser:** per-dimension mean and std estimated on ~50k frames (half random policy, half teacher rollouts), saved to disk and never updated. A drifting normaliser would make "the same features" false when two critics are compared in Peldaño 2.
- Design the features "for the linear critic, not only for PPO convergence".

**Measured compute budget** (1,024 envs; one iteration = 51,200 env-steps):

| Component | s/iteration |
|---|---|
| Simulation only (state) | 1.05 |
| Simulation + RGB-D render at 128² | 3.36 (the render alone costs 3.2×) |
| PPO update | ~1.85 |
| DINOv2 ViT-S/14 at 126², fp32 / **bf16** | 8.68 / **4.84** |
| DINOv2 ViT-S/14 at 224², bf16 | 15.65 |
| DINOv2 ViT-B/14 at 126² / 224², bf16 | 12.33 / 40.94 |

- The direct approach costs ≈ 10.1 s/iteration vs 2.9 for the privileged baseline, i.e. **3.5× slower per sample**.
- **Cuts:**
  - 1,500 iterations instead of 2,500 (the baseline plateaus by iteration ~1,500);
  - bf16 autocast;
  - screen the ablation arms on 1 seed and run 3 seeds only for the winner;
  - optionally a perception frame-skip of k = 2, which changes the POMDP and must then be reported.
- **Plan:** ~26 GPU-hours in total.

**Exit criterion:** a visual policy at **0.64–0.73 deterministic success** (70–80% of the baseline), where differences under 0.06 are ties.
- **Pre-committed Plan B trigger:** if deterministic success is below **0.30 at iteration 1,500**, switch to Plan B. Between 0.30 and 0.64, allow one extension to 2,500 iterations.
- **Plan B, teacher–student:** the teacher is the best privileged checkpoint (seed 44, iteration 2,500, 0.944 deterministic). The student is trained by BC, then DAgger, then PPO fine-tuning. Cloning error and success are reported separately. This is the same distillation pattern as privileged-teacher locomotion ([Legged Locomotion](../../concepts/legged-locomotion.md)).
- **Modality ablation:** DINO only / depth only / both. The plan anticipates that **depth alone may do surprisingly well** (a cube on a flat table is easy to segment by depth) and commits to reporting that honestly.

### Peldaño 2: SwiftTD in the critic. Not started
- Replace the PPO `ValueNet` with a **linear critic on the Peldaño 1 feature vector**, trained by `SwiftTDNonSparse`. GAE still consumes V(s); only *who learns V and how* changes.
- **Temporal-stream constraint:** SwiftTD is online with eligibility traces, so it needs **one instance (or trace set) per env**. Transitions from different envs must never be mixed in one stream.
- Start from the package defaults (`eta = 0.1`, `decay = 0.999`, `meta_step_size = 1e-3`), which tests the paper's hyperparameter-robustness claim.
- **Comparison, same actor and seeds:** success and convergence speed, **critic compute** (FLOPs and wall-clock, where SwiftTD should win), value quality (lifetime error vs realised returns), and the credit heatmap.
- **Main risk:** SwiftTD may be unstable under PPO's non-stationarity (the policy changes every iteration). Mitigations are refreshing traces per rollout or lowering η. "If it happens, it IS the finding."

### Peldaño 3: Write-up
A 4–6 page workshop-style report (RLC or CoRL workshop, or a blog post), plus a clean public repo.

### Peldaño 4: Extensions (only if 1–3 succeed)
Per-feature step-size optimisation in the **actor** (extending SwiftTD to policy gradient), Genesis sim-to-sim transfer, more ManiSkill tasks.

## How This Connects to the Wiki

- **The γ decision** was made *for* the SwiftTD study: at γ = 0.8 the trace half-life was ~2 steps and V ≈ r_t. See [TD(λ)](../../concepts/td-lambda-and-eligibility-traces.md).
- **A linear critic with per-feature step sizes** on 3,560 frozen features is exactly the regime where IDBD-style step-size adaptation acts as a **relevance detector**, which is the premise of the credit-heatmap question. See [Step-Size Adaptation](../../concepts/step-size-adaptation.md).
- **PPO's changing policy** makes the critic's target non-stationary, which is the tracking setting in [Continual Learning & Tracking](../../concepts/continual-learning-and-tracking.md).

> **Verify:** the plan cites the package defaults as `eta=0.1, decay=0.999, meta_step_size=1e-3`. These are not in the compiled SwiftTD paper summary. Check them against the `SwiftTD` package before Peldaño 2.

## See Also

- [rl_lab Overview](_overview.md)
- [Evaluation Protocol](evaluation-protocol.md)
- [PPO Agent](ppo-agent.md)
- [Javed et al. (2024) — SwiftTD](../../papers/javed_2024_swifttd.md)
- [Step-Size Adaptation](../../concepts/step-size-adaptation.md)
- [Pretrained Visual Representations for Robotics](../../concepts/pretrained-visual-representations.md)
- [Zhou et al. (2024) — DINO-WM](../../papers/zhou_2024_dino_wm.md)
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../../open_questions/object-centric-swifttd-critic.md) — proposed alternative/extra arm: persistent superquadric + DINO slots, tile-coded relations for a SwiftTD critic
