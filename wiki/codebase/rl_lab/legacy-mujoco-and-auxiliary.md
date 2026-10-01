---
title: rl_lab — Legacy MuJoCo SO-100 Env & Auxiliary Code (SAC stub, Genesis, JEPA/SIGReg)
type: codebase
tags: [rl_lab, mujoco, so100, reward-shaping, sac, genesis, sigreg, jepa, identifiability]
related: [codebase/rl_lab/_overview.md, codebase/rl_lab/ppo-agent.md, concepts/sigreg.md, papers/balestriero_2025_lejepa.md, concepts/value-based-rl.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/repos/rl_lab/envs/, raw/repos/rl_lab/scripts/train_ppo.py, raw/repos/rl_lab/scripts/eval_ppo.py, raw/repos/rl_lab/scripts/train_sac.py, raw/repos/rl_lab/scripts/sanity_check.py, raw/repos/rl_lab/src/ex4_sac.py, raw/repos/rl_lab/utils/train_jepa.py, raw/repos/rl_lab/assets/mujoco/]
---

# rl_lab — Legacy MuJoCo SO-100 Env & Auxiliary Code

This code predates or sits beside the ManiSkill pipeline. None of it feeds the frozen protocol or the SwiftTD plan.

## 1. Custom MuJoCo SO-100 Pick-and-Place Env (`envs/so100_rl_env.py`, `envs/so100_mdp_utils.py`)

This was the project's first pick-and-place environment, before the move to ManiSkill in `89d05d3` (2026-08-06).

- **Model:** `assets/mujoco/so100_pos_ctrl.xml`, built on the `trs_so_arm100` MJCF of the Standard Open Arm-100. A torque-control variant also exists.
- **Control:** actions in [−1, 1] are **linearly mapped to *absolute* joint targets** over each joint's full range (`process_action`), with `ctrl_decimation = 50` physics steps per control step and 10 s episodes. This differs from ManiSkill's small `pd_joint_delta_pos` deltas.
- **Observation (26-d, robot base frame):**
  - qpos 6;
  - end-effector position 3 and quaternion 4;
  - object position 3, end-effector→object 3;
  - goal position 3, object→goal 3;
  - `grasp_score` 1.
- **Grasp detection:** `grasp_score` ∈ {0, 0.5, 1} is the fraction of jaw sides (fixed or moving pads) touching the object. A value of 1.0 means a real pinch, not a one-finger push.
- **Staged reward with strict dominance:** each stage's floor exceeds the previous stage's ceiling, so progressing always pays more than camping.
  - reach $0.1(1-\tanh 10d)$, ceiling 0.1;
  - grasp $+0.3\cdot$`grasp_score`;
  - lift $+0.5$ if holding and $z > 0.04$ (binary, so lowering the cube to place it costs nothing);
  - transport $+0.7(1-\tanh 3d_g) + 0.3(1-\tanh 15 d_g)$ while holding;
  - success $+5.0$ per step when $d_g < 0.025$;
  - minus $0.01\max q̇^2$.

  The ceilings run 0.1 → 0.4 → 0.9 → ~1.9 → ~5.2. The same lesson as the ManiSkill `ignore_terminations` fix applies: success must *keep paying*.
- **Scripts:** `train_ppo.py` (CPU, the `PPO_PARAMETERS` config with 30k iterations × 512 steps, γ = 0.99, and fixed-seed eval by reseeding global numpy and restoring the RNG), `eval_ppo.py`, `sanity_check.py` (random actions in the viewer), and `view_scene.py`.

> **Stale docstrings:** `sanity_check.py` says the lift bonus fires at `obj_z > 0.08` and success needs the object "lifted". The code uses `_LIFT_HEIGHT = 0.04`, and success depends only on the object–goal distance.

## 2. SAC: Unfinished Exercise (`src/ex4_sac.py`, `src/ex4_sac_config.py`, `scripts/{train,eval}_sac.py`)

- The class structure and docstrings describe standard SAC: a squashed-Gaussian actor with tanh log-probability correction, twin Q networks with Polyak-averaged targets, and automatic temperature tuning with target entropy −|A|.
- **The core methods are placeholders (`...`):** `sample_action`, the critic, actor and alpha losses, and more. **The SAC scripts cannot run as-is.** The networks in `rl/networks.py` (`SquashedGaussianActor`, `DoubleQNet`) and `ReplayBuffer` are implemented.
- See [Value-Based RL](../../concepts/value-based-rl.md) for the double-Q idea it builds on.

## 3. Genesis Experiments (`envs/genesis_basics.py`, `envs/grasp_env.py`)

- **`genesis_basics.py`:** a script that renders RGB, depth, segmentation and normals of a Franka in Genesis and records an orbiting-camera video.
- **`grasp_env.py`:** a batched **Franka** grasp environment adapted from Genesis's examples.
  - A stereo camera pair uses the Madrona batch renderer when available, and lazily rendered `get_stereo_rgb_images()` returns 6 channels.
  - IK is either Genesis's solver or damped least squares, with the DLS solve moved to CPU on Apple MPS, where it is ~300× faster.
  - Reward is a **keypoint-alignment reward** $\exp(-\sum\|k_{finger} - k_{object}\|)$.
- **Not referenced by any script** in the repo, and `genesis` is not in `requirements.txt`. This matches the plan's "Genesis on the watchlist, not a migration".

> The observation list comment in `grasp_env.get_observations` labels `obj_pos` / `obj_quat` as "goal position/orientation". That is a copy-over from the source example, since the goal *is* the object pose in that task.

## 4. JEPA Identifiability Utility (`utils/train_jepa.py`)

- **What it trains:** a self-supervised **CartPole image encoder** (4 strided conv layers + BN/GELU, pooled projector to a 4-d latent). The loss is $(1-\lambda)\,\mathcal{L}_{align}(h(o), h(o')) + \lambda\,\mathrm{SIGReg}(h(o))$, i.e. an alignment loss between paired observations plus the **SIGReg** isotropic-Gaussian regulariser from LeJEPA.
  - SIGReg: 128 random unit projections, and an Epps–Pulley-style characteristic-function statistic on a 17-knot grid over [0, 3] with Gaussian window weights.
- **Identifiability check:** fit a linear regression from embeddings to the true 4-d CartPole state (and the reverse) and report held-out R². The best-R² encoder is checkpointed.
- **Context:** the removed `IdentJEPA.pdf` and DQN/CartPole utilities suggest a side study on whether JEPA/SIGReg latents linearly recover the true state. `raw/papers/pdf/IdentJEPA.pdf` exists in this knowledge base but is **not yet compiled**.
- **Portability issues:**
  - hardcoded dataset paths on another machine (`/media/jbhayet/...`);
  - outputs written to `logs/dqn/debug/`;
  - an unused `rho` argument;
  - it depends on `sklearn`, which is not in `requirements.txt`.
- See [SIGReg](../../concepts/sigreg.md) and [Balestriero & LeCun (2025) — LeJEPA](../../papers/balestriero_2025_lejepa.md).

## See Also

- [rl_lab Overview](_overview.md)
- [PPO Agent](ppo-agent.md)
- [SIGReg](../../concepts/sigreg.md)
- [Balestriero & LeCun (2025) — LeJEPA](../../papers/balestriero_2025_lejepa.md)
- [Value-Based Reinforcement Learning](../../concepts/value-based-rl.md)
