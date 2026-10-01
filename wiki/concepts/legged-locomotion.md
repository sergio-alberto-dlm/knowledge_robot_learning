---
title: Legged Locomotion & Perceptive Control
type: concept
tags: [robot-learning, legged-locomotion, quadruped, humanoid, cross-embodiment, in-context-adaptation, sim2real, teacher-student, perceptive-locomotion, curriculum-rl, depth-camera, parkour]
related: [papers/cheng_2023_extreme_parkour.md, concepts/sim-to-real.md, concepts/policy-gradient-methods.md, concepts/intrinsic-motivation.md, papers/liu_2025_locoformer.md, concepts/in-context-adaptation.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/extreme_parkour.pdf, raw/papers/pdf/locoformer.pdf]
---

# Legged Locomotion & Perceptive Control

## Problem Definition

**Legged locomotion** is the problem of controlling a multi-limb robot (biped, quadruped, hexapod, and wheeled-legged hybrids) to navigate terrain by stepping — a task fundamentally different from wheeled or manipulation robotics because the contact schedule, balance, and gait must all be controlled simultaneously. **Perceptive locomotion** extends this with exteroceptive sensing (cameras, LIDAR, elevation maps) to handle terrain geometry the robot cannot sense via proprioception alone.

## Classical vs. Learning-Based Approaches

**Classical approach:**
1. Fuse depth + odometry → elevation map
2. Plan footstep locations using the map
3. Execute with model-based controller (e.g., ZMP, whole-body control)

Limitation: each component must be engineered with very low tolerances; the pipeline fails when sensing or actuation is imprecise, and cannot generalize to terrain not anticipated at design time.

**Learning-based approach (RL + sim-to-real):**
- Train a policy end-to-end in simulation to map observations → joint commands
- Transfer to real hardware via domain randomization and teacher-student distillation
- Advantage: the policy implicitly learns to compensate for imprecise sensing and actuation; no explicit map needed

## Teacher-Student / Privileged Learning

The dominant training paradigm for perceptive locomotion:

**Phase 1 — Teacher RL with privileged information:**
- Teacher policy observes proprioception + *scandots* (dense height map from simulated LIDAR) + oracle heading direction
- Trained with PPO and an automatic terrain curriculum
- ROA (Regularized Online Adaptation) / RMA: an adaptation module learns to estimate environment properties (friction, mass) from the history of proprioceptive observations, collapsing two-phase training into one

**Phase 2 — Student distillation to deployable sensors:**
- Student policy replaces scandots with depth images (convnet-GRU backbone)
- Trained via DAgger: student rolls out in environment, teacher labels actions
- Actor initialized from teacher weights to prevent distribution drift

Key challenge: **heading direction** — the teacher knows exact waypoint directions; the student must infer these from visible terrain geometry. The **Mixture of Teacher and Student (MTS)** strategy blends teacher and student observations during distillation to prevent catastrophic distribution shift:

$$obs_\theta = \begin{cases} \theta_{\text{pred}}, & \text{if } |\theta_{\text{pred}} - \hat{d}_w| < 0.6 \\ \hat{d}_w, & \text{otherwise} \end{cases}$$

## Generalist / Cross-Embodiment Locomotion

The paradigm above trains **one specialist per robot** with narrow randomization, and its adaptation modules see only ~hundreds of ms of history. **LocoFormer** (Liu, Pathak & Agarwal, CoRL 2025) replaces this with a single **omni-bodied** policy:
- **Training data:** PPO on ~100k **procedurally generated** bipeds, quadrupeds and wheeled variants under aggressive dynamics randomization. Observations and actions live in a **unified superset joint space**.
- **Architecture:** a **Transformer-XL** policy with ~18 s of memory at 50 Hz, trained on a **multi-trial** objective where memory persists across falls and resets.
- **Results:** zero-shot, it reaches 0.96 normalized displacement on 10 unseen robots in simulation (0.98 with 5 s of adaptation; the per-robot expert gets 0.99; a GRU trained the same way gets 0.37). On real hardware it runs zero-shot on G1, H1, Go2 and Go2-W.
- **Emergent adaptation:** it recovers from a locked knee in 2–3 s, from cut lower legs in 7–8 s, handles stilts, switches from rolling to walking when its wheels lock, and learns to balance a no-ankle biped across trials.

The shift is from *robustness through privileged distillation* to **adaptation through in-context inference**. See [In-Context Adaptation & Cross-Embodiment Policies](in-context-adaptation.md).

## Reward Design for Locomotion

**Velocity tracking (standard):** $r = \min(\langle \mathbf{v}_{\text{base}}, \mathbf{v}_{\text{cmd}} \rangle, v_{\text{cmd}})$ — simple but can be exploited.

**World-frame inner-product tracking (Extreme Parkour):** $r = \min(\langle \mathbf{v}_{\text{world}}, \hat{\mathbf{d}}_w \rangle, v_{\text{cmd}})$ — tracks velocity in world frame toward waypoints; prevents the robot from learning to turn around obstacles instead of jumping them.

**Foot clearance penalty:** $r = -\sum c_i \cdot M[p_i]$ — penalizes foot contacts near terrain edges (within 5 cm); essential for real-world stability on gaps and steps.

**Style reward (inner-product design principle):** $r = W \cdot [0.5 \cdot \langle \hat{\mathbf{v}}_{\text{fwd}}, \hat{\mathbf{c}} \rangle + 0.5]^2$ — drives the robot's forward axis toward a desired orientation; $\hat{\mathbf{c}} = [0,0,-1]^T$ produces handstand behavior. The inner-product design generalizes: any desired body configuration can be expressed as a target vector.

The inner-product reward principle is **universal**: the same formulation induces arbitrarily diverse behaviors (walking, jumping, handstand, ramp traversal) by changing only the target direction vector — no per-skill reward engineering.

## Automatic Terrain Curriculum

Robots are arranged on courses of increasing difficulty:
- **Promote**: if robot traverses >50% of the level length
- **Demote**: if robot travels <50% of expected distance $v^{\text{cmd}} \cdot T$

Applied separately to each terrain type (tilted ramps, gaps, hurdles, high steps), allowing the policy to improve on each in parallel.

## Emergent Behaviors from Unified Reward

With no explicit priors, the following behaviors emerge from pure RL with the unified reward:
- **High jump** (2× hip height): stride shortening → rear kick → front pull-up → rear tuck
- **Long jump** (2× body length): edge alignment → rear kick → mid-air extension → landing
- **Handstand** (bipedal on front legs): weight shift → rear kick to vertical → proprioceptive balance

These behaviors are qualitatively similar to how human athletes learn parkour: the reward specifies *what* to achieve (reach the waypoint), not *how* to move.

## Perceptive Locomotion Sensors

| Sensor | Frequency | Advantages | Disadvantages |
|--------|-----------|------------|---------------|
| Scandots (simulated LIDAR) | — | Accurate, noise-free in sim | Not available on low-cost hardware |
| Elevation map (depth+odometry) | ~10 Hz | Real sensor | Noise, drift, map artifacts |
| Egocentric depth (single camera) | 10±2 Hz | Low cost, lightweight | Jitter, latency, limited FoV |
| Proprioception only | 50–1000 Hz | No sim-to-real gap | Blind to terrain geometry |

The extreme parkour result demonstrates that a single low-cost depth camera (RealSense D435, 10 Hz, 58×87 pixels) is sufficient for precise athletic behaviors when combined with a strong RL-trained backbone — the policy learns to compensate for noise and jitter.

## Key Results Benchmark

From Extreme Parkour (Cheng et al., 2023) on Unitree A1:

| Maneuver | Achievement | Prior SOTA |
|----------|------------|-----------|
| High jump | 0.5m = 2.0× hip height | 1.6× (Zhuang et al.) |
| Long jump | 0.8m = 2.0× body length | 1.5× (Zhuang et al.) |
| Tilted ramp | 37° | Not demonstrated |
| Handstand | ✓ (4-leg → 2-leg transition) | Not demonstrated |

## See Also

- [Extreme Parkour](../papers/cheng_2023_extreme_parkour.md) — the paper introducing the unified reward and dual distillation for parkour
- [Sim-to-Real Transfer](sim-to-real.md) — domain randomization and the reality gap
- [Policy Gradient Methods](policy-gradient-methods.md) — PPO used in Phase 1 training
- [Intrinsic Motivation & Curiosity-Driven RL](intrinsic-motivation.md) — alternative exploration strategies complementary to curriculum RL
- [Liu et al. (2025) — LocoFormer](../papers/liu_2025_locoformer.md) — one Transformer-XL policy for many legged/wheeled bodies, with long-context in-context adaptation
- [In-Context Adaptation & Cross-Embodiment Policies](in-context-adaptation.md) — how LocoFormer's adaptation compares with RMA-style short-history modules
