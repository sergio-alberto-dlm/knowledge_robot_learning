---
title: "Extreme Parkour with Legged Robots"
type: paper
tags: [robot-learning, legged-locomotion, parkour, sim2real, teacher-student, perceptive-locomotion, depth-camera, curriculum-rl, ppo]
related: [concepts/legged-locomotion.md, concepts/sim-to-real.md, concepts/policy-gradient-methods.md, papers/liu_2025_locoformer.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/extreme_parkour.pdf]
---

# Extreme Parkour with Legged Robots

**Authors:** Xuxin Cheng\*, Kexin Shi\*, Ananye Agarwal, Deepak Pathak (\* equal contribution)  
**Institution:** Carnegie Mellon University  
**Venue:** arXiv 2309.14341v1 [cs.RO], 25 Sep 2023  
**Website:** https://extreme-parkour.github.io/

## Abstract

Humans can perform parkour by traversing obstacles in a highly dynamic fashion requiring precise eye-muscle coordination. Classical approaches to robot parkour independently engineer perception, actuation, and control systems to very low tolerances, restricting them to tightly controlled lab settings. This paper takes a learning-based approach to developing robot parkour on a small low-cost robot with imprecise actuation and a single front-facing depth camera that is low-frequency, jittery, and prone to artifacts. A single neural network policy operating directly from camera images, trained in simulation with large-scale RL, overcomes imprecise sensing and actuation to output highly precise control behavior end-to-end. The robot performs a high jump over obstacles 2× its height (0.5m), long jumps across gaps 2× its length (0.8m), a handstand (bipedal walking on front two legs), and generalizes to novel obstacle courses.

## Key Contributions

- **Unified inner-product reward principle**: A single general-purpose reward formulation (velocity tracking via inner product in the world frame + foot clearance penalty + optional style term) from which all diverse parkour behaviors emerge automatically — no per-skill reward engineering required.
- **Dual distillation (Phase 2)**: A two-phase training scheme that distills both (1) motor commands (scandots → depth image, via convnet-GRU + DAgger) and (2) heading direction (oracle waypoint direction → predicted direction from depth), enabling fully autonomous deployment with no human direction commands.
- **Mixture of Teacher and Student (MTS)**: A curriculum strategy for heading direction distillation that avoids catastrophic distribution shift by blending teacher (oracle) and student (predicted) heading observations during Phase 2 training.
- **Automatic terrain curriculum**: Robots are promoted to harder terrain levels when they traverse >half the level length and demoted when they travel <half the expected distance, enabling stable exploration across diverse obstacle types.
- **New SOTA for learning-based parkour**: 2× robot height jumps, 2× robot length gaps, tilted ramps (37°), and handstand — all on the low-cost Unitree A1 with a single egocentric depth camera.

## Methodology

### System Overview

A single neural network maps depth images + proprioception → joint angle commands at 50 Hz. Training uses a two-phase teacher-student framework:

- **Phase 1**: RL in simulation with privileged information (scandots, oracle heading)
- **Phase 2**: Supervised distillation to a student that operates only from depth + proprioception

### Reward Design (Unified Inner-Product Principle)

**Waypoint direction** (computed from terrain waypoints placed in the simulator):

$$\hat{\mathbf{d}}_w = \frac{\mathbf{p} - \mathbf{x}}{||\mathbf{p} - \mathbf{x}||}  \quad (1)$$

where **p** = next waypoint, **x** = robot position in world frame.

**Velocity tracking reward** (inner product in world frame):

$$r_{\text{tracking}} = \min(\langle \mathbf{v}, \hat{\mathbf{d}}_w \rangle,\ v_{\text{cmd}})  \quad (2)$$

Critically, velocity is tracked in the **world frame** (not base frame) — this prevents the robot from exploiting the reward by turning around an obstacle. The inner-product formulation means the robot is free to choose the optimal heading rather than following a fixed velocity command.

**Foot clearance penalty** (prevents dangerous edge-stepping):

$$r_{\text{clearance}} = -\sum_{i=0}^{4} c_i \cdot M[p_i]  \quad (3)$$

where $c_i = 1$ if the i-th foot touches the ground, $M$ is a boolean function that is 1 iff foot position $p_i$ lies within 5 cm of a terrain edge.

**Style reward** (for handstand/bipedal mode):

$$r_{\text{stylized}} = W \cdot \left[0.5 \cdot \langle \hat{\mathbf{v}}_{\text{fwd}},\ \hat{\mathbf{c}} \rangle + 0.5\right]^2  \quad (4)$$

where $\hat{\mathbf{v}}_{\text{fwd}}$ = unit forward vector of robot body, $\hat{\mathbf{c}}$ = desired direction unit vector, $W \in \{0,1\}$ = binary enable. For handstand: $\hat{\mathbf{c}} = [0,0,-1]^T$ (drive body downward, forcing front-leg support). $W$ is sampled randomly during training and controlled by operator at deployment.

Additional regularization terms from [Cheng et al., ICRA 2023] are also included.

### Phase 1: RL from Scandots

Policy inputs: proprioception **x**, scandots **m** (height map from simulated LIDAR), target heading $\hat{\mathbf{d}}$ (from waypoints), walking flag $W$, commanded speed $v^{\text{cmd}}$.

- Trained with PPO (model-free RL)
- ROA (Regularized Online Adaptation) trains an adaptation module to estimate environment properties from observation history
- Terrain types: tilted ramps (up to 37°), gaps (up to 0.8m), hurdles, high steps (up to 0.5m)
- Curriculum: promote robot to harder levels if it traverses >half the level length; demote if it travels <half expected distance $v^{\text{cmd}} \cdot T$

### Phase 2: Distilling Direction and Exteroception

The phase 1 teacher relies on two privileged quantities unavailable at deployment:
1. **Scandots** (LIDAR height map) — replaced by depth images via convnet-GRU backbone
2. **Oracle heading** $\hat{\mathbf{d}}_w$ from waypoints — must be inferred from visible terrain geometry

**Exteroception distillation**: Replace scandots input with a convnet-GRU processing depth images (58×87 pixels). Trained via DAgger with phase 1 teacher actions as supervision. Actor network initialized from phase 1 weights to prevent drift.

**Heading direction distillation**: Train a separate heading predictor that outputs $\theta_{\text{pred}}$ (desired yaw angle) from the depth encoding.

**Mixture of Teacher and Student (MTS)** — prevents distribution shift in heading distillation:

$$obs_\theta = \begin{cases} \theta_{\text{pred}}, & \text{if } |\theta_{\text{pred}} - \hat{\mathbf{d}}_w| < 0.6 \\ \hat{\mathbf{d}}_w, & \text{otherwise} \end{cases}$$

When student prediction is close to the oracle, use the student prediction; otherwise fall back to oracle. This keeps the data distribution consistent during distillation training.

### Hardware and Deployment

- **Robot**: Unitree A1 (12 joints), hip joint height 26 cm, body length 40 cm
- **Camera**: Intel RealSense D435 mounted in robot head, 10±2 Hz, cropped to 58×87 pixels
- **Compute**: Depth server (Jetson NX) processes images at 10 Hz, sends latent + heading to base policy at 50 Hz
- **Latency enforcement**: Constant depth latency of 0.08 s (pause if $t_p < 0.08$) to prevent jitter artifacts
- **Proprioception latency**: 0.016 s
- **Training time**: Single NVIDIA 3090 GPU, <20 hours

## Experimental Results

### Emergent Behaviors (Qualitative)

The unified reward imposes no priors; these behaviors emerge from the optimization:

**High jump (0.5m = 2× hip height):**
1. Stride shortens as robot approaches; aligns front and rear feet at correct distance from edge
2. Rear legs kick out with high torque to propel body upward
3. Front legs extend to clear top of obstacle, then pull the body up
4. Rear legs tuck to clear the object boundary
5. Returns to stable walking gait

**Long jump (0.8m = 2× body length):**
1. Front feet align with near edge; rear feet also move forward to maximize jumping distance
2. Hind legs kick to propel forward and upward; front legs extend toward far side
3. Hind legs extend to maximize force application duration; then tuck mid-air
4. Both sides land on far side; front legs extend again to resume normal gait

**Handstand (bipedal on front legs):**
1. Robot bends forward, shifts weight onto front legs
2. Rear legs kick upward just enough to reach vertical position
3. Rear legs held in neutral pose with tiny active adjustments for balance
4. Robust across indoor, outdoor, and deformable surfaces (grass); can descend stairs without vision

### Simulation Results (Table 2)

Obstacle course: series of each terrain in increasing difficulty. 256 robots spawned, run for 30 s.
Metrics: **MXD** (mean x-displacement ↑), **MEV** (mean edge violation ↓).

| Terrain | Ours MXD | NoInner MXD | NoClear MXD | Noisy MXD | Ours MEV | NoClear MEV | Noisy MEV |
|---------|----------|-------------|-------------|-----------|----------|-------------|-----------|
| Hurdle  | 0.99±0.05 | 0.90±0.12  | 1.00±0.03  | 0.78±0.26 | 0.04±0.21 | 0.12±0.38  | 0.31±0.58 |
| Step    | 0.99±0.07 | 0.14±0.00  | 0.92±0.24  | 0.84±0.29 | 0.04±0.20 | 0.07±0.27  | 0.15±0.38 |
| Gap     | 0.96±0.14 | 0.86±0.26  | 0.96±0.12  | 0.87±0.24 | 0.02±0.14 | 0.07±0.32  | 0.06±0.25 |
| Ramps   | 1.00±0.04 | 0.92±0.24  | 1.00±0.04  | 0.79±0.31 | 0.01±0.11 | 0.04±0.19  | 0.14±0.41 |
| **Total** | **0.98±0.09** | 0.75±0.36 | 0.99±0.06 | 0.82±0.29 | **0.03±0.18** | 0.08±0.32 | 0.20±0.50 |

- **NoInner** (base-frame velocity tracking): learns to walk *around* hurdles; fails completely on steps (collide-and-retry behavior); lowest MEV on hurdles is artifact of avoidance behavior
- **NoClear** (no foot clearance penalty): near-identical MXD but dangerously high MEV — steps on edges, unstable in real world
- **Noisy** (elevation map from fused depth+odometry): works but high variance and high MEV; feet clearance helps somewhat but map noise remains

### Real-World Results (Table 3)

| Method | MXD ↑ | MEV ↓ |
|--------|--------|--------|
| Both (always predicted yaw) | 0.12±0.07 | 0.26±0.57 |
| Mask (yaw zeroed) | 0.05±0.07 | 0.00±0.00 |
| **Ours (MTS)** | **0.92±0.19** | 0.09±0.33 |
| Oracle (ground-truth yaw) | 0.94±0.19 | 0.10±0.32 |

- **Ours** matches oracle performance (MTS effectively closes the gap)
- **Both** (always use predicted yaw): distribution drift causes large failures (noisy yaw → robot spins)
- **Mask** (no direction input): robot cannot navigate at all
- **NoDir** (human joystick direction): fails specifically on tilted ramps (quick direction changes out-of-distribution for the operator) and on last-minute yaw adjustments for jumps

**Success rate vs. difficulty** (Figure 7, 5 trials per terrain per difficulty): Ours achieves 20–80% higher success rate on the most difficult instance of each terrain compared to all baselines.

### Comparison to Prior Work (Table 1)

| Method | Robot | Climb (×height) | Gap (×length) | Ramp | Handstand |
|--------|-------|-----------------|---------------|------|-----------|
| Rudin et al. | AnymalC | 1.1 | 0.75 | ✗ | ✗ |
| Hoeller et al.* | AnymalC | 2.0 | 1.5 | ✗ | ✗ |
| Zhuang et al.* | Unitree-A1 | 1.6 | 1.5 | ✗ | ✗ |
| **Ours** | **Unitree-A1** | **2.0** | **2.0** | **37°** | **✓** |

*Concurrent work. AnymalC is a more expensive industry-standard robot; Extreme Parkour achieves the same climb/gap ratio on the cheaper A1.

## Limitations & Open Questions

- **Sensor quality**: Relies on the Intel RealSense D435; outdoor strong-sunlight and featureless-surface scenarios degrade depth quality.
- **Handstand trained without exteroception**: The handstand policy is separate from the parkour policy and does not use depth — it cannot respond to terrain changes visually.
- **Single front-facing camera**: No lateral awareness; cannot respond to side obstacles or multi-direction parkour.
- **No replanning**: Waypoints must be pre-placed in the environment; the system does not build maps or plan at the semantic level.
- **Modular head/tail policies**: The handstand and parkour skills are not jointly optimized; transitioning between arbitrary skills mid-course is not demonstrated.
- **Future direction**: Extending this paradigm to mobile manipulation (manipulating objects while navigating) is identified as the natural next step.
- **Open question**: Can the same approach scale to even more extreme behaviors (wall runs, vaults) with more capable hardware?
- **Open question**: How does the teacher-student distillation framework scale when the privileged teacher information is more abstract (e.g., semantic scene understanding)?

## See Also

- [Legged Locomotion & Perceptive Control](../concepts/legged-locomotion.md) — concept article for the broader RL-for-legged-robots paradigm
- [Sim-to-Real Transfer](../concepts/sim-to-real.md) — sim-to-real methodology, domain randomization, teacher-student distillation
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — PPO used in Phase 1 RL training
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — comparable sim2real pipeline for arm/hand control
- [Liu et al. (2025) — LocoFormer](liu_2025_locoformer.md) — generalist cross-embodiment follow-up from overlapping authors (Agarwal, Pathak)
