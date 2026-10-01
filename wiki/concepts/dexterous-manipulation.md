---
title: Dexterous Manipulation
type: concept
tags: [robot-learning, dexterous-manipulation, functional-grasping, sim2real, eigengrasp, leap-hand, affordances, in-hand-reorientation]
related: [papers/agarwal_2023_dex_func_grasp.md, concepts/sim-to-real.md, concepts/policy-gradient-methods.md, papers/fedele_2025_superdec.md, concepts/superquadric-scene-representations.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/dex_func_grasp.pdf, raw/papers/pdf/SuperDec.pdf]
---

# Dexterous Manipulation

## What It Is

Dexterous manipulation refers to the ability to pick up, reorient, and functionally use objects with a multi-fingered robotic hand — mimicking the capability of the human hand. This contrasts with the dominant paradigm in robot manipulation, which uses two-fingered parallel-jaw grippers or suction cups, devices that are limited in the range of objects they can grasp and cannot provide the in-hand control needed to apply force for tool use.

A dexterous hand (e.g., the LEAP hand with 16 DOF — 4 joints on each of 3 fingers and a thumb) provides:
- The ability to grasp thin, irregularly shaped objects
- In-hand pose adjustment without releasing the object
- The stable power grasp needed to apply force during tool use (drilling, hammering, screwing)

## Functional vs. Geometric Grasping

**Geometric grasping** optimizes grasp stability metrics (form closure, force closure) without regard for how the object will be used. A hammer can be geometrically stably grasped from its head or handle — but only the handle grasp is *functionally* valid for hammering.

Geometric grasping needs an object model. One lightweight option is to decompose a scanned point cloud into **superquadric primitives** and run an analytic superquadric grasp planner on the chosen part. SuperDec (Fedele et al. 2025) does this for arbitrary objects in real room scans and grasps a milk bottle with a Spot arm, without a learned grasp network. This covers the *geometric* half only; which part serves the object's function still needs semantics. See [Superquadric & Primitive-Based Scene Representations](superquadric-scene-representations.md).

**Functional grasping** requires understanding **affordances**: the regions of an object relevant for its intended use. This understanding:
- Cannot be derived from geometry alone
- Requires semantic knowledge (how humans use tools)
- Must be paired with precise low-level control to execute the grasp

The standard approach decomposes functional grasping into three stages:

```
Pre-grasp pose → Grasp execution → Post-grasp tool use
(WHERE to grasp) (HOW to grasp)   (WHAT to do after)
```

## Functional Affordances and One-Shot Detection

**Affordances** describe functionally relevant object regions. Obtaining them from human video/demos is noisy and does not scale to low-level control. A more scalable approach:

**One-shot DINOv2 feature matching** (Hadjivelichkov et al. 2022, AffCorrs; Agarwal et al. 2023):
1. Annotate one exemplar image per category with a functional mask (e.g., "hammer handle")
2. For a new object instance, match DINOv2 ViT features to find the corresponding region
3. Intersect with a segmentation mask (e.g., from DETIC) to keep predictions within the object boundary
4. Derive the 3D pre-grasp keypoint from depth projection and robot frame transform

DINOv2's self-supervised features capture part-level correspondences across instances, enabling generalization from a single annotation. This outperforms CLIP-based methods (CLIPPort, CLIPSeg) which operate at object level rather than part level.

**Multi-axis grasping:** For objects in arbitrary orientations, run affordance matching from multiple camera angles (e.g., top, front, side views) and select the approach axis with the highest affordance score.

## Eigengrasp Action Space

Training dexterous manipulation directly in the full joint space (16 DOF for LEAP) is challenging:
- High-dimensional search space → sample-inefficient RL
- Easy to fall into physically inconsistent poses (self-collisions, unnatural configurations)
- Physically unrealistic poses do not transfer to real hardware

**Eigengrasps** (Ciocarlie & Allen 2007) are a PCA-based compression of the hand pose space:

1. Collect a small motion-capture dataset $\mathcal{D}$ of 16-DOF hand joint sequences from human VR demonstrations
2. Run PCA on all poses → 9 principal components $\mathbf{e}_1, \ldots, \mathbf{e}_9$ (eigengrasps)
3. RL policy operates in this 9-dim eigengrasp space: $\mathbf{a}_t \in \mathbb{R}^9$
4. Recover joint angles as: $\mathbf{q}_t = \sum_{k=1}^{9} (\mathbf{a}_t)_k \mathbf{e}_k$

**Key advantages over VAE or unconstrained training:**
- PCA is linear and can extrapolate beyond the training poses (unlike a generative VAE)
- Any convex combination of eigenvectors is approximately realistic
- The RL agent can still discover optimal behavior not present in the demos
- Dramatically reduces exploration variance: ~15× lower reward standard deviation across seeds
- Forces smooth, contact-consistent motions that transfer better to hardware

## Blind Grasping Policy

Once the arm is in the pre-grasp pose, the grasping policy operates **blindly** (no visual input), using only proprioception:
- Observations: 7D end-effector pose (position + quaternion) + 16D finger joint angles
- Actions: 9D eigengrasp coordinates

This decoupling works because grasping is a locally reactive behavior — once close to the object, touch and finger position signals suffice. Visual input would require high-frequency closed-loop cameras at the fingertips, which is currently impractical.

A **recurrent (GRU) policy** is preferred over feedforward because:
- Joint velocity is not directly available from the LEAP hand hardware
- The GRU implicitly integrates joint position history to estimate velocity
- The RNN adapts its hidden state to domain randomization during rollout

## Reward Engineering for Grasping

Functional grasping requires pulling the object into the palm (not just touching it). A two-term reward works well in practice:

$$r_\text{hand-obj}(t) = \sum_{i=1}^{3} \exp\!\left(-\frac{\|\mathbf{r}_\text{obj} - \mathbf{r}_\text{hand}\|}{d_i}\right) - 4\|\mathbf{r}_\text{obj} - \mathbf{r}_\text{hand}\|$$

with $d_1 = 10\text{cm}$, $d_2 = 5\text{cm}$, $d_3 = 1\text{cm}$, combined with a binary pickup signal. The multi-scale exponential encourages progressive approach at different distances. With eigengrasp parameterization, no additional shaping is needed.

## Sim-to-Real for Dexterous Manipulation

Dexterous manipulation presents unique sim2real challenges compared to locomotion or rigid-body tasks:
- Continuous surface contacts require accurate friction/stiffness modeling
- High forces during power grasps amplify model errors
- Self-contact between fingers is hard to simulate accurately
- Simple reward functions that work for locomotion lead to unrealistic finger gaiting

Key techniques (beyond standard domain randomization):
1. **Eigengrasp action space** (see above) — restricts exploration to physically realistic poses
2. **Spinning augmentation** — arm spins the grasped object in a circle during training, producing policies that adaptively re-grasp in response to orientation changes
3. **Domain randomization** of object scale, mass, friction, stiffness, damping

See [[sim-to-real]] for a broader treatment of sim-to-real techniques.

## Hardware: LEAP Hand

The LEAP hand (Shaw, Agarwal & Pathak, RSS 2023) is a low-cost, anthropomorphic dexterous hand designed for robot learning:
- 16 actuated DOF (4 joints per finger, 3 fingers + thumb)
- Anthropomorphic kinematics — finger gaiting similar to human hands
- Designed for sim2real transfer with low-cost off-the-shelf components
- Mounted on xArm6 (6-DOF arm) in Agarwal et al. 2023

The LEAP hand is the platform used in Agarwal et al. (2023) for functional grasping experiments.

## See Also

- [Agarwal et al. (2023) — Dexterous Functional Grasping](../papers/agarwal_2023_dex_func_grasp.md) — full paper combining DINOv2 affordances, eigengrasp RL, and sim2real for tool grasping
- [Sim-to-Real Transfer](sim-to-real.md) — covers domain randomization and the reality gap in robot learning
- [Policy Gradient Methods](policy-gradient-methods.md) — PPO is used as the RL training algorithm for grasping policies
- [Joint Embedding Predictive Architecture (JEPA)](joint-embedding-predictive-architecture.md) — DINOv2 (used for affordance matching) is trained with a JEPA-style SSL objective
- [Fedele et al. (2025) — SuperDec](../papers/fedele_2025_superdec.md) — superquadric decomposition of real scans enabling analytic geometric grasping
- [Superquadric & Primitive-Based Scene Representations](superquadric-scene-representations.md)
