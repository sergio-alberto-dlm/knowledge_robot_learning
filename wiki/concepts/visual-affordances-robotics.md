---
title: Visual Affordances for Robotics
type: concept
tags: [robot-learning, affordances, human-video, egocentric-video, contact-points, trajectory, representation-learning, manipulation, versatile-representation]
related: [concepts/learning-from-human-videos.md, concepts/dexterous-manipulation.md, concepts/intrinsic-motivation.md, concepts/value-based-rl.md, concepts/vision-language-action-models.md, concepts/superquadric-scene-representations.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/video_learning.pdf]
---

# Visual Affordances for Robotics

## Concept Origin

The term *affordance* originates with J.J. Gibson (1979): "the meaning or value of a thing consists of what it affords... what we perceive when we look at objects are their affordances, not their qualities." A chair affords sitting; a door handle affords pulling. In robotics, this ecological concept is operationalized as actionable knowledge about *where* and *how* to interact with an object.

## The Robot-Centric Affordance Representation

For a robot to use affordances, they must be defined in terms the robot can act on. A useful robot-centric representation is the **(contact point, post-contact trajectory)** pair:

| Symbol | Meaning | Key property |
|--------|---------|-------------|
| **c** | Contact point — pixel location where the robot should make initial contact with the object | Spatially grounded; projects to 3D via depth |
| **τ** | Post-contact trajectory — sequence of relative end-effector displacements after contact | Relative (not absolute) positions, transferable across scenes |

This pair is agnostic to human morphology: it does not model hands, fingers, or arm kinematics — only the object-relevant outcome. A robot uses (c, τ) by: (1) navigating to the 3D point corresponding to c, (2) grasping, and (3) following the trajectory τ.

**Why not model full human motion?** Human body motion is morphologically incompatible with robot kinematics. Modeling it would require solving the embodiment gap (see [[learning-from-human-videos]]). The (c, τ) abstraction cuts through this problem: the robot-centric decomposition is the same regardless of whether the demonstrator was a human or a different robot.

## Learning Affordances from Egocentric Video

Large datasets of egocentric human-interaction video (e.g., Epic-Kitchens) provide free supervision for affordance learning:

### Automated Label Extraction

Given video V = {I_1, …, I_T}:

1. **Find contact timestep** t_contact using a hand-object interaction detector (100DOH on Faster-RCNN).
2. **Extract contact points** from frame I_{t_contact}: apply skin-color segmentation around the hand bounding box; collect all periphery pixels intersecting the object surface → set {c^i}^N.
3. **Fit GMM** to contact candidates to model multi-modality (a scene may afford multiple interaction styles):
   ```
   p(c) = argmax Σ_i Σ_k α_k N(c^i | μ_k, Σ_k)
   ```
4. **Extract trajectory** τ: pixel positions {h_t}_{t_contact}^T of the hand bounding box. Compensate camera ego-motion via homography H_t between consecutive frames: τ = H_t ∘ {h_t}_{t_contact}.

### Domain Shift: Human-Less Frame

Training videos contain human bodies; robot deployment images do not. This *visual domain shift* is addressed by a simple trick: warp all affordance labels back to the first frame where the human has not yet entered the scene (via the same homography procedure). The affordance model is then conditioned on a human-free image at both training and inference time.

### Affordance Model Architecture

```
Input: I_t (human-less scene image, cropped around task-relevant region)
      ↓
ResNet encoder g_θ^conv  →  spatial latent z_t
      ↓                              ↓
K deconv heads g_θ^deconv    Transformer T_θ (self-attention)
      ↓                              ↓
Heatmaps H_t = σ_2D(·)        τ_pred (relative trajectory)
(contact point distribution)
```

**Loss functions:**
```
L_contact = || μ_i - σ_2D(g_θ^deconv(g_θ^conv(I_t))) ||_2
L_traj    = || τ - T_θ(z_t) ||_2
```

Training uses *local crops* around contact points rather than full images to prevent spurious correlations and improve generalization to new scene configurations.

## Affordances as a Versatile Interface for Four Robot Paradigms

The power of the (c, τ) representation is that a single trained model f_θ can bootstrap four fundamentally different robot learning approaches:

### 1. Offline Imitation Learning (Data Collection Quality)

Standard IL from human data requires expensive teleoperation. With affordances:
- Query f_θ(scene) → execute (c, τ) → collect robot interaction dataset
- Downstream: k-Nearest Neighbors (find trajectories near goal in feature space) or Behavior Cloning
- Affordance quality directly determines data quality → better IL performance

### 2. Reward-Free Exploration

Exploration from scratch wastes time on irrelevant actions. Affordances provide a focused prior:
- Bias exploration toward predicted contact regions and trajectories
- Rank trajectories by environment-change metric: EC(I_i, I_j) = ||φ(I_i) - φ(I_j)||_2 (robot masked)
- Iteratively fit distribution h over top-ranked (c, τ) values; subsequent episodes sample from h with probability p
- Achieves 3×–10× improvement over random exploration (VRB, Bahl et al. 2023)

This is related to curiosity-driven exploration ([[intrinsic-motivation]]) but grounded in a task-relevant prior rather than pure novelty.

### 3. Goal-Conditioned Learning

Goal images specify desired object states (e.g., an opened drawer). Affordances guide search:
- Rank trajectories by feature distance to goal image: ||ψ(I_goal) - ψ(I_T)||_2
- Sample (c, τ) from the distribution over highest-ranked trajectories
- Iteratively re-collect data as policy improves toward the goal

### 4. Affordance as Action Space Parameterization

Continuous end-effector action spaces are hard to search for RL. Affordances discretize them meaningfully:
- Query f_θ many times → GMM over (c, τ) → discrete set of semantically meaningful actions
- Train DQN ([[value-based-rl]]) over this reduced action space
- Confines RL search to human-informed interaction points; eliminates physically meaningless actions

## Relationship to Other Affordance Approaches

| Approach | Representation | Transfer to robot | Multi-modal | Ego-motion handled |
|----------|---------------|------------------|-------------|-------------------|
| Hotspots [80] | Frequency heatmap | No | No | No |
| HAP [39] | Hand heatmap | No | No | No |
| HOI [66] | Contact + hand pose | Partially | No | No |
| **VRB (c, τ)** | Contact + trajectory | Yes (robot-first) | Yes (GMM) | Yes (homography) |

The critical differences: VRB is robot-first (not human-centric), explicitly multi-modal, and handles camera motion — all of which matter for deployment in unstructured environments.

## Connection to One-Shot Affordance Detection (DINOv2)

A complementary approach (Agarwal et al. 2023, [[dexterous-manipulation]]) uses DINOv2 feature matching to detect *functional* affordances (e.g., hammer handle vs. head) in a one-shot, training-free manner. VRB vs. DINOv2 affordances differ in:
- **VRB**: learned from video, captures dynamic interaction (trajectory), generalizes across task categories, requires egocentric video data.
- **DINOv2 matching**: training-free, works from a single annotated exemplar image, captures static part semantics, no trajectory information.

## Open Questions

- Can affordance representations scale to internet-scale YouTube video with diverse viewpoints?
- Can physical affordances (force, compliance, deformability) be extracted from video using audio or motion signals?
- How to chain multiple (c, τ) affordances for long-horizon multi-contact tasks?
- Can affordance models be fine-tuned with robot experience to correct for domain shift without losing the internet-video generalization?

## See Also

- [Learning Robot Manipulation from Human Videos](learning-from-human-videos.md) — broader context; WHIRL (predecessor) and VRB's affordance pipeline
- [Dexterous Manipulation](dexterous-manipulation.md) — one-shot DINOv2 affordance detection; complementary approach without learned video model
- [Intrinsic Motivation & Curiosity-Driven RL](intrinsic-motivation.md) — affordance-seeded exploration relates to curiosity-driven exploration
- [Value-Based Reinforcement Learning](value-based-rl.md) — DQN used over the affordance-discretized action space in VRB paradigm D
- [Bahl et al. (2023) — VRB](../papers/bahl_2023_vrb.md) — primary instantiation of robot-centric visual affordances
- [Superquadric & Primitive-Based Scene Representations](superquadric-scene-representations.md) — geometry-centric counterpart: compact explicit primitives rather than learned interaction affordances
