---
title: "Dexterous Functional Grasping"
type: paper
tags: [robot-learning, dexterous-manipulation, functional-grasping, sim2real, affordances, eigengrasp, dinov2, leap-hand, ppo, corl]
related: [concepts/dexterous-manipulation.md, concepts/sim-to-real.md, concepts/policy-gradient-methods.md, concepts/pretrained-visual-representations.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/dex_func_grasp.pdf]
---

# Dexterous Functional Grasping

**Authors:** Ananye Agarwal, Shagun Uppal, Kenneth Shaw, Deepak Pathak  
**Affiliation:** Carnegie Mellon University  
**arXiv:** 2312.02975v1 — 5 Dec 2023  
**Published:** CoRL 2023 (7th Conference on Robot Learning)  
**Project page:** https://dexfunc.github.io/

---

## Abstract

While significant strides have been made in dexterous manipulation, most is limited to benchmark tasks like in-hand reorientation of limited utility in the real world. The main benefit of dexterous hands over two-fingered grippers is their ability to pick up tools and other objects (including thin ones) and grasp them firmly in order to apply force. This task requires a complex understanding of functional affordances as well as precise low-level control. Prior work obtains affordances from human data (not scalable to low-level control) or uses simulation training (which cannot give the robot an understanding of real-world semantics). This paper combines both: affordances are obtained by matching corresponding regions of different objects, and a low-level policy trained in sim grasps the object. A novel eigengrasp action space reduces the RL search space using a small amount of human data, leading to more stable and physically realistic motion. Eigengrasp action spaces beat baselines in simulation and outperform hardcoded grasping in real, matching or outperforming a trained human teleoperator.

---

## Key Contributions

- **Modular three-stage pipeline:** Decomposes functional grasping into (1) pre-grasp affordance estimation, (2) blind sim2real grasping policy, and (3) post-grasp trajectory execution — each stage optimized independently with the right data source.
- **One-shot affordance model via DINOv2 feature matching:** Annotate one exemplar image per category; localize the functional region on new instances by matching DINOv2 features. Handles arbitrary object orientations via 3-camera multi-axis setup.
- **Eigengrasp action space:** PCA on a small VR motion-capture dataset of 16-DOF hand poses yields 9 principal components (eigengrasps). RL policy operates in this 9-dim space; forces physically realistic poses, halves the search space, and stabilizes training across random seeds.
- **Recurrent grasping policy (GRU):** A stateful RNN policy outperforms a feedforward policy because it can implicitly capture joint velocity (unavailable from the hand hardware) and adapt to domain randomization without explicit velocity inputs.
- **Real-world performance:** Matches or exceeds a trained teleoperator (20 hours of experience with a VR glove) on stapler, screwdriver, and hammer; achieves 70–100% success rate on 7 diverse objects out-of-distribution at test time.

---

## Methodology

### Overview: Three-Stage Pipeline

```
Internet data → [1. Affordance Model] → Pre-grasp pose
                                             ↓
VR hand demos → [2. Sim2Real RL Policy] → Grasp execution
                 (eigengrasp space)          ↓
MoCap/keypoints → [3. Post-grasp trajectory] → Tool use
```

### Phase 1: Pre-Grasp Pose from Affordances

**Functional affordances** describe the region of an object relevant for its intended use (e.g., the handle of a hammer, not the head). These cannot be inferred from geometry alone and require semantic understanding.

**Method: One-shot DINOv2 feature matching** (following Hadjivelichkov et al. 2022, AffCorrs):
1. Annotate one exemplar image per object category with a functional mask.
2. For a new instance, match DINOv2 ViT features pixel-by-pixel to find the region corresponding to the annotated mask.
3. Intersect with DETIC segmentation to prevent the mask bleeding across the object boundary.
4. Take the center of the resulting mask as the keypoint $(x_\text{img}, y_\text{img})$.
5. Project to depth image → camera intrinsics/extrinsics → robot frame $(x_\text{robot}, y_\text{robot}, z_\text{robot})$.
6. Compute hand orientation **q** as perpendicular to the largest principal component of the object mask.

**Multi-axis grasping:** Three D435 cameras along the x, y, z axes. Run affordance matching from all three views; select the axis with highest affordance score. This handles upright objects (drills, mugs) where top-down approach is invalid.

**Pre-grasp execution:** Move hand to $(x_\text{robot}, y_\text{robot}, z_\text{robot}) + \delta\mathbf{v}$ (fixed offset along grasp axis), then set finger joints to midrange as a neutral pre-grasp pose. The policy adapts to errors in this pose during grasping.

### Phase 2: Sim2Real Grasping Policy

**The challenge:** Dexterous grasping involves continuous surface contacts and high forces — unlike locomotion or in-hand reorientation where simple reward functions and direct sim2real transfer suffice. Standard sim2real yields unrealistic "finger-gaiting" that does not transfer.

#### Eigengrasp Action Space

A small VR motion-capture dataset $\mathcal{D} = \{\tau_1, \ldots, \tau_n\}$ of 16-DOF hand poses is collected. PCA on all poses yields 9 eigenvectors (eigengrasps) $\mathbf{e}_1, \ldots, \mathbf{e}_9$.

The policy outputs $\mathbf{a}_t \in \mathbb{R}^9$; raw joint angles are reconstructed as:

$$\mathbf{q}_t = \sum_{k=1}^{9} (\mathbf{a}_t)_k \mathbf{e}_k$$

**Why this works better than VAE or unconstrained:**
- PCA (unlike a generative VAE) can extrapolate to poses not seen in the dataset — any convex combination of eigenvectors is realistic.
- Forces the exploration space to contain only physically plausible poses, eliminating self-collisions and erratic finger gaiting.
- Halves the effective search space (16 → 9 dims), reducing sample complexity exponentially.
- Does not constrain the policy to imitate the demos — it can still discover optimal behavior from suboptimal data.

#### Reward Function

Two-term reward encouraging the object to be close to the palm and lifted:

$$r_\text{hand-obj}(t) = \sum_{i=1}^{3} \exp\!\left(-\frac{\|\mathbf{r}_\text{obj} - \mathbf{r}_\text{hand}\|}{d_i}\right) - 4\|\mathbf{r}_\text{obj} - \mathbf{r}_\text{hand}\|$$

with $d_1 = 10\text{cm}$, $d_2 = 5\text{cm}$, $d_3 = 1\text{cm}$.

Binary pickup signal:
$$r_\text{threshold}(t) = \mathbb{1}[(\mathbf{r}_\text{obj}(t))_z \geq 0.04\text{cm}]$$

Overall: $r(t) = r_\text{hand-obj}(t) + 0.1 \cdot r_\text{threshold}(t) + 1$

No reward shaping beyond this — the eigengrasp parameterization makes additional shaping unnecessary.

#### Policy Architecture and Training

- **Policy:** Stateful GRU with layer norm (256 hidden), followed by MLP (512, 256, 128). Observes 7D end-effector pose + 16D finger joint angles ($\mathbf{o}_t \in \mathbb{R}^{16}$), outputs $\mathbf{a}_t \in \mathbb{R}^9$.
- **Blind policy:** No visual input during grasping — purely proprioceptive. The affordance model handles "where to grasp"; the policy handles "how to grasp."
- **Trained with PPO** using IsaacGym (8192 parallel environments), 400 epochs.
- **Domain randomization:** Object scale [0.8, 1.2], mass [0.5, 1.5], friction [0.7, 1.3], stiffness [0.75, 1.5], damping [0.3, 3.0]. Gaussian noise added to observations and actions.
- **Training object:** Procedurally generated hammers only — all other objects (screwdriver, drill, stapler, saucepan) are out-of-distribution at test time.
- **Spinning augmentation:** During training the arm spins the grasped object in a circle, producing emergent adaptive grasps that adjust to orientation changes.

### Phase 3: Post-Grasp Trajectory

Once firmly grasped, the 6-DOF arm executes a task trajectory (hammering, drilling, screwing) via motion capture or keypoint interpolation. The arm can be moved arbitrarily in space once the object is secured.

---

## Experimental Results

### Simulation (Table 1)

Trained on hammer; evaluated on hammer, drill, screwdriver across 5 random seeds.

| Method | Hammer Reward | Drill Reward | Hammer SR | Drill SR | Screwdriver SR |
|--------|--------------|--------------|-----------|----------|----------------|
| Unconstrained (16D) | 213 ± 169 | 102 ± 36 | 0.60 | 0.09 | 0.46 |
| VAE (latent space) | 140 ± 109 | 83 ± 43 | 0.30 | 0.08 | 0.25 |
| Feed-forward | 232 ± 175 | 104 ± 44 | 0.60 | 0.21 | 0.56 |
| **Ours (Eigengrasp + RNN)** | **327 ± 11** | **211 ± 11** | **1.00** | **0.23** | **0.95** |

Key findings:
- Our method has ~15× lower reward variance across seeds — nearly deterministic performance.
- Unconstrained and VAE methods fluctuate widely; VAE is worst because it cannot extrapolate to unseen poses.
- RNN beats feedforward because it implicitly captures finger joint velocity (not available from hardware).

### Real World (Table 2)

10 trials per object, orientation randomized in $[-\pi, \pi]$, position randomized in $1\text{m} \times 0.5\text{m}$.

| Object | Teleop Oracle | Hardcoded | **Ours** |
|--------|--------------|-----------|---------|
| Hammer (heavy) | 0.5 | 0.0 | **0.8** |
| Hammer (light) | 0.6 | 0.3 | **0.9** |
| Saucepan | 0.9 | 0.3 | **0.9** |
| Drill (heavy) | 0.9 | 0.2 | 0.5 |
| Drill (light) | 0.9 | 0.3 | **0.8** |
| Stapler | 0.9 | 0.3 | **1.0** |
| Screwdriver | 0.5 | 0.0 | **0.7** |

Notable: Our method **exceeds** the teleop oracle on stapler, screwdriver, and heavy hammer — heavy objects require swift forceful motion that humans execute poorly via VR glove but the policy does reliably.

### Affordance Matching (Table 3)

Evaluated on CLIPPort dataset; Hammer is unseen, Spatula and Frying Pan are seen.

| Method | Hammer Pick↑ | IoU↑ | Spatula Pick↑ | IoU↑ |
|--------|-------------|------|--------------|------|
| CLIPPort | 2/10 | 0.034 | 6/10 | 0.15 |
| CLIPSeg | 1/10 | 0.05 | 2/10 | 0.06 |
| **Ours** | **8/10** | **0.33** | **8/10** | **0.23** |

CLIPSeg fails because it segments the whole object, not the functional part. CLIPPort's language grounding is imprecise. DINOv2 feature matching captures part-level correspondences.

### Emergent Behavior

Despite a simple reward, the policy exhibits complex adaptive behaviors:
- Detects thumb-object collision from proprioception and repositions the thumb autonomously.
- Adjusts thumb position when the heavy drill is moved upright during grasping.

---

## Limitations & Open Questions

- **Large pre-grasp errors unrecoverable:** The blind grasping policy cannot recover if the arm is far from the object. Adding a wrist camera for local fine-tuning of the grasp pose is a natural next step.
- **No joint pose information from affordance model:** The pre-grasp pose is computed from 2D/3D keypoints but not object joint structure — this limits generalization to very thin objects (coins, credit cards).
- **Policy trained only on hammers:** Remarkable generalization to screwdrivers, drills, staplers suggests good domain randomization, but systematic evaluation on more diverse object categories is needed.
- **Post-grasp trajectory is scripted:** Task trajectories (hammering, drilling) are obtained from MoCap or keypoint interpolation, not learned end-to-end. Full task learning from image goals remains future work.
- **Heavy drill is still the failure case:** 50% success rate reflects the narrow grip region and unbalanced weight distribution — the most challenging geometry for the current approach.
- **Open question:** Can eigengrasp action spaces generalize to whole-body manipulation (arm + hand joint PCA), and would larger motion-capture datasets yield qualitatively better grasping policies?

---

## See Also

- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — concept article covering dexterous hands, functional grasping, eigengrasp action spaces, and LEAP hand
- [Sim-to-Real Transfer](../concepts/sim-to-real.md) — covers domain randomization, the reality gap, and techniques for bridging simulation and real-world deployment
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — PPO is used as the base RL algorithm for training the grasping policy
- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md) — DINOv2 (used for affordance matching) is trained with a JEPA-style SSL objective
- [Pretrained Visual Representations for Robotics](../concepts/pretrained-visual-representations.md) — DINOv2 dense features for perception and world models
