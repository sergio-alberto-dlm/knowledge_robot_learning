---
title: "Bahl et al. (2023) — VRB: Affordances from Human Videos as a Versatile Representation for Robotics"
type: paper
tags: [robot-learning, affordances, human-video, visual-imitation, egocentric-video, imitation-learning, exploration, goal-conditioned-rl, action-space, contact-points]
related: [concepts/visual-affordances-robotics.md, concepts/learning-from-human-videos.md, concepts/dexterous-manipulation.md, concepts/intrinsic-motivation.md, concepts/value-based-rl.md]
created: 2026-05-17
updated: 2026-09-25
sources: [raw/papers/pdf/video_learning.pdf]
---

# Bahl et al. (2023) — VRB: Affordances from Human Videos as a Versatile Representation for Robotics

**Venue:** arXiv 2304.08488v1 [cs.RO], 17 Apr 2023  
**Authors:** Shikhar Bahl*, Russell Mendonca*, Lili Chen, Unnat Jain, Deepak Pathak (* equal contribution) — CMU, Meta AI  
**Project:** https://robo-affordances.github.io

---

## Abstract

VRB (Vision-Robotics Bridge) aims to bridge the gap between computer vision models trained on human video data and robot learning in the wild. It learns *visual affordances* — defined as contact points (c) and post-contact trajectories (τ) — from large-scale egocentric human interaction videos in a robot-centric, scalable manner. These affordances are actionable: they encode *where* and *how* to interact with objects, and are agnostic to human-robot morphology differences. VRB demonstrates that this single representation can be seamlessly deployed across four distinct robot learning paradigms: offline imitation learning, reward-free exploration, goal-conditioned learning, and action space parameterization. Evaluated across 10 real-world manipulation tasks in 4 environments on 2 robot platforms (Franka Emika Panda + Hello Robot Stretch), VRB substantially outperforms prior affordance methods, achieving a 57% average success rate on offline imitation and 3×–10× exploration improvement over random baselines.

---

## Key Contributions

- **Robot-centric affordance representation**: Defines affordances as (contact point c, post-contact trajectory τ) — where c is a pixel-space contact location and τ is a sequence of relative hand displacements after contact — chosen specifically because both are easily transferable to any robot regardless of morphology.
- **Scalable data-driven affordance learning from egocentric video**: Uses off-the-shelf hand-object detection (100DOH) to automatically extract supervision from videos (no manual annotation), with a GMM fit over contact points to handle multi-modality and homography compensation for camera ego-motion.
- **Domain shift solution**: Warps extracted affordances back to the *human-less* first frame (before the human enters the scene), so the affordance model is conditioned on robot-viewpoint images at inference, not on images containing human bodies.
- **Versatility across four paradigms**: Shows the same trained affordance model f_θ plugs into: (a) offline IL data collection, (b) exploration bootstrapping, (c) goal-conditioned RL, and (d) discretized action space for DQN.
- **Visual representation quality**: The VRB encoder (ResNet trained on egocentric affordance prediction) produces feature spaces that outperform R3M on behavior cloning fine-tuning in simulated Franka environments, with only 2K fine-tuning steps vs. 20K for R3M.

---

## Methodology

### 3.1 Actionable Affordance Representation

Affordances are defined robot-first as the pair **(c, τ)**:

| Symbol | Meaning | Space |
|--------|---------|-------|
| c | Contact point — *where* to interact | 2D pixel coords on object surface |
| τ | Post-contact trajectory — *how* to move | Sequence of relative pixel displacements from t_contact |

This avoids modeling full human body motion (human-centric, non-transferable) and instead captures only the interaction outcome. The contact phase is decomposed as:
- **τ_pre-interaction**: robot moves to contact point
- **τ_interaction**: robot follows post-contact trajectory  
- **τ_post-interaction**: (not explicitly modeled)

### 3.2 Extracting Affordances from Egocentric Video

Given video V = {I_1, …, I_T} of a human performing a task:

**Step 1 — Find contact:** Run 100DOH hand-object detection on each frame. Contact variable o_t indicates contact type. Find first contact timestep t_contact.

**Step 2 — Extract contact points (c):** At frame I_{t_contact}, use skin-color segmentation around the hand bounding box h_t to find all periphery pixels intersecting the interacted object. Fit a **Gaussian Mixture Model (GMM)** to these N contact candidates {c^i}^N:

```
p(c) = argmax_{μ_1,...,μ_K, Σ_1,...,Σ_K} Σ_{i=1}^N Σ_{k=1}^K α_k N(c^i | μ_k, Σ_k)   (1)
```

K clusters target the f_θ GMM heads. Covariance matrices are fixed (not learned) for stability.

**Step 3 — Extract post-contact trajectory (τ):** Pixel positions {h_t}_{t=t_contact}^T constitute τ. Camera ego-motion is compensated via homography H_t (matching features between consecutive frames): τ = H_t ∘ {h_t}_{t_contact}^t′.

**Step 4 — Domain shift fix:** Human-less frame (first frame before human enters) is used to condition the affordance model. Affordances are projected onto this frame via the same homography. If the human is always in frame, crop/discard those frames.

### 3.3 Training the Affordance Model f_θ

Architecture: **ResNet encoder** g_θ^conv → spatial latent z_t → **K deconvolutional heads** for contact heatmaps + **Transformer trajectory network** T_θ.

```
z_t = g_θ^conv(I_t)
H_t = g_θ^deconv(z_t)          # K heatmaps via deconv + spatial softmax σ_2D
τ_pred = T_θ(z_t)              # Transformer with self-attention
```

Losses:
```
L_contact = || μ_i - σ_2D(g_θ^deconv(g_θ^conv(I_t))) ||_2     (2)
L_traj = || τ - T_θ(z_t) ||_2
```

Training on local crops of I_t around contact points (not full images) prevents spurious correlations and improves generalization. Multi-modal contact predictions (K GMM components) are critical since a scene may admit multiple possible interactions (e.g., lift vs. push a lid).

### 3.4 Four Robot Learning Paradigms (Fig. 3)

**A. Offline Imitation Learning (Data Collection Quality)**
1. Query f_θ on scene → produces (c, τ) predictions → execute as robot trajectory
2. Collect dataset D = {(I_t, (c, τ))}
3. Train policy via **k-Nearest Neighbors** (filter D by feature distance to goal image, take 10-closest) or **Behavior Cloning** π(c, τ | I_t)
4. Affordance quality determines data quality → better downstream IL performance

**B. Reward-Free Exploration**
- Seed exploration using f_θ to focus on interaction-relevant scene regions
- Rank collected trajectories by environment-change metric:
  ```
  EC(I_i, I_j) = ||φ(I_i) - φ(I_j)||_2    (first vs. last frame, robot masked)
  ```
- Fit distribution h over (c, τ) of top-ranked trajectories; subsequent data collection samples from h (prob p) or f_θ (prob 1-p)
- Bootstraps from highly exploratory trajectories iteratively

**C. Goal-Conditioned Learning**
- Task specified by a goal image I_g (e.g., opened door)
- Rank trajectories by minimizing distance to goal in feature space: EC(I_T, I_g) or ||ψ(I_g) - ψ(I_T)||_2
- Sample (c, τ) from affordance distribution h filtered toward goal
- Re-collects data online as policy improves

**D. Affordance as Action Space**
- Query f_θ many times on scene → fit GMM → discrete set of (c, τ) candidates
- Train **Deep Q-Network (DQN)** over this discretized action space
- Robot executes: move to c, grasp, follow τ
- Structured action space is much smaller and semantically meaningful than raw end-effector space

### 3.5 Robot Deployment

- **Robot 1**: Franka Emika Panda arm — two distinct play kitchen environments
- **Robot 2**: Hello Robot Stretch — in-the-wild environments (kitchens, offices)
- **Perception**: Intel RealSense D415i (RGB-D); pixel-space (c, τ) projected to 3D via calibrated camera
- **Control**: 6-DOF end-effector space; rotate to c, grasp, follow τ trajectory
- **Image input**: Task-specific image crop via bounding box detection [135]

---

## Experimental Results

### Tasks (10 total, 4 environments)

Franka: Cabinet, Knife, Veg, Shelf, Pot  
Hello/Stretch (in-the-wild): Door, Lid, Drawer, Garbage Can, Dishwasher, Stovepot

### Baselines

| Baseline | Description |
|----------|-------------|
| HOI [66] | Human-object interaction detector, predicts contact + hand pose |
| HAP [39] | "Hands as Probes" — heatmap of where hands go |
| Hotspots [80] | Heatmap of interaction frequency |
| Random | Random affordance sampling |

### A. Imitation Learning — Offline Data Collection Quality (Table 1)

Success rates (k-NN / Behavior Cloning, 8 environments):

| Method | Cabinet | Knife | Veg | Shelf | Pot | Door | Lid | Drawer | Avg |
|--------|---------|-------|-----|-------|-----|------|-----|--------|-----|
| HOI (k-NN) | 0.2 | 0.1 | 0.1 | 0.6 | 0.0 | 0.4 | 0.0 | 0.6 | — |
| HAP (k-NN) | 0.3 | 0.0 | 0.1 | 0.0 | 0.1 | 0.4 | 0.0 | 0.1 | — |
| Hotspots (k-NN) | 0.4 | 0.0 | 0.1 | 0.0 | **0.5** | 0.4 | 0.3 | 0.5 | — |
| **VRB (k-NN)** | **0.6** | **0.3** | **0.6** | **0.8** | 0.4 | **1.0** | **0.4** | **1.0** | **~57%** |
| HOI (BC) | 0.3 | 0.0 | 0.3 | 0.0 | 0.1 | 0.2 | 0.0 | 0.1 | — |
| HAP (BC) | 0.5 | 0.0 | **0.4** | 0.0 | 0.3 | 0.1 | 0.0 | 0.1 | — |
| Hotspots (BC) | 0.2 | 0.0 | 0.1 | 0.0 | 0.2 | 0.1 | **0.8** | 0.7 | — |
| **VRB (BC)** | **0.6** | **0.1** | 0.1 | **0.3** | **0.3** | **0.8** | 0.2 | **0.9** | — |

VRB outperforms all baselines on 7/8 tasks (k-NN) and 6/8 tasks (BC).

### B. Reward-Free Exploration (Fig. 5)

Coincidental success (reaching goal without having access to goal image) on 4 tasks:
- VRB achieves 3×–10× improvement over random exploration on all tasks
- Consistent improvement over HAP on Cabinet, Knife, Stovepot, Shelf

### C. Goal-Conditioned Learning (Fig. 6)

VRB leads to faster learning and higher final performance on all 6 tested tasks (Door, Veggies, Lid, Dishwasher, Drawer, Garbage Can) compared to HAP and Random.

### D. Action Space + DQN (Fig. 7)

VRB action space discretization leads to more successes on Cabinet and Veggies tasks compared to HAP, confirming that affordance-structured action spaces are easier for DQN to search.

### E. Visual Representation Quality (Table 2)

Behavior cloning with frozen encoder (2K fine-tuning steps):

| Method | Microwave | Slide-door | Door-open |
|--------|-----------|------------|-----------|
| R3M [84] | 0.10 | 0.70 | 0.11 |
| **VRB (ours)** | **0.16** | **0.84** | **0.13** |

VRB outperforms R3M on all 3 simulated Franka tasks with 10× fewer fine-tuning steps, suggesting the egocentric affordance pretraining yields representations useful for robot control.

### F. Feature Space Distance (Fig. 8)

VRB's feature distance to goal image decreases monotonically during successful episodes (cabinet opening, door opening), correlating well with task progress and enabling reliable reward shaping for goal-conditioned learning.

### G. Failure Mode Analysis (Fig. 9)

VRB has ~2× more successes than HAP/Random, and crucially >6× more *partial successes* — indicating the robot makes meaningful progress even when it doesn't fully complete the task. This is in contrast to baselines which mostly produce outright failures.

---

## Limitations & Open Questions

- **Contact + trajectory coverage**: Assumes most manipulation tasks can be decomposed into a contact point followed by a trajectory. Tasks requiring multi-stage contact, re-grasping, or non-contact manipulation (pushing with palm) may need richer representations.
- **Force and tactile**: The representation captures geometric affordances only. Future work should incorporate physical signals (force, compliance, deformation) for tasks like squeezing, unscrewing, or precise insertion.
- **Camera calibration sensitivity**: Projecting pixel-space (c, τ) to 3D robot actions requires accurate camera calibration; errors propagate directly to action quality.
- **Multi-stage tasks**: The current work addresses single-contact tasks. Long-horizon tasks requiring task planning and multiple contact events are not evaluated.
- **Internet-scale**: Data is collected from egocentric datasets (Epic-Kitchens scale). Truly internet-scale training from YouTube would require handling far more diverse camera viewpoints and compositions.

> **Verify:** The paper reports spanning "several hundred hours of robot running time" for evaluations — this is one of the largest real-world robot learning evaluations in the literature at time of publication (Apr 2023).

---

## See Also

- [Visual Affordances for Robotics](../concepts/visual-affordances-robotics.md) — general concept article for (contact point, trajectory) affordance representations from human video
- [Learning Robot Manipulation from Human Videos](../concepts/learning-from-human-videos.md) — predecessor WHIRL paper (Bahl et al. 2022) from the same group; VRB generalizes the affordance extraction and applies it to 4 paradigms
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — DINOv2-based one-shot affordance detection (Agarwal et al. 2023); complements VRB's learned affordances
- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — VRB's reward-free exploration paradigm builds on curiosity/environment-change metrics
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — VRB uses DQN over the affordance-discretized action space in paradigm D
- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md) — modern alternative for scalable robot manipulation; VRB uses affordances rather than language as the generalization mechanism
