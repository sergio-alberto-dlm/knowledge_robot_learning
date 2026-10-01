---
title: "Bahl et al. (2022) — WHIRL: In-the-Wild Human Imitating Robot Learning"
type: paper
tags: [robot-learning, imitation-learning, human-video, visual-imitation, embodiment-gap, real-world-rl, one-shot, manipulation, exploration]
related: [concepts/learning-from-human-videos.md, concepts/dexterous-manipulation.md, concepts/sim-to-real.md, concepts/intrinsic-motivation.md, concepts/model-predictive-control.md, concepts/vision-language-action-models.md]
created: 2026-05-17
updated: 2026-09-25
sources: [raw/papers/pdf/HRIW.pdf]
---

# Bahl et al. (2022) — WHIRL: In-the-Wild Human Imitating Robot Learning

**Venue:** arXiv 2207.09450v1 [cs.RO], 19 Jul 2022  
**Authors:** Shikhar Bahl, Abhinav Gupta, Deepak Pathak — Carnegie Mellon University  
**Project:** https://human2robot.github.io

---

## Abstract

WHIRL is an efficient real-world algorithm for one-shot visual imitation in the wild. Given a single third-person human video of a manipulation task, WHIRL extracts a *prior* over the demonstrator's intent (hand positions, interaction waypoints, wrist orientation, gripper open/close), uses it to initialize an agent policy, and then iteratively refines the policy via real-world interactions. A key challenge is the embodiment mismatch between human and robot; WHIRL addresses this with an *agent-agnostic* video representation (obtained by inpainting the agent out of both human and robot videos) and a high-level action recognition model as the embedding space. The method is demonstrated on 20 diverse manipulation tasks across 3 real-world environments, achieving 83% success on drawer opening and 92% on door opening, strongly outperforming offline RL and behavior cloning baselines.

---

## Key Contributions

- **One-shot in-the-wild imitation**: Learns robot manipulation from a *single* third-person human video, with no robot demonstrations, no task labels, and no reward engineering.
- **Human prior extraction pipeline**: Structured extraction of hand position (100DOH detector on Faster-RCNN), contact timing (contact variable c_t smoothed with Savitzky-Golay filter), and wrist orientation (MANO parameterization) from video, mapped to robot waypoints via a learned f_map function.
- **Agent-agnostic video alignment objective**: Inpaints the agent (human or robot) out of videos using Copy-Paste Networks, then measures distance in an action recognition embedding space (SlowFast 3D ResNets / Multi-Moments): `|Φ(V) - Φ(R_k)||_2`. This bridges the embodiment gap without paired human-robot data.
- **Dual-policy structure**: A *task policy* (CVAE outputting residual ΔΨ_k over the prior) and an *exploration policy* (maximizes frame-wise visual change in the environment) are trained simultaneously, avoiding local minima around the prior.
- **Zeroth-order real-world optimization**: A CEM-style sampling procedure samples M action residuals, executes them, ranks by agent-agnostic cost, and fits the CVAE to the top-k elite set — enabling safe, sample-efficient real-world learning without traditional RL infrastructure.

---

## Methodology

### Three-Phase Loop: Watch → Improve → Repeat

```
Watch:   Human video V_k  →  Extract prior Ψ_k  (hand positions, wrist, gripper)
Improve: Optimize π, π_exp using agent-agnostic objective
Repeat:  Execute robot actions, collect videos R_{k,m}
```

### 1. Human Prior Extraction

Each task video V is processed frame-by-frame to extract:

| Signal | Method | Output |
|--------|--------|--------|
| Hand position h_t = (x_t, y_t, z_t) | 100DOH detector (Faster-RCNN) + depth image d_t | 3D position in camera frame |
| Contact c_t ∈ {no contact, portable, fixed, self} | 100DOH contact classifier + Savitzky-Golay smoothing | Discrete contact event |
| Wrist orientation θ_hand^(t) | MANO parameterization | Yaw-pitch-roll |

The prior is assembled as interaction waypoints **h** = (h_interaction, h_mid, h_end) and mapped to robot frame:

```
f_map(h, θ_hand, o_1:T) = (w, θ_YPR, g_1:T)  ≜  Ψ_k
```

where **w** are Cartesian waypoints, θ_YPR is wrist rotation, g_1:T are gripper open/close commands.

### 2. Policy Structure

**Task Policy (π):** A CVAE conditioned on the human video embedding φ(V_k) and latent z ~ N(0,1). Outputs residual ΔΨ_k. At inference: π = p(x|z,c) with c = φ(V_k).

**Exploration Policy (π_exp):** Maximizes the "change" caused in the environment:

```
c_k = max_{i,j} || Φ_f(R_{k,i}) - Φ_f(R_{k,j}) ||_2
```

where Φ_f is a frame-level embedding (VGG16 features). The exploration policy is also a CVAE with the same architecture, fitted to the top-"change" trajectories.

### 3. Agent-Agnostic Video Alignment

To compare human and robot performance without paired data:
1. Train instance segmentation models to produce human and robot masks.
2. Inpaint the agent out of each video using Copy-Paste Networks (Lee et al., 2019).
3. Embed inpainted videos with an action recognition model Φ (Multi-Moments or SlowFast 3D ResNets).
4. Objective for task policy: minimize `|Φ(V) - Φ(R_k)||_2`.

The exploration policy uses frame-wise Φ_f rather than video-level Φ, augmented with multiple video-level augmentations (salt-pepper jitter, random crop, Gaussian blur, flips) for robustness.

### 4. Training Algorithm (Algorithm 1)

```
for each video V_k (k = 1..K):
    compute prior Ψ_k = f_map(V_k)
    for m = 1..M:
        sample ΔΨ_{k,m} from π_exp (prob p) or π (prob 1-p)
        execute a_{k,m} = Ψ_k + ΔΨ_{k,m}, collect R_{k,m}
    rank by Cost(Φ(R_{k,m}), Φ(V_k)) for all m
    E = top-10 elite set
    fit π as CVAE to Ψ_{k,m} ∈ E
    E_exp = top-"change" trajectories by Φ_f
    fit π_exp as CVAE to Ψ_{k,m} ∈ E_exp
```

M = 30 samples per iteration, top 10 elites, 3 iterations. Each episode ~1 minute on Stretch robot.

### Hardware

- **Robot:** Hello Robot Stretch (mobile base, 6-DOF arm, suction cup gripper)
- **Perception:** Intel RealSense D415 (RGB-D)
- **Control:** Cartesian position control for wrist translation; built-in orientation controller for wrist rotation
- **Policy:** 4-layer MLP (inputs: video embedding + prior Ψ_k); CVAE with latent dim=4

---

## Experimental Results

### Tasks

20 manipulation tasks across 3 environments in the wild (kitchens, labs, offices):
- Articulated objects: Drawer, Door, Dishwasher, Fridge
- Object placement: Shelf Pick-and-Place, Ball-in-Hoop, Stacking Cups, Stacking Dice
- Deformable/cloth: Fold Shirt, Remove Shirt from Hanger
- Other: Cleaning Whiteboard, Garbage Can, Open Tap, Turn Off Light, Place Hat, Remove Lid, Pull Plug, Arrange Chair, Pulling Garbage Bag, Toaster

### Comparison to Baselines (Table I, 30 trials, 3 iterations)

| Method | Drawer | Door |
|--------|--------|------|
| Behavior Cloning | 0.53 | 0.30 |
| Offline RL (CQL-ours) | 0.47 | 0.30 |
| Offline RL (CQL-CycleGAN) | 0.23 | 0.30 |
| Offline RL (CQL-TCN) | 0.27 | 0.20 |
| Offline RL (CQL) | 0.33 | 0.13 |
| WHIRL (no agent-agnostic obj.) | 0.47 | 0.53 |
| WHIRL (no exploration policy) | 0.60 | 0.73 |
| **WHIRL (full)** | **0.83** | **0.92** |

- Starting success rates (prior only): ~43% drawer, ~40% door
- After 2 iterations: 83% drawer, 92% door

### Ablations

- **Agent-agnostic objective essential**: Removing it → essentially no improvement from prior (purple curves in Fig. 7). Standard video alignment focuses too much on agent appearance rather than task outcome.
- **Exploration policy accelerates learning**: Without it, improvement is slower but still possible. The exploration policy prevents convergence to local minima near the prior.
- **Cost function discriminates success**: Agent-agnostic cost is strictly ordered: Failure > Partial Success > Success (Fig. 7c), confirming it provides a meaningful reward signal.

### Generalization

- **New instances** (held-out drawer/door): Success rate lower than train but improves with iterations; prior biases may differ between instances.
- **New scenes** (different part of kitchen, different camera view): Drawer generalizes better than door; door policy in new scene improves from 20% → 57%.
- **Task-level generalization**: Drawer policy transfers to door task and vice versa (Fig. 6f), suggesting shared structure.
- **Multi-task policy** (shared π over 3 tasks): Achieves improvement over all iterations; generalization easier with more data.

---

## Limitations & Open Questions

- **Online interaction required**: WHIRL cannot learn purely from offline video; real-world rollouts are necessary for improvement. Future work aims to remove this requirement.
- **New scene geometry**: Performance degrades significantly in new scenes where camera calibration and geometry change, especially for door tasks. A more geometry-robust prior would help.
- **Precision-sensitive tasks**: Shelf pick-and-place is much harder than kitchen tasks because waypoint errors of a few centimeters cause the robot to get stuck; the prior extraction noise is a bottleneck.
- **Inpainting as bottleneck**: Video inpainting takes 4-6 hours total training time and is the main software bottleneck. End-to-end differentiable agent removal would be preferable.
- **Human prior accuracy**: The 100DOH detector and MANO parameterization can produce noisy estimates in unstructured settings; the sampling approach partially compensates but better priors (e.g., heat sensing, richer keypoint models) would improve performance.
- **Agent-agnostic cost not zero at success**: The embedding distance never reaches zero even for successful trajectories, meaning there is still noise in the reward signal.

> **Verify:** The paper claims this is the first work to take manipulation out of the lab at this scale (20 tasks, 3 environments in the wild). Earlier in-the-wild grasping work (Song et al., 2020) covered grasping only; WHIRL handles arbitrary manipulation.

---

## See Also

- [Learning from Human Videos](../concepts/learning-from-human-videos.md) — general framework and concepts behind in-the-wild visual imitation
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — manipulation with dexterous hands; functional grasping (Agarwal et al., 2023)
- [Sim-to-Real Transfer](../concepts/sim-to-real.md) — alternative paradigm: train in sim, deploy in real; WHIRL sidesteps simulation entirely
- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md) — modern VLAs for imitation learning; RECAP for RL fine-tuning
- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — the exploration policy objective (maximizing environment change) is related to curiosity-driven exploration
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md) — CEM-style zeroth-order optimization used in WHIRL's sampling procedure
