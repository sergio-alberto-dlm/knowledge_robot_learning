---
title: "von Hartz et al. (2025) — MiDiGaP: Mixture of Discrete-Time Gaussian Processes for Robot Policy Learning"
type: paper
tags: [imitation-learning, gaussian-processes, robot-manipulation, policy-representation, multimodal, few-shot, task-parameterization, cross-embodiment, inference-time-adaptation]
related: [concepts/discrete-time-gaussian-processes.md, concepts/sim-to-real.md, concepts/dexterous-manipulation.md, concepts/vision-language-action-models.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/midigap.pdf]
---

# von Hartz et al. (2025) — MiDiGaP

**arXiv:** 2505.03296v1 (6 May 2025, cs.RO)
**Authors:** Jan Ole von Hartz, Adrian Röfer, Joschka Boedecker, Abhinav Valada (University of Freiburg)
**Code:** https://midigap.cs.uni-freiburg.de

## Abstract

MiDiGaP (Mixture of Discrete-time Gaussian Processes) is a novel approach for flexible policy representation and imitation learning in robot manipulation. It enables learning from as few as five demonstrations using only camera observations and generalizes across a wide range of challenging tasks: long-horizon behaviors (making coffee), highly constrained motions (opening doors), dynamic actions (scooping with a spatula), and multimodal tasks (hanging a mug). Training runs on a CPU in under one minute and scales linearly to large datasets. The method also provides a rich suite of inference-time steering tools using evidence such as collision signals and robot kinematic constraints, enabling novel generalization including obstacle avoidance and cross-embodiment policy transfer.

## Key Contributions

1. **DiGaP** (Discrete-time Gaussian Process): a finite sequence of per-timestep Gaussians on Riemannian manifolds — a flexible, kernel-free unimodal trajectory distribution that models oscillatory, piecewise, non-stationary, and discontinuous functions where CoGaP and GMMs fail.
2. **MiDiGaP**: mixture of DiGaPs for multimodal policy learning with automatic mode discovery (no prior knowledge of number of modes required).
3. **Modal partitioning**: three strategies (Riemannian GMM-BIC, k-means, DBSCAN) for unsupervised clustering of demonstrations into trajectory modes; k-means recommended as robust default.
4. **Skill segmentation and sequencing** via TAPAS integration: automatic decomposition of long-horizon tasks into skill sequences, with learned transition probabilities and novel skill composition via KL-divergence.
5. **Constrained Gaussian Updating**: inference-time updating of MiDiGaP using (a) modal evidence (zeroing infeasible modes) and (b) convex evidence (moment-matching to adjust Gaussian parameters) — handles collision, reachability, and scene object constraints.
6. **VAPOR** (Variance-Aware Path Optimization): leverages MiDiGaP's predicted variance to guide IK toward kinematically feasible joint trajectories, enabling effective cross-embodiment transfer.

## Methodology

### DiGaP: Discrete-time Gaussian Process

A DiGaP is a finite sequence of T Gaussian components fit to a dataset of N trajectories:

$$\mathcal{GP} = \bigl((\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)\bigr)_{t=1}^T$$

Fitting on the Riemannian manifold $\mathcal{M}_\text{pose} = \mathbb{R}^3 \times S^3$ (robot end-effector pose):

$$\boldsymbol{\mu}_t = \arg\min_{\mathbf{x}\in\mathcal{M}} \sum_{n=1}^N d_\mathcal{M}(\mathbf{x}, \mathbf{z}_t^n)^2$$

$$\hat{\boldsymbol{\Sigma}}_t = \frac{1}{N-1}\sum_{n=1}^N \bigl(\mathrm{Log}_{\boldsymbol{\mu}_t}(\mathbf{z}_t^n)\bigr)\bigl(\mathrm{Log}_{\boldsymbol{\mu}_t}(\mathbf{z}_t^n)\bigr)^\top, \quad \boldsymbol{\Sigma}_t = \mathrm{diag}(\hat{\boldsymbol{\Sigma}}_t)$$

The covariance is **diagonalized**: this suppresses spurious inter-dimensional correlations (a failure mode of GMMs on Riemannian data). Operating at 20 Hz, fitting scales $O(N)$ in samples and inference costs $\sim$1 ms.

**Why DiGaP beats CoGaP (continuous GP):** CoGaP requires a kernel function which restricts expressivity to locally linear or smooth functions. DiGaP models oscillatory, piecewise-linear, non-stationary, discontinuous, and chaotic functions (see Fig. 3 comparison). It also avoids CoGaP's cubic training and quadratic inference complexity.

### MiDiGaP: Mixture of Discrete-time Gaussian Processes

$$\mathcal{GPM} = \Bigl\{\pi^m, \bigl\{\boldsymbol{\mu}_t^m, \boldsymbol{\Sigma}_t^m\bigr\}_{t=1}^T\Bigr\}_{m=1}^M$$

where $\pi^m = |\mathcal{DM}_m|/|\mathcal{D}|$ (fraction of demonstrations belonging to mode $m$).

**Inference:** sample a mode $m \sim \pi$ and follow the most likely trajectory $\{\boldsymbol{\mu}_t^m\}_{t=1}^T$.

### Modal Partitioning

Demonstrations are embedded as concatenated trajectory vectors $\mathbf{v}_n \in \mathcal{M}^{T'}$ and clustered:

1. **Riemannian GMM + BIC**: iterative EM, most principled, slow for large datasets.
2. **Riemannian k-means + BIC** *(recommended)*: near-linear scaling, robust; completes in 0.3–24 s.
3. **DBSCAN with geodesic distance**: fastest (<1 s), but fails when modes are separable only in the second half of trajectories (misses variance information).

**Mode discovery:** modal partitioning can reveal *unexpected* modes. On `TurnTap`, expected 2 modes (one per valve); found 4 (two approach styles per valve), which are important for hardware safety.

### Skill Segmentation and Sequencing

Long-horizon tasks are segmented using TAPAS into a sequence of skill datasets $\mathcal{D}_1,\ldots,\mathcal{D}_n$, each fit with a MiDiGaP $\mathcal{GPM}_j$. Transition probabilities between modes across consecutive skills:

$$\pi_j(k, l) = \frac{|\mathcal{DM}_k \cap \mathcal{DM}_l|}{|\mathcal{DM}_k|}$$

For novel skill sequences not in the training set, two MiDiGaPs are sequenced by comparing modes via KL-divergence:

$$\pi_j(k,l) \propto \exp\!\Bigl(-D_{KL}\,\mathcal{N}(\boldsymbol{\mu}_{T_i}^k, \boldsymbol{\Sigma}_{T_i}^k) \,\|\, \mathcal{N}(\boldsymbol{\nu}_0^l, \boldsymbol{\Lambda}^{0l})\Bigr)$$

### Constrained Gaussian Updating

**Moment matching** (for convex constraints): sample $\{\mathbf{x}_i\} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, keep feasible set $\mathbb{S}_R$, re-estimate mean and covariance, estimate truncation strength $p_R = |\mathbb{S}_R|/N$.

Posterior mode weight:

$$\tilde{\pi}^m \propto \pi^m \left(\frac{1}{T}\sum_{t=1}^T p_R(\boldsymbol{\mu}_t^m, \boldsymbol{\Sigma}_t^m)^q\right)^{1/q}$$

with $q=1$ used in experiments (average truncation across timesteps).

**Convex constraints implemented:**
- **Reachability** $R_\text{Reach}$: sphere around robot base of radius $r$
- **Collision** $R_\text{CC}$: halfspace constraint $\mathbf{n}^\top(\mathbf{x}-\mathbf{p}) \geq d_\text{safe}$ per obstacle
- **Combined collision** $R_\text{CCC}$: intersection of multiple halfspaces

**Modal constraints** (adjust weights only, not Gaussian parameters):
- **Heuristic self-collision** $R_\text{HSC}$: minimum distance from end-effector to robot base
- **Scene object collision** $R_\text{SOC}$: occupancy-based threshold on voxel grid / point cloud

### VAPOR: Variance-Aware Path Optimization

Path optimization minimizes trajectory NLL under the DiGaP while satisfying joint limits:

$$\min_{\mathbf{q}_1,\ldots,\mathbf{q}_T} \sum_{t=1}^T \tilde{d}\!\left(\boldsymbol{\xi}_t^\text{des}, \boldsymbol{\xi}(\mathbf{q}_t)\right) + \lambda_q \|\mathbf{q}_{t+1} - \mathbf{q}_t\|^2$$
$$\text{s.t.} \quad \mathbf{q}_\min \leq \mathbf{q}_t \leq \mathbf{q}_\max, \quad \mathbf{e}_{\mathbf{q}_t} < z\,\mathrm{diag}(\boldsymbol{\Sigma}_t)^{1/2}$$

with $\boldsymbol{\xi}_t^\text{des} = \boldsymbol{\mu}_t$ and $\tilde{d}$ using weight $W = \boldsymbol{\Sigma}_t^{-1}$ (normalized by $\sigma_\text{max}$). Solved via Augmented Lagrangian (Kineverse). The deviation constraint (Eq. 31) restricts the optimized trajectory to remain within the policy's predicted confidence region.

The VAPOR trajectory's likelihood under each MiDiGaP mode is used as **modal evidence** to select the most kinematically feasible mode for the target embodiment.

## Experimental Results

### Benchmarks and Datasets

- **Simulation:** RLBench tasks — 5 mildly constrained (`OpenDrawer`, `StackWine`, `SlideBlock`, `SweepToDustpan`, `PlaceCups`) + 5 highly constrained (`OpenMicrowave`, `ToiletSeatUp`, `WipeDesk`, `TurnTap`, `ScoopWithSpatula`) + 4 multimodal tasks
- **Baselines:** Diffusion Policy (100 / 5 demos), LSTM (100 demos), ARP (5 demos), TAPAS-GMM (5 demos)
- **Real robot:** Franka Emika + Realsense D435, 11 tasks (mildly constrained, highly constrained, multimodal, visual generalization)

### Unimodal RLBench (Table III) — 5 demonstrations

| Method | Mild Avg. | Highly Constrained Avg. |
|---|---|---|
| Diffusion Policy (100 demos) | 0.67 | 0.48 |
| Diffusion Policy (5 demos) | 0.40 | 0.09 |
| ARP (5 demos) | 0.23 | 0.25 |
| TAPAS-GMM (5 demos) | 0.93 | 0.19 |
| **MiDiGaP (5 demos)** | **0.98** | **0.95** |

MiDiGaP surpasses Diffusion Policy trained on 100 demos by 31 pp (mild) and 47 pp (highly constrained) while using 1/20th of the data.

### Trajectory Smoothness (Table IV)

MiDiGaP achieves ~67% lower total positional end-effector acceleration than TAPAS-GMM (e.g., OpenDrawer: 61±28 vs. 346±48 m/s²). Diffusion Policy produces the highest jerk.

### Multimodal RLBench (Table VI) — 15 demonstrations

| Method | Avg. (4 tasks) |
|---|---|
| Diffusion Policy (100 demos) | 0.46 |
| ARP (15 demos) | 0.33 |
| **MiDiGaP (15 demos)** | **0.86** |
| **MiDiGaP + Modal Evidence** | **0.94** |

### Constrained Gaussian Updating (Table VII)

| Method | Collision Avg. | Reachability Avg. | Overall Avg. |
|---|---|---|---|
| Diffusion Policy | 0.20 | 0.17 | 0.17 |
| Diffusion Policy + ITPS | 0.22 | 0.15 | 0.14 |
| MiDiGaP (no updating) | 0.34 | 0.58 | 0.57 |
| **MiDiGaP + Constrained Updating** | **0.97** | **1.00** | **0.98** |

### Cross-Embodiment Transfer Franka → UR5 (Tables VIII–IX)

| Method | Unimodal Avg. | Multimodal Avg. |
|---|---|---|
| Diffusion Policy | 0.00 | 0.00 |
| TAPAS-GMM | 0.10 | 0.00 |
| MiDiGaP Naïve | 0.32 | 0.33 |
| **MiDiGaP + VAPOR** | **0.68** | **0.73** |

VAPOR-transferred policies often outperform policies trained directly on UR5 demonstrations, since Franka demonstrations encode more variance (UR5 demos are low-variance due to demo collection failures).

### Real-World Robot (Tables X–XI) — Franka Emika

| Method | Unimodal Avg. | Multimodal Avg. |
|---|---|---|
| LSTM / Diffusion / ARP | 0.00 | 0.00 |
| TAPAS-GMM | 0.95 | 0.00 |
| **MiDiGaP** | **1.00** | **0.99** |

Real-world collision avoidance (Table XII): MiDiGaP + Constrained Updating achieves 0.99 avg; Diffusion Policy + ITPS 0.00.

### Timing (Table II)

| Method | Training (5 demos) | Inference |
|---|---|---|
| MiDiGaP | 0–1 min (CPU) | 1 ms |
| TAPAS-GMM | 1–2 min (CPU) | 5 ms |
| Diffusion Policy | 1–72 h (A6000 GPU) | 200 ms |
| ARP | 19–45 min (GPU) | 200 ms* |

MiDiGaP training is ~3 orders of magnitude faster than Diffusion Policy; inference is ~50× faster.

## Limitations & Open Questions

1. **Perception bottleneck:** sub-millimeter trajectory accuracy is possible in principle, but real-world performance is bounded by the quality of the perception system (pose estimation, DINO keypoints, FoundationPose).
2. **Skill segmentation dependency:** relies on TAPAS for automatic segmentation; does not generalize to tasks where individual skills cannot be segmented.
3. **End-effector space only:** designed for task-parameterized end-effector motions; not applicable to joint-space trajectories (locomotion, walking) without fundamental changes.
4. **No emergent capabilities:** unlike scaled deep learning methods, MiDiGaP cannot exhibit emergent behaviors (e.g., retry logic) — though these could be added externally by monitoring execution.
5. **Self-collision in VAPOR:** the joint-space optimization does not explicitly account for self-collision; future work could add self-collision cost terms.
6. **Open question:** can DiGaP representations be extended to RGB/point-cloud inputs without explicit pose estimation, to remove the perception bottleneck?

## See Also

- [Discrete-Time Gaussian Processes for Imitation Learning](../concepts/discrete-time-gaussian-processes.md)
- [Sim-to-Real Transfer](../concepts/sim-to-real.md)
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md)
- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md)
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md)
