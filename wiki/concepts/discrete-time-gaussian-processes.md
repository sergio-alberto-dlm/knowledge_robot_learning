---
title: Discrete-Time Gaussian Processes for Imitation Learning
type: concept
tags: [imitation-learning, gaussian-processes, policy-representation, robot-manipulation, multimodal, few-shot, riemannian-geometry, inference-time-adaptation, cross-embodiment]
related: [papers/vonhartz_2025_midigap.md, concepts/dynamic-movement-primitives.md, papers/bahl_2020_ndp.md, concepts/sim-to-real.md, concepts/dexterous-manipulation.md, concepts/vision-language-action-models.md, concepts/model-predictive-control.md]
created: 2026-05-17
updated: 2026-05-18
sources: [raw/papers/pdf/midigap.pdf]
---

# Discrete-Time Gaussian Processes for Imitation Learning

## Overview

A **Discrete-time Gaussian Process (DiGaP)** is a finite sequence of per-timestep Gaussian distributions fit independently to a set of robot trajectory demonstrations. Unlike continuous Gaussian Processes (CoGaP), DiGaP requires no kernel function and no initialization, models arbitrary trajectory shapes (oscillatory, piecewise, non-stationary, discontinuous), and scales linearly in dataset size with constant inference cost. Its **mixture variant**, MiDiGaP, extends this to multimodal trajectory distributions — i.e., tasks where the robot can execute a goal in qualitatively different ways.

Introduced by von Hartz et al. (2025) in the context of task-parameterized imitation learning on Riemannian manifolds.

## DiGaP: Formal Definition

A DiGaP with $T$ timesteps is:

$$\mathcal{GP} = \bigl((\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)\bigr)_{t=1}^T$$

Given N demonstrations $\{(\mathbf{z}_t^n)_{t=1}^T\}_{n=1}^N$ on Riemannian manifold $\mathcal{M}$, the parameters are:

$$\boldsymbol{\mu}_t = \arg\min_{\mathbf{x}\in\mathcal{M}}\sum_{n=1}^N d_\mathcal{M}(\mathbf{x}, \mathbf{z}_t^n)^2 \quad \text{(Fréchet mean)}$$

$$\boldsymbol{\Sigma}_t = \mathrm{diag}\!\left(\frac{1}{N-1}\sum_{n=1}^N \mathrm{Log}_{\boldsymbol{\mu}_t}(\mathbf{z}_t^n)\,\mathrm{Log}_{\boldsymbol{\mu}_t}(\mathbf{z}_t^n)^\top\right)$$

The **diagonal covariance** is crucial: off-diagonal entries model temporal correlations between dimensions which can be spurious on Riemannian data (e.g., quaternion components), degrading prediction quality.

For robot end-effector trajectories, $\mathcal{M}_\text{pose} = \mathbb{R}^3 \times S^3$, and the Log map gives Lie-algebra tangent vectors. In practice, demonstrations are subsampled to a fixed length $T$ (e.g., 20 Hz) before fitting.

## Why DiGaP vs. Other Representations

| Approach | Oscillatory | Non-stationary | Discontinuous | Multimodal | Linear Scaling | Probabilistic |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CoGaP (continuous GP) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| GMM | ✗ | ✗ | ✗ | ✓ (limited) | ✓ | ✓ |
| Diffusion Policy | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| **DiGaP** | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| **MiDiGaP** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**CoGaP failure modes:** kernel functions (RBF, Matérn) enforce global smoothness, preventing oscillatory, piecewise, or long-range-dependent functions. CoGaP also has $O(N^3)$ training and $O(N^2)$ inference complexity.

**GMM failure modes:** locally linear dynamics assumption; prone to spurious inter-dimensional correlations on Riemannian data; poor initialization sensitivity; converges to local optima in multimodal settings.

**Diffusion Policy failure modes:** needs ~100 demonstrations to generalize; inference takes 200 ms (GPU); point estimate at each step; very hard to steer at inference time (ITPS fails consistently in practice).

## MiDiGaP: Mixture of Discrete-time Gaussian Processes

$$\mathcal{GPM} = \Bigl\{\pi^m, \bigl\{\boldsymbol{\mu}_t^m, \boldsymbol{\Sigma}_t^m\bigr\}_{t=1}^T\Bigr\}_{m=1}^M$$

Each mode $m$ has prior weight $\pi^m = |\mathcal{DM}_m|/|\mathcal{D}|$ (fraction of demonstrations). Inference samples a mode $m \sim \pi$ and follows $\{\boldsymbol{\mu}_t^m\}$.

### Modal Partitioning (Automatic Mode Discovery)

Demonstrations are embedded as concatenated vectors $\mathbf{v}_n = (\mathbf{z}_1^n, \ldots, \mathbf{z}_{T'}^n) \in \mathcal{M}^{T'}$ and clustered:

1. **Riemannian GMM + BIC:** most principled, detects all modes (accounts for data variance), slow ($O(n^2)$ per EM iteration).
2. **Riemannian k-means + BIC** *(recommended):* near-linear scaling with $n$ and $k$; robust; 0.3–24 s for 100 demos.
3. **DBSCAN:** fastest (<1 s for 100 demos), but ignores variance — fails when modes are only separable in the second half of trajectories.

**Mode discovery in practice:** on `TurnTap`, expected 2 modes (one per valve) but found 4 — two approach styles per valve, which generate hardware-unsafe velocities if the wrong one is selected. MiDiGaP's automatic partitioning surfaced this otherwise-hidden structure.

### Skill Sequencing for Long-Horizon Tasks

Long-horizon tasks are segmented via TAPAS into skill sequences $(\mathcal{D}_1, \ldots, \mathcal{D}_n)$, each modeled by a MiDiGaP. Mode transitions between consecutive skills are encoded as:

$$\pi_j(k, l) = \frac{|\mathcal{DM}_k \cap \mathcal{DM}_l|}{|\mathcal{DM}_k|}$$

Novel skill sequences can be composed by comparing MiDiGaPs via KL-divergence between boundary Gaussians.

## Inference-Time Adaptation

### Constrained Gaussian Updating

MiDiGaP supports two forms of evidence integration at inference time:

**1. Modal updating** (adjusts prior weights only): zeroes the likelihood of modes that violate modal constraints (e.g., heuristic self-collision, occupancy-based scene object collision).

**2. Convex updating** (adjusts Gaussian parameters + weights): uses **moment matching**:
- Sample $\{\mathbf{x}_i\}_{i=1}^N \sim \mathcal{N}(\boldsymbol{\mu}_t^m, \boldsymbol{\Sigma}_t^m)$
- Keep feasible set $\mathbb{S}_R = \{x_i \mid x_i \in R\}$
- Re-estimate mean and variance from $\mathbb{S}_R$
- Estimate truncation strength $p_R = |\mathbb{S}_R|/N$

Posterior weight: $\tilde{\pi}^m \propto \pi^m \left(\frac{1}{T}\sum_t p_R(\boldsymbol{\mu}_t^m, \boldsymbol{\Sigma}_t^m)\right)$

**Convex constraints supported:**
- Reachability sphere: $R_\text{Reach} = \{\mathbf{x} \mid \|\mathbf{x} - \mathbf{b}\| \leq r\}$
- Collision halfspace: $R_\text{CC} = \{\mathbf{x} \mid \mathbf{n}^\top(\mathbf{x}-\mathbf{p}) \geq d_\text{safe}\}$
- Combined collision: intersection of multiple halfspaces
- Scene object collision: occupancy threshold on voxel grid / point cloud

Key advantage over Diffusion Policy guidance (ITPS): Diffusion Policy is brittle to out-of-distribution samples, so ITPS guidance often pushes it OOD. MiDiGaP's Gaussian structure allows exact (convex) or approximate (moment-matching) Bayesian updating without OOD risk.

### VAPOR: Variance-Aware Path Optimization

Converts probabilistic end-effector trajectories $\{(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)\}$ to kinematically feasible joint trajectories by:

$$\min_{\mathbf{q}_1,\ldots,\mathbf{q}_T} \sum_{t=1}^T \tilde{d}\!\left(\boldsymbol{\xi}_t^\text{des}, \boldsymbol{\xi}(\mathbf{q}_t)\right) + \lambda_q \|\mathbf{q}_{t+1} - \mathbf{q}_t\|^2$$
$$\text{s.t.} \quad \mathbf{q}_\min \leq \mathbf{q}_t \leq \mathbf{q}_\max, \quad \mathbf{e}_{\mathbf{q}_t} < z\,\mathrm{diag}(\boldsymbol{\Sigma}_t)^{1/2}$$

where $\tilde{d}$ uses $W = \boldsymbol{\Sigma}_t^{-1}$ (normalized by $\sigma_\text{max}$) to let the optimizer deviate more along high-variance dimensions and less along low-variance (tightly constrained) dimensions.

**Cross-embodiment transfer:** given a policy trained on a Franka arm, VAPOR re-optimizes joint trajectories for a UR5 arm. The VAPOR trajectory's NLL under each MiDiGaP mode is used as modal evidence to select the most kinematically achievable mode on the target embodiment. Achieves 63–68% avg unimodal / 73% avg multimodal success rate on Franka→UR5 transfer, often outperforming policies trained directly on UR5 demonstrations (which have lower variance due to demo collection failures).

Optimization completes in ~50 ms using Augmented Lagrangian (Kineverse library).

## Task Parameterization Compatibility

MiDiGaP is fully compatible with task-parameterized multi-stream learning (TAPAS-GMM framework). Object poses from DINO keypoints or FoundationPose are used to define task parameters; end-effector trajectories are expressed in local coordinate frames and transformed to world frame by combining per-frame models via product of Gaussians. This enables robust visual generalization (instance-level and environment-level) without retraining.

## Practical Properties

- **Sample efficiency:** as few as 5 demonstrations for unimodal tasks; 10–15 for multimodal
- **Training speed:** 0–1 min on CPU for 5 demonstrations; scales linearly
- **Inference speed:** 1 ms per query (excluding perception)
- **Probabilistic:** explicit trajectory distribution (density, variance, log-likelihood)
- **Interpretable:** per-timestep mean and variance are directly inspectable; modes correspond to human-interpretable behavior clusters
- **Steerable at inference time:** Bayesian updating with new evidence (collision, reachability) is efficient and reliable

## See Also

- [von Hartz et al. (2025) — MiDiGaP](../papers/vonhartz_2025_midigap.md)
- [Model Predictive Control for Robot Learning](model-predictive-control.md)
- [Sim-to-Real Transfer](sim-to-real.md)
- [Dexterous Manipulation](dexterous-manipulation.md)
- [Vision-Language-Action Models (VLAs)](vision-language-action-models.md)
