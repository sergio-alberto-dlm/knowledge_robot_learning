---
title: Sparse Gaussian Process Approximations
type: concept
tags: [gaussian-processes, sparse-gp, pseudo-datapoints, variational-inference, kl-divergence, scalable-inference, time-series, spatial-inference]
related: [papers/bui_2014_tree_gp.md, concepts/gaussian-belief-propagation.md, concepts/discrete-time-gaussian-processes.md, concepts/model-predictive-control.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/GP_w_GBP.pdf]
---

# Sparse Gaussian Process Approximations

## Overview

Gaussian Processes (GPs) are flexible nonparametric probabilistic models for functions, used extensively in robotics and ML for regression, classification, and motion planning. Their main limitation is computational: exact GP regression costs $\mathcal{O}(N^3)$ in training and $\mathcal{O}(N^2)$ in prediction for $N$ data points. **Sparse GP approximations** address this by summarizing the observed data with a smaller **pseudo-dataset** of $M \ll N$ points, reducing costs to $\mathcal{O}(NM^2)$ training and $\mathcal{O}(M^2)$ prediction. The tree-structured GP (Bui & Turner, 2014) further reduces inference to $\mathcal{O}(N)$ by imposing a tree structure on the pseudo-dataset.

## GP Regression Fundamentals

Given $N$ observations $y_n = f(\mathbf{x}_n) + \epsilon_n$, $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$, the GP posterior is:

$$m_f(\mathbf{x}) = \mathbf{K}_{\mathbf{xf}}(\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$$
$$k_f(\mathbf{x}, \mathbf{x}') = k(\mathbf{x},\mathbf{x}') - \mathbf{K}_{\mathbf{xf}}(\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})^{-1}\mathbf{K}_{\mathbf{fx}'}$$

The bottleneck is inverting $\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I} \in \mathbb{R}^{N \times N}$ ($\mathcal{O}(N^3)$). Hyperparameters are learned by maximizing the marginal likelihood $p(\mathbf{y}|\theta, \sigma) = \mathcal{N}(\mathbf{y}; \mathbf{0}; \mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})$.

## Pseudo-Datapoint Approximations: The Core Idea

All sparse GP methods introduce a pseudo-dataset $\{\tilde{\mathbf{x}}_m, u_m\}_{m=1}^M$ that acts as a bottleneck in the graphical model. Two categories:

**Indirect (prior) approximations**: modify the generative model to make inference tractable, then calibrate it to the original. The approximate model has a modified prior over the function values.

**Direct (posterior) approximations**: keep the generative model intact but approximate the posterior distribution directly with a simpler form.

### Key Methods

| Method | KL objective | Posterior form | Notes |
|---|---|---|---|
| FITC | $\mathrm{KL}(p(\mathbf{f},\mathbf{u})\|q(\mathbf{u})\prod_n p(f_n|\mathbf{u}))$ | $q(\mathbf{u}) = p(\mathbf{u}),\ q(f_n|\mathbf{u}) = p(f_n|\mathbf{u})$ | Fully Independent Training Conditional |
| PIC | $\mathrm{KL}(p(\mathbf{f},\mathbf{u})\|q(\mathbf{u})\prod_k p(\mathbf{f}_{C_k}|\mathbf{u}))$ | Blocks partially independent | Partially Independent Conditional |
| PP/VFE | $\mathrm{KL}(q(\mathbf{f}|\mathbf{u})p(\mathbf{u})\|p(\mathbf{f},\mathbf{u}|\mathbf{y}))$ | $q(\mathbf{u}) \propto p(\mathbf{u})\exp(\langle\log p(\mathbf{y}|\mathbf{f})\rangle)$ | Variational Free Energy; unbiased |
| EP | $\mathrm{KL}(q(\mathbf{f};\mathbf{u})p(y_m|f_m)/q_m(f_m)\|q(\mathbf{f};\mathbf{u}))$ | $q(\mathbf{f};\mathbf{u}) \propto p(\mathbf{u})\prod_m p(u_m|f_m)$ | Expectation Propagation |
| **Tree** | $\mathrm{KL}(p(\mathbf{f},\mathbf{u})\|\prod_k q(\mathbf{f}_{C_k}|\mathbf{u}_{B_k})q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}(B_k)}))$ | Tree-structured prior; GBP inference | Linear-in-N |

### The Locality Problem

Every pseudo-datapoint is locally effective: it sculpts the posterior in a small region of input space of radius $\sim l_d$ (the kernel length-scale). To maintain accuracy across a dataset of spatial extent $L_d$ in dimension $d$, roughly $M \gtrsim \prod_d L_d/l_d$ pseudo-datapoints are needed. For:
- **Time series**: $L \propto N$, so $M \propto N$ — no computational gain
- **2D spatial**: $M \propto N$ in each dimension — cubic memory and compute

This is why standard methods fail on large time-series and spatial datasets.

## Tree-Structured GP Approximation (Bui & Turner, 2014)

### Architecture

Partition $M$ pseudo-datapoints into $K$ blocks $\{\mathbf{u}_{B_k}\}$ arranged in a tree, and $N$ function values into $K$ blocks $\{\mathbf{f}_{C_k}\}$ assigned to corresponding tree nodes. The approximate model:

$$q(\mathbf{u}) = \prod_{k=1}^K q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}(B_k)}), \quad q(\mathbf{f}|\mathbf{u}) = \prod_{k=1}^K q(\mathbf{f}_{C_k}|\mathbf{u}_{B_k})$$

Each block's pseudo-data conditions on its parent, propagating global dependencies through the tree structure rather than through a monolithic global inducing set.

### KL-Calibrated Optimal Distributions

Minimizing the forward KL divergence gives closed-form conditionals equal to the true GP conditionals:

$$q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}}) = \mathcal{N}(\mathbf{A}_k\mathbf{u}_{\mathrm{par}}, \mathbf{Q}_k), \quad \mathbf{A}_k = \mathbf{K}_{\mathbf{u}_k\mathbf{u}_l}\mathbf{K}_{\mathbf{u}_l\mathbf{u}_l}^{-1}$$
$$q(\mathbf{f}_{C_k}|\mathbf{u}_{B_k}) = \mathcal{N}(\mathbf{C}_k\mathbf{u}_{B_k}, \mathbf{R}_k), \quad \mathbf{C}_k = \mathbf{K}_{\mathbf{f}_k\mathbf{u}_k}\mathbf{K}_{\mathbf{u}_k\mathbf{u}_k}^{-1}$$

The Schur complement residuals $\mathbf{Q}_k$ and $\mathbf{R}_k$ capture the uncertainty not explained by the parent and the local pseudo-inputs respectively.

### Inference via GBP

After marginalizing $\mathbf{f}$, the effective local likelihood is Gaussian: $p(\mathbf{y}_k|\mathbf{u}_k) = \mathcal{N}(\mathbf{y}_k; \mathbf{C}_k\mathbf{u}_k, \mathbf{R}_k + \sigma^2\mathbf{I})$. The model is now a tree-structured linear Gaussian model — the canonical setting for GBP. The up-down algorithm (leaves → root for aggregation, root → leaves for smoothing) produces exact marginals in two passes.

**Complexity**: $\mathcal{O}(KD^3)$ where $D$ is average block size. For fixed $D$, this is $\mathcal{O}(N)$ — **linear in dataset size**.

For chain-structured temporal data (special case), inference is equivalent to running the Kalman smoother at the same cost.

### Hyperparameter Learning

The marginal likelihood factors recursively over blocks. Its gradient decomposes into local terms involving only the GBP-computed marginal posteriors:

$$\frac{d}{d\theta}\log p(\mathbf{y}|\theta) = \sum_{k=1}^K \left[\left\langle\frac{d}{d\theta}\log p(\mathbf{u}_k|\mathbf{u}_l)\right\rangle_{p(\mathbf{u}_k,\mathbf{u}_l|\mathbf{y})} + \left\langle\frac{d}{d\theta}\log p(\mathbf{y}_k|\mathbf{u}_k)\right\rangle_{p(\mathbf{u}_k|\mathbf{y})}\right]$$

This locality is key: gradients are computed without global matrix inversions. Optimized with BFGS.

## Common Kernels in Practice

- **Exponentiated quadratic (RBF)**: $k(t, t') = \sigma^2 \exp\!\left(-\frac{(t-t')^2}{2l^2}\right)$ — smooth functions, stationary
- **Matérn**: adjustable smoothness, stationary
- **Spectral mixture**: $k(t,t') = \sum_k \sigma_k^2\cos(\omega_k(t-t'))\exp\!\left(-\frac{(t-t')^2}{2l^2}\right)$ — captures oscillatory/multi-frequency patterns; used in audio tasks
- **Anisotropic RBF**: separate length-scales per input dimension; used in 2D spatial tasks

## Connection to DiGaP

DiGaP (von Hartz et al., 2025) is a conceptually related but distinct approach: instead of using a continuous-time GP with a kernel and pseudo-datapoints, it directly fits per-timestep Gaussians from demonstrations without a kernel function. DiGaP avoids the locality problem entirely by fitting one Gaussian per timestep (dense trajectory) rather than selecting a sparse inducing set. However, DiGaP is restricted to imitation learning (trajectory modeling), while GPs with GBP can be applied to any regression problem with a kernel.

## Limitations

1. **Kernel choice matters**: tree-GP still inherits the usual GP limitation of requiring an appropriate kernel; spectral mixture kernels improve expressivity but add parameters.
2. **Block size cubic**: $\mathcal{O}(D^3)$ per block — blocks must be kept small for efficiency.
3. **Euclidean tree structure**: Kruskal's algorithm uses Euclidean input distances; non-Euclidean or very high-dimensional inputs may produce poor trees.
4. **No closed-form for non-Gaussian likelihoods**: classification or count data requires additional approximations on top of the tree structure.

## See Also

- [Bui & Turner (2014) — Tree-structured GP Approximations](../papers/bui_2014_tree_gp.md)
- [Gaussian Belief Propagation & Factor Graphs](gaussian-belief-propagation.md)
- [Discrete-Time Gaussian Processes for Imitation Learning](discrete-time-gaussian-processes.md)
- [Model Predictive Control for Robot Learning](model-predictive-control.md)
