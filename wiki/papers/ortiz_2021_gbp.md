---
title: "Ortiz et al. (2021) — A Visual Introduction to Gaussian Belief Propagation"
type: paper
tags: [bayesian-inference, gaussian-belief-propagation, factor-graphs, distributed-inference, probabilistic-graphical-models, slam, robot-perception]
related: [concepts/gaussian-belief-propagation.md, concepts/latent-world-models.md, papers/nabarro_2024_gbp_learning.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/GBP.pdf]
---

# Ortiz et al. (2021) — A Visual Introduction to Gaussian Belief Propagation

**arXiv:** 2107.02308v1 (5 Jul 2021, cs.AI)
**Authors:** Joseph Ortiz, Talfan Evans, Andrew J. Davison (Imperial College London; DeepMind)
**Interactive article:** https://gaussianbp.github.io

> **Note:** This is an expository/tutorial paper, not a primary research paper. It presents no novel algorithm, but makes a compelling case for GBP as the right inference framework for future distributed hardware and provides a comprehensive visual introduction with interactive figures.

## Abstract

This article presents a visual introduction to Gaussian Belief Propagation (GBP), an approximate probabilistic inference algorithm that operates by passing messages between nodes of arbitrarily structured factor graphs. GBP is a special case of loopy belief propagation: GBP updates rely only on local information and converge independently of the message schedule. The key argument is that, given recent trends in computing hardware toward parallel, heterogeneous, and distributed systems, GBP has the right computational properties to act as a scalable distributed probabilistic inference framework for future machine learning systems.

## Key Contributions

1. **Unified GBP tutorial**: connects factor graphs, Gaussian models, linear algebra, and loopy belief propagation into one coherent framework accessible to ML practitioners.
2. **Hardware motivation**: argues that the "Bitter Lesson" — general methods that leverage computation win — points toward GBP as the right algorithm for emerging heterogeneous/distributed hardware (neuromorphic chips, graph processors, multi-core, edge).
3. **Four key GBP properties**: local, probabilistic, iterative+convergent, asynchronous — all naturally matched to distributed hardware without global coordination.
4. **Practical extensions**: covers non-linear factors (linearization), non-Gaussian distributions (covariance scaling, Huber energy), message scheduling strategies, and multiscale/coarse-to-fine acceleration.
5. **Variational derivation** (Appendix A): shows loopy BP is equivalent to constrained minimization of the Bethe free energy approximation to the KL divergence.

## Methodology

### Factor Graphs

The Hammersley-Clifford theorem guarantees that any positive joint distribution factorizes as:

$$p(X) = \prod_i f_i(X_i)$$

**Factor graphs** are bipartite graphs with variable nodes (circles) and factor nodes (squares), where edges connect each factor to the variables it depends on. The conditional independence structure is explicit: variables not connected through a common factor are conditionally independent. Factors can also be written as energy-based models: $f_i(X_i) \propto e^{-E_i(X_i)}$, so MAP inference minimizes $\sum_i E_i(X_i)$.

### Belief Propagation Algorithm

BP computes marginal posteriors via three iterative operations:

**1. Factor-to-variable message** (aggregates all other variable messages, marginalizes):
$$m_{f_j \to x_i} = \sum_{X_j \setminus x_i} f_j(X_j) \prod_{k \in N(j) \setminus i} m_{x_k \to f_j}$$

**2. Variable-to-factor message** (product of all other incoming factor messages):
$$m_{x_i \to f_j} = \prod_{s \in N(i) \setminus j} m_{f_s \to x_i}$$

**3. Belief update** (product of all incoming factor messages):
$$b_i(x_i) = \prod_{s \in N(i)} m_{f_s \to x_i}$$

On tree-structured graphs, one forward-backward sweep yields exact marginals. On loopy graphs, iterative application yields **loopy BP** — empirically effective but without convergence guarantees in general.

### Gaussian Models and GBP

Gaussian factors have two equivalent parameterizations:

| | Moments form | Canonical form |
|---|---|---|
| Parameters | $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ | $\boldsymbol{\eta} = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$, $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ |
| Energy | $E(x) = \frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)$ | $E(x) = \frac{1}{2}x^\top\Lambda x - \eta^\top x$ |
| Marginalization | Easy | Expensive |
| Conditioning | Expensive | Easy |
| Product | Expensive | Easy (additive) |

The canonical form is preferred for inference: products of factors are computed by simply adding $(\eta, \Lambda)$ pairs. Messages between Gaussian nodes are also Gaussians, represented as small vectors and matrices.

**GBP is guaranteed to compute exact marginal means on convergence**, though variances can be overconfident in very loopy graphs.

### Connection to Linear Algebra

The joint Gaussian model has posterior $P(X) \propto \exp(-\frac{1}{2}X^\top\Lambda X + \eta^\top X)$. Inference reduces to:

- **MAP inference**: solve $\Lambda\mu = \eta$ → equivalent to $Ax = b$
- **Marginal inference**: compute $\mu = \Lambda^{-1}\eta$ and diagonal elements of $\Lambda^{-1} = \Sigma$

GBP is therefore a distributed iterative solver for linear systems — comparable to Jacobi or Gauss-Seidel iteration, but fully local and probabilistic.

### Non-Linear Factors

Most real problems involve non-linear measurement functions $h(X)$. Given data $d \sim h(X) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \Sigma_n)$:

$$E(X) = \frac{1}{2}(h(X)-d)^\top\Sigma_n^{-1}(h(X)-d)$$

Linearize via first-order Taylor at current estimate $X_0$: $h(X) \approx h(X_0) + \mathbf{J}(X - X_0)$, giving:

$$\Lambda = \mathbf{J}^\top\Sigma_n^{-1}\mathbf{J}, \quad \eta = \mathbf{J}^\top\Sigma_n^{-1}(d - c), \quad c = h(X_0) - \mathbf{J}X_0$$

GBP's locality allows **just-in-time relinearization**: factors relinearize individually when their adjacent variable estimates stray from the linearization point, avoiding global recomputation.

### Non-Gaussian Distributions: Covariance Scaling

For robust estimation (e.g., handling outliers), the Huber energy replaces the squared loss:

$$E_\text{huber}(r) = \begin{cases}\frac{1}{2}r^\top\Sigma_n^{-1}r, & |r| < t \\ A + B|r|, & \text{otherwise}\end{cases}$$

To retain the Gaussian form, the covariance is scaled to match the true energy at the current residual:

$$\Sigma_\text{sc} = \begin{cases}\Sigma_n, & |r| < t \\ \frac{2E_\text{huber}(r)}{r^\top\Sigma_n^{-1}r}\Sigma_n, & \text{otherwise}\end{cases}$$

This allows GBP to handle outliers robustly (e.g., sharp image denoising, bundle adjustment) while keeping the Gaussian message structure intact.

### Message Scheduling

Messages can be passed in any order; GBP does not require a global clock. Three strategies:

1. **Synchronous**: all nodes broadcast simultaneously — simple, parallelizable.
2. **Random**: any node passes messages in any order — convergence-improving, asynchronous-hardware-friendly.
3. **Sweep**: left-to-right then back (for chains) — optimal for tree graphs.
4. **Residual Belief Propagation (RBP)**: priority queue ordered by $\|m_t - \tilde{m}_{t-1}\|$ — focuses compute on most information-carrying messages.

**Message damping** improves convergence in very loopy graphs by exponential averaging:
$$\tilde{\eta}_t = \beta\eta_t + (1-\beta)\tilde{\eta}_{t-1}, \quad \tilde{\Lambda}_t = \beta\Lambda_t + (1-\beta)\tilde{\Lambda}_{t-1}$$
with $\beta = 1$ recovering standard BP.

### Multiscale Learning

Information propagates at one hop per iteration, so long-range dependencies require many iterations in grid graphs. Coarse-to-fine acceleration (analogous to multigrid methods): solve a coarsened graph first, use its solution to initialize the fine-grained graph. Demonstrated for stereo, optical flow, and image restoration with multiscale BP.

### Variational Derivation (Bethe Free Energy)

Loopy BP is equivalent to minimizing the **Bethe free energy** — an approximation to the KL divergence between a variational distribution $q(X)$ and the true posterior. Because the Bethe energy is bounded from below, BP always has a fixed point. The fixed-points are stationary (not necessarily global minima) of the Bethe free energy.

## Demonstrations

This paper presents interactive rather than quantitative experiments. Applications shown:

- **Geometric estimation**: GBP on chain, loop, and grid graphs for 2D pose estimation; converges to true marginals regardless of message schedule
- **Image denoising**: GBP on 300×300 pixel grid graph with attention-driven scheduling; Huber energy yields sharp edges vs. blurred results from squared loss
- **Robot pose graph**: 2D SLAM pose graph where GBP infers robot positions and landmark locations from relative measurements
- **Bundle adjustment**: referenced result showing 24x speedup over standard methods on a graph processor (Ortiz et al., CVPR 2020)

## Comparison to Related Methods

| Method | Local | Probabilistic | Iterative | Asynchronous |
|---|:---:|:---:|:---:|:---:|
| Gaussian elimination / Cholesky | ✗ | ✗ | ✗ | ✗ |
| Gradient descent | ✓ | ✗ | ✓ | ✓ |
| Conjugate gradient | ✗ | ✗ | ✓ | ✗ |
| Jacobi / Gauss-Seidel | ✓ | ✗ | ✓ | ✓ |
| Expectation Propagation | ✓ (partial) | ✓ | ✓ | ✗ |
| **GBP** | **✓** | **✓** | **✓** | **✓** |

GBP is the extreme case that maximizes parallelism and minimizes communication.

## Limitations & Open Questions

1. **Convergence not guaranteed**: in very loopy graphs GBP may not converge or may give overconfident variances; theoretical conditions exist but are restricted.
2. **Linearization quality**: non-linear factors require repeated relinearization; poor initialization leads to local optima (same as in non-linear least squares).
3. **Discrete variables not handled**: GBP as described applies only to Gaussian (continuous) variables; extending to discrete or mixed models requires different methods.
4. **No learned factors**: standard GBP uses hand-designed factors; combining with learned factors (e.g., neural networks as $h(X)$) is an open direction.
5. **Hierarchical structure**: long-range dependencies require many iterations; multiscale acceleration requires graph-specific coarsening strategies.
6. **Open questions (from authors):** improving theoretical convergence guarantees; using learned factors [DeepFactors]; combining GBP with GNNs; investigating discrete variables; GBP for distributed learning in overparameterized networks; unifying iterative inference with test-time self-supervised learning.

## See Also

- [Gaussian Belief Propagation](../concepts/gaussian-belief-propagation.md)
- [Latent World Models](../concepts/latent-world-models.md)
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md)
- [Nabarro et al. (2024) — Learning in Deep Factor Graphs with GBP](nabarro_2024_gbp_learning.md) — follow-up from the same lab that realises this paper's "GBP for learning in overparameterised networks" direction
