---
title: Gaussian Belief Propagation & Factor Graphs
type: concept
tags: [bayesian-inference, gaussian-belief-propagation, factor-graphs, probabilistic-graphical-models, distributed-inference, slam, robot-perception, message-passing, learning-as-inference]
related: [papers/ortiz_2021_gbp.md, papers/nabarro_2024_gbp_learning.md, papers/bui_2014_tree_gp.md, concepts/continual-learning-and-tracking.md, concepts/latent-world-models.md, concepts/model-predictive-control.md, concepts/discrete-time-gaussian-processes.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/GBP.pdf, raw/papers/pdf/GP_w_GBP.pdf, raw/papers/pdf/GBP_learning.pdf]
---

# Gaussian Belief Propagation & Factor Graphs

## Overview

**Gaussian Belief Propagation (GBP)** is an approximate probabilistic inference algorithm for computing marginal posteriors in Gaussian graphical models. It operates by iteratively passing local messages between nodes of a **factor graph** — a bipartite graph that makes the conditional independence structure of a joint distribution explicit. GBP's key properties are that it is **local** (no global view needed), **probabilistic** (outputs full distributions, not point estimates), **iterative** (runs continuously, tolerates asynchrony), and **asynchronous** (message order does not affect fixed points). These properties make GBP well suited for distributed/heterogeneous hardware and for robotics problems (SLAM, sensor fusion, structure from motion).

## Factor Graphs

Any positive joint distribution $p(X)$ factorizes (Hammersley-Clifford theorem) as:

$$p(X) = \prod_i f_i(X_i)$$

A **factor graph** represents this factorization as a bipartite graph with:
- **Variable nodes** (circles): the unknown variables $x_i$
- **Factor nodes** (squares): the functions $f_i$ encoding constraints/observations
- **Edges**: connect each factor to the variables it depends on

Factors are often written in energy-based form: $f_i(X_i) \propto e^{-E_i(X_i)}$. MAP inference becomes minimizing total energy $\sum_i E_i(X_i)$.

The conditional independence structure is directly visible: two variables with no path through a common factor are conditionally independent. The precision matrix $\Lambda$ of the joint Gaussian has a sparsity pattern matching the factor graph — entry $(i,j)$ is zero iff variables $x_i$ and $x_j$ share no factor.

## Gaussian Models: Two Parameterizations

| | **Moments form** | **Canonical form** |
|---|---|---|
| Parameters | $\boldsymbol{\mu}$ (mean), $\boldsymbol{\Sigma}$ (covariance) | $\boldsymbol{\eta} = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$ (information vector), $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ (precision matrix) |
| Energy | $E(x) = \frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)$ | $E(x) = \frac{1}{2}x^\top\Lambda x - \eta^\top x$ |
| Marginalization | Easy | Expensive (requires $\Lambda^{-1}$) |
| Conditioning | Expensive | Easy |
| Product (combining factors) | Expensive | Easy: add $(\eta_1 + \eta_2,\, \Lambda_1 + \Lambda_2)$ |

The canonical form is standard for GBP: all messages are $(\eta, \Lambda)$ pairs and factor products are computed by addition.

## Belief Propagation Algorithm

Three operations, applied iteratively:

**Factor-to-variable message** (aggregates information from neighboring variables, marginalizes out others):
$$m_{f_j \to x_i} = \sum_{X_j \setminus x_i} f_j(X_j) \prod_{k \in N(j) \setminus i} m_{x_k \to f_j}$$

**Variable-to-factor message** (what the variable would believe without this factor):
$$m_{x_i \to f_j} = \prod_{s \in N(i) \setminus j} m_{f_s \to x_i}$$

**Belief update** (product of all incoming factor messages):
$$b_i(x_i) = \prod_{s \in N(i)} m_{f_s \to x_i}$$

On trees, one forward-backward sweep gives **exact marginals**. On loopy graphs, iterative application gives **loopy BP** — an approximation that minimizes the Bethe free energy (an approximation to the KL divergence between a variational distribution and the true posterior). The fixed-points are stationary points of the Bethe energy; BP always has a fixed point because the Bethe energy is bounded below.

## Connection to Linear Algebra

For a Gaussian factor graph with joint $P(X) \propto \exp(-\frac{1}{2}X^\top\Lambda X + \eta^\top X)$:

- **MAP inference**: $X_\text{MAP} = \Lambda^{-1}\eta = \mu$ — equivalent to solving the linear system $\Lambda\mu = \eta$
- **Marginal inference**: same $\mu$ plus the diagonal of $\Sigma = \Lambda^{-1}$

GBP is therefore a distributed iterative linear solver, analogous to Jacobi or Gauss-Seidel, but additionally computing per-variable uncertainty estimates. Among iterative Gaussian solvers, GBP is the extreme case that maximizes locality and parallelism.

## GBP: Four Key Properties

1. **Local**: each node computes using only messages from immediate neighbors. No global view or coordination needed. Naturally exploits conditional independence.
2. **Probabilistic**: maintains full Gaussian beliefs (mean + variance) for each variable. Uncertainty propagates through the graph.
3. **Iterative and convergent**: not run for a fixed number of steps. Can run continuously in the background; new data/factors can be added without interrupting inference.
4. **Asynchronous**: convergence is reached regardless of message order (synchronous, random, sweep, or residual-priority). Compatible with distributed systems without a global clock.

## Practical Extensions

### Non-Linear Factors (Linearization)

Most real problems have non-linear measurement functions $h(X)$. Given $d \sim h(X) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \Sigma_n)$:

$$E(X) = \frac{1}{2}(h(X)-d)^\top\Sigma_n^{-1}(h(X)-d)$$

Linearize at current estimate $X_0$: $h(X) \approx h(X_0) + \mathbf{J}(X - X_0)$, yielding:
$$\Lambda = \mathbf{J}^\top\Sigma_n^{-1}\mathbf{J}, \quad \eta = \mathbf{J}^\top\Sigma_n^{-1}(d - h(X_0) + \mathbf{J}X_0)$$

GBP's locality enables **just-in-time relinearization**: each factor independently relinearizes when adjacent estimates drift, without touching other factors.

### Robust Energy Functions (Non-Gaussian Distributions)

For heavy-tailed distributions (e.g., outlier-resistant image denoising, bundle adjustment), the **Huber energy** transitions from quadratic to linear for large residuals:

$$E_\text{huber}(r) = \begin{cases}\frac{1}{2}r^\top\Sigma_n^{-1}r, & |r| < t \\ A + B|r|, & \text{otherwise}\end{cases}$$

To maintain Gaussian messages, scale the covariance to match the true energy at the current residual:

$$\Sigma_\text{sc} = \begin{cases}\Sigma_n, & |r| < t \\ \frac{2E_\text{huber}(r)}{r^\top\Sigma_n^{-1}r}\Sigma_n, & \text{otherwise}\end{cases}$$

This covariance scaling is the standard approach to using GBP with M-estimators.

### Message Damping (Convergence Improvement)

In very loopy graphs, standard GBP can oscillate. Exponential averaging of messages reduces oscillation:

$$\tilde{\eta}_t = \beta\eta_t + (1-\beta)\tilde{\eta}_{t-1}, \quad \tilde{\Lambda}_t = \beta\Lambda_t + (1-\beta)\tilde{\Lambda}_{t-1}$$

$\beta = 1$: standard BP. $\beta \to 0$: no update (keep old message). Damping does not change fixed points but improves convergence rate.

### Message Dropout

Nabarro et al. (2024) stabilise GBP in deep, non-linear factor graphs by combining damping (0.7–0.9) with **dropout on factor→variable messages** (rate 0.5–0.6). They report that this was enough for stable training, though, like damping, it is a heuristic with no convergence guarantee.

> **Verify:** The paper does not define message dropout precisely. The most likely reading is that a random subset of messages keeps its previous value at each iteration. Check the gbp_learning code.

### Message Scheduling

- **Synchronous**: all nodes in parallel — simple, standard.
- **Random**: any order — asynchronous-hardware-friendly.
- **Sweep**: ordered traversal — optimal for chains/trees.
- **Residual Belief Propagation (RBP)**: priority queue by $\|\eta_t - \tilde{\eta}_{t-1}\|$ — focuses compute on highest-information messages; analogous to attention in neural networks; compatible with energy/compute-constrained hardware.

### Multiscale / Coarse-to-Fine

Low-frequency errors (long-range dependencies) decay slowly in local message passing. Coarse-to-fine acceleration: build a coarser graph, solve it, use the solution to initialize the fine graph. Applicable to grid-structured graphs (image segmentation, PDEs). For general unstructured graphs, hierarchical abstraction can be learned or embedded as additional nodes.

## Learning with GBP: Parameters as Variables

GBP is not limited to state estimation. If model **parameters** are added as variable nodes, then *training* is also inference: the posterior $p(\Theta\mid\mathcal{D})$ is just another set of marginals. **GBP Learning** (Nabarro et al. 2024) builds factor graphs that mirror NN architectures (conv, transposed conv, max-pool, upsample, dense, softmax). Inputs, activations, outputs and weights are all variables, and training vs. prediction differ only in which variables are observed.

- **Layer factors** enforce local consistency, $E = \|\mathbf{x}_{l-1} - f(\mathbf{x}_l,\Theta_l)\|^2/2\sigma_l^2$. These factors are non-linear (bilinear in $\mathbf{x}$ and $\Theta$, plus activation $g$) and are relinearized as described above. The input-dependent Jacobian makes factor strengths "soft-switch", much like NN non-linearities.
- **No global gradient / backprop:** only per-factor Jacobians. There is no backward locking, and training tolerates random asynchronous layer schedules (98.11% vs. 98.16% on MNIST).
- **Efficiency:** the low-rank linearized precision plus diagonal message precision lets the Woodbury identity cut factor→variable updates from $O(V(V-1)^3)$ to $O(VM^3)$. For $L$ dense layers of width $C$ and batch $B$, a full message update costs $O(BLC^2)$, the same as a backprop pass.
- **Continual learning = Bayesian filtering:** after each batch or task, the parameter marginals become unary prior factors for the next. Data is seen once and then discarded, and this doubles as minibatching.
- **Hyperparameters replace the learning rate:** factor strengths $\sigma$, prior widths, damping, dropout, and iteration counts (see the [gradients/hyperparameters report](../../outputs/reports/2026-09-25_gbp-learning-gradients-hyperparameters.md)).

This contrasts with [Bui & Turner (2014)](../papers/bui_2014_tree_gp.md), where GBP does inference over latent function values but kernel hyperparameters are still fit by gradient-based marginal-likelihood optimisation (BFGS).

## Applications in Robotics

- **Factor graph SLAM**: GBP is the natural inference algorithm for robot pose graph optimization (SLAM), where variable nodes are robot/landmark poses and factor nodes are sensor measurements.
- **Bundle adjustment**: Ortiz et al. (CVPR 2020) demonstrated 24x speedup over standard methods by running GBP on a graph processor.
- **Sensor fusion**: any multi-sensor system with Gaussian noise can be expressed as a factor graph and solved with GBP.
- **Dense 3D reconstruction**: DeepFactors uses GBP on a factor graph where a neural network provides learned factors over depth maps.
- **Learned deep factor graphs**: Nabarro et al. (2024) train convolutional factor graphs with GBP. They beat a hand-designed pairwise smoother on video denoising and reach 98.16% on MNIST in a single continual-learning epoch, which points to learned factors that could live inside robotics factor graphs.
- **Sparse GP approximations**: Bui & Turner (2014) use GBP as the inference engine for tree-structured GP approximations, where pseudo-datapoints form a tree and GBP's up-down algorithm replaces the Kalman smoother, achieving linear-in-N inference for temporal and spatial GP regression.

## Limitations

1. **No convergence guarantee on loopy graphs**: GBP may not converge or may give overconfident variances; known convergence conditions are restrictive.
2. **Linearization sensitivity**: repeated relinearization can diverge from poor initialization; equivalent to local minima in non-linear least squares.
3. **Gaussian-only**: standard GBP handles only continuous Gaussian variables; discrete or mixed models require different methods.
4. **Long-range dependencies slow**: global consensus takes many iterations proportional to graph diameter; multiscale methods help but require graph structure.
5. **No discrete variables**: extending to mixed discrete-continuous models is non-trivial.
6. **Slow on today's hardware for learning**: GBP Learning takes ~3 h on an RTX 3090 for single-epoch MNIST vs. minutes for a CNN with backprop. Scalar variable nodes also discard correlations between weights (Nabarro et al. 2024).

## See Also

- [Ortiz et al. (2021) — A Visual Introduction to GBP](../papers/ortiz_2021_gbp.md)
- [Nabarro et al. (2024) — Learning in Deep Factor Graphs with GBP](../papers/nabarro_2024_gbp_learning.md)
- [Bui & Turner (2014) — Tree-structured GP Approximations](../papers/bui_2014_tree_gp.md)
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md)
- [Sparse Gaussian Processes](sparse-gaussian-processes.md)
- [Latent World Models](latent-world-models.md)
- [Model Predictive Control for Robot Learning](model-predictive-control.md)
- [Discrete-Time Gaussian Processes for Imitation Learning](discrete-time-gaussian-processes.md)
