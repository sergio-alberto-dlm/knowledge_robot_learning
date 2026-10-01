---
title: "Bui & Turner (2014) — Tree-structured Gaussian Process Approximations"
type: paper
tags: [gaussian-processes, sparse-gp, pseudo-datapoints, belief-propagation, variational-inference, time-series, spatial-inference, kl-divergence]
related: [concepts/sparse-gaussian-processes.md, concepts/gaussian-belief-propagation.md, concepts/discrete-time-gaussian-processes.md, papers/nabarro_2024_gbp_learning.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/GP_w_GBP.pdf]
---

# Bui & Turner (2014) — Tree-structured Gaussian Process Approximations

**Venue:** NeurIPS 2014
**Authors:** Thang Bui, Richard Turner (Computational and Biological Learning Lab, University of Cambridge)
**Code:** http://mlg.eng.cam.ac.uk/thang/ [Tree+VFE]

## Abstract

Gaussian process regression can be accelerated by constructing a small pseudo-dataset to summarize observed data. Standard pseudo-datapoint approximations require the number of pseudo-datapoints to scale with the range of the input space — leading to quadratic or cubic computational costs for time-series and large spatial datasets. This paper devises an approximation whose complexity grows **linearly with the number of pseudo-datapoints** by imposing a **tree or chain structure** on the pseudo-dataset and calibrating the approximation via KL minimization. Inference and learning are performed efficiently using Gaussian belief propagation. The method dominates the speed-accuracy frontier over a large number of existing approximations on audio missing data imputation and 2D spatial terrain reconstruction tasks.

## Key Contributions

1. **Tree-structured GP prior approximation**: a new indirect posterior approximation that partitions pseudo-datapoints into $K$ blocks arranged in a tree, enabling conditional independence structure that allows linear-in-$N$ inference.
2. **KL-calibrated approximation**: the tree approximate prior is calibrated against the exact GP prior via forward KL divergence minimization, yielding closed-form optimal distributions.
3. **GBP inference and learning**: exact inference in the tree-structured model is performed via the GBP up-down algorithm; marginal likelihood derivatives are also computed via GBP, making hyperparameter learning tractable.
4. **Unified framework**: the tree approximation subsumes FITC, VFE, PIC (full and local), and the full GP as special cases (see Table 1 and supplementary material).
5. **Dominant speed-accuracy frontier**: on three challenging datasets, the tree approximation Pareto-dominates all compared methods across the full range of training/test time budgets.

## Methodology

### GP Regression Background

Standard GP regression with $N$ training points $\{\mathbf{x}_n, y_n\}$ assumes:
$$y_n = f(\mathbf{x}_n) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)$$

The posterior function distribution is Gaussian with:
$$m_f(\mathbf{x}) = \mathbf{K}_{\mathbf{xf}}(\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})^{-1}\mathbf{y}, \quad k_f(\mathbf{x},\mathbf{x}') = k(\mathbf{x},\mathbf{x}') - \mathbf{K}_{\mathbf{xf}}(\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})^{-1}\mathbf{K}_{\mathbf{fx}'}$$

The bottleneck: forming the Cholesky decomposition of $\mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I}$ costs $\mathcal{O}(N^3)$ for training and $\mathcal{O}(N^2)$ for prediction. Hyperparameters $\theta$ and noise $\sigma^2$ are learned by maximizing the marginal likelihood $p(\mathbf{y}|\theta, \sigma) = \mathcal{N}(\mathbf{y}; \mathbf{0}; \mathbf{K}_{\mathbf{ff}} + \sigma^2\mathbf{I})$.

### Why Standard Pseudo-Datapoint Methods Fail at Scale

Pseudo-datapoint methods (FITC, VFE, PIC) reduce cost to $\mathcal{O}(NM^2)$ training and $\mathcal{O}(M^2)$ prediction using $M \ll N$ pseudo-inputs. However, each pseudo-datapoint only sculpts the posterior in a local region of input space. To maintain accuracy, $M$ must scale with $\prod_d L_d / l_d$ (ratio of data range to posterior dependency length per dimension). For time-series ($L \propto N$) and large 2D spatial datasets, $M \gg N$ is required — negating all computational savings. The rule-of-thumb estimated required pseudo-datasets for the paper's three tasks: $\{1400, 1000, 5000\}$ — all at the upper limit of tractable standard approximations.

### Tree-Structured Approximation

The key structural change: **impose a tree (or chain) on the pseudo-datapoints**.

**Partition** the $M$ pseudo-datapoints into $K$ disjoint blocks $\{\mathbf{u}_{B_k}\}_{k=1}^K$ and the $N$ function values into $K$ blocks $\{\mathbf{f}_{C_k}\}_{k=1}^K$. The approximate model is:

$$q(\mathbf{u}) = \prod_{k=1}^K q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}(B_k)}), \quad q(\mathbf{f}|\mathbf{u}) = \prod_{k=1}^K q(\mathbf{f}_{C_k}|\mathbf{u}_{B_k}), \quad p(\mathbf{y}|\mathbf{f}) = \prod_{n=1}^N p(y_n; f_n, \sigma^2)$$

where $\mathbf{u}_{\mathrm{par}(B_k)}$ denotes the pseudo-datapoints in the parent block of $B_k$.

**Calibration via KL minimization**: minimize the forward KL divergence between the true model prior and the approximation. The optimal distributions are the conditional distributions from the unapproximated augmented model:

$$q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}(B_k)}) = \mathcal{N}(\mathbf{u}_{B_k}; \mathbf{A}_k\mathbf{u}_{\mathrm{par}(B_k)}, \mathbf{Q}_k)$$
$$q(\mathbf{f}_{C_k}|\mathbf{u}_{B_k}) = \mathcal{N}(\mathbf{f}_{C_k}; \mathbf{C}_k\mathbf{u}_{B_k}, \mathbf{R}_k)$$

with parameters (where $\mathbf{u}_k = \mathbf{u}_{B_k}$, $\mathbf{u}_l = \mathbf{u}_{\mathrm{par}(B_k)}$, $\mathbf{f}_k = \mathbf{f}_{C_k}$):

$$\mathbf{A}_k = \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_l}\mathbf{K}_{\mathbf{u}_l,\mathbf{u}_l}^{-1}, \quad \mathbf{Q}_k = \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k} - \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_l}\mathbf{K}_{\mathbf{u}_l,\mathbf{u}_l}^{-1}\mathbf{K}_{\mathbf{u}_l,\mathbf{u}_k}$$
$$\mathbf{C}_k = \mathbf{K}_{\mathbf{f}_k,\mathbf{u}_k}\mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k}^{-1}, \quad \mathbf{R}_k = \mathbf{K}_{\mathbf{f}_k,\mathbf{f}_k} - \mathbf{K}_{\mathbf{f}_k,\mathbf{u}_k}\mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k}^{-1}\mathbf{K}_{\mathbf{u}_k,\mathbf{f}_k}$$

After marginalizing out $\mathbf{f}$, the effective local likelihood is:
$$p(\mathbf{y}_k|\mathbf{u}_k) = \mathcal{N}(\mathbf{y}_k; \mathbf{C}_k\mathbf{u}_k, \mathbf{R}_k + \sigma^2\mathbf{I})$$

This is a **tree-structured Gaussian model with latent variables** $\mathbf{u}$ and observations $\mathbf{y}$, exactly the structure suited to GBP.

### Complexity

Inference via the GBP up-down algorithm (leaves → root → leaves): $\mathcal{O}(KD^3) \approx \mathcal{O}(ND^2)$, where $D$ is the average block size (observations per block). Since $D$ is fixed (independent of $N$), total cost is **linear in $N$** — a fundamental improvement over the $\mathcal{O}(NM^2)$ cost of FITC/VFE when $M \propto N$.

### Tree Construction

1. **Block assignment**: use k-means with Euclidean distance to assign the $N$ observations to $K$ blocks. For regular-grid time-series, regular blocks suffice. Block sizes chosen to accelerate inference.
2. **Pseudo-input selection**: randomly select a subset of data within each block as pseudo-inputs.
3. **Tree structure**: compute pairwise distances between block centers, build minimum spanning tree via Kruskal's algorithm on the fully connected graph, select a random root.
4. **Matrix parameters**: $\{\mathbf{A}_k, \mathbf{Q}_k, \mathbf{C}_k, \mathbf{R}_k\}$ are computed by traversing the tree root-to-leaves during training and recomputed at each step.

### Inference and Hyperparameter Learning

**Inference**: GBP up-down algorithm on the tree. Messages are Gaussian (information vector + precision matrix). Two passes suffice (upward from leaves to root, then downward). For chain structures, equivalent to the Kalman smoother.

**Marginal likelihood**: has recursive form $p(\mathbf{y}_{1:K}|\theta) = \prod_{k=1}^K p(\mathbf{y}_k|\mathbf{y}_{1:k-1}, \theta)$. Derivatives involve only local moments:
$$\frac{d}{d\theta}\log p(\mathbf{y}|\theta) = \sum_{k=1}^K \left[\left\langle\frac{d}{d\theta}\log p(\mathbf{u}_k|\mathbf{u}_l)\right\rangle_{p(\mathbf{u}_k,\mathbf{u}_l|\mathbf{y})} + \left\langle\frac{d}{d\theta}\log p(\mathbf{y}_k|\mathbf{u}_k)\right\rangle_{p(\mathbf{u}_k|\mathbf{y})}\right]$$

Hyperparameters optimized with BFGS to a local maximum of the marginal likelihood.

### Unification of Existing Methods

The tree approximation includes as special cases:
- **Full GP**: single block
- **FITC**: $\mathbf{A}_k = \mathbf{0}$, $\mathbf{Q}_k = \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k}$ (all blocks share one root, no tree structure)
- **VFE**: $\mathbf{A}_k = \mathbf{0}$, $\mathbf{Q}_k = \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k}$, diagonal $\mathbf{R}_k$
- **PIC (local)**: $\mathbf{A}_k = \mathbf{0}$, $\mathbf{Q}_k = \mathbf{K}_{\mathbf{u}_k,\mathbf{u}_k}$
- **Chain/tree GP**: full tree-structured parameterization

## Experimental Results

### Datasets and Baselines

Three real-world regression tasks with challenging spatial/temporal extent:

| Task | Dataset | $N$ | Missing regions | Kernel |
|---|---|---|---|---|
| Audio sub-band imputation | TIMIT speech (152Hz channel) | 50,000 | 25 sections of 80 samples | Exponentiated quadratic |
| Audio filter imputation | TIMIT filtered (50Hz bandwidth) | 50,000 | Missing sections of 150 samples | Spectral mixture (2 components) |
| 2D terrain reconstruction | UK OS terrain-50 (20km×30km) | ~240,000 | 80 blocks of 1km² | Anisotropic exp. quadratic |

**Baselines**: Chain, Local (PIC), FITC, VFE, SSGP, SDE (Kalman smoother via SDEs, only for Exp. 1).

**Metric**: Standardized Mean Squared Error (SMSE). Speed-accuracy trade-off curves (SMSE vs. training time and test time).

### Results Summary

**Experiment 1 (Audio sub-band)**:
- Chain method outperforms all baselines at every training/test time budget
- At training time = 100s: chain achieves 3× lower SMSE than the local PIC method (next best)
- Global approximations (FITC, VFE, SSGP) cannot tractably handle enough pseudo-datapoints

**Experiment 2 (Audio filter, spectral mixture)**:
- Chain substantially more accurate than FITC, VFE, and local PIC at all budgets
- Spectral mixture kernels do not affect the relative rankings

**Experiment 3 (2D terrain)**:
- Tree approximation with {2500, 10000, 20000} pseudo-datapoints used (estimated required: 5000)
- Tree dominates speed-accuracy frontier; global methods cannot reach sufficient pseudo-datapoint counts
- Local PIC faster at small budgets but introduces block-boundary artifacts; tree eliminates these

**Overall**: "The speed-accuracy frontier for the new approximation scheme dominates those produced by the other methods over a wide range for each of the three datasets."

## Limitations & Open Questions

1. **High-dimensional inputs**: tree structure built on Euclidean distances in input space; unclear whether the tree generalizes well when input dimensionality is large.
2. **Block size cubic cost**: complexity is $\mathcal{O}(ND^2)$ — linear in $N$ but still cubic in block size $D$; very dense blocks remain expensive.
3. **Fixed blocks during learning**: block assignment and pseudo-input selection are fixed after initialization; joint optimization of block structure with hyperparameters is not done.
4. **Decoupled prediction tree**: the tree used for prediction could differ from the tree used for training (using the same pseudo-datapoints), potentially improving prediction — unexplored.
5. **Online/stochastic learning**: the tree structure naturally enables distributed stochastic gradient updates, but this was not implemented.
6. **Open question**: does the tree structure generalize to GP classification, multi-output GPs, or deep GPs?

## See Also

- [Sparse Gaussian Processes](../concepts/sparse-gaussian-processes.md)
- [Gaussian Belief Propagation & Factor Graphs](../concepts/gaussian-belief-propagation.md)
- [Discrete-Time Gaussian Processes for Imitation Learning](../concepts/discrete-time-gaussian-processes.md)
- [Nabarro et al. (2024) — Learning in Deep Factor Graphs with GBP](nabarro_2024_gbp_learning.md): GBP also learns the model parameters, with no outer gradient loop
