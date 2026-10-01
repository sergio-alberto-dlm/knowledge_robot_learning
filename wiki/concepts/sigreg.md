---
title: SIGReg (Sketched Isotropic Gaussian Regularization)
type: concept
tags: [self-supervised-learning, distribution-matching, isotropic-gaussian, characteristic-function, regularization, theory]
related: [concepts/joint-embedding-predictive-architecture.md, papers/balestriero_2025_lejepa.md, codebase/rl_lab/legacy-mujoco-and-auxiliary.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/LeJEPA.pdf]
---

# SIGReg: Sketched Isotropic Gaussian Regularization

## What It Is

SIGReg is a regularization objective introduced in LeJEPA (Balestriero & LeCun, 2025) that enforces the embedding distribution $P_\theta = f_\theta(x)_\#P_x$ to match an **isotropic Gaussian** $\mathcal{N}(0, I_K)$. It achieves this via *random projection sketching* — testing that all 1D marginals of the embeddings match their Gaussian counterparts — which bypasses the curse of dimensionality while maintaining full distributional expressiveness.

**Motivation:** Balestriero & LeCun prove (Theorem 1) that the isotropic Gaussian is the unique distribution minimizing worst-case downstream prediction risk (for linear, k-NN, and kernel probes). SIGReg is the practical mechanism to enforce this property during training.

## Construction

### Step 1 — Cramér-Wold Sketching

The **Hyperspherical Cramér-Wold theorem** (Lemma 3) states that for $\mathbb{R}^d$-valued random vectors:

$$\langle u, X \rangle \stackrel{d}{=} \langle u, Y \rangle, \quad \forall u \in \mathbb{S}^{d-1} \iff X \stackrel{d}{=} Y$$

This allows reducing a $K$-dimensional distribution test to an aggregation of scalar (1D) tests.

### Step 2 — Directional Test Statistics

For a unit-norm direction $a \in \mathbb{S}^{K-1}$, define the projected samples $z_i = a^\top f_\theta(x_i)$ and test the hypothesis:

$$H_0(a): P_\theta^{(a)} = Q^{(a)} \quad \text{vs.} \quad H_1(a): P_\theta^{(a)} \neq Q^{(a)} \tag{3}$$

The global test statistic maximizes over a set $\mathbb{A}$ of directions:

$$T_\mathbb{A}(\{f_\theta(x_n)\}_{n=1}^N) \triangleq \max_{a \in \mathbb{A}} T(\{a^\top f_\theta(x_n)\}_{n=1}^N) \tag{4}$$

**Theorem 2 (Sufficiency):** Equation (4) is a valid consistent test: if $P=Q$, it is level-$\alpha$; if $P \ne Q$, it has power 1 as $n \to \infty$.

### Step 3 — SIGReg Definition

To avoid sparse gradients from the max in (4), SIGReg replaces the maximum with an average:

$$\text{SIGReg}_T(\mathbb{A}, \{f_\theta(x_n)\}_{n=1}^N) \triangleq \frac{1}{|\mathbb{A}|} \sum_{a \in \mathbb{A}} T(\{a^\top f_\theta(x_n)\}_{n=1}^N) \tag{SIGReg}$$

## Choice of Test Statistic T

Three families of tests are considered and the **Epps-Pulley (EP)** test is selected:

| Family | Examples | Issue |
|--------|----------|-------|
| Moment-based | Jarque-Bera, EJB | Gradient norm scales as $O(k)$ for $k$-th moment; insufficient by Theorem 3 (finite moments cannot identify distribution) |
| CDF-based | Cramér-von Mises, Anderson-Darling, Watson | Require sorting → non-differentiable, $O(N \log N)$, breaks DDP |
| **CF-based** | **Epps-Pulley (EP)** | **Differentiable, O(N), bounded gradients, DDP-friendly** |

### Epps-Pulley Statistic

$$EP = N \int_{-\infty}^{\infty} |\hat{\phi}_X(t) - \phi_{\mathcal{N}}(t)|^2 w(t)\, dt, \quad w(t) = e^{-t^2/2}$$

where $\hat{\phi}_X(t) = \frac{1}{n} \sum_{j=1}^n e^{itX_j}$ is the empirical CF and $\phi_{\mathcal{N}}(t) = e^{-t^2/2}$ is the standard Gaussian CF. The integral is approximated by trapezoidal quadrature (17 points, domain $[-5,5]$ recommended).

**Theorem 4 (Stability of EP Test):**

$$\left|\frac{\partial EP}{\partial z_i}\right| \le \frac{4\sigma^2}{N}, \quad \left|\frac{\partial^2 EP}{\partial z_i^2}\right| \le \frac{C\sqrt{\pi}\sigma^3}{2N}$$

Gradients and curvature are **bounded regardless of input distribution**, unlike moment-based tests.

## Defeating the Curse of Dimensionality

**Theorem 5 (Unified Error Bounds):** When the embedding distribution has Sobolev regularity $\alpha$, the number of directions $|\mathbb{A}|$ needed scales as $O(K)$ — linear in the embedding dimension. Critically:
- The target distribution (isotropic Gaussian) is smooth ($\alpha$ large), so SIGReg's error bounds decay rapidly with $|\mathbb{A}|$ (Figure 4).
- SGD compounds this: the cumulative number of sampled directions grows linearly with training time, providing tight distributional matching even with $|\mathbb{A}| = 16$ per step.

## PyTorch Implementation (Algorithm 1)

```python
def SIGReg(x, global_step, num_slices=256):
    dev = dict(device=x.device)
    g = torch.Generator(**dev)
    g.manual_seed(global_step)
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)
    # Epps-Pulley statistic
    t = torch.linspace(-5, 5, 17, **dev)
    exp_f = torch.exp(-0.5 * t * t*2)    # theoretical CF for N(0,1)
    x_t = (x @ A).unsqueeze(2) * t       # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size
    T = torch.trapz(err, t, dim=1) * N
    return T
```

The `global_step` seed ensures directions are **resampled each step** (avoiding fixed-direction bias) while being synchronized across DDP workers. Total memory and compute: O(N · num_slices · 17) — linear in batch size.

## Relation to Other Methods

- **VICReg:** Setting $T(\{x_n\}) = \text{mean}(\{x_n\})^2 + (\text{std}(\{x_n\}) - 1)^2$ in SIGReg recovers VICReg in the limit of large slices. However, Theorem 3 shows finite moments are insufficient to identify the distribution → shortcut solutions not eliminated.
- **Sliced Wasserstein:** SIGReg uses the same 1D projection principle but with a CF-based statistic and a fixed target (Gaussian), not sample-to-sample comparison.
- **Kernel MMD:** When the EP integral is computed exactly (not quadrature), each slice value recovers a kernel MMD with a specific kernel — but with quadratic complexity instead of linear.

## Practical Recommendations

- **num_slices:** 1024 (competitive at 512; more slices give marginal gains)
- **Integration:** domain $[-5, 5]$, 17 quadrature points
- **Gradient bias:** $O(1/N)$; negligible for batch sizes ≥ 16
- **No register tokens needed** (unlike DINO-based methods)

## See Also

- [Joint Embedding Predictive Architecture (JEPA)](joint-embedding-predictive-architecture.md) — framework that SIGReg regularizes
- [LeJEPA (Balestriero & LeCun, 2025)](../papers/balestriero_2025_lejepa.md) — paper introducing SIGReg
- [rl_lab — Legacy & Auxiliary Code](../codebase/rl_lab/legacy-mujoco-and-auxiliary.md) — a standalone SIGReg implementation used in a CartPole encoder identifiability experiment
