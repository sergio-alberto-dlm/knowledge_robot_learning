---
title: Dynamic Movement Primitives & Trajectory-Space Policy Parameterization
type: concept
tags: [robot-learning, policy-representation, dynamic-movement-primitives, trajectory-space, action-reparameterization, classical-robotics, imitation-learning, reinforcement-learning]
related: [papers/bahl_2020_ndp.md, concepts/discrete-time-gaussian-processes.md, papers/vonhartz_2025_midigap.md, concepts/policy-gradient-methods.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/neural_dyn_policies.pdf]
---

# Dynamic Movement Primitives & Trajectory-Space Policy Parameterization

## Core Idea

Rather than predicting one action per timestep (the dominant deep learning paradigm), **trajectory-space policy parameterization** has the policy predict parameters of a *trajectory representation*, from which a sequence of actions is generated. This shifts reasoning to a lower-dimensional space of behaviorally coherent motion plans.

The key insight: a trajectory is not an arbitrary sequence of positions — it has physical structure (smoothness, attractor dynamics, momentum). Parameterizing policies in a space that respects this structure makes learning more sample-efficient and produces more dynamically plausible motions.

---

## Dynamic Movement Primitives (DMPs)

DMPs are the canonical trajectory representation from classical robotics (Ijspeert et al. 2002/2013, Schaal 2006). A DMP encodes a motion as a second-order ODE:

$$\ddot{y} = \alpha\bigl(\beta(g - y) - \dot{y}\bigr) + f(x, g)$$

where:
- **y, ẏ, ÿ** — generalized coordinate, velocity, acceleration
- **g** — goal/attractor (trajectory converges to g as x → 0)
- **α, β** — damping coefficients; typically set for critical damping (β = α/4)
- **f(x, g)** — nonlinear *forcing function* that shapes the trajectory away from a straight-line attractor; parameterized by **w** (basis function weights)
- **x** — phase variable, decaying via ẋ = −aₓx (makes trajectories time-invariant)

### Forcing Function

Standard implementation uses weighted Gaussian RBFs:

$$f(x, g) = \frac{\sum_i \psi_i w_i}{\sum_i \psi_i}\, x(g - y_0), \qquad \psi_i = e^{-h_i(x - c_i)^2}$$

The basis functions span the phase trajectory; their weights **w** capture all the shape complexity of the motion. Given (**w**, **g**), numerically integrating the ODE produces a complete trajectory.

### Key Properties of DMPs

| Property | Description |
|----------|-------------|
| **Goal-attractor** | Trajectory always converges to g; change g to retarget the motion |
| **Phase-invariant** | Parameterized by phase x, not wall-clock time → robust to execution speed changes |
| **Smooth** | Second-order ODE guarantees continuous velocity; no discontinuities |
| **Compact** | Full trajectory encoded in O(N) parameters (N basis weights + goal g) |
| **Generalizable** | Scale/translate a demonstrated trajectory by modifying g and y₀ |

### Limitations of Classical DMPs

- **Unimodal**: one DMP = one trajectory mode; cannot represent distributions or multimodal behaviors
- **Fixed kernel**: Gaussian RBFs may not capture all trajectory types (oscillatory, chaotic, non-stationary)
- **No uncertainty**: classical DMPs have no probabilistic interpretation
- **Per-demonstration**: one DMP is typically fit to one demonstration; generalizing to new states requires separate learning
- **Not end-to-end**: classical fitting is a post-hoc optimization, not integrated into a policy network

---

## Embedding DMPs in Neural Policies

### Neural Dynamic Policies (NDPs — Bahl et al. 2020)

NDPs are the first method to embed a DMP as a **differentiable layer** inside a deep network, enabling end-to-end training:

$$\pi(a | s; \theta) = \Omega\!\bigl(\text{DE}(\Phi(s; \theta))\bigr)$$

- **Φ(s; θ)**: neural network → predicts DMP parameters (w, g) from state/image
- **DE(w, g)**: forward integrator → solves the ODE for k steps → trajectory {y, ẏ, ÿ}
- **Ω(y, ẏ, ÿ)**: inverse controller → converts to executable actions

**Differentiability**: closed-form gradients ∂f/∂w and ∂f/∂g allow backpropagation through the ODE:
$$\frac{\partial f}{\partial w_i} = \frac{\psi_i}{\sum_j \psi_j}\,x(g - y_0), \qquad \frac{\partial f}{\partial g} = \frac{\sum_j \psi_j w_j}{\sum_j \psi_j}\,x$$

**Multi-action critic**: when using RL (PPO), k separate critic heads estimate values at each step of the NDP rollout — necessary because intermediate steps lack direct environment feedback under sparse rewards.

**Inference speed**: NDP operates at 0.5–5 kHz (k× the environment frequency), producing smoother motions than step-by-step policies at 100 Hz.

---

## Trajectory-Space Policy Design Space

A broader family of methods parameterizes policies in trajectory space rather than raw action space:

| Method | Trajectory Rep. | Distribution | Differentiable | RL-compatible |
|--------|----------------|-------------|----------------|---------------|
| DMP (classical) | Weighted RBFs + ODE | None | No | No |
| **NDP** (Bahl 2020) | DMP as layer | Unimodal | Yes | Yes (PPO) |
| CNN-DMP (Pahic 2018) | Image→DMP params | Unimodal | Partial | No |
| **DiGaP** (von Hartz 2025) | Per-timestep Gaussians | Gaussian (unimodal) | — | — |
| **MiDiGaP** (von Hartz 2025) | Mixture of DiGaPs | Multi-modal | — | — |
| Diffusion Policy (Chi 2023) | Denoising trajectory | Multi-modal | Partial | Hard |
| VLA action expert (π₀) | Flow matching chunks | Multi-modal | Partial | Via RL token |

### Key Design Axes

**Trajectory representation**:
- ODE-based (DMP/NDP): physically interpretable, compact, smooth, unimodal
- Per-timestep Gaussian (DiGaP): no kernel constraints, captures non-stationary/oscillatory/chaotic
- Denoising/flow (Diffusion Policy, VLA): highly expressive, implicit distribution

**Distribution**:
- Deterministic / unimodal (DMP, NDP): efficient, but cannot capture demonstration diversity
- Mixture (MiDiGaP, Diffusion): captures multi-modal behavior; important when demonstrations vary by strategy

**End-to-end trainability**:
- Critical for RL; DMPs are differentiable when w and g are predicted by a neural network
- DiGaP/MiDiGaP are trained by IL only; no current RL formulation

**Action space vs. trajectory space trade-off**:
- Action space: flexible, reactive, handles any dynamics — but high-dimensional, no physical inductive bias
- Trajectory space: compact, smooth, physically plausible — but harder to use with sparse rewards, requires explicit multi-modality for complex tasks

---

## When to Use DMP-Based Policies

**Use DMPs / NDPs when:**
- Task is inherently continuous and trajectory-like (writing, throwing, reaching)
- Long-horizon execution matters and dynamical smoothness is desired
- Prior demonstrations have clear goal-directed structure
- RL is needed and action-space dimensionality is a bottleneck (NDP reduces effective search space)

**Prefer DiGaP/MiDiGaP when:**
- Demonstrations are highly multimodal or strategically diverse
- Riemannian geometry of the action space matters (SO(3) orientations, etc.)
- Few-shot learning from <10 demos is required
- Inference-time adaptation (collision avoidance, reachability constraints) is needed

**Prefer Diffusion/Flow policy when:**
- Maximum expressiveness over action distribution is required
- Large demonstration datasets are available
- Uncertainty over trajectory shape is complex and multi-modal

---

## See Also

- [NDP (Bahl et al. 2020)](../papers/bahl_2020_ndp.md) — first end-to-end trainable DMP policy for both IL and RL
- [Discrete-Time Gaussian Processes for Imitation Learning](discrete-time-gaussian-processes.md) — DiGaP/MiDiGaP: later trajectory-space methods addressing DMP limitations
- [MiDiGaP (von Hartz et al. 2025)](../papers/vonhartz_2025_midigap.md) — mixture of DiGaPs; state-of-the-art few-shot IL
- [Policy Gradient Methods](policy-gradient-methods.md) — PPO used as RL backbone in NDPs
