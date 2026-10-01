---
title: "Neural Dynamic Policies for End-to-End Sensorimotor Learning"
type: paper
tags: [robot-learning, policy-representation, dynamic-movement-primitives, trajectory-space, action-reparameterization, imitation-learning, reinforcement-learning, end-to-end, ppo]
related: [concepts/dynamic-movement-primitives.md, concepts/policy-gradient-methods.md, concepts/discrete-time-gaussian-processes.md, papers/vonhartz_2025_midigap.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/neural_dyn_policies.pdf]
---

# Neural Dynamic Policies for End-to-End Sensorimotor Learning

**Authors:** Shikhar Bahl, Mustafa Mukadam, Abhinav Gupta, Deepak Pathak  
**Affiliations:** CMU, FAIR  
**Venue:** NeurIPS 2020  
**arXiv:** 2012.02788  
**Project:** https://shikharbahl.github.io/neural-dynamic-policies/

---

## Abstract

Current RL and IL policies reason in raw action spaces (torques, joint angles, end-effector positions), making decisions individually at each timestep. This limits scalability to long-horizon, high-dimensional, continuous tasks. NDPs reparameterize the action space of a deep network via second-order differential equations: the network predicts parameters of a DMP, the DMP forward integrator generates a trajectory, and the full system is trained end-to-end by differentiating through the ODE. NDPs outperform prior state-of-the-art in terms of efficiency or performance on several robotic control tasks for both IL and RL.

---

## Key Contributions

- **Neural Dynamic Policy (NDP)**: embeds a Dynamic Movement Primitive (DMP) as a fully differentiable layer inside a neural network policy, enabling end-to-end training in trajectory space
- **Multi-action critic**: k critic heads predicting different Q-value estimates for each step in the NDP rollout — handles the mismatch between NDP's high-frequency action generation and sparse environment rewards
- **Closed-form gradients through the DMP**: analytical expressions for ∂f/∂w and ∂f/∂g allow backpropagation from trajectory loss to network parameters θ
- Works identically for imitation learning (BC loss on trajectory) and RL (PPO with multi-action critic)
- NDP runs at inference at 0.5–5 kHz vs. environment at 100 Hz, producing dynamically smooth motions

---

## Method

### Dynamic Movement Primitives (DMP Review)

A DMP represents motion via a second-order ODE with a nonlinear forcing function:

$$\ddot{y} = \alpha\bigl(\beta(g - y) - \dot{y}\bigr) + f(x, g) \tag{1}$$

where:
- y, ẏ, ÿ — robot state, velocity, acceleration (generalized coordinate)
- g — goal/attractor state
- α, β — global damping parameters (set for critical damping: β = α/4)
- f(x, g) — forcing function capturing trajectory shape; x is a phase variable decaying as ẋ = −aₓx

The forcing function is a weighted sum of Gaussian RBFs:

$$f(x, g) = \frac{\sum_i \psi_i w_i}{\sum_i \psi_i}\, x(g - y_0), \quad \psi_i = e^{-h_i(x - c_i)^2} \tag{3}$$

where cᵢ = e^{−iπ/n} are horizontal shifts and hᵢ = n/π is the width. **The weights w = {w₁,...,wₙ} and goal g are the learnable parameters of the DMP**; given w and g, numerically integrating Eq. (1) produces a complete trajectory.

### Neural Dynamic Policy Architecture

$$\pi(a | s;\theta) \doteq \Omega\!\bigl(\text{DE}(\Phi(s;\theta))\bigr)$$

1. **Neural network Φ(s; θ)**: takes unstructured input s (image or state) → predicts (w, g)
2. **Forward integrator DE(w, g)**: solves Eq. (1) via Euler for m steps → trajectory {y, ẏ, ÿ}_{t=1..k}
3. **Inverse controller Ω(y, ẏ, ÿ)**: converts trajectory states to executable robot actions
   - Joint angle space ↔ joint angles: Ω = identity
   - Joint angles ↔ torques: Ω = inverse dynamics

The NDP produces k actions per forward pass (rollout length k). During online RL, the environment is stepped k times with these actions while the world is observed only once every k steps.

### Differentiating Through the DMP

Euler integration gives:

$$\dot{y}_t = \dot{y}_{t-1} + \ddot{y}_t\,dt, \quad y_t = y_{t-1} + \dot{y}_t\,dt \tag{5}$$

Closed-form partial derivatives from the forcing function (Eq. 3):

$$\frac{\partial f(x_t, g)}{\partial w_i} = \frac{\psi_i}{\sum_j \psi_j}\,x_t(g - y_0), \qquad \frac{\partial f(x_t, g)}{\partial g} = \frac{\sum_j \psi_j w_j}{\sum_j \psi_j}\,x_t \tag{6}$$

A recursive relationship (similar to Pahic et al.) then gives ∂yₜ/∂wᵢ and ∂yₜ/∂g from ∂yₜ₋₁/∂wᵢ and ∂yₜ₋₁/∂g, enabling full backpropagation through the ODE to θ.

### Training for Imitation Learning

Given demonstrated trajectory τ_target, minimize squared trajectory error:

$$\mathcal{L}_\text{imitation} = \sum_s \|\pi(s) - \tau_\text{target}(s)\|^2 \tag{7}$$

Gradients flow from trajectory loss → DMP parameters (w, g) → network θ.

### Training for Reinforcement Learning

Uses PPO as the RL optimizer. Key design choice: **multi-action critic** — k separate critic heads, each estimating the value at one step of the NDP rollout. Found to work better in practice than a single shared value for all k steps (because intermediate steps have different values under sparse rewards).

```
Algorithm 1: Training NDPs for RL
for episode = 1, 2, ...
  for t = 0, k, 2k, ...
    w, g = Φ(sₜ)               # predict DMP params from obs
    yₜ, ẏₜ from sₜ (pos, vel)
    for m = 1, ..., M           # M integration steps
      estimate ẋₘ via Eq.(2)
      estimate ÿₘ, ẏₘ, yₘ via Eqs.(4),(5)
      a = Ω(yₘ, yₘ, yₘ₋₁)     # inverse controller
      apply a → get s'
      store transition (s, a, s', r)
    compute policy gradient ∇θJ
    θ ← θ + η∇θJ
```

---

## Experiments

### Environments

| Task | Robot | Type | Notes |
|------|-------|------|-------|
| Throwing | Kinova Jaco 6-DoF | Dynamic | Toss cube into bin; random start positions |
| Picking | Kinova Jaco 6-DoF | Dynamic | Pick cube and lift; random start positions |
| Pushing | Sawyer (Meta-World) | Quasi-static | Push object to goal |
| Faucet Open | Sawyer (Meta-World) | Quasi-static | Rotate faucet |
| Soccer | Sawyer (Meta-World) | Quasi-static | Kick ball to goal |
| MT50 | Sawyer (Meta-World) | Multi-task | All 50 tasks jointly |

All use MuJoCo simulation. All environments have randomized goals (or positions for Throwing/Picking). RL setup uses k=5 rollout; IL digit writing uses k=300.

### Baselines

- **PPO**: plain PPO in raw action space
- **PPO-multi**: PPO with same multi-action architecture as NDP but no DMP structure
- **VICES**: Variable Impedance Control in End-Effector Space — reparameterizes as PD controller parameters
- **Dyn-E**: Dynamics-Aware Embeddings — learns lower-dimensional action embeddings from environment dynamics
- **CNN / CNN-DMP**: for imitation only; CNN-DMP maps image to a single DMP (Pahic et al. 2018)

### Imitation Learning Results

| Method | Throw | Pick | Push | Soccer | Faucet |
|--------|-------|------|------|--------|--------|
| NN | 0.528 | **0.672** | 0.002 | 0.885 | 0.532 |
| **NDP** | **0.642** | 0.408 | **0.208** | **0.890** | **0.790** |

NDP outperforms NN on 4/5 tasks. Picking is the exception — high-dimensional precise contact makes BC harder.

**Digit writing (imitation)**: input = digit image; output = end-effector trajectory

| Method | Train loss | Test loss |
|--------|-----------|----------|
| CNN | 10.42 ± 5.26 | 10.59 ± 4.63 |
| CNN-DMP | 9.44 ± 4.59 | 8.46 ± 8.45 |
| **NDP** | **0.70 ± 0.36** | **0.74 ± 0.34** |

NDP outperforms baselines by ~14× on test loss; produced trajectories are smoother and qualitatively much closer to the digit shape.

### Reinforcement Learning Results

NDP gains in both efficiency (fewer samples) and final performance on most tasks:
- **Throwing**: NDP shows clear data efficiency and higher final performance
- **Picking**: NDP most efficient; PPO-multi fails
- **Pushing, Faucet**: NDP best or tied; PPO-multi inconsistent
- **Soccer**: PPO achieves higher final performance, but NDP is ~2× more sample-efficient
- **MT50**: NDP slightly higher absolute performance; no efficiency gains for any method (transfer hard for all)

NDP outperforms VICES (which struggles with high action dimensionality) and Dyn-E (which fails on complex dynamics tasks like Picking).

### Ablations

Conducted on the Pushing task (RL), varying:

| Hyperparameter | Range tested | Finding |
|---------------|-------------|---------|
| RBF kernel type | Gaussian, linear, multiquadric, inv. quadratic, inv. multiquadric | All similar; NDP robust to this choice |
| # basis functions N | {2, 6, 10, 15, 20} | No large effect; small N may lack representation power |
| Integration steps | {15, 25, 35, 45} | No large effect |
| Rollout length k | {3, 5, 7, 10, 15} | k=3 too short to capture motion; k=5–15 similar |

**Component ablations** (Throwing, Push, Soccer, Faucet, Pick):
- **only-g** (force w=0, learn goal only): significantly less sample-efficient; converges to slightly lower performance
- **learn-α** (also learn global damping α): worse — fixed critical damping is a useful inductive bias

---

## Limitations & Open Questions

- **MT50 multitask**: NDP does not show efficiency gains on all 50 tasks jointly; cross-task transfer remains hard
- **Picking (IL)**: worse than baseline — contact-rich, high-precision tasks may require richer trajectory representations
- **Fixed α**: while ablations show this is good, the optimal damping may vary per task; adaptive α is not explored
- **DMP structure**: DMPs cannot represent arbitrary trajectories (no periodic motions in basic form, no multi-modal distributions). MiDiGaP (von Hartz et al. 2025) addresses these limitations with per-timestep Gaussian modeling.
- **Single trajectory per forward pass**: NDP generates one trajectory segment per observation; handling multi-modal behavior requires explicit mixture models
- **Open question**: Can the DMP be replaced by a richer dynamical structure (e.g., Riemannian Motion Policies) while remaining differentiable?

---

## Relationship to Other Policy Representations

| Method | Structure | Training | Modality | Distribution |
|--------|-----------|----------|----------|-------------|
| Raw action (PPO) | None | RL/IL | Reactive | Gaussian per step |
| NDP (this paper) | DMP (2nd-order ODE) | RL+IL e2e | Trajectory | Unimodal trajectory |
| DiGaP / MiDiGaP | Per-timestep Gaussians on Riemannian manifold | IL (few-shot) | Trajectory | Multi-modal, no DMP constraints |
| Diffusion Policy | Denoising diffusion | IL | Trajectory | Expressive multi-modal |
| VLA action expert | Flow matching | IL+RL | Chunk | Multi-modal, 50Hz |

NDP is the earliest work to embed DMP structure end-to-end in deep networks for both IL and RL. Its limitations (unimodal, fixed kernel, no Riemannian geometry) motivated subsequent trajectory-space approaches.

---

## See Also

- [Dynamic Movement Primitives & Trajectory-Space Policies](../concepts/dynamic-movement-primitives.md) — concept article covering DMPs and structured action reparameterization
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — PPO used as RL backbone
- [Discrete-Time Gaussian Processes for Imitation Learning](../concepts/discrete-time-gaussian-processes.md) — DiGaP: later trajectory-space approach addressing NDP's limitations
- [MiDiGaP (von Hartz et al. 2025)](vonhartz_2025_midigap.md) — mixture of DiGaPs; per-timestep Gaussian with multi-modal support
