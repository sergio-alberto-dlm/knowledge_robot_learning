---
title: "Proximal Policy Optimization Algorithms"
type: paper
tags: [reinforcement-learning, policy-gradient, ppo, trust-region, actor-critic, continuous-control, atari, mujoco]
related: [concepts/policy-gradient-methods.md, concepts/intrinsic-motivation.md, concepts/model-predictive-control.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/PPO.pdf]
---

# Proximal Policy Optimization Algorithms

**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
**Affiliation:** OpenAI  
**arXiv:** 1707.06347v2 — 28 Aug 2017

---

## Abstract

PPO proposes a new family of policy gradient methods that alternate between sampling data through environment interaction and optimizing a "surrogate" objective with stochastic gradient ascent. Unlike standard policy gradient (one gradient update per data sample), PPO enables multiple epochs of minibatch updates on the same trajectory data. The key innovation is a clipped probability ratio objective that forms a pessimistic lower bound on the unclipped objective, preventing destructively large policy updates without the complexity of TRPO's second-order constrained optimization. PPO achieves the reliability of TRPO while being simpler to implement, more general (compatible with shared architectures and dropout), and achieving better empirical performance.

---

## Key Contributions

- **Clipped surrogate objective ($L^{CLIP}$):** Clips the probability ratio $r_t(\theta)$ to $[1-\epsilon, 1+\epsilon]$ and takes the minimum with the unclipped objective — a pessimistic bound that penalizes large policy updates with first-order optimization only.
- **Multiple epochs of minibatch SGD:** Reuses each batch of environment transitions for $K$ epochs of minibatch optimization, dramatically improving sample efficiency versus vanilla policy gradient.
- **Systematic surrogate objective comparison:** Empirically establishes that clipping ($\epsilon = 0.2$, avg score 0.82) outperforms adaptive KL penalty (0.74), fixed KL penalty (0.71), and no regularization (−0.39) on 7 MuJoCo tasks.
- **Broad applicability:** First-order optimization means PPO is compatible with parameter sharing between policy and value function, recurrent networks, dropout, and auxiliary losses — unlike TRPO.
- **State-of-the-art on continuous control and competitive on Atari:** Outperforms A2C, TRPO, and CEM on MuJoCo; wins 30/49 Atari games on average-training-reward vs. 18/49 for ACER and 1/49 for A2C.

---

## Methodology

### Background: Policy Gradient

The standard policy gradient estimator (Williams 1992):

$$\hat{g} = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t\right]$$

implemented by differentiating the surrogate objective:

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t\left[\log \pi_\theta(a_t \mid s_t)\hat{A}_t\right]$$

Performing multiple gradient steps on $L^{PG}$ with the same trajectory is not well-justified and leads to destructively large policy updates.

### Background: TRPO

TRPO maximizes a ratio-based surrogate subject to a KL constraint:

$$\text{maximize}_\theta \quad \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t\right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] \leq \delta$$

Requires conjugate gradient + line search (second-order). Provides a monotonic improvement guarantee but is incompatible with dropout and parameter sharing.

### Clipped Surrogate Objective

Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ (probability ratio; $r(\theta_\text{old}) = 1$).

Unclipped surrogate (conservative policy iteration):
$$L^{CPI}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right]$$

**PPO clipped objective:**
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Intuition:
- $\hat{A}_t > 0$ (good action): increasing $r_t$ improves $L^{CPI}$, but clipped at $1+\epsilon$ — no incentive to push the ratio further.
- $\hat{A}_t < 0$ (bad action): decreasing $r_t$ improves $L^{CPI}$, but clipped at $1-\epsilon$.
- Taking the **minimum** ensures $L^{CLIP}$ is a lower bound on $L^{CPI}$: ratio changes that would make the objective worse are never ignored.

Best default: $\epsilon = 0.2$.

### Adaptive KL Penalty (Alternative)

KL-penalized objective with adaptive coefficient $\beta$:

$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t - \beta\,\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]\right]$$

After each policy update, compute $d = \hat{\mathbb{E}}_t[\text{KL}]$:
- If $d < d_\text{targ}/1.5$: $\beta \leftarrow \beta/2$
- If $d > 1.5 \cdot d_\text{targ}$: $\beta \leftarrow \beta \times 2$

Underperforms clipping in practice (best adaptive KL score: 0.74 vs. 0.82).

### Combined Objective (Shared Architecture)

When policy and value function share parameters:

$$L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t\!\left[L^{CLIP}_t(\theta) - c_1\underbrace{(V_\theta(s_t) - V_t^\text{targ})^2}_{L^{VF}_t} + c_2\,S\big[\pi_\theta(\cdot\,|\,s_t)\big]\right]$$

$S$ is an entropy bonus discouraging premature convergence to a deterministic policy.

### Advantage Estimation (GAE)

Generalized Advantage Estimation (Schulman et al. 2015a):

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda = 0$ → 1-step TD (low variance, high bias); $\lambda = 1$ → Monte Carlo (high variance, low bias). Default: $\lambda = 0.95$.

### Algorithm

```
Algorithm 1: PPO, Actor-Critic Style
for iteration = 1, 2, ... do
    for actor = 1, ..., N do
        Run policy π_θ_old in environment for T timesteps
        Compute advantage estimates Â_1, ..., Â_T
    end for
    Optimize surrogate L w.r.t. θ, with K epochs and minibatch size M ≤ NT
    θ_old ← θ
end for
```

**MuJoCo hyperparameters (Table 3):** T=2048, Adam lr=3×10⁻⁴, K=10, M=64, γ=0.99, λ=0.95  
**Atari hyperparameters (Table 5):** T=128, N=8, K=3, M=256, ε=0.1×α (annealed 1→0), entropy c₂=0.01, γ=0.99

---

## Experimental Results

### 6.1 Surrogate Objective Comparison (7 MuJoCo Tasks, 1M Timesteps)

| Algorithm | Avg. Normalized Score |
|-----------|----------------------|
| No clipping or penalty | −0.39 |
| Clipping, ε = 0.1 | 0.76 |
| **Clipping, ε = 0.2** | **0.82** |
| Clipping, ε = 0.3 | 0.70 |
| Adaptive KL, d_targ = 0.01 | 0.74 |
| Fixed KL, β = 1.0 | 0.71 |

Scores normalized so random policy = 0, best result = 1, averaged over 21 runs × 7 environments.

### 6.2 Continuous Control — MuJoCo (1M Timesteps)

PPO (ε=0.2) vs. A2C, A2C+Trust Region, CEM, Vanilla PG (adaptive stepsize), TRPO. PPO outperforms all other methods on almost all of HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d.

### 6.3 3D Humanoid Control — Roboschool

Three tasks of increasing difficulty trained up to 100M timesteps:
1. RoboschoolHumanoid (forward locomotion only)
2. RoboschoolHumanoidFlagrun (target position changes randomly every 200 steps)
3. RoboschoolHumanoidFlagrunHarder (robot pelted by cubes, must recover from ground)

PPO successfully learns robust locomotion and goal-directed behavior on all three, demonstrating scaling to high-dimensional continuous control.

### 6.4 Atari — 49 Games (40M Frames / 10M Timesteps)

Compared against tuned A2C and ACER implementations using the same policy network architecture:

| Metric | A2C | ACER | **PPO** |
|--------|-----|------|---------|
| Games won (avg. training reward) | 1 | 18 | **30** |
| Games won (last 100 episodes) | 1 | **28** | 19 |

PPO wins more games by average training reward (fast learning); ACER edges out on final-episode reward (asymptotic). Notable PPO wins: Atlantis (2.3M vs 1.8M for ACER), Enduro (758 vs 0), Kangaroo (9928 vs 50), MontezumaRevenge (42 vs 0.3), TimePilot (4342 vs 4175).

---

## Limitations & Open Questions

- **No theoretical monotonic improvement guarantee:** Unlike TRPO, PPO's clipping heuristic lacks a formal performance bound. The choice of ε = 0.2 is empirically motivated.
- **ε not problem-adaptive:** Performance degrades clearly outside ε ∈ [0.1, 0.2]; no principled method to set ε per-task.
- **Asymptotic performance on Atari:** ACER outperforms PPO on final-episode scores (28 vs 19 games), suggesting PPO learns faster but may not converge to the strongest policy in all settings.
- **Sample complexity vs. wall-time:** PPO reuses data for K epochs, improving wall-time efficiency, but the on-policy nature means it cannot match off-policy methods (ACER, DQN) on pure sample efficiency.
- **Value function training not fully studied:** The paper does not investigate clipping the value function update (as done in PPO implementations in OpenAI Baselines), which subsequent work found important for stability.
- **Open question:** How does PPO perform with off-policy data or with replay buffers? PPO-based extensions (e.g., APPO, PPG) address this in subsequent work.

---

## See Also

- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — concept article covering REINFORCE, TRPO, PPO, GAE, and actor-critic architectures
- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — uses PPO as the base RL algorithm for curiosity experiments (Burda et al. 2018)
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md) — planning-based alternative to policy gradient for robot control
