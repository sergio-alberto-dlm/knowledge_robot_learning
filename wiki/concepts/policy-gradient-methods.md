---
title: Policy Gradient Methods
type: concept
tags: [reinforcement-learning, policy-gradient, ppo, trpo, actor-critic, advantage-estimation, gae, continuous-control]
related: [papers/schulman_2017_ppo.md, concepts/intrinsic-motivation.md, concepts/model-predictive-control.md, concepts/value-based-rl.md, concepts/dexterous-manipulation.md, concepts/sim-to-real.md, concepts/step-size-adaptation.md, papers/liu_2025_locoformer.md, codebase/rl_lab/ppo-agent.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/PPO.pdf]
---

# Policy Gradient Methods

## What They Are

Policy gradient methods directly optimize the parameters $\theta$ of a stochastic policy $\pi_\theta(a \mid s)$ by gradient ascent on expected cumulative reward. Unlike value-based methods (Q-learning, DQN) that learn a value function and derive a policy implicitly, policy gradient methods differentiate through the policy directly and work naturally with continuous action spaces.

The fundamental gradient estimator (REINFORCE / Williams 1992):

$$\hat{g} = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\hat{A}_t\right]$$

where $\hat{A}_t$ is the **advantage function** — how much better action $a_t$ is compared to the average action in state $s_t$. This estimator is unbiased but has high variance; all practical methods reduce variance via a learned value-function baseline.

## The Core Problem: Large Policy Updates

A critical failure mode of policy gradient is **catastrophic update**: taking too large a gradient step collapses the policy, and unlike supervised learning, there is no recovery — the new policy collects bad data that perpetuates the collapse. Standard policy gradient computes one gradient step per batch; performing multiple steps on the same batch is not theoretically justified and often causes collapse.

Two major solutions:

### Trust Region Policy Optimization (TRPO, Schulman et al. 2015)

Maximizes a probability-ratio surrogate subject to a KL divergence constraint on the policy update:

$$\text{maximize}_\theta \quad \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t\right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}, \pi_\theta]] \leq \delta$$

Solved via conjugate gradient + line search (second-order). Provides a monotonic improvement guarantee. **Limitations:** second-order optimization is incompatible with parameter sharing between policy and value function, dropout, or auxiliary losses.

### Proximal Policy Optimization (PPO, Schulman et al. 2017)

PPO achieves TRPO's stability with first-order optimization. Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

**Clipped surrogate objective:**

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

This forms a **pessimistic lower bound** on the unclipped objective: ratio changes that move $r_t$ outside $[1-\epsilon, 1+\epsilon]$ in a direction that *improves* the objective are ignored; those that *hurt* the objective are included. Default: $\epsilon = 0.2$.

**Why PPO over TRPO?**
- First-order only → compatible with shared architectures, dropout, auxiliary tasks
- Multiple epochs of minibatch SGD on the same data batch
- Simpler: a few lines of code change from vanilla policy gradient
- Empirically competitive or better on both continuous control and Atari

## Advantage Estimation (GAE)

The advantage $\hat{A}_t = Q(s_t, a_t) - V(s_t)$ measures how much better action $a_t$ is than average. In practice, computed using **Generalized Advantage Estimation** (GAE, Schulman et al. 2015a):

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda$ interpolates between:
- $\lambda = 0$: 1-step TD advantage — low variance, high bias
- $\lambda = 1$: Monte Carlo return — high variance, zero bias

$\lambda = 0.95$ works well across continuous control and Atari. A learned value function $V_\phi(s)$ serves as the baseline.

## Actor-Critic Architecture

Most modern policy gradient algorithms are actor-critic:
- **Actor:** $\pi_\theta(a \mid s)$ — selects actions
- **Critic:** $V_\phi(s)$ — estimates state values for advantage computation

When actor and critic share parameters (standard for visual inputs), the combined objective is:

$$L(\theta) = \hat{\mathbb{E}}_t\!\left[L^{CLIP}_t - c_1(V_\theta(s_t) - V_t^\text{targ})^2 + c_2\,S\big[\pi_\theta(\cdot\,|\,s_t)\big]\right]$$

where $S$ is an entropy bonus preventing premature collapse to a deterministic policy.

**PPO Algorithm (Actor-Critic style):**
```
for iteration = 1, 2, ... do
    for actor = 1, ..., N do
        Run π_θ_old for T timesteps; compute advantages Â_1, ..., Â_T
    end for
    Optimize L w.r.t. θ for K epochs, minibatch size M ≤ NT
    θ_old ← θ
end for
```

## Comparison to Value-Based Methods

| Property | Policy Gradient (PPO) | Value-Based (DQN/DDQN) |
|----------|----------------------|------------------------|
| Action spaces | Continuous or discrete | Discrete only (standard) |
| Sample efficiency | Lower (on-policy) | Higher (off-policy replay) |
| Stability | Good with clipping | Can diverge (deadly triad) |
| Parallel environments | Natural fit | Less natural |
| Architecture sharing | Easy (PPO) | Not applicable |

PPO is the standard choice for continuous action spaces (robotics, locomotion). DQN-family methods remain strong on discrete tasks when sample efficiency is critical.

## Practical Recommendations (PPO)

- **Clipping:** ε = 0.2 is a robust default; ε ∈ [0.1, 0.3] is the useful range
- **Epochs:** K = 3–10 per data collection; more epochs = higher data efficiency but more off-policy drift
- **Advantage normalization:** zero mean, unit std per minibatch — critical for stability
- **Value function:** train with MSE loss; optionally clip value updates similarly to policy
- **Entropy bonus:** c₂ = 0.01 helps avoid premature convergence on discrete tasks
- **Parallel actors:** N ≥ 8; 32–128 for humanoid-scale or exploration-heavy tasks
- **GAE parameters:** γ = 0.99, λ = 0.95 work broadly across domains

## Relation to Curiosity-Driven RL

PPO is the canonical base RL algorithm for curiosity and intrinsic motivation methods (Burda et al. 2018, Pathak et al. 2017). Its on-policy nature makes it natural to add intrinsic reward alongside extrinsic reward within the same batch of rollouts. The clipping mechanism also interacts well with non-stationary intrinsic rewards: small policy steps prevent the dynamics model from changing too rapidly between updates. See [[intrinsic-motivation]].

## Relation to Model-Based RL

Policy gradient methods are *model-free* — they optimize the policy directly from environment interactions without learning an explicit world model. This contrasts with **Model Predictive Control (MPC)** approaches (used in JEPA-WM and V-JEPA 2-AC) that maintain a learned dynamics model and plan actions at inference time. Model-free (PPO) methods typically require more environment interactions but produce policies that are fast to execute; MPC methods are more sample-efficient but slow at inference due to online planning. See [[model-predictive-control]].

## See Also

- [Schulman et al. (2017) — PPO](../papers/schulman_2017_ppo.md) — original PPO paper with systematic surrogate objective comparison and full experimental results
- [Intrinsic Motivation & Curiosity-Driven RL](intrinsic-motivation.md) — uses PPO as the base RL algorithm; intrinsic rewards augment PPO's objective
- [Value-Based Reinforcement Learning](value-based-rl.md) — contrasting family: Q-learning, DQN, Double DQN; off-policy, discrete actions, higher sample efficiency
- [Dexterous Manipulation](dexterous-manipulation.md) — application domain where PPO + eigengrasp action space enables sim2real grasping of tools
- [Sim-to-Real Transfer](sim-to-real.md) — PPO with parallel actors is the standard training algorithm for sim2real robot learning
- [Model Predictive Control for Robot Learning](model-predictive-control.md) — planning-based alternative to policy gradient; contrast in sample efficiency vs. inference cost
- [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md) — meta-gradient alternative to Adam-style normalizers for setting step sizes; contrast with PPO's fixed schedule plus clipping
- [Liu et al. (2025) — LocoFormer](../papers/liu_2025_locoformer.md) — PPO at scale with a Transformer-XL actor-critic over ~100k procedurally generated robots
- [rl_lab — PPO Agent](../codebase/rl_lab/ppo-agent.md) — a from-scratch PPO in the personal repo. Why a hard log-std floor beats an entropy bonus for a state-independent σ, why success must keep paying (terminations ignored), and γ 0.8 → 0.95 doubling pick-and-place success
