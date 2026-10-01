---
title: Value-Based Reinforcement Learning
type: concept
tags: [reinforcement-learning, q-learning, dqn, double-dqn, temporal-difference, value-function, experience-replay, overestimation]
related: [papers/van_hasselt_2016_ddqn.md, concepts/policy-gradient-methods.md, concepts/intrinsic-motivation.md, concepts/step-size-adaptation.md, concepts/continual-learning-and-tracking.md, concepts/td-lambda-and-eligibility-traces.md]
created: 2026-05-17
updated: 2026-09-24
sources: [raw/papers/pdf/DDQN.pdf]
---

# Value-Based Reinforcement Learning

## What It Is

Value-based RL learns an estimate of the **optimal action-value function** $Q_*(s,a)$ — the expected discounted return when taking action $a$ in state $s$ and following the optimal policy thereafter:

$$Q_*(s,a) = \mathbb{E}\!\left[R_1 + \gamma R_2 + \gamma^2 R_3 + \ldots \mid S_0=s, A_0=a, \pi^*\right]$$

An optimal policy is immediately derivable: $\pi^*(s) = \text{argmax}_a Q_*(s,a)$. Value-based methods are primarily used for **discrete action spaces** and are sample-efficient because they learn off-policy (reusing experience from a replay buffer) — in contrast to policy gradient methods like PPO that discard data after each update.

## Q-learning

Q-learning (Watkins 1989) is the foundational value-based algorithm. Starting from the Bellman optimality equation:

$$Q_*(s,a) = \mathbb{E}\!\left[R + \gamma \max_{a'} Q_*(S', a') \mid S=s, A=a\right]$$

Q-learning updates towards this target in a stochastic gradient step:

$$\theta \leftarrow \theta + \alpha\!\left(\underbrace{R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta)}_{Y_t^Q \text{ (target)}} - Q(S_t, A_t; \theta)\right)\nabla_\theta Q(S_t, A_t; \theta)$$

The max operator makes this an **off-policy** method: the update target assumes optimal future behavior regardless of the behavior policy used to collect data.

## Deep Q-Networks (DQN)

DQN (Mnih et al. 2015) scales Q-learning to high-dimensional visual observations with two key stabilization techniques:

### Experience Replay
Transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ are stored in a circular replay buffer of size $\sim$1M and sampled uniformly at random for each update. This breaks temporal correlations in the data (preventing oscillation) and allows each transition to be reused many times.

### Target Network
A separate parameter vector $\theta^-$ (a periodically frozen copy of the online network $\theta$, updated every $\tau \approx$ 10,000 steps) provides stable regression targets:

$$Y_t^{DQN} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-)$$

Without the target network, the regression target changes with every gradient step, creating a moving target problem that causes divergence.

**Architecture (Atari):** 84×84×4 grayscale frames → 3 conv layers → 512-unit FC → one Q-value output per action. Optimized with RMSProp.

## The Overestimation Problem

Q-learning's max operator uses the **same values for both selecting the best action and evaluating its value**. This induces an upward bias:

**Theorem (van Hasselt et al. 2016):** If $Q_t(s,a)$ are unbiased on average but imperfect (mean squared error $C > 0$ across $m$ actions), then:

$$\max_a Q_t(s,a) \geq V_*(s) + \sqrt{\frac{C}{m-1}}$$

This lower bound on overestimation:
- Grows with estimation error $C$ (e.g., early in training)
- Decreases with the number of actions $m$ (but not to zero for finite $m$)
- Is zero for Double Q-learning under the same conditions

The overestimation propagates via bootstrapping: overestimated values in one state raise estimates in predecessor states, leading to a cascade of inflated values that can destabilize training (visible as sudden score collapses in Atari games like Asterix and Wizard of Wor).

## Double Q-learning and Double DQN

**Double Q-learning** (van Hasselt 2010) decomposes the max operation:
- Use one estimator $\theta$ to **select** the greedy action: $a^* = \text{argmax}_a Q(S', a; \theta)$
- Use a separate estimator $\theta'$ to **evaluate** that action: $Q(S', a^*; \theta')$

This eliminates the selection bias because the action selected by $\theta$ is unlikely to also be the maximum under $\theta'$ unless it is truly optimal.

**Double DQN** (van Hasselt et al. 2016) adapts this to DQN with a single line change: the target network $\theta^-$ is used for evaluation, but the online network $\theta$ is used for selection:

$$Y_t^{DoubleDQN} \equiv R_{t+1} + \gamma\, Q\!\left(S_{t+1},\, \underbrace{\text{argmax}_a Q(S_{t+1}, a; \theta_t)}_{\text{online network selects}};\, \underbrace{\theta_t^-}_{\text{target network evaluates}}\right)$$

This is the *only* difference from DQN. No additional networks, no additional memory, negligible computational overhead.

**Results on Atari (no-op evaluation):**

| Method | Median normalized | Mean normalized |
|--------|------------------|-----------------|
| DQN | 93.5% | 241.1% |
| Double DQN | 114.7% | 330.3% |
| Double DQN (tuned) | 116.7%* | 475.2%* |

*Human-start evaluation (harder generalization test).

## The Deadly Triad

Value-based RL with function approximation faces the **deadly triad** (Sutton & Barto 2018): the combination of (1) function approximation, (2) bootstrapping (TD targets), and (3) off-policy data can cause divergence, even with linear approximators. DQN and DDQN mitigate this with target networks and replay buffers but have no formal convergence guarantee.

## Comparison to Policy Gradient Methods

| Property | Value-Based (DQN/DDQN) | Policy Gradient (PPO) |
|----------|------------------------|----------------------|
| Action spaces | Discrete (standard) | Continuous or discrete |
| Data reuse | High (off-policy replay) | Low (on-policy only) |
| Stability | Can diverge (deadly triad) | Good with clipping |
| Parallelism | Less natural | Natural (N actors) |
| Continuous control | Not standard | Standard |
| Atari asymptotic perf. | Strong | Competitive |

DDQN and other value-based extensions (Dueling networks, Prioritized Replay, Distributional RL, n-step) are typically preferred when sample efficiency on discrete tasks is critical. PPO is preferred for continuous action spaces and robot learning.

## Practical Recommendations

- **Always use Double DQN over vanilla DQN:** trivial change, consistent improvement, no downside
- **Target network update frequency:** τ = 10,000 steps (DQN default); increase to 30,000 for further overestimation reduction (tuned DDQN)
- **Replay buffer size:** 1M transitions provides a good diversity-memory tradeoff; scale up with more compute
- **Exploration:** ε-greedy annealing (1.0 → 0.1 over 1M steps during training); use ε = 0.001–0.05 for evaluation
- **Extensions (Rainbow order of importance):** Double DQN > Prioritized Replay > Dueling Networks > n-step returns > Distributional RL > Noisy nets

## See Also

- [van Hasselt et al. (2016) — Double DQN](../papers/van_hasselt_2016_ddqn.md) — original Double DQN paper; overestimation theory and Atari results
- [Policy Gradient Methods](policy-gradient-methods.md) — contrasting family of RL algorithms; on-policy, supports continuous actions
- [Intrinsic Motivation & Curiosity-Driven RL](intrinsic-motivation.md) — exploration methods that add intrinsic rewards to Q-learning or policy gradient objectives
- [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md) — learning the step size $\alpha$ online instead of tuning it; IDBD (Sutton 1992) and the per-feature relevance signal it produces
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md) — bootstrapping against a moving target makes TD learning intrinsically non-stationary
- [TD(λ) & Eligibility Traces](td-lambda-and-eligibility-traces.md) — the multi-step ($\lambda > 0$) online branch of TD learning; λ-returns, True Online TD(λ), and the correction ratio
