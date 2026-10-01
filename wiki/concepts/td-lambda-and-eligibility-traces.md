---
title: TD(λ) & Eligibility Traces
type: concept
tags: [temporal-difference, td-lambda, eligibility-traces, lambda-return, online-learning, prediction, credit-assignment]
related: [papers/javed_2024_swifttd.md, concepts/value-based-rl.md, concepts/step-size-adaptation.md, concepts/continual-learning-and-tracking.md, codebase/rl_lab/maniskill-training-pipeline.md]
created: 2026-09-24
updated: 2026-09-29
sources: [raw/papers/pdf/SwiftTD-RLC.pdf]
---

# TD(λ) & Eligibility Traces

## The Problem: Delayed Feedback

Many useful predictions — *will it rain in two hours?*, *what score will this episode reach?* — cannot be checked until the outcome arrives. The naive approach stores experience and waits, which scales poorly in memory and delays all learning. **Temporal Difference learning** (Sutton 1988) instead updates a prediction towards a *bootstrapped* target built from the next prediction, so learning happens online, in constant memory per step, without waiting for the outcome.

Because predictions can be learned from experience alone — no ground-truth labels needed — predictive knowledge is a scalable way to encode knowledge about a world. TD is also the building block inside Sarsa(λ), Q-learning, actor-critic, and PPO's advantage estimation, so improvements to TD propagate broadly.

## The λ-Return

The λ-return interpolates over *all* $n$-step returns with geometrically decaying weights:

$$G^\lambda_t \;\overset{\text{def}}{=}\; (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}, \qquad G_{t:t+n} \overset{\text{def}}{=} r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1}r_{t+n} + \gamma^{n}v_{t+n}$$

$\lambda = 0$ recovers the one-step TD target; $\lambda = 1$ approaches Monte Carlo. Intermediate $\lambda$ trades bias against variance, and in practice is often the best-performing setting.

What makes λ-returns special is not the interpolation itself but that **eligibility traces give a computationally efficient way to learn from them** — $O(n)$ work per step with no storage of past examples.

## Forward View, Backward View, and the Trace

- **Forward view:** conceptually, each state's update looks *ahead* to its λ-return. Clean to define, not directly implementable online.
- **Backward view:** equivalently, maintain an **eligibility vector** $z$ that accumulates a decaying record of recently active features, and apply each TD error to all of them at once. Traces decay by $\gamma\lambda$ per step and are incremented by the current features.

The eligibility vector is the mechanism of **temporal credit assignment**: it answers "which features were responsible for what I'm now surprised about?" A feature active three steps ago still holds trace, so a surprise now still reaches it.

## Why True Online TD(λ)

Plain TD(λ)'s weight updates are *not* the same as learning directly from λ-returns — they are only a good approximation **when the step size is small**. This is precisely the wrong failure mode for fast online learning, where large step sizes are the whole point.

**True Online TD(λ)** (Van Seijen et al. 2016) closes the gap: it is *exactly* equivalent to the Online λ-return algorithm (which uses no traces at all), and it outperforms TD(λ) when learning with large step sizes.

That exactness turns out to have consequences beyond accuracy. In SwiftTD (Javed et al. 2024) it is **load-bearing**: because True Online TD(λ) behaves exactly like a learner with undelayed targets, a bound on how much a single update may correct the prediction can be derived and applied — an implementation whose safety guarantee simply fails for approximate TD(λ). Empirically, TD(λ) with the same bound still diverged on several Atari prediction problems while True Online TD(λ) with it never diverged on any. The lesson generalizes: **exact equivalences are worth their complexity when you want to build further guarantees on top of them.**

## Step Sizes and the Correction Ratio

Two quantities recur when TD is pushed towards fast online learning:

**Per-feature step sizes.** A single scalar step size forces a bad trade: large enough to learn quickly *is* large enough to let noisy or irrelevant features wreck the weights. Per-feature step sizes dissolve the conflict, at the cost of needing to be *learned* rather than tuned (millions of them). See [Step-Size Adaptation](step-size-adaptation.md).

**The correction ratio.** For a linear learner with per-component step sizes, the fraction of the current prediction error removed by an update has a closed form computable *before* updating:

$$\tau_t = \sum_i \alpha_t[i]\,\phi_t[i]^2$$

$\tau = 1$ lands the prediction exactly on the target; $\tau = 0.5$ moves halfway; $\tau > 1$ **overshoots**, and repeated overshooting is what divergence looks like from close up. Bounding $\tau$ is therefore a natural stability device. For TD this has to be applied when **incrementing the eligibility vector**, not at weight-update time, and requires the semi-gradient assumption (treating the bootstrap target as fixed) — the target otherwise depends on the very weights being updated.

## Online Evaluation

TD's natural evaluation is **lifetime error** over the stream (see [Continual Learning & Tracking](continual-learning-and-tracking.md)), not final error on a held-out set: every prediction is made before its ground truth exists, so the stream already provides an honest test, and the metric rewards learning *fast* as well as learning *well*.

## See Also

- [Javed et al. (2024) — SwiftTD](../papers/javed_2024_swifttd.md) — True Online TD(λ) plus step-size optimization, the overshoot bound, and step-size decay
- [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md) — how per-feature step sizes get learned instead of tuned
- [Value-Based Reinforcement Learning](value-based-rl.md) — Q-learning/DQN, the one-step ($\lambda = 0$) replay-based branch of the same family
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md) — lifetime error and the online evaluation protocol
- [rl_lab — ManiSkill Training Pipeline](../codebase/rl_lab/maniskill-training-pipeline.md) — trace half-life ln0.5/ln(γλ) used to choose γ for a planned SwiftTD critic (2.1 steps at γ = 0.8 vs 4.4 at 0.95)
