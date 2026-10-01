---
title: Continual Learning & Tracking in Non-Stationary Tasks
type: concept
tags: [continual-learning, non-stationarity, tracking, online-learning, meta-learning, evaluation-methodology, bayesian-filtering]
related: [papers/sutton_1992_idbd.md, papers/javed_2024_swifttd.md, papers/nabarro_2024_gbp_learning.md, concepts/gaussian-belief-propagation.md, concepts/step-size-adaptation.md, concepts/td-lambda-and-eligibility-traces.md, concepts/value-based-rl.md, concepts/in-context-adaptation.md, concepts/rl-evaluation-methodology.md]
created: 2026-09-24
updated: 2026-09-29
sources: [raw/papers/pdf/sutton-92a.pdf, raw/papers/pdf/SwiftTD-RLC.pdf, raw/papers/pdf/GBP_learning.pdf]
---

# Continual Learning & Tracking in Non-Stationary Tasks

> **Status:** partial — seeded from Sutton (1992), extended with the lifetime-error protocol from SwiftTD (Javed et al. 2024).

## What It Is

A **tracking task** is a supervised- or concept-learning task whose target function *drifts over time* and must be continually re-estimated, rather than learned once and finished. The learner sees an unending stream of examples, processes each one and discards it, and is measured on its ongoing error rather than on final convergence.

Contrast the two evaluation regimes:

| | Conventional (train-once) task | Tracking / continual task |
|---|---|---|
| Target function | Fixed | Drifts |
| Data | Fixed training set, revisited | One-pass stream, examples discarded |
| Success measure | Final test error after convergence | Asymptotic error *while tracking* |
| Right thing to do with step size | Anneal it towards zero | Keep it at a useful standing value per feature |

## Why It Matters for Bias Learning

This setting is not an exotic special case — it is the setting in which *learning to learn* is measurable at all. Sutton's (1992) argument: a learner can only acquire appropriate bias automatically by generating it from previous learning experience, and that is possible **only if the learner encounters a series of different problems requiring the same or similar biases**. A single train-once task offers no such series, so meta-learning effects there are small and second-order. On a continuing sequence of related problems they can be first-order — on his drifting tracking task, learning the biases cut squared error by roughly 60%.

Sutton's broader methodological claim: single learning tasks have been extremely useful but are *limited* as vehicles for studying representation change and the identification of relevant vs. irrelevant features, because those issues barely register when there is only one problem to solve. **Cross-task learning may be key to human-level learning ability.**

## A Canonical Testbed

The synthetic drifting-relevance task from Sutton (1992), a useful minimal benchmark for any online step-size or relevance method:

- $n$ inputs drawn i.i.d. $\mathcal{N}(0,1)$ (Sutton: $n = 20$).
- Target is a sparse sign-weighted sum of a fixed relevant subset (Sutton: the first 5 inputs, weights $\pm1$); the remaining inputs have weight exactly 0.
- Every $k$ examples (Sutton: $k = 20$), one relevant weight flips sign.
- So the *relevance structure is stationary* while the *mapping is non-stationary* — which cleanly separates "which features matter" (learnable bias) from "what the function currently is" (must be tracked).
- Measurement: run past initial transients (20,000 examples), then average squared error over a long window (10,000 examples). Since it is a tracking problem, one long run suffices — there is no separate test set.

Variants worth building when this article is expanded: drift in *which* features are relevant (not just their signs), gradual drift instead of abrupt flips, correlated or unequal-variance inputs, and abrupt distribution shift in the inputs themselves.

## Lifetime Error: the Online Evaluation Metric

Javed et al. (2024) make the evaluation argument concrete with **lifetime error**:

$$\text{Lifetime error}(T) = \frac{1}{T}\sum_{t=1}^{T}\left(v_t - \sum_{j=t+1}^{T}\gamma^{\,j-t-1}r_j\right)^{2}$$

Each prediction $v_t$ is scored against the actual discounted future of the stream, and the errors are averaged over the
agent's whole lifetime $T$. Two properties matter:

- It measures **how quickly the solution was found**, not only its final quality — a learner that converges to the same
  place more slowly gets a worse score, which is exactly the property a train/test split throws away.
- It needs **no held-out set**. Splitting data is a device for offline learning where the learner sees the whole dataset;
  online, every prediction is made *before* its ground truth arrives, so the stream is already an honest test. This also
  removes the awkwardness of defining a "test set" for a target that drifts.

$T$ and $\gamma$ are part of the problem definition, not tuning knobs. In the Atari Prediction Benchmark, $\gamma = 0.98$
and $T = 210{,}000$ steps (~2 hours of gameplay at 30 fps) — long enough that transient early error is amortized but short
enough that learning speed dominates the score.

A corollary for method design: under lifetime error, **replay is not free**. Doing multiple updates per data point buys
sample efficiency but costs computation, makes the agent less reactive (feedback is not reflected in behavior
immediately), and degrades in big worlds — which is the gap SwiftTD targets with single-update online learning.

## Relation to Neighbouring Ideas

- **Step-size adaptation** is the most direct response to non-stationarity for a linear learner: see [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md).
- **Catastrophic forgetting / plasticity loss** in deep continual learning is the same pressure from the other side — networks trained long on a stream lose the ability to adapt. Step-size adaptation addresses the plasticity side of that tension explicitly.
- **In-context adaptation** handles non-stationarity *without weight updates*. LocoFormer (Liu et al. 2025) conditions on seconds of history and re-identifies its body when a joint locks or its wheels stop working mid-run. It is a complementary, activation-space answer to tracking. See [In-Context Adaptation](in-context-adaptation.md).
- **RL is intrinsically non-stationary** even with a fixed environment: as the policy improves, the value-function regression target moves. This is one reason tracking-oriented methods keep reappearing in RL.

- **Bayesian filtering over parameters** is a replay-free alternative to gradient-based continual learning. The posterior after task $t-1$ becomes the prior for task $t$, so data can be discarded after one pass. Nabarro et al. (2024) do this with GBP on deep factor graphs. Single-epoch MNIST reaches 98.16%, matching a CNN + Adam with a 6,000-example replay buffer. Continual learning across video frames also beats per-frame learning on denoising. The weakness mirrors plasticity loss: posterior precision keeps growing, so updates shrink. When a shallow model overfit to noise, the authors relaxed each new prior back toward the original one ($\mu,\sigma \leftarrow \alpha\,\text{prior} + (1-\alpha)\,\text{posterior}$, with $\alpha=0.5$). That is a forgetting/process-noise device, in the same role as a floor on the step size.

> **Verify:** the plasticity-loss framing above is not argued in either compiled source (Sutton 1992; Javed et al. 2024) — it still needs a dedicated continual-learning paper before being cited. The RL non-stationarity point, by contrast, is standard and is implicit in SwiftTD's motivation.

## See Also

- [Sutton (1992) — IDBD](../papers/sutton_1992_idbd.md) — introduces the drifting tracking testbed and the argument for non-stationary evaluation
- [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md) — the algorithmic response to this setting
- [Javed et al. (2024) — SwiftTD](../papers/javed_2024_swifttd.md) — lifetime error on the Atari Prediction Benchmark; the case for single-update, replay-free online learning
- [TD(λ) & Eligibility Traces](td-lambda-and-eligibility-traces.md) — the online prediction machinery evaluated this way
- [Value-Based Reinforcement Learning](value-based-rl.md) — bootstrapping against a moving target as an intrinsic source of non-stationarity
- [Nabarro et al. (2024) — Learning in Deep Factor Graphs with GBP](../papers/nabarro_2024_gbp_learning.md) — continual learning as Bayesian filtering over parameters, replay-free and single-epoch
- [Gaussian Belief Propagation & Factor Graphs](gaussian-belief-propagation.md)
- [In-Context Adaptation & Cross-Embodiment Policies](in-context-adaptation.md) — adapting through history conditioning instead of weight updates
- [RL Evaluation Methodology](rl-evaluation-methodology.md) — the episodic train-then-evaluate counterpart to lifetime error
