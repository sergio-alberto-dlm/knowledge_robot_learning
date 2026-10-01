---
title: Step-Size Adaptation & Meta-Learning of Learning Rates
type: concept
tags: [meta-learning, step-size-adaptation, idbd, lms, delta-rule, online-learning, optimization, feature-relevance, gradient-descent]
related: [papers/sutton_1992_idbd.md, papers/javed_2024_swifttd.md, concepts/td-lambda-and-eligibility-traces.md, concepts/continual-learning-and-tracking.md, concepts/value-based-rl.md, concepts/policy-gradient-methods.md, papers/nabarro_2024_gbp_learning.md, codebase/rl_lab/research-roadmap.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-24
updated: 2026-09-29
sources: [raw/papers/pdf/sutton-92a.pdf, raw/papers/pdf/SwiftTD-RLC.pdf]
---

# Step-Size Adaptation & Meta-Learning of Learning Rates

## What It Is

Step-size adaptation treats the learning rate not as a hyperparameter to be tuned by the user, but as a **parameter to be learned online from the learning process itself**. The framing that motivates it (Sutton 1992): *bias* — what a learner brings to a problem before seeing data — is the key to fast learning and generalization, and the only way for a learner to acquire good bias automatically is to generate it from **previous learning experience** on a series of related problems.

In a linear learner, one important and concrete form of bias is the **per-input learning rate**. Learning about irrelevant inputs acts as noise that interferes with learning about relevant ones; with equal-variance inputs, learning time scales with the *sum of squares of the learning rates*. Learning rates are therefore a resource to distribute: small for inputs likely to be irrelevant, large for inputs likely to be relevant. Adapting them online is a form of **online feature-relevance learning**.

## The Base Learner: LMS / Delta Rule

A single linear unit with output $y(t) = \sum_i w_i(t)x_i(t)$ and error $\delta(t) = y^*(t) - y(t)$. LMS (also known as the delta rule, ADALINE, Widrow-Hoff, and — in psychology — the Rescorla-Wagner rule) is gradient descent on the sample squared error:

$$w_i(t+1) = w_i(t) + \alpha\,\delta(t)x_i(t)$$

Per-input step sizes generalize this to $w_i(t+1) = w_i(t) + \alpha_i(t+1)\delta(t)x_i(t)$, and the question becomes how to set the $n$ rates $\alpha_i$.

## Two-Level Structure: Meta-Learning

Algorithms in this family are **meta-learning** algorithms in a specific, narrow sense: a base level learns the weights, and a meta level learns the base level's step sizes. This is *not* the "learn across a task distribution with an outer loop" sense of MAML-style meta-learning — everything happens in one online stream, on one learner, with no episode boundaries.

### Delta-Bar-Delta (Jacobs 1988; lineage: Kesten 1958; Barto & Sutton 1981; Sutton 1982, 1986)

Core heuristic: if the **current weight change correlates positively with recent weight changes**, the past steps were too small — increase the step size. If it correlates negatively, the learner is overshooting and re-correcting — decrease it. DBD applies this batch-by-batch (after a full presentation of a training set) and has three free parameters. It was designed for nonlinear networks.

### IDBD (Sutton 1992)

IDBD makes the same idea **fully incremental** (one example at a time, examples discarded afterwards) and reduces it to **one** free parameter. Two mechanisms do this:

**1. Exponential parameterization.** $\alpha_i = e^{\beta_i}$, and the meta level updates $\beta_i$. This keeps $\alpha_i$ positive automatically and makes a fixed step in $\beta_i$ a fixed *fraction* change in $\alpha_i$ (e.g. ±10%). That matters because the needed rates span orders of magnitude across inputs — no single additive step size in $\alpha$ works for all of them.

**2. A presence-gated trace.** The decay of the recent-weight-change trace $h_i$ is gated by $x_i^2$, so the trace fades only to the extent that input $i$ is actually present, and the decay rate is *tied to the current learning rate* rather than being a separate hyperparameter.

$$\beta_i(t+1) = \beta_i(t) + \theta\,\delta(t)x_i(t)h_i(t)$$
$$\alpha_i(t+1) = e^{\beta_i(t+1)}$$
$$w_i(t+1) = w_i(t) + \alpha_i(t+1)\delta(t)x_i(t)$$
$$h_i(t+1) = h_i(t)\big[1 - \alpha_i(t+1)x_i^2(t)\big]^+ + \alpha_i(t+1)\delta(t)x_i(t)$$

$\theta$ is the meta-learning rate; $[x]^+ = \max(x,0)$; $h_i$ is initialized to 0. Cost: three memories per input, ~3× the memory and compute of plain LMS, still **linear** in the number of inputs.

## Why It Is Gradient Descent (Meta-Gradient)

IDBD is *derived*, not heuristic: it is stochastic gradient descent on the sample error in $\beta$-space,

$$\beta_i(t+1) = \beta_i(t) - \tfrac{1}{2}\theta\,\frac{\partial\delta^2(t)}{\partial\beta_i}$$

where $\partial/\partial\beta_i$ means the derivative with respect to an infinitesimal change in $\beta_i$ *at all time steps* (the same device as in gradient analyses of recurrent networks). The trace has an exact interpretation:

$$h_i(t) \equiv \frac{\partial w_i(t)}{\partial \beta_i}$$

i.e. **how much the current weight depends on its own step-size parameter**. The derivation needs one approximation — $\partial w_j/\partial\beta_i \approx 0$ for $i \ne j$, since the primary effect of changing the $i$-th learning rate is on the $i$-th weight — and then the $h_i$ update falls out of the product rule.

Consequences: the algorithm is stable at local optima of the error with respect to $\beta$, and reduces expected error elsewhere. These are necessary but **not sufficient** conditions for a good algorithm — step size at the meta level still matters, and convergence to a *global* optimum in step-size space is not claimed.

### A family, not a single algorithm

Multiplying the $\beta_i$ increment by $\alpha_i^p$ for any $p$ yields other algorithms that are still **descent** algorithms (positive inner product with the gradient) though no longer *steepest*-descent. Sutton notes experiments in progress suggesting some members of this family find optimal learning rates more efficiently than IDBD. The general lesson: both the magnitude *and* the direction of a meta-step can be modified without losing the descent property.

## Interpretation: Incremental Hold-One-Out Cross Validation

Leave-one-out CV holds out one example, trains on the rest, measures generalization to the held-out one, and repeats. IDBD is the online analogue: the newest example *is* the held-out example, and the meta-update steps $\beta$ to improve generalization to it. This makes such rules categorically different from ordinary learning rules — **performance on the new example is optimized without using the new example for the base update**. It is also why the proper evaluation of a step-size adapter is its tracking error on a stream, not its fit to a fixed training set.

## What the Empirical Result Looks Like

On a drifting linear tracking task (20 Gaussian inputs, 5 relevant with signs flipping every 20 examples, 15 irrelevant), IDBD roughly **halves the asymptotic squared error** of the best fixed-$\alpha$ LMS (≈1.5 vs ≈3.5, i.e. ~60% error reduction), and does so over a broad range of $\theta$. Given long enough, it drives the irrelevant inputs' rates towards zero and converges the relevant inputs' rates to the value an exhaustive fixed-$\alpha$ sweep identifies as optimal (≈0.13 on that task). See [Sutton (1992)](../papers/sutton_1992_idbd.md) for the full protocol and numbers.

## Relation to Modern Adaptive Optimizers

Degris et al. (2024) name this distinction directly: IDBD does **step-size optimization**, while RMSProp does **step-size normalization** — articulating the difference with a simple problem on which IDBD adapts its step sizes to optimize the loss while RMSProp ignores the loss landscape entirely when adapting them.

Adam, RMSProp, AdaGrad and friends are also per-parameter step-size schemes, but they are **normalizers**, not meta-learners: they rescale by running statistics of gradient magnitude (second moment), which does not distinguish an input that is irrelevant from one that is relevant but currently quiescent. IDBD's signal is different in kind — it is the *correlation between successive weight changes*, i.e. a directional, sign-sensitive statistic derived by differentiating the error with respect to the step size itself.

Practical corollaries of that difference:

- **Normalizers cannot drive a step size to zero for being useless**; a meta-gradient method can, and that is exactly what makes it a relevance detector.
- **Meta-gradient methods pay off under non-stationarity**, where "the right step size" is not a fixed quantity to be annealed away but a standing property of each feature. On a stationary train-once problem their effect is second-order.
- **The hyperparameter count does not go to zero** — $\theta$ remains — but it goes from $n$ to 1.

### The lineage, as documented by SwiftTD

Javed, Sharifnassab & Sutton (2024) audit the attempts to carry IDBD to temporal-difference learning, and the record is
unusually messy for a 30-year-old idea:

| Work | Status per SwiftTD |
|---|---|
| Mahmood et al. (2012) | Proposed a bound on the **correction ratio** for linear regression — the ancestor of SwiftTD's overshoot bound |
| Thill (2015) | Incorrect: mistake in deriving the meta-gradient update rule |
| Kearney et al. (2018) — TIDBD | Meta-gradient correct, but uses the **TD(0) objective** while learning with TD(λ); can fail to raise the right features' step sizes |
| Young et al. (2019) | First correct extension of IDBD to TD(λ) |
| Javed et al. (2024) — SwiftTD | First extension to **True Online TD(λ)**; adds the overshoot bound and step-size decay |

Two general lessons from that table. First, a meta-gradient must be taken with respect to **the objective the base learner is
actually using** — mismatching TD(0) against TD(λ) silently corrupts the credit signal. Second, the meta-gradient alone is
not enough for stability: SwiftTD needed a non-differentiable safety bound *plus* a non-gradient decay fallback, because the
step sizes that meta-learning finds can themselves grow large enough to diverge, and the clip that saves you is not something
the meta-gradient can see.

SwiftTD also demonstrates what the meta-learned step sizes are *for* beyond speed: the per-feature lifetime credit
$\text{Credit}^T_i = \sum_t e^{\beta_t[i]}\phi_t[i]^2$ visualizes as a meaningful saliency map over Atari pixels (the ball's
trajectory in Pong; dots and enemies in MsPacman). Learned step sizes *are* a readable relevance signal, which is what
[IDBD](../papers/sutton_1992_idbd.md) claimed in 1992 and what a normalizer cannot give you.

## Practical Notes

- **Initialization:** $h_i = 0$; $\beta_i$ set so the initial $\alpha_i$ is a reasonable global rate (0.05 in Sutton's experiments). The initialization does not affect *asymptotic* tracking performance, only the transient.
- **Numerical guards:** floor each $\beta_i$ (e.g. at $-10$) to avoid underflow, and optionally clip $|\Delta\beta_i|$ per step (e.g. to 2). Neither was needed for Sutton's reported results.
- **Instability:** as with plain LMS, too large a step-size parameter diverges; $\theta$ has a usable range, just a broad one.
- **Where it applies as-is:** linear learners over a fixed feature set with a continuing input stream. Nonlinear extension was left as future work in the 1992 paper.

## See Also

- [Sutton (1992) — IDBD](../papers/sutton_1992_idbd.md) — the source paper: algorithm, gradient derivation, tracking experiments
- [Javed et al. (2024) — SwiftTD](../papers/javed_2024_swifttd.md) — IDBD's step-size optimization carried to True Online TD(λ), plus the overshoot bound and step-size decay
- [TD(λ) & Eligibility Traces](td-lambda-and-eligibility-traces.md) — the base learner SwiftTD adapts step sizes for; the correction ratio as a stability device
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md) — the problem setting where step-size adaptation earns its keep
- [Value-Based Reinforcement Learning](value-based-rl.md) — TD learning, where per-feature step sizes and stability interact with bootstrapping
- [Policy Gradient Methods](policy-gradient-methods.md) — modern deep-RL practice: Adam plus trust-region/clipping heuristics instead of explicit meta-gradients
- [Nabarro et al. (2024) — Learning in Deep Factor Graphs with GBP](../papers/nabarro_2024_gbp_learning.md): Bayesian filtering, where per-weight update size comes from posterior precision (a Kalman gain) rather than a meta-learned step size
- [rl_lab — Research Roadmap](../codebase/rl_lab/research-roadmap.md) — planned test of per-feature SwiftTD step sizes as a relevance detector over 3,560 frozen DINOv2 + depth features, with credit heatmaps
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md) — step-size credit as object × relation feature selection; soft vs hard selection
