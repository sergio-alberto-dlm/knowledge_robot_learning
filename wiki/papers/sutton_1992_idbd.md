---
title: "Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta"
type: paper
tags: [meta-learning, step-size-adaptation, idbd, lms, delta-rule, online-learning, non-stationarity, tracking, gradient-descent, feature-relevance]
related: [concepts/step-size-adaptation.md, concepts/continual-learning-and-tracking.md, concepts/td-lambda-and-eligibility-traces.md, concepts/value-based-rl.md, papers/javed_2024_swifttd.md]
created: 2026-09-24
updated: 2026-09-24
sources: [raw/papers/pdf/sutton-92a.pdf]
---

# Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta

**Author:** Richard S. Sutton
**Affiliation:** GTE Laboratories Incorporated, Waltham, MA
**Published:** Proceedings of the Tenth National Conference on Artificial Intelligence (AAAI-92), pp. 171–176, MIT Press, 1992

---

## Abstract

Appropriate bias is widely viewed as the key to efficient learning and generalization. The paper presents a new algorithm, **Incremental Delta-Bar-Delta (IDBD)**, for learning appropriate biases based on previous learning experience. IDBD is developed for a simple linear learning system — the LMS or delta rule with a *separate learning-rate parameter for each input* — and adjusts those learning-rate parameters, which are an important form of bias for this system. Because bias is adapted from previous learning experience, the appropriate testbeds are **drifting or non-stationary** learning tasks. On such tasks IDBD performs better than ordinary LMS and in fact finds the optimal learning rates. IDBD extends and improves prior work by Jacobs and by Sutton in that it is **fully incremental** and has **only a single free parameter**. The paper also derives IDBD as gradient descent in the space of learning-rate parameters, and offers a novel interpretation of IDBD as an incremental form of hold-one-out cross validation.

---

## Key Contributions

- **The IDBD algorithm:** a meta-learning rule that adapts one learning rate per input of an LMS learner, online, example-by-example, with no stored training set.
- **Exponential learning-rate parameterization:** $\alpha_i = e^{\beta_i}$, so that meta-updates take *geometric* steps in $\alpha_i$ and $\alpha_i$ is guaranteed positive. This is what lets some learning rates shrink towards zero while others stay large under a single meta step-size.
- **Fully incremental where Delta-Bar-Delta was batch:** Jacobs's (1988) DBD updates after a full presentation of a training set and has **three** free parameters; IDBD works on one-by-one examples and has **one** ($\theta$). The trick is defining the trace $h_i$ so that it decays *only to the extent that input $x_i$ is present* (via the $x_i^2$ factor), and tying the decay rate to the current learning rate instead of making it a separate hyperparameter.
- **Derivation as stochastic gradient descent in $\beta$-space** (meta-gradient descent), replacing earlier heuristic justifications of DBD-style rules; this also gives a recipe for deriving bias-learning rules for *other* base learners.
- **Empirical demonstration that IDBD finds near-optimal learning rates**, not merely better-than-any-single-rate ones: it drives the 15 irrelevant inputs' rates towards 0 and converges to $\alpha \approx 0.13$ on the 5 relevant inputs, matching an exhaustive fixed-$\alpha$ sweep.
- **A family of algorithms:** multiplying the $\beta_i$ increment by any $\alpha_i^p$ yields other descent (though no longer *steepest*-descent) algorithms; Sutton notes experiments in progress suggesting some members find optimal learning rates more efficiently than IDBD itself.
- **Reinterpretation as incremental hold-one-out cross validation:** each new example is implicitly the held-out example, and the meta-update optimizes generalization *to that example without using it for the base update* — a fundamental difference from ordinary learning rules.
- **Framing argument:** meta-learning issues are second-order on single learning tasks but first-order on *sequences of related tasks*; this motivates non-stationary tracking testbeds over conventional train-once benchmarks.

---

## Methodology

### Base learner: LMS / delta rule with per-input learning rates

A single linear unit outputs $y(t) = \sum_{i=1}^{n} w_i(t) x_i(t)$, with error $\delta(t) = y^*(t) - y(t)$. Ordinary LMS uses one global rate:

$$w_i(t+1) = w_i(t) + \alpha \, \delta(t) x_i(t)$$

IDBD replaces $\alpha$ with a per-input, time-varying $\alpha_i$:

$$w_i(t+1) = w_i(t) + \alpha_i(t+1)\,\delta(t) x_i(t)$$

(The $t+1$ index signals that the rate is updated *before* the weight update.)

**Why learning rates are bias.** Learning about irrelevant inputs acts as noise that interferes with learning about relevant ones; as a rule of thumb, learning time is proportional to the *sum of squares of the learning rates* (equal-variance inputs). Learning rates are therefore "a valuable resource that must be distributed carefully": likely-irrelevant inputs should get small rates, likely-relevant inputs large ones.

### The exponential parameterization

$$\alpha_i(t) = e^{\beta_i(t)}$$

Two advantages: $\alpha_i > 0$ automatically, and a fixed step in $\beta_i$ is a fixed *fraction* change in $\alpha_i$ (e.g. ±10%), which is essential because the required rates span orders of magnitude across inputs.

### The meta-update

$$\beta_i(t+1) = \beta_i(t) + \theta\,\delta(t)x_i(t)h_i(t)$$

$$h_i(t+1) = h_i(t)\big[1 - \alpha_i(t+1)x_i^2(t)\big]^+ + \alpha_i(t+1)\delta(t)x_i(t)$$

where $[x]^+ = \max(x, 0)$ and $\theta > 0$ is the **meta-learning rate** — the algorithm's only free parameter. $h_i$ is a decaying trace of recent changes to $w_i$; the decay factor $[1 - \alpha_i x_i^2]^+$ is normally a positive fraction, and vanishes only when $x_i$ is present.

**Intuition.** The $\beta_i$ increment is the product of the *current* weight change $\delta(t)x_i(t)$ and a *trace of recent* weight changes $h_i(t)$, so the accumulated change in $\beta_i$ is proportional to the **correlation between current and recent weight changes**. Positively correlated steps ⇒ past steps were too small ⇒ raise $\beta_i$. Negatively correlated steps ⇒ overshooting and re-correcting ⇒ lower $\beta_i$. This is the same intuition as Jacobs's DBD; the contribution is making it incremental and single-parameter.

### Pseudocode (Figure 2)

```
Initialize h_i to 0, and w_i, β_i as desired,  i = 1..n
Repeat for each new example (x_1..x_n, y*):
    y ← Σ_i w_i x_i
    δ ← y* − y
    Repeat for i = 1..n:
        β_i ← β_i + θ δ x_i h_i
        α_i ← e^{β_i}
        w_i ← w_i + α_i δ x_i
        h_i ← h_i [1 − α_i x_i²]⁺ + α_i δ x_i
```

Cost: memory and computation increase by roughly a factor of **three** over plain LMS, and remain **linear** in $n$ (three per-input memories: $w_i$, $\beta_i$, $h_i$).

**Practical notes from the paper:** it is often useful to bound each $\beta_i$ below (e.g. at $-10$) to prevent arithmetic underflow, and prudent to clip $\Delta\beta_i$ per step (e.g. to $\pm 2$) — though neither bound was needed for the reported results.

### Derivation as gradient descent (meta-gradient)

Just as LMS is gradient descent on the sample error in $w$-space, IDBD is gradient descent on the sample error in $\beta$-space:

$$\beta_i(t+1) = \beta_i(t) - \tfrac{1}{2}\theta\,\frac{\partial \delta^2(t)}{\partial \beta_i}$$

Here $\partial/\partial\beta_i$ without a time index means the derivative with respect to an infinitesimal change in $\beta_i$ **at all time steps** (the same device used in gradient analyses of recurrent networks, cf. Williams & Zipser 1989). Expanding through all weights and then applying the key approximation $\partial w_j(t)/\partial \beta_i \approx 0$ for $i \ne j$ ("the primary effect of changing the $i$-th learning rate should be on the $i$-th weight"):

$$\beta_i(t+1) \approx \beta_i(t) - \tfrac{1}{2}\theta\,\frac{\partial \delta^2(t)}{\partial w_i(t)}\frac{\partial w_i(t)}{\partial \beta_i} = \beta_i(t) + \theta\,\delta(t)x_i(t)h_i(t), \qquad h_i(t) \equiv \frac{\partial w_i(t)}{\partial \beta_i}$$

Differentiating the base update and using the product rule, with $\partial\delta(t)/\partial\beta_i \approx -h_i(t)x_i(t)$ under the same approximation:

$$h_i(t+1) \approx h_i(t)\big[1 - \alpha_i(t+1)x_i^2(t)\big] + \alpha_i(t+1)\delta(t)x_i(t)$$

which is exactly the $h_i$ rule (6) once the positive-bounding operation is added. So IDBD is **stochastic gradient descent in the learning-rate parameters**: stable at local optima, error-reducing elsewhere. Sutton stresses these are necessary but *not sufficient* properties — step size still matters, and the step *direction* can be modified by any factor $\alpha_i^p$ while remaining a descent direction (positive inner product with the gradient), generating a whole family of algorithms.

### Interpretation: incremental hold-one-out cross validation

Conventional leave-one-out CV holds out one of $N$ examples, trains on the rest, measures generalization to the held-out one, repeats $N$ times, and steps the hyperparameters to optimize that. IDBD does the incremental analogue: at each step the *new* example is the held-out example, and $\beta$ is stepped to improve generalization to it. Such methods "differ fundamentally from ordinary learning algorithms in that performance on the new example is optimized **without using the new example**."

---

## Experimental Results

Both experiments use a synthetic **tracking task** (non-stationary supervised learning, cf. Schlimmer 1987) rather than a fixed benchmark dataset — the point is to test whether biases learned early are exploited later, which requires a *continuing* problem.

**Task.** 20 real-valued inputs drawn i.i.d. $\mathcal{N}(0,1)$; one output. Target concept:

$$y^* = s_1x_1 + s_2x_2 + s_3x_3 + s_4x_4 + s_5x_5 + 0\cdot x_6 + \cdots + 0\cdot x_{20}, \qquad s_i \in \{+1,-1\}$$

Every 20 examples, one of the five $s_i$ is chosen at random and **flipped in sign**. So the same 5 inputs are always relevant and 15 always irrelevant, but the relevant mapping keeps drifting.

### Experiment 1 — Does IDBD help?

Protocol: one long run per setting; 20,000 examples to pass initial transients, then average MSE over the next 10,000 examples as the asymptotic tracking measure. LMS swept over $\alpha$; IDBD swept over $\theta$ (with $\alpha_i$ initialized to 0.05, which cannot affect asymptotic performance).

| Algorithm | Best asymptotic MSE |
|---|---|
| LMS (best $\alpha$) | ≈ 3.5 |
| IDBD (broad range of $\theta$) | ≈ 1.5 |

Standard errors of all means < 0.1, so the gap is highly significant — roughly a **60% reduction in squared error**, and IDBD beats LMS's best over a *broad* range of $\theta$ rather than at one tuned point (Figure 3). Above the plotted parameter ranges, both algorithms can become unstable.

### Experiment 2 — Does IDBD find the *optimal* $\alpha_i$?

Protocol: $\theta = 0.001$, 250,000 examples, $\alpha_i$ initialized to 0.05.

- Learning rates of the **15 irrelevant** inputs all < 0.007 after 250k steps and still heading towards zero (they cannot reach exactly zero unless $\beta_i = -\infty$) — clearly optimal.
- Learning rates of the **5 relevant** inputs all converged to $\alpha \approx 0.13 \pm 0.015$ (Figure 4 shows the time course for one relevant and one irrelevant input).
- **Ground-truth sweep:** fixing irrelevant rates to 0 and sweeping the relevant rate over 0.05–0.25 (same 20k + 10k protocol) gives an error minimum near $\alpha = 0.13 \pm 0.01$ (Figure 5).

So IDBD recovers both the relevance structure and the numerically optimal step size for this task.

---

## Limitations & Open Questions

Stated by the author:

- **Linear base learner only.** Only a linear (single-unit LMS) version is explored. Sutton does not foresee great difficulty extending IDBD to the nonlinear case, but that is out of scope; Jacobs's DBD, by contrast, was designed for nonlinear networks.
- **Not shown to be the best or fastest** way to find optimal learning rates. The $\alpha_i^p$ family may contain more efficient members; "there is little reason beyond simplicity for preferring the IDBD algorithm over these other possibilities a priori."
- **Optimality is empirical, not proved.** Experiment 2 confirms near-optimal rates only for a special case; a general result is acknowledged as possibly difficult.
- **Approximations in the derivation.** The gradient derivation assumes $\partial w_j/\partial\beta_i \approx 0$ for $i \ne j$, i.e. it ignores cross-input effects of a learning-rate change. It refines but does not eliminate the approximations of earlier analyses.
- **Meta-step-size still hand-picked.** IDBD removes $n$ learning rates in favor of one $\theta$, but $\theta$ itself is a free parameter — the meta-level regress is reduced, not closed.
- **Synthetic testbed.** Results are on one family of artificial drifting tracking tasks with i.i.d. unit-variance Gaussian inputs; no real dataset.
- **Stability.** Both LMS and IDBD "can become unstable" for step-size parameters above the plotted ranges, and practical runs may need $\beta_i$ floors and per-step $\Delta\beta_i$ clipping.

Directions the paper raises:

- Deriving bias-learning algorithms for **other base learners**, notably instance-based methods where the inter-instance distance metric parameters are the bias (currently set by people or offline CV).
- Using IDBD's learning rates as a **feature-utility/relevance signal** to direct constructive induction and representation change.
- IDBD as a **psychological model** of relevance learning / stimulus associability, augmenting LMS-style models of human and animal learning (Gluck, Glauthier & Sutton, in preparation; cf. Kruschke 1992, Hurwitz 1990).

**Resolved by a later source.** [SwiftTD](javed_2024_swifttd.md) (Javed, Sharifnassab & Sutton 2024) documents what became of this line, and confirms Sutton's open items were real:

- The **nonlinear extension** Sutton expected to be straightforward was still unresolved 32 years later: SwiftTD applies step-size optimization only to a network's **last layer**, training the convolutional kernels with ordinary TD(λ).
- Carrying IDBD to **TD learning** proved error-prone — Thill (2015) derived the meta-gradient incorrectly; Kearney et al. (2018, TIDBD) derived it correctly but against the TD(0) objective while learning with TD(λ); Young et al. (2019) got TD(λ) right; SwiftTD extended it to True Online TD(λ).
- Sutton's suspicion that other members of the $\alpha^p$ family might behave better foreshadows SwiftTD's finding that the **meta-gradient alone is insufficient**: it needs a non-differentiable overshoot bound (descended from Mahmood et al. 2012's correction-ratio bound for linear regression) plus a non-gradient step-size decay to stay stable.
- Degris et al. (2024) formalize Sutton's implicit claim as **step-size optimization vs. step-size normalization**, the axis on which IDBD differs from RMSProp/Adam.

---

## See Also

- [Step-Size Adaptation & Meta-Learning of Learning Rates](../concepts/step-size-adaptation.md) — concept article covering LMS, DBD, IDBD, the $\alpha^p$ family, and the line of descendants
- [Javed et al. (2024) — SwiftTD](javed_2024_swifttd.md) — the direct descendant: IDBD's step-size optimization inside True Online TD(λ), with the stability machinery IDBD lacked
- [Continual Learning & Tracking in Non-Stationary Tasks](../concepts/continual-learning-and-tracking.md) — why drifting targets, not train-once benchmarks, are the right testbed for bias learning
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — where the same step-size question reappears for TD learning (per-feature step sizes, the deadly triad)
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — optimizer/step-size choices in the modern deep-RL setting (Adam, clipping) as a contrast to explicit meta-gradient adaptation
