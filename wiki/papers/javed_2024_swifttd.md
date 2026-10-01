---
title: "SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning"
type: paper
tags: [temporal-difference, td-lambda, eligibility-traces, step-size-adaptation, idbd, meta-learning, online-learning, prediction, atari, continual-learning]
related: [concepts/step-size-adaptation.md, concepts/td-lambda-and-eligibility-traces.md, concepts/continual-learning-and-tracking.md, papers/sutton_1992_idbd.md, concepts/value-based-rl.md, codebase/rl_lab/research-roadmap.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-24
updated: 2026-09-29
sources: [raw/papers/pdf/SwiftTD-RLC.pdf]
---

# SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning

**Authors:** Khurram Javed, Arsalan Sharifnassab, Richard S. Sutton
**Affiliation:** Alberta Machine Intelligence Institute (Amii), Department of Computing Science, University of Alberta
**Published:** Reinforcement Learning Journal / Reinforcement Learning Conference (RLC) 2024, pp. 840–863

---

## Abstract

Learning to make temporal predictions is a key component of reinforcement learning algorithms, and the dominant paradigm for learning predictions from an online stream of data is Temporal Difference (TD) learning. This work introduces **SwiftTD**, a TD algorithm that learns more accurate predictions than existing algorithms. SwiftTD combines **True Online TD(λ)** with **per-feature step-size parameters**, **step-size optimization**, a **bound on the update to the eligibility vector**, and **step-size decay**. Per-feature step sizes and step-size optimization improve credit assignment by increasing the step sizes of important signals and reducing them for irrelevant ones; the bound on the eligibility-vector update prevents overcorrections; step-size decay shrinks step sizes that are too large. SwiftTD is benchmarked on the Atari Prediction Benchmark, where even with linear function approximation it learns accurate predictions, and it performs well across a wide range of its hyperparameters. It can also be used in the last layer of neural networks to improve their performance.

---

## Key Contributions

- **SwiftTD**, a single algorithm combining True Online TD(λ) with three mechanisms, none of which suffices alone: step-size optimization, the overshoot bound, and step-size decay. The authors state this took two years to find and that it is specifically their *combination* that works.
- **The first correct extension of IDBD-style step-size optimization to True Online TD(λ)** (their extension to plain TD(λ) turned out to coincide with Young et al. 2019, discovered after the fact).
- **The overshoot bound:** a bound on the *correction ratio* of each update, enforced by bounding the increment to the **eligibility vector** rather than the weight update. This is the mechanism that makes large step sizes safe; Mahmood et al. (2012) had proposed a similar bound for linear regression, and the contribution here is making it work for TD learning with delayed, weight-dependent targets.
- **Step-size decay:** a non-gradient fallback that multiplicatively shrinks step sizes, in proportion to each feature's contribution to the overcorrection, whenever the bound triggers — because the bound's clipping operation is non-differentiable and so cannot be handled by the meta-gradient.
- **A fourth option for online learning.** The paper frames existing practice as three unsatisfactory choices — small step size (stable but slow), large step size (fast but divergent), or small step size with replay (sample-efficient but computationally wasteful, less reactive, and poor in big worlds). SwiftTD aims at learning quickly, without divergence, and **without multiple updates per data point**, removing the need to store and replay data.
- **Strong empirical result on the Atari Prediction Benchmark:** lower lifetime error than True Online TD(λ) on essentially every game, by up to an order of magnitude, with linear function approximation over 201,619 binary features.
- **Robustness across hyperparameters:** in a 3,025-configuration sweep on Pong, SwiftTD never diverged for *any* combination of initial step size and meta-step size, whereas step-size optimization alone diverged over much of that space.
- **Drop-in improvement for neural networks:** used in the last layer only (kernels trained with ordinary TD(λ), à la Tesauro 1995), SwiftTD beat a True Online TD(λ) last layer on almost all games — suggesting existing deep-RL systems could benefit by swapping only their final layer.
- **Interpretable credit assignment:** per-feature lifetime credit $\text{Credit}^T_i = \sum_t e^{\beta_t[i]}\phi_t[i]^2$ visualizes as semantically meaningful pixel maps (the ball's trajectory in Pong; dots and enemies in MsPacman; enemies, bullets and the passing UFO in SpaceInvaders).

---

## Methodology

### Problem formulation: online prediction, measured by lifetime error

The agent perceives $\phi_t \in \mathbb{R}^n$ and emits a scalar prediction $v_t$; the target is the discounted sum of a **cumulant** (any scalar component of the observation, commonly reward). Performance is the **lifetime error**:

$$\text{Lifetime error}(T) = \frac{1}{T}\sum_{t=1}^{T}\left(v_t - \sum_{j=t+1}^{T}\gamma^{\,j-t-1}r_j\right)^{2}$$

This measures not just the final solution quality but **how fast the agent got there**. The authors argue explicitly that the train/test split is a device for offline learning and is *unnecessary* online, since every prediction is made before its ground truth arrives.

Predictions are linear: $v_t = \sum_i w_{t-1}[i]\phi_t[i]$ (weights indexed $t-1$ because the prediction precedes the update).

### Background: λ-returns and True Online TD(λ)

The λ-return mixes all $n$-step returns, $G^\lambda_t = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}$. TD(λ)'s weight updates only *approximate* learning from λ-returns, and only when the step size is small. **True Online TD(λ)** (Van Seijen et al. 2016) is exactly equivalent to the Online λ-return algorithm, and performs better than TD(λ) at large step sizes. The paper shows this exact equivalence is not a nicety but **load-bearing**: it is what makes the overshoot bound applicable at all, and TD(λ) with the same bound still diverges.

### Idea 1: Step-size optimization

Step sizes are parameterized as $\alpha[i] = e^{\beta[i]}$ (as in IDBD) and updated by meta-gradient descent on the squared error to the λ-return:

$$\beta_t[i] = \beta_{t-1}[i] - \frac{\theta}{e^{\beta[i]}}\,\frac{\partial\left(v_t - G^\lambda_t\right)^2}{\partial\beta[i]}$$

The $1/e^{\beta[i]}$ normalization of the meta-step-size $\theta$ is deliberate: the scale of the meta-gradient with respect to $\beta[i]$ is itself proportional to $e^{\beta[i]}$. The exact meta-gradient is computationally infeasible, so it is approximated by **forward-view differentiation** (Williams & Zipser 1989) as in IDBD, under the **semi-gradient assumption** (ignoring the target's dependence on the weights).

**Prior attempts to carry IDBD to TD, per the paper:**

| Work | Status |
|---|---|
| Thill (2015) | Incorrect — mistake in deriving the meta-gradient update rule |
| Kearney et al. (2018) — TIDBD | Meta-gradient derived correctly, but uses the **TD(0) objective** for the meta-gradient while learning with TD(λ); the discrepancy can fail to raise the right features' step sizes |
| Young et al. (2019) | Correct extension of IDBD to TD(λ); coincides with this paper's TD(λ) version |
| This paper | Extension to **True Online TD(λ)** is novel |

### Idea 2: The overshoot bound

Define the **correction ratio** as the fraction of the prediction error removed by an update. For a linear learner with per-component step sizes it simplifies to a quantity computable *before* the update:

$$\tau_t = \sum_i \alpha_t[i]\,\phi_t[i]^2$$

$\tau = 1$ means the prediction jumps exactly onto the target; $\tau = 0.5$ means halfway. Bounding $\tau \le 1$ guarantees no overshoot. Two TD-specific obstacles and their resolutions:

1. **Delayed targets** — resolved by True Online TD(λ)'s exact equivalence to the Online λ-return algorithm. Crucially, the bound **cannot be applied at parameter-update time**; it must be applied when *adding to the eligibility vector*, as `min(1, (η/τ) e^{β[i]} φ[i])`.
2. **Targets depend on the weights** — resolved by the semi-gradient assumption, justified on the grounds that the goal of TD learning "is not to minimize the error but to propagate credit to the correct features by matching predictions to targets."

Dabney & Barto (2012) had attempted a correction-ratio bound for TD without the semi-gradient assumption and using the one-step target; the authors tried it and found it does not fix divergence, because it can permit arbitrarily large weight changes when consecutive time steps' features are highly correlated.

### Idea 3: Step-size decay

The bound's `min` is non-differentiable, so the meta-gradient cannot learn from the fact that it fired. Instead, whenever $\tau > \eta$, step sizes are shrunk multiplicatively **in proportion to each feature's contribution** to the overcorrection:

$$\alpha_{t+1}[i] = \alpha_t[i]\,\epsilon^{\,\phi_t[i]^2}$$

with $\epsilon \approx 0.99$ a reasonable decay rate. This differs from the Step-size Ratchet (Ghiassian 2022) in three ways: Ratchet decays abruptly rather than gradually, uses the one-step bootstrapped target rather than the λ-return for the overcorrection ratio, and uses a scalar step size not decayed per-feature.

### The combined algorithm

SwiftTD = True Online TD(λ) + all three ideas + two refinements: the bound triggers at a tunable $\eta$ (default 0.1) rather than at 1, and every $\beta[i]$ is clipped to $[\ln \eta_{\min}, \ln \eta]$ each step.

Pseudocode defaults (Algorithm 1): $\epsilon = 0.99$, $\eta = 0.1$, $\eta_{\min} = e^{-15}$, $\alpha_{\text{init}} = 10^{-7}$, plus $\lambda$, $\gamma$, $\theta$. The algorithm loops only over nonzero features (the benchmark's features are binary and sparse), but carries **nine** per-feature vectors — $w$, $\beta$, $h$, $h_{\text{old}}$, $h_{\text{temp}}$, $z$, $\bar z$, $z^\delta$, $p$ — versus three for True Online TD(λ).

> **Verify:** the 9-vs-3 vector count is read off Algorithm 1's initialization line, not stated as a cost figure by the authors; the paper does not report wall-clock or memory overhead versus True Online TD(λ) anywhere.

---

## Experimental Results

### Benchmark: the Atari Prediction Benchmark (Javed et al. 2023)

Prediction problems (not control) built on the Arcade Learning Environment. Actions come from **pretrained Rainbow-DQN policies** from the ChainerRL model zoo, so the policy is fixed and only prediction is learned.

- **Features:** 210×160×3 frame → resized to 105×80×3 → each channel lossily one-hot coded into 8 bins of 32 pixel values → stacked to 105×80×24 → flattened to 201,600 **binary** components, plus the previous one-hot action (18) and the cumulant (1) = **201,619 features**.
- **Cumulant:** +1 for positive ALE reward, −1 for negative, 0 otherwise.
- **γ = 0.98**; **lifetime T = 210,000** steps ≈ 2 hours of gameplay at 30 fps.
- Linear-function-approximation runs are deterministic, so single runs suffice; conv-net runs use 5-run sweeps and then 15 runs for the reported best configuration (±2 SE).

### Step-size optimization alone (Pong, Figure 3)

3,025 configurations: 55 values each of $\alpha_{\text{init}}$ and $\theta$ from $\{0.7^x : x = 0..54\}$.

- Error improves as $\theta$ rises from $10^{-8}$ to $10^{-2}$; **diverges above $\theta \approx 10^{-2}$**, and for $\alpha_{\text{init}} > 10^{-4}$.
- TD(λ) and True Online TD(λ) perform *similarly* once step-size optimization is added — so optimization alone does not explain SwiftTD's advantage.
- Conclusion drawn: optimization helps, but its divergence zone is unacceptable, since hyperparameters tuned on Pong are unlikely to transfer.

### Overshoot bound alone (subset of games, Figure 4)

- **True Online TD(λ) + bound never diverged for any $\alpha$**, and was *not conservative* — it left performance at the best $\alpha$ unchanged and only engaged beyond it.
- **TD(λ) + bound performed poorly** (Frostbite, Kangaroo, Breakout) and **diverged** (VideoPinball, DemonAttack, BattleZone). This is the evidence that True Online TD(λ) rather than TD(λ) is required.

### Full SwiftTD (Pong sensitivity, Figure 5)

With $\eta = 0.1$, $\epsilon = 0.999$: **no divergence for any $(\alpha_{\text{init}}, \theta)$ combination**, and reasonable error almost everywhere. **SwiftTD without step-size decay** also avoided divergence but performed poorly when either $\alpha_{\text{init}}$ or $\theta$ was large — the ablation establishing that all three ideas are needed.

### All Atari games (Figures 6, 7)

Hyperparameters swept and tuned **per game** for both algorithms, with a deliberately coarser grid for SwiftTD so the configuration counts match (Table 1). Errors are normalized by True Online TD(λ)'s error.

- SwiftTD achieved **equal or lower lifetime error on all games**, in some cases **an order of magnitude lower**.
- Learning curves (8 games) show lower lifetime error at almost every value of $T$, not just at the end.
- On **Atlantis and Pooyan, True Online TD(λ) failed completely for every hyperparameter setting** while SwiftTD learned accurate predictions. Predictions on Pong and Pooyan were near-perfect (Figure 1); on harder games like SpaceInvaders they still anticipated reward onsets.

### With a convolutional network (Figure 8)

One conv layer (25 kernels of 3×3×24, stride 2, weights $\sim U(-1,1)$), ReLU, flattened to 52,000 features, then a linear head. **SwiftTD in the last layer only**, kernels updated by TD(λ); baseline uses True Online TD(λ) in the last layer. SwiftTD helped in almost all games, with the kernel step size tuned independently of head hyperparameters.

### Semi-gradient vs full-gradient meta-updates (Appendix E, Figure 10)

Semi-gradient step-size optimization beat full-gradient **on average**, in some environments by more than 2×; in the worst case (Atlantis) full-gradient was only 20% better. This supports the semi-gradient choice empirically as well as conceptually.

---

## Limitations & Open Questions

- **Prediction, not control.** Every experiment learns value predictions under *fixed, pretrained* Rainbow-DQN policies. SwiftTD is never used to learn a policy, and no claim about control performance is tested — despite the motivating argument that improving TD improves Sarsa(λ)/Q-learning/PPO/actor-critic.
- **Derived for linear learners.** The neural-network result is explicitly a workaround: SwiftTD runs in the last layer only, with the conv kernels trained by ordinary TD(λ). A genuinely nonlinear SwiftTD is not developed. (The same limitation constrained [IDBD](sutton_1992_idbd.md) 32 years earlier.)
- **Hyperparameter count grows.** SwiftTD adds $\theta$, $\eta$, $\epsilon$, $\eta_{\min}$ and $\alpha_{\text{init}}$. The robustness claim is the mitigation — performance is flat across wide ranges — but the comparison is made fair only by *coarsening* SwiftTD's grid relative to True Online TD(λ)'s step-size grid.
- **Per-game tuning.** Hyperparameters are tuned individually per game. The authors state they verified results "did not change qualitatively" under a single shared setting, but that comparison is not shown.
- **Approximations stacked.** The meta-gradient is approximate (forward-view, IDBD-style), the semi-gradient assumption discards the target's weight dependence, and the overshoot bound is a non-differentiable clip. No convergence or stability theorem is offered — robustness is entirely empirical, on one benchmark at one $\gamma$ (0.98) and one lifetime (210k steps).
- **No reported compute/memory cost.** Nine per-feature vectors versus three is a real overhead the paper does not quantify against the replay-based baselines it argues are "computationally wasteful."
- **Internal inconsistency in reported hyperparameters.** The text gives defaults $\eta = 0.1$ and $\epsilon \approx 0.99$ (and Figure 5 uses $\epsilon = 0.999$), while the sweep in Table 1 lists $\eta \in \{1.0, 0.5\}$ and $\epsilon \in \{0.9, 0.8\}$ — i.e. the all-games results appear to come from a grid that excludes the recommended defaults. Table 1 also gives only two values for SwiftTD's initial step-size vector ($10^{-4}, 10^{-5}$) against Algorithm 1's default of $10^{-7}$.
  > **Verify:** this looks like a genuine discrepancy between §6 / Algorithm 1 and Appendix D rather than a reading error, but it is worth re-checking against the published version before citing either set of values as "the" SwiftTD defaults. Minor typo in the same vein: Eq. (9) writes $G_{t:t+1}$ where $G_{t:t+n}$ is meant.
- **Future direction named by the authors:** combining SwiftTD with efficient online RNN learning (Menick et al. 2021; Javed et al. 2023) for **replay-free state construction** from an online stream.

---

## See Also

- [Sutton (1992) — IDBD](sutton_1992_idbd.md) — the algorithm SwiftTD's step-size optimization descends from; same $\alpha = e^\beta$ trick, same forward-view trace
- [Step-Size Adaptation & Meta-Learning of Learning Rates](../concepts/step-size-adaptation.md) — concept article: IDBD, the optimization-vs-normalization distinction, and the IDBD→TD lineage this paper corrects
- [TD(λ) & Eligibility Traces](../concepts/td-lambda-and-eligibility-traces.md) — λ-returns, forward/backward views, and why True Online TD(λ)'s exactness matters here
- [Continual Learning & Tracking in Non-Stationary Tasks](../concepts/continual-learning-and-tracking.md) — lifetime error as an online evaluation metric; the case against train/test splits for streams
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — the replay-based deep-RL paradigm SwiftTD positions itself against
- [rl_lab — Research Roadmap](../codebase/rl_lab/research-roadmap.md) — planned use of SwiftTDNonSparse as a linear PPO critic on frozen visual features (one trace stream per env)
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md) — tile-coded object relations to match SwiftTD's validated sparse-binary regime; the 1,024-stream batching problem
