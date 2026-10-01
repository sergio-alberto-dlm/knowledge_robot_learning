---
title: "Learning with GBP — Are Gradients Needed, and What Plays the Role of the Learning Rate?"
type: open_question
tags: [gaussian-belief-propagation, factor-graphs, hyperparameters, learning-rate, message-damping, step-size-adaptation, bayesian-inference, gaussian-processes]
related: [concepts/gaussian-belief-propagation.md, papers/ortiz_2021_gbp.md, papers/bui_2014_tree_gp.md, concepts/sparse-gaussian-processes.md, concepts/step-size-adaptation.md, papers/sutton_1992_idbd.md, concepts/continual-learning-and-tracking.md]
created: 2026-09-25
updated: 2026-09-25
sources: [raw/papers/pdf/GBP.pdf, raw/papers/pdf/GP_w_GBP.pdf]
---

# Learning with GBP: Are Gradients Needed, and What Plays the Role of the Learning Rate?

## Question

In a model that learns with Gaussian Belief Propagation (GBP), message passing estimates the posterior, so inference and training run as the same process. That raises three questions:
1. Does the model need to compute any gradients?
2. What are its hyperparameters?
3. Which of those hyperparameters is closest to the learning rate in ordinary deep learning?

> **Scope note:** The wiki has two GBP papers. Neither one is a "learning as inference" paper where network weights are variable nodes in the graph:
> - **Ortiz et al. (2021)** is a tutorial on GBP *inference*. It lists "GBP for distributed learning in overparameterized networks" as an open direction.
> - **Bui & Turner (2014)** uses GBP for inference in a tree-structured GP. Its hyperparameters are learned *separately*, by following gradients of the marginal likelihood.
>
> The paper the question most likely refers to is Nabarro, van der Wilk & Davison, *"Learning in Deep Factor Graphs with Gaussian Belief Propagation"* (ICML 2024). It is **not in the wiki yet** (see Gaps).
>
> This answer explains the general mechanism: any parameter you add to the factor graph as a variable node gets "learned" by inference. It then shows how each wiki paper fits that picture.

## Short answer

| Question | Answer |
|---|---|
| Is a gradient of a global loss needed? | **No.** When parameters are variable nodes, GBP updates them with local messages: sums of $(\eta, \Lambda)$ pairs and small marginalizations. There is no backpropagation and no global loss gradient. |
| Are *any* derivatives needed? | **Only local Jacobians, and only for non-linear factors.** Each non-linear factor $h(X)$ is linearized at its current estimate, which needs $\mathbf{J} = \partial h / \partial X$ for that factor's own variables. Nothing is chained across the graph. Linear-Gaussian factors need no derivatives at all. |
| Are there exceptions? | **Yes: hyperparameters that stay outside the graph.** In Bui & Turner, kernel hyperparameters $\theta$ and noise $\sigma^2$ are fit by maximizing the marginal likelihood with **gradients + BFGS**. GBP supplies the local posteriors those gradients need. |
| Is there a learning-rate analogue? | **There are two**, and they answer different questions:<br>(a) **Message damping $\beta$** is the step size of the message-passing iteration. It is the analogue in the optimization sense.<br>(b) **The ratio of factor (noise) precision to accumulated parameter precision** sets how far one datum moves a parameter. It is the analogue in the statistical sense, and it works like an automatic, per-parameter, decaying learning rate (a Kalman gain). |

---

## 1. Are gradients necessary?

### 1.1 Learning as inference: parameters become variable nodes

In a factor graph, "training" amounts to adding the unknown parameters $W$ as variable nodes and adding the data as factors connected to them. The posterior $p(W \mid \mathcal{D})$ is then just another marginal, and GBP computes it the same way it computes every other marginal:

- **Variable → factor:** the product of the other incoming messages, i.e. **a sum** of $(\eta, \Lambda)$ pairs.
- **Factor → variable:** condition on the incoming messages, then marginalize the other variables. In canonical form this is a **Schur complement** on a small local block.
- **Belief:** $b_i = \prod_s m_{f_s \to x_i}$, i.e. again a sum of $(\eta, \Lambda)$.

(See [GBP concept](../../wiki/concepts/gaussian-belief-propagation.md), "Belief Propagation Algorithm".)

None of these steps is a gradient step. For a Gaussian model, GBP is a distributed iterative solver for $\Lambda\mu = \eta$, closer to Jacobi or Gauss-Seidel than to gradient descent. Ortiz et al. make the contrast explicit in their comparison table: gradient descent is local and iterative but **not probabilistic**, while GBP is local, probabilistic, iterative, and asynchronous.

### 1.2 Where derivatives come back in: non-linear factors

Deep models are not linear-Gaussian. A layer factor such as $y = \phi(W x)$ is non-linear, and even the bilinear term $Wx$ is non-linear when both $W$ and $x$ are unknowns. GBP keeps messages Gaussian by **linearizing each factor at its current estimate $X_0$**:

$$h(X) \approx h(X_0) + \mathbf{J}(X - X_0), \qquad \Lambda = \mathbf{J}^\top \Sigma_n^{-1} \mathbf{J}, \qquad \eta = \mathbf{J}^\top \Sigma_n^{-1}\big(d - h(X_0) + \mathbf{J}X_0\big)$$

So a derivative **is** computed. It differs from backprop in three ways:
- It is the **Jacobian of one factor's measurement function** with respect to that factor's own variables. It is not the gradient of a global loss.
- It is **never chained through the network**. Information travels between layers as messages, not as a chain-rule product.
- It is recomputed only when a factor **relinearizes**, and each factor does this independently ("just-in-time relinearization").

You can compute the Jacobian analytically or with autodiff on the small factor function. Either way, the cost and data flow are local.

### 1.3 The GP case (Bui & Turner 2014): inference by GBP, hyperparameters by gradients

In the tree-structured GP there are two kinds of unknowns, and each is handled differently:

| Unknown | How it is obtained | Gradient? |
|---|---|---|
| Pseudo-datapoints $\mathbf{u}$ (and the implied function values) | GBP up-down pass on the tree: exact, two sweeps, $\mathcal{O}(KD^3)$ | **No** |
| Kernel hyperparameters $\theta$, noise $\sigma^2$ | Type-II maximum likelihood with **BFGS** | **Yes** |

The paper says: *"We obtain point estimates of the hyperparameters by finding a (local) maximum of the marginal likelihood using the BFGS algorithm."* GBP still helps here, because the gradient splits into terms that need only the **local posteriors GBP has already computed**:

$$\frac{d}{d\theta}\log p(\mathbf{y}\mid\theta) = \sum_{k}\Big[\big\langle \tfrac{d}{d\theta}\log p(\mathbf{u}_k\mid\mathbf{u}_l)\big\rangle_{p(\mathbf{u}_k,\mathbf{u}_l\mid\mathbf{y})} + \big\langle \tfrac{d}{d\theta}\log p(\mathbf{y}_k\mid\mathbf{u}_k)\big\rangle_{p(\mathbf{u}_k\mid\mathbf{y})}\Big]$$

This is an EM-like split: GBP does the E-step, and a gradient optimizer does the M-step. Hyperparameters like lengthscales are hard to put inside the graph because they are positive-valued and enter the kernel matrices in a highly non-linear way, so they don't fit Gaussian messages. The authors suggest one way to bring learning closer to inference: **online stochastic updates** that use gradients from single blocks instead of full up-down passes. They do not implement this.

**Takeaway:** "inference and training in one framework" is fully true only for quantities you put **inside** the graph as Gaussian variables. Anything left outside the graph (kernel hyperparameters, sometimes noise levels) still needs an outer optimizer, usually gradient-based.

---

## 2. The hyperparameters

### 2.1 Modelling hyperparameters (they define the posterior)

| Hyperparameter | Role | Deep-learning analogue |
|---|---|---|
| **Factor noise covariance $\Sigma_n$** (one per factor type: data/likelihood factors, layer/smoothness factors, …) | How strongly each constraint pulls. It weights the energies $E_i$. | Loss weights. A per-datum "confidence". |
| **Prior precision on parameters $\Lambda_0$** (and prior mean) | Anchors the weights. Its size sets how easily data can move them. | Weight decay / L2 regularization. Also initialization scale. |
| **Robust-loss threshold $t$** (Huber, applied through covariance scaling) | Past $t$, residuals are treated as outliers and down-weighted. | Huber loss δ / gradient clipping. |
| **Graph structure** (which factors connect which variables, layer widths, and in Bui & Turner the number of blocks $K$, block size $D$, pseudo-points $M$, and the spanning tree) | Sets which dependencies the model can express. It also sets compute cost ($\mathcal{O}(KD^3)$). | Architecture. |
| **Non-linearity choice in the factors** | Changes how accurate linearization is. | Activation function. |
| **GP only:** kernel lengthscales, signal variance, noise $\sigma^2$, spectral-mixture parameters | Define the prior over functions. | Architecture + regularization. These are learned by the outer BFGS loop, not by GBP. |

### 2.2 Algorithmic hyperparameters (they affect how you reach the posterior, not what it is)

| Hyperparameter | Role | Deep-learning analogue |
|---|---|---|
| **Message damping $\beta \in (0,1]$** | Exponential averaging of messages. Improves convergence in loopy graphs and **does not change fixed points**. | **Learning rate / step size** (with a momentum flavour). See §3. |
| **Number of GBP iterations / convergence tolerance** | How long messages are passed before reading beliefs. | Number of training steps / early stopping. |
| **Message schedule** (synchronous, random, sweep, residual BP) | The order of updates. Fixed points do not depend on it; convergence speed does. | Data ordering / curriculum / prioritized updates. |
| **Relinearization threshold** | How far estimates may drift before a factor recomputes its Jacobian. | A bit like how often you refresh curvature in second-order methods. |
| **Multiscale / coarse-to-fine levels** | Speeds up long-range propagation. | Progressive growing / multigrid warm starts. |
| **Initial beliefs** | The starting linearization points. With non-linear factors this affects which local optimum you reach. | Weight initialization. |

---

## 3. Which hyperparameter is analogous to the learning rate?

The learning rate in SGD does two jobs at once:
- **(i)** it sets the **step size of an iterative optimizer**;
- **(ii)** it sets **how far a single example moves the weights**, which is the plasticity/stability trade-off.

In GBP these two jobs belong to two different quantities.

### 3.1 Optimization sense: message damping $\beta$

$$\tilde{m}_t = \beta\, m_t + (1-\beta)\,\tilde{m}_{t-1} \;\;\Longleftrightarrow\;\; \tilde{m}_t = \tilde{m}_{t-1} + \beta\,\big(m_t - \tilde{m}_{t-1}\big)$$

The right-hand form is a relaxation step toward the new message, with **$\beta$ as the step size**. It behaves the way a learning rate does:
- $\beta = 1$ is the full step, i.e. plain GBP. It is fast, but it can **oscillate or diverge** on very loopy graphs, much like a learning rate that is too high.
- $\beta \to 0$ means no update, i.e. the model stops changing, much like a learning rate that is too low.
- The exponential average of past messages is effectively **momentum/EMA**.

**One important difference:** damping *does not change the fixed points* (Ortiz et al.: it improves convergence "without affecting the fixed points of GBP"). In non-convex deep learning, the learning rate *does* change which solution you reach. With non-linear factors and relinearization, $\beta$ can again affect the trajectory and therefore the local optimum, so the difference is smaller in the deep case.

> **Verify:** Whether $\beta$ changes the solution reached in deep, repeatedly relinearized factor graphs, as opposed to linear-Gaussian ones, is not covered by the wiki sources. Check the Nabarro et al. (2024) experiments.

### 3.2 Statistical sense: precision ratio = automatic per-parameter learning rate

Take a single scalar weight $w$ with belief $\mathcal{N}(\mu, \lambda^{-1})$, and a new datum factor $d \sim \mathcal{N}(w, \lambda_d^{-1})$ where $\lambda_d = \sigma_n^{-2}$. The GBP/Bayes update is:

$$\mu' = \frac{\lambda\mu + \lambda_d d}{\lambda + \lambda_d} = \mu + \underbrace{\frac{\lambda_d}{\lambda + \lambda_d}}_{\alpha_\text{eff}}\,(d - \mu), \qquad \lambda' = \lambda + \lambda_d$$

This has **exactly the form of an LMS/delta-rule step** $w \leftarrow w + \alpha(d - w)$, with an **effective learning rate $\alpha_\text{eff} = \lambda_d / (\lambda + \lambda_d)$**, which is a Kalman gain. Consequences:
- **Hyperparameters that set it:** the **factor noise $\Sigma_n$** (smaller noise → larger step) and the **prior precision $\Lambda_0$** (stronger prior → smaller initial step).
- **It adapts per parameter:** weights that already have a lot of evidence (high $\lambda$) move little, and uncertain weights move a lot.
- **It anneals itself:** $\lambda$ grows with every datum, so $\alpha_\text{eff}$ shrinks roughly like $1/n$. That is the classic Robbins-Monro schedule, and you don't have to tune it.

So **$\Sigma_n$ relative to $\Lambda_0$ is the closest analogue of the learning rate in terms of what gets learned**. $\beta$ is the closest analogue in terms of how the iteration moves.

This links directly to [step-size adaptation / IDBD](../../wiki/concepts/step-size-adaptation.md). IDBD *meta-learns* per-weight step sizes by gradient descent. In a Bayesian/GBP model, per-weight step sizes come out of the posterior precision automatically.

**Caveat for non-stationary / continual settings:** because $\alpha_\text{eff} \to 0$, a GBP learner eventually stops adapting. That is the same loss of plasticity discussed in [continual learning & tracking](../../wiki/concepts/continual-learning-and-tracking.md). The standard Kalman-filter fix is to add **process noise** (a forgetting factor) that inflates $\Sigma$ at every step. It becomes one more hyperparameter and sets a **floor on the effective learning rate**.

> **Verify:** Process noise / forgetting for GBP-trained weights is standard Kalman-filter practice. It is not described in either wiki source; check whether Nabarro et al. (2024) or follow-up work uses it.

### 3.3 Summary mapping

| Deep learning | GBP learning | Notes |
|---|---|---|
| Learning rate (step size) | Damping $\beta$ | Controls stability and speed of the iteration. In the linear case it does not change the fixed point. |
| Learning rate (plasticity) | $\Sigma_n$ vs. accumulated precision $\Lambda$ ($\alpha_\text{eff}$ = Kalman gain) | Automatic, per-parameter, decaying. |
| LR schedule / warm-up / decay | Growth of posterior precision (+ process noise for a floor) | Comes out of the inference itself. |
| Momentum | Damping as an EMA of messages | Same exponential-averaging form. |
| Weight decay | Prior precision $\Lambda_0$ | |
| Loss weighting / Huber δ | $\Sigma_n$ per factor type; Huber threshold $t$ | |
| Backprop | Messages + local Jacobians | No global chain rule. |
| Epochs / steps | GBP iterations to convergence | GBP can run continuously and absorb new factors. |

---

## Sources used

- [Gaussian Belief Propagation & Factor Graphs](../../wiki/concepts/gaussian-belief-propagation.md): message equations, linearization, damping, scheduling, robust factors
- [Ortiz et al. (2021) — Visual Introduction to GBP](../../wiki/papers/ortiz_2021_gbp.md), plus `raw/papers/pdf/GBP.pdf` (damping footnote: "improves convergence without affecting the fixed points"; open question "GBP for distributed learning in overparameterized networks")
- [Bui & Turner (2014) — Tree-structured GP Approximations](../../wiki/papers/bui_2014_tree_gp.md), plus `raw/papers/pdf/GP_w_GBP.pdf` (hyperparameters by BFGS on the marginal likelihood; proposed online stochastic hyperparameter updates)
- [Sparse Gaussian Processes](../../wiki/concepts/sparse-gaussian-processes.md)
- [Step-Size Adaptation & Meta-Learning of Learning Rates](../../wiki/concepts/step-size-adaptation.md): LMS/IDBD comparison for the effective-learning-rate argument

## Gaps identified

1. **The paper that actually does "learning = inference with GBP" is missing from `raw/`.** This is most likely Nabarro, van der Wilk & Davison, *Learning in Deep Factor Graphs with Gaussian Belief Propagation* (ICML 2024), where weights are variable nodes. Adding it would confirm:
   - exactly which hyperparameters it tunes (factor strengths per layer, damping, dropout on messages, number of iterations, …);
   - whether it uses any gradient-based outer loop.
2. **No empirical data in the wiki** on how sensitive GBP learning is to $\beta$ compared with $\Sigma_n$. Both claims in §3 are derived from the algorithm, not measured.
3. **Overconfident variances on loopy graphs** (a known GBP limitation) would make $\alpha_\text{eff}$ shrink too fast, i.e. too-early "learning-rate decay". No wiki source measures this.
4. **Process noise / forgetting for continual GBP learning** is not covered in the wiki sources.

## Follow-up questions

- In Nabarro et al. (2024), how do factor-strength hyperparameters in different layers interact? Is there a "per-layer learning rate" effect?
- Can the Bui & Turner GP hyperparameters ($\theta$, $\sigma^2$) be pulled into the graph, e.g. log-lengthscales as linearized variable nodes, so that everything is GBP?
- How does the GBP effective learning rate $\alpha_\text{eff}$ compare with IDBD/SwiftTD's meta-learned step sizes on a non-stationary tracking task?
- Is GBP learning a good fit for the online, on-robot settings in [continual learning & tracking](../../wiki/concepts/continual-learning-and-tracking.md), given that it is asynchronous and can take new factors at any time?

---

## Addendum (2026-09-29): Nabarro et al. (2024) now compiled

The gap flagged above is closed: [Nabarro et al. (2024) — GBP Learning](../../wiki/papers/nabarro_2024_gbp_learning.md). What it confirms or changes:

- **Gradients:** confirmed. Weights are variable nodes. Only local Jacobians of the relinearized conv/dense/pool factors are computed, and there is no backprop or global loss.
- **Hyperparameters actually tuned:** per-layer factor strengths $\sigma$ (recon, pixel/input obs, class obs), weight and activation prior σ, robust thresholds $N_\text{rob}$, **damping 0.7–0.9**, **message dropout 0.5–0.6** (not in the list above), GBP iterations per batch (train and test), batch size, and the prior-interpolation $\alpha$. All were tuned on validation splits, with no gradient-based outer loop reported.
- **§3.1 Verify (does damping change the solution?):** still open. The paper uses damping only as a stabiliser and does not ablate it.
- **§3.2 Verify (process noise / forgetting):** partly answered. For the single-layer continual denoiser, the prior at each frame is interpolated back toward the original prior ($\mu,\sigma \leftarrow 0.5\,\text{prior} + 0.5\,\text{posterior}$) to stop overfitting. This forgetting device puts a floor on $\alpha_\text{eff}$. The deeper model did not need it.
- **Iteration budget (the "epochs" row):** ~200 train iterations per batch suffice if test-time inference runs ≥200 iterations. The best result used 1,600.
- **Per-layer learning-rate effect (follow-up question):** not studied. Factor strengths differ by layer (e.g. recon σ 0.12 → 0.07 → 0.03 in the 5-layer denoiser), but there is no sensitivity analysis.
