---
title: "TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning"
type: paper
tags: [zero-shot-rl, unsupervised-rl, successor-features, latent-predictive, jepa, temporal-difference, representation-learning, model-based-rl]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/successor-features.md, concepts/zero-shot-unsupervised-rl.md, concepts/latent-world-models.md, concepts/value-based-rl.md, papers/assran_2025_vjepa2.md, papers/balestriero_2025_lejepa.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/TD-JEPA.pdf]
---

# TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning

**Authors:** Marco Bagatella, Matteo Pirotta, Ahmed Touati, Alessandro Lazaric, Andrea Tirinzoni
**Affiliation:** FAIR at Meta, ETH Zurich, Max Planck Institute for Intelligent Systems
**arXiv:** 2510.00739v1 [cs.LG], October 1, 2025

---

## Abstract

Latent prediction — agents learning by predicting their own latents — has emerged as a powerful paradigm for training general representations. In RL this has been used as an auxiliary loss for reward-based RL, behavior cloning, and world modeling, but prior methods are limited to single-task learning, one-step prediction, or on-policy data. TD-JEPA shows that temporal-difference (TD) learning enables representations predictive of long-term latent dynamics across multiple policies from **offline, reward-free transitions**. TD-JEPA trains a state encoder, a task encoder, a policy-conditioned multi-step predictor, and a set of parameterized policies entirely in latent space. At test time it performs zero-shot optimization of **any** reward function. Theoretically, an idealized variant avoids collapse, recovers a low-rank factorization of long-term policy dynamics, and the predictor approximates successor features. Empirically, TD-JEPA matches or outperforms SOTA zero-shot RL baselines across 65 tasks, 13 datasets (ExoRL + OGBench), especially in the challenging pixel-based setting.

---

## Key Contributions

- **TD-JEPA loss** (Eq. 9): a novel off-policy, multi-step, policy-conditioned latent-predictive loss using TD bootstrapping — enabling learning from any offline dataset without reward or on-policy sampling.
- **Asymmetric two-encoder architecture**: separate state encoder φ and task encoder ψ trained via symmetric cross-prediction, decoupling low-level dynamics information from high-level task context.
- **Zero-shot policy extraction**: at test time, reward function r is projected onto the task space via closed-form linear regression to recover z_r, then policy π_{z_r} is directly executed — no policy fine-tuning needed.
- **Theoretical guarantees**: (1) non-collapse under proper initialization; (2) recovery of a low-rank factorization of successor measures; (3) policy evaluation error bounded by the successor measure approximation loss.
- **SOTA on pixel-based zero-shot RL**: strongest aggregate performance among all compared methods in RGB settings across locomotion, navigation, and manipulation.

---

## Methodology

### Problem Setup

Consider a reward-free MDP M = (S, A, P, γ). A set of parameterized policies {π_z}_{z∈Z} is trained on an offline dataset D = {(s, a, s')}. At test time, given a small reward dataset D_rwd = {(s, r)}, the goal is to recover the optimal zero-shot policy without further environment interaction.

The key connection used throughout: the Q-value of any linear reward r(s) = ψ(s)^T z_r decomposes as:

$$Q_r^\pi(s,a) = \mathbb{E}_{s^+ \sim M^\pi(\cdot|s,a)}[\psi(s^+)]^\top z_r = F_\psi^\pi(s,a)^\top z_r$$

where F_ψ^π is the **successor feature** function. If a predictor approximates F_ψ^π, then optimal policies for all rewards in the span of ψ can be extracted directly.

### One-Step JEPA Loss (baseline)

The standard latent-predictive loss for a single policy π:

$$\mathcal{L}_{\text{one-step}}(\phi, T) = \mathbb{E}_{s \sim \rho, a \sim \pi(\cdot|s), s' \sim P(\cdot|s,a)} \left[\|T(\phi(s)) - \overline{\phi(s')}\|^2\right]$$

where $\overline{\phi}$ denotes stop-gradient. This is on-policy and captures only one-step dynamics.

### Multi-Step Policy-Conditioned JEPA (MC-JEPA)

To handle multiple policies and long-term dynamics, introduce a policy-conditioned predictor T_φ: R^{d_φ} × A × Z → R^{d_φ}:

$$\mathcal{L}_{\text{MC-JEPA}}(\phi, T_\phi) = \mathbb{E}_{(s,a) \sim \mathcal{D}, z \sim \mathcal{Z}, s^+ \sim M^{\pi_z}(\cdot|s,a)} \left[\|T_\phi(\phi(s), a, z) - \overline{\phi(s^+)}\|^2\right]$$

**Proposition 1** establishes that predictors trained via this loss approximate the successor features F_φ^{π_z}(s,a) in the latent space induced by φ itself. However, this loss requires on-policy sampling from successor measures — infeasible from offline data.

### TD-JEPA Loss (the key innovation)

Using the Bellman equation for successor features F_φ^{π_z}(s,a) = E_{s'~P(·|s,a), a'~π_z(·|s')}[φ(s') + γ F_φ^{π_z}(s',a')], define the off-policy TD loss:

$$\mathcal{L}_{\text{TD-JEPA}}(\phi, T_\phi) = \mathbb{E}_{\substack{(s,a,s') \sim \mathcal{D} \\ z \sim \mathcal{Z},\, a' \sim \pi_z(\cdot|s')}} \left[\|T_\phi(\phi(s), a, z) - \overline{\phi(s')} - \gamma\overline{T_\phi(\phi(s'), a', z)}\|^2\right]$$

This only requires **one-step transitions** from any offline dataset. The target is a bootstrapped TD target (current next-state representation + discounted bootstrapped predictor output) — analogous to Q-learning's Bellman target but entirely in latent space.

### Asymmetric State and Task Encoders (Full TD-JEPA)

In practice, separate encoders φ (state, captures low-level dynamics) and ψ (task, captures high-level context) are trained symmetrically. The full loss is:

$$\mathcal{L}_{\text{TD-JEPA}}(\phi, T_\phi, \psi) = \mathbb{E}_{\substack{(s,a,s') \sim \mathcal{D} \\ z \sim \mathcal{Z},\, a' \sim \pi_z(\cdot|s')}} \left[\|T_\phi(\phi(s), a, z) - \overline{\psi(s')} - \gamma\overline{T_\phi(\phi(s'), a', z)}\|^2\right]$$

with φ and T_φ optimized via L_{TD-JEPA}(φ, T_φ, ψ), and ψ and T_ψ via L_{TD-JEPA}(ψ, T_ψ, φ) (roles swapped). The task latent space Z ⊆ R^{d_ψ} then defines the space of reward functions of interest.

### Full Algorithm (Algorithm 1)

**Training** (all from offline data D):
1. Sample batch {(s_i, a_i, s'_i)} from D, latents {z_i} from Z, actions {a'_i} from {π(·|φ^-(s'_i), z_i)}
2. Compute L_{TD-JEPA}(φ, T_φ, ψ) and L_{TD-JEPA}(ψ, T_ψ, φ)
3. Orthonormality regularization (prevents collapse):
   $$\hat{\mathcal{L}}_{\text{REG}}(\phi) = \frac{1}{2B(B-1)}\sum_{i \ne j}(\phi(s_i)^\top\phi(s_j))^2 - \frac{1}{B}\sum_i \phi(s_i)^\top\phi(s_i)$$
4. Actor loss: $\hat{\mathcal{L}}_{\text{actor}}(\pi) = -\frac{1}{B}\sum_i T_\phi(\phi(s_i), \hat{a}_i, z_i)^\top z_i$
5. Update φ, T_φ with L_{TD-JEPA} + λ L_{REG}; similarly for ψ, T_ψ; update π; update EMA target networks

**Test time (zero-shot)**:
- Given D_rwd = {(s, r)}, compute z_r via linear regression:
  $$z_r = \operatorname{argmin}_{z}\,\mathbb{E}_{(s,r) \sim \mathcal{D}_{\text{rwd}}}\left[(r - \psi(s)^\top z)^2\right]$$
  Closed form: $z_r = \mathbb{E}[\psi(s)\psi(s)^\top]^{-1}\mathbb{E}[\psi(s)r(s)]$
- Execute policy $\pi_{z_r}(s) = \operatorname{argmax}_a T_\phi(\phi(s), a, z_r)^\top z_r$

---

## Theoretical Analysis

**Theorem 1 (MC case — non-collapse & successor measure recovery):** Under uniform state distribution and symmetric policies, the optimal predictors for L_{MC-JEPA} and the successor measure approximation loss L_{SM} match, and their gradients w.r.t. φ, ψ are identical. Thus gradient descent on L_{MC-JEPA} improves the successor measure approximation.

**Theorem 2 (Non-collapse for TD losses):** In a continuous-time relaxation of Eq. 9, when predictors are trained faster than encoders, the covariance matrices φ_t^T φ_t and ψ_t^T ψ_t are constant over time — preventing collapse to trivial solutions (φ = ψ = 0) when initialized with unit covariance.

**Theorem 3 (TD case — successor measure recovery):** Under the same assumptions as Theorem 1, TD-JEPA's optimal predictors and gradients match those of the non-latent-predictive forward-backward (FB) successor measure objective (Touati & Ollivier, 2021). Unlike the MC case (orthogonal projection), here the projection is oblique.

**Theorem 4 (Zero-shot optimality bound):** With identity covariance representations, the policy evaluation error (how suboptimal the zero-shot policy is) is bounded above by 2×L_{SM}(φ, T_z, ψ) — the successor measure approximation loss that TD-JEPA minimizes. This justifies TD-JEPA as a sound approach for zero-shot RL.

---

## Experimental Results

### Setup
- **Benchmarks**: 13 datasets total — 4 from ExoRL/DMControl (locomotion: walker, cheetah, quadruped, pointmass; high-coverage data) and 9 from OGBench (antmaze variants, cube tasks, scene, puzzle; goal-reaching, low-coverage)
- **Observations**: both proprioceptive (state) and RGB pixel inputs
- **Metric**: normalized return (DMC) and success rate (OGBench), 4–8 tasks per domain
- **Baselines**: Laplacian, HILP (task-encoder-only), FB (Touati & Ollivier 2021, contrastive), RLDP (Jajoo et al. 2025), BYOL* (one-step latent-predictive), BYOL-γ* (multi-step behavioral), ICVF* (multilinear decomposition via expectile regression)

### Main Results (Table 1)

| Setting | TD-JEPA | Best baseline |
|---|---|---|
| DMC_RGB avg | **628.8 ± 5.5** | BYOL-γ* 582.4 ± 9.8 |
| DMC (prop) avg | **661.2 ± 6.3** | BYOL* 645.4 ± 10.5 |
| OGBench_RGB avg | 41.34 ± 0.45 | BYOL* 41.58 ± 0.64 |
| OGBench (prop) avg | 37.98 ± 0.77 | BYOL-γ* 37.98 ± 1.11 |

**TD-JEPA is consistently among the top algorithms across all suites.** Key observations:
- Latent-predictive methods (TD-JEPA, BYOL*, BYOL-γ*) tend to dominate in RGB domains
- TD-JEPA is only slightly behind FB/HILP from proprioception but significantly outperforms in visual domains
- BYOL-γ* is slightly better than TD-JEPA on OGBench_RGB but significantly worse on DMC_RGB and OGBench
- Probability-of-improvement analysis (Fig. 2): TD-JEPA has the highest probability of outperforming any individual baseline

### Ablation: What dynamics to model? (§5.2)

BYOL* (one-step behavioral), BYOL-γ* (multi-step behavioral), and TD-JEPA (multi-step zero-shot policy dynamics) differ in prediction target. Directly modeling policy-conditional successor measures (TD-JEPA) is **on average beneficial** vs. modeling behavioral dynamics (BYOL-γ*), especially for out-of-distribution tasks in ExoRL.

### Ablation: Shared vs. separate encoders (§5.3)

The symmetric single-encoder variant (shared state/task encoder) performs comparably on most tasks but using distinct encoders tends to improve empirical performance more often than not, particularly on tasks requiring different abstraction levels.

### Fast Adaptation (§5.4)

Fine-tuning from zero-shot initialization (both offline TD3 and online TD3) achieves much faster learning and higher asymptotic performance than training from scratch. Notably, **frozen representations** are often sufficient for downstream learning — the pre-trained state encoder is expressive enough that only the policy/value head needs updating.

---

## Limitations & Open Questions

- **Symmetry assumption**: formal guarantees (Theorems 1–4) rely on symmetric successor measures (M^{π_z} = M^{π_z}^T), which holds only for symmetric policies. Extension to asymmetric successor measures is noted as important future work.
- **Linear reward span**: zero-shot optimality is guaranteed only for rewards in the span of ψ; rewards outside this span may not be recoverable.
- **Offline data only during pre-training**: the offline dataset must have reasonable coverage of the state-action space — poor coverage datasets (like OGBench) limit performance.
- **Real robot evaluation**: TD-JEPA is evaluated only in simulation; benchmarking on large-scale real robotic datasets is identified as an important open problem.
- **Computational cost at test time**: the closed-form linear regression for z_r is cheap, but the quality of the zero-shot policy degrades if the reward dataset D_rwd is very small or noisy.

---

## See Also

- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md) — the broader SSL paradigm; TD-JEPA instantiates it with TD objectives for RL
- [Successor Features & Successor Measures](../concepts/successor-features.md) — the foundational theory that TD-JEPA's predictor recovers
- [Zero-Shot / Unsupervised RL](../concepts/zero-shot-unsupervised-rl.md) — the problem setting: pre-train without reward, solve any task at test time
- [Latent World Models](../concepts/latent-world-models.md) — related latent-space dynamics modeling; TD-JEPA differs by targeting successor features rather than one-step prediction
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — temporal difference learning and Q-functions; TD-JEPA applies TD bootstrapping in latent space
- [Assran et al. (2025) — V-JEPA 2](assran_2025_vjepa2.md) — action-conditioned world model using JEPA for robot manipulation via planning
- [Balestriero & LeCun (2025) — LeJEPA](balestriero_2025_lejepa.md) — theoretical framework for JEPA; TD-JEPA extends JEPA theory to RL / TD setting
- [Hansen et al. (2025) — Newt / MMBench](hansen_2025_newt.md) — self-predictive world model (TD-MPC2 family) for multitask RL; related but uses reward during training
