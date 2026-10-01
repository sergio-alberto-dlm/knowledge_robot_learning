---
title: Successor Features & Successor Measures
type: concept
tags: [reinforcement-learning, zero-shot-rl, successor-features, representation-learning, linear-reward, temporal-difference, unsupervised-rl]
related: [concepts/zero-shot-unsupervised-rl.md, concepts/value-based-rl.md, concepts/joint-embedding-predictive-architecture.md, papers/bagatella_2025_tdjepa.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/TD-JEPA.pdf]
---

# Successor Features & Successor Measures

## Core Idea

Successor features (SFs) decouple the dynamics of an MDP from the reward function, enabling **zero-shot policy transfer** across tasks that share the same transition structure but differ in reward.

Given a feature function ψ: S → R^d and a policy π, the **successor feature function** is:

$$F_\psi^\pi(s,a) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t \psi(s_{t+1}) \;\middle|\; s_0=s, a_0=a, \pi\right] \in \mathbb{R}^d$$

For any reward function in the linear span r(s) = ψ(s)^T z_r, the Q-value factorizes as:

$$Q_r^\pi(s,a) = F_\psi^\pi(s,a)^\top z_r$$

This means: **learn F_ψ^π once for the current dynamics, then zero-shot transfer to any task** by projecting the new reward onto ψ and reading off the optimal policy.

## Successor Measures

The **successor measure** M^π(·|s,a) is the (unnormalized) discounted distribution over future states:

$$M^\pi(\mathcal{X} \mid s, a) = \sum_{t=0}^\infty \gamma^t \Pr(s_{t+1} \in \mathcal{X} \mid s, a, \pi) \quad \forall \mathcal{X} \subseteq \mathcal{S}$$

The Q-value for any bounded measurable reward r then decomposes as:

$$Q_r^\pi(s,a) = \int_{s^+} M^\pi(ds^+ \mid s,a)\, r(s^+) = \mathbb{E}_{s^+ \sim M^\pi(\cdot|s,a)}[r(s^+)]$$

Successor features are a finite-dimensional approximation to successor measures: F_ψ^π(s,a) = E_{s^+~M^π(·|s,a)}[ψ(s^+)], which is exact for rewards in the span of ψ.

## Learning Successor Features

### Bellman Equation for SFs

Successor features satisfy a **vector-valued Bellman equation**:

$$F_\psi^\pi(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a),\, a' \sim \pi(\cdot|s')}\left[\psi(s') + \gamma F_\psi^\pi(s', a')\right]$$

This enables TD-style learning of SFs from off-policy data, analogous to Q-learning but predicting a vector rather than a scalar.

### Unsupervised / Zero-Shot RL Setup

In the **unsupervised RL** setting, a family of parameterized policies {π_z}_{z∈Z} is trained from reward-free data. Successor features are learned for all these policies simultaneously, producing a **policy-conditioned** successor feature function F(s, a; z) ≈ F_ψ^{π_z}(s,a).

At test time, given reward samples {(s_i, r_i)}, the task vector is recovered by linear regression:

$$z_r = \operatorname{argmin}_z \mathbb{E}[(r - \psi(s)^\top z)^2] = \mathbb{E}[\psi(s)\psi(s)^\top]^{-1}\mathbb{E}[\psi(s)r(s)]$$

The optimal policy is then π_{z_r}(s) = argmax_a F(s, a; z_r)^T z_r.

## Key Methods in This Framework

| Method | Approach |
|---|---|
| Successor Features (Barreto et al., 2017) | SFs with hand-coded ψ; TD learning; multi-task transfer |
| Forward-Backward (Touati & Ollivier, 2021) | Bilinear F = φ^T ψ via contrastive learning |
| HILP (Park et al., 2024) | Task encoder ψ from Laplacian/contrastive; no φ training |
| BYOL* / BYOL-γ* (Grill et al. 2020 / Lawson et al. 2025) | φ via one-step / multi-step latent-predictive learning |
| ICVF* (Ghosh et al., 2023) | Multilinear decomposition via expectile regression |
| **TD-JEPA** (Bagatella et al., 2025) | Both φ and ψ via TD latent-predictive JEPA loss |

## Theoretical Properties

**Completeness**: For rewards exactly in the span of ψ, SFs give the exact Q-function and optimal policy. For general rewards, performance degrades gracefully with the projection error.

**Zero-shot optimality** (Theorem 4, TD-JEPA): The policy evaluation error is bounded by 2× the successor measure approximation loss — so minimizing the TD-JEPA loss directly improves zero-shot policy quality.

**Relationship to JEPA**: The multi-step latent-predictive JEPA predictor, when trained via TD, approximates F_ψ^π in the latent space induced by the encoder (Proposition 1 in TD-JEPA). This connects the JEPA paradigm to the successor feature theory.

## See Also

- [Zero-Shot / Unsupervised RL](zero-shot-unsupervised-rl.md) — the broader problem setting; successor features are the dominant solution paradigm
- [Value-Based Reinforcement Learning](value-based-rl.md) — TD learning background; SFs extend scalar Q-functions to vector-valued predictions
- [Joint Embedding Predictive Architecture (JEPA)](joint-embedding-predictive-architecture.md) — TD-JEPA uses JEPA-style latent prediction to train successor feature approximators
- [Bagatella et al. (2025) — TD-JEPA](../papers/bagatella_2025_tdjepa.md) — trains both state and task encoders via TD-JEPA loss to recover successor measures
