---
title: Zero-Shot / Unsupervised Reinforcement Learning
type: concept
tags: [zero-shot-rl, unsupervised-rl, successor-features, representation-learning, task-agnostic, offline-rl, robot-learning]
related: [concepts/successor-features.md, concepts/latent-world-models.md, concepts/intrinsic-motivation.md, concepts/value-based-rl.md, papers/bagatella_2025_tdjepa.md, papers/hansen_2025_newt.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/TD-JEPA.pdf]
---

# Zero-Shot / Unsupervised Reinforcement Learning

## Problem Formulation

**Unsupervised (reward-free) RL** separates the RL pipeline into two phases:

1. **Pre-training phase**: The agent explores a reward-free MDP M = (S, A, P, γ), collecting a dataset D = {(s, a, s')} of transitions. No task reward is provided. The agent learns general-purpose representations and policies.

2. **Test time (zero-shot)**: Given a small **reward dataset** D_rwd = {(s, r)} specifying the task, the agent must recover an (approximately) optimal policy **without additional environment interaction**.

This framing is attractive for robotics: a robot can explore its environment freely (or be given an offline dataset), then be reassigned to new tasks with minimal reward engineering.

## The Zero-Shot Policy Extraction Paradigm

Most current zero-shot RL methods build on the **successor feature framework**: learn a task encoder ψ: S → R^d during pre-training such that reward functions of interest can be expressed as r(s) = ψ(s)^T z_r, and train policies {π_z}_{z∈Z} that are optimal for all tasks in the span of ψ.

At test time:
1. Project the new reward onto the task space: z_r = argmin_z E[(r - ψ(s)^T z)²] (linear regression)
2. Retrieve the pre-trained policy: π_{z_r}(s)

**Key insight**: if the task encoder ψ spans the rewards of interest, and if the policies {π_z} have been optimized for all z∈Z, then any new reward can be solved zero-shot.

## Design Axes

### 1. Task Encoder Training
How is ψ learned?
- **Contrastive/forward-backward** (FB, Touati & Ollivier 2021): bilinear F = φ^T ψ trained with contrastive loss over state pairs
- **Laplacian** (Wu et al. 2019): ψ = eigenvectors of the graph Laplacian of the state-transition graph
- **HILP** (Park et al. 2024): ψ via quasimetric/IQE objectives
- **Latent-predictive** (BYOL*, TD-JEPA): ψ via self-supervised prediction in latent space

### 2. State Encoder Training
Some methods (Laplacian, HILP, FB) focus only on ψ and use ψ directly as state encoder. Others (TD-JEPA, ICVF) train a **separate state encoder** φ for low-level perception, feeding φ(s) into the successor feature estimator F(φ(s), a; z). Separate encoders allow different abstraction levels for dynamics vs. task information.

### 3. Prediction Target
- **One-step behavioral** (BYOL*): predict the representation of the next state under the behavioral policy
- **Multi-step behavioral** (BYOL-γ*): predict multi-step future states under the behavioral policy
- **Multi-step zero-shot policy** (TD-JEPA): predict future states under the *zero-shot policies* {π_z}, enabling the predictor to approximate successor features F^{π_z}

Modeling zero-shot policy dynamics (TD-JEPA) is generally better than behavioral dynamics for tasks that are out-of-distribution from the offline data.

### 4. Offline vs. Online Pre-training
Methods differ in whether they require on-policy data (Monte Carlo rollouts from π_z are expensive) or can learn from any offline dataset. TD-JEPA's TD bootstrapping avoids Monte Carlo sampling from successor measures, enabling learning from arbitrary offline datasets.

## Representative Methods (zero-shot RL)

| Method | ψ training | φ training | Data requirement |
|---|---|---|---|
| Laplacian | Graph Laplacian | None | Offline |
| FB (Touati & Ollivier 2021) | Contrastive (bilinear) | Contrastive | Online/offline |
| HILP (Park et al. 2024) | Quasimetric IQE | None | Offline |
| BYOL* | One-step latent prediction | Same as ψ | Offline |
| BYOL-γ* | Multi-step latent prediction | Same as ψ | Offline |
| ICVF* (Ghosh et al. 2023) | Expectile regression | Expectile | Offline |
| **TD-JEPA** (Bagatella et al. 2025) | TD latent-predictive | TD latent-predictive | Offline |

## Relationship to Related Paradigms

**vs. Intrinsic motivation**: Intrinsic motivation also trains without extrinsic reward, but focuses on exploration (curiosity/count-based bonuses) rather than task-agnostic policy pre-training. Zero-shot RL aims to solve downstream tasks without further interaction; intrinsic methods still require environment interaction at test time.

**vs. Latent world models (JEPA-WM, V-JEPA 2-AC)**: World models predict future states conditioned on actions for planning. Zero-shot RL methods train a family of policies directly, enabling instant policy retrieval without planning at test time. World models require test-time search (CEM); zero-shot RL trades this for pre-training compute.

**vs. Meta-RL**: Meta-RL also adapts to new tasks quickly, but requires task rewards at meta-training time and typically requires a short adaptation phase with environment interaction. Zero-shot RL is strictly harder: no task rewards at training time, no interaction at test time.

## Benchmarks

- **ExoRL / DMControl**: 4 locomotion and navigation domains (walker, cheetah, quadruped, pointmass) with high-coverage offline data. Used to evaluate aggregate zero-shot performance.
- **OGBench**: 9 goal-reaching domains (antmaze variants, cube tasks, scene, puzzle) with low-coverage data. More challenging; tests generalization to unseen goal configurations.

## Open Questions

- How well does zero-shot RL scale to real robotics datasets (diverse, high-dimensional, partially observable)?
- Can asymmetric successor measures (non-symmetric M^π) be handled without losing theoretical guarantees?
- How does the quality of the offline dataset (coverage, diversity) bound zero-shot performance?

## See Also

- [Successor Features & Successor Measures](successor-features.md) — the theoretical backbone of most zero-shot RL methods
- [Latent World Models](latent-world-models.md) — alternative test-time planning approach; complementary to zero-shot policy retrieval
- [Intrinsic Motivation & Curiosity-Driven RL](intrinsic-motivation.md) — reward-free exploration; different goal from zero-shot RL
- [Value-Based Reinforcement Learning](value-based-rl.md) — TD learning and Q-functions that zero-shot RL generalizes
- [Bagatella et al. (2025) — TD-JEPA](../papers/bagatella_2025_tdjepa.md) — instantiates zero-shot RL via TD latent-predictive JEPA objectives
- [Hansen et al. (2025) — Newt / MMBench](../papers/hansen_2025_newt.md) — multitask world-model approach; related but uses reward during training
