---
title: "Large-Scale Study of Curiosity-Driven Learning"
type: paper
tags: [intrinsic-motivation, curiosity, exploration, reinforcement-learning, prediction-error, forward-dynamics, feature-learning, atari, mario]
related: [concepts/intrinsic-motivation.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/curiosity_robots.pdf]
---

# Large-Scale Study of Curiosity-Driven Learning

**Authors:** Yuri Burda\*, Harri Edwards\*, Deepak Pathak\*, Amos Storkey, Trevor Darrell, Alexei A. Efros (\*equal contribution)  
**Affiliations:** OpenAI, UC Berkeley, University of Edinburgh  
**arXiv:** 1808.04355v1 — 13 Aug 2018

---

## Abstract

Reinforcement learning algorithms rely on carefully engineering environment rewards that are extrinsic to the agent. However, annotating each environment with hand-designed, dense rewards is not scalable, motivating the need for developing reward functions that are intrinsic to the agent. Curiosity is a type of intrinsic reward function which uses prediction error as reward signal. This paper: (a) performs the first large-scale study of purely curiosity-driven learning — *without any extrinsic rewards* — across 54 standard benchmark environments, including the Atari game suite; (b) investigates the effect of using different feature spaces for computing prediction error, showing that random features are sufficient for many popular RL game benchmarks, but learned features generalize better (e.g., to novel game levels in Super Mario Bros.); (c) demonstrates limitations of prediction-based rewards in stochastic setups (the "noisy-TV problem").

---

## Key Contributions

- **First large-scale study of pure curiosity:** 54 diverse simulated environments tested with zero extrinsic reward and no end-of-episode signal, demonstrating that dynamics-based curiosity consistently produces nontrivial behavior.
- **Surprising effectiveness of random features:** A frozen randomly-initialized CNN (Random Features / RF) is a simple yet strong baseline for modeling curiosity — competitive with or better than learned features in ~67% of Atari games while being more stable due to stationarity.
- **IDF generalizes better across environments:** Inverse Dynamics Features (IDF) outperform RF in transfer to novel Mario levels, suggesting that learned features capture more environment-relevant structure despite training instability.
- **Infinite-horizon curiosity ("death is not the end"):** Removing the end-of-episode (done) signal prevents the agent from exploiting it as a proxy for extrinsic reward, enabling cleaner measurement of pure exploration behavior.
- **Empirical characterization of the noisy-TV problem:** Stochastic environment elements attract prediction-error-based curiosity agents away from productive exploration, demonstrated concretely in a Unity maze with a TV displaying random channels.
- **Scalability:** Training with 2048 parallel environments causes Mario agent to discover 11 game levels — purely by curiosity.

---

## Methodology

### Dynamics-Based Curiosity

The intrinsic reward is the **prediction error** of a forward dynamics model. Given an observation $x_t$, action $a_t$, and next observation $x_{t+1}$:

1. An embedding network $\phi$ maps observations to a feature representation.
2. A forward dynamics model $f$ predicts $\phi(x_{t+1})$ from $\phi(x_t)$ and $a_t$.
3. The intrinsic reward is defined as the **surprisal**:

$$r_t = -\log p(\phi(x_{t+1}) \mid x_t, a_t)$$

With a fixed-variance Gaussian density model, this reduces to mean-squared prediction error:

$$r_t = \| f(x_t, a_t) - \phi(x_{t+1}) \|_2^2$$

### Feature Spaces Compared

Good feature spaces for curiosity should be **compact** (low-dimensional), **sufficient** (information-preserving), and **stable** (non-stationary targets make learning harder):

| Method | Compact | Sufficient | Stable |
|--------|---------|------------|--------|
| **Pixels** | No | Yes | Yes |
| **Random Features (RF)** | Yes | Maybe | Yes |
| **Inverse Dynamics Features (IDF)** | Yes | Maybe | No |
| **Variational Autoencoder (VAE)** | Yes | Yes | No |

- **Pixels:** $\phi(x) = x$. Sufficient but not compact; prediction errors dominated by irrelevant details.
- **Random Features (RF):** $\phi$ is a CNN with fixed random weights. Stable because the target is stationary, compact because architecture controls dimension. May lack sufficiency for complex environments.
- **Inverse Dynamics Features (IDF):** Given transition $(s_t, s_{t+1}, a_t)$, predict $a_t$ from $\phi(s_t)$ and $\phi(s_{t+1})$. Features correspond to aspects under the agent's immediate control. Not stable (features change as the network trains).
- **VAE:** Learned latent $z$ from $p(z|x)$. Compact and sufficient but not stable.

### Practical Training Considerations

To reduce non-stationarity and stabilize training across all environments with minimal hyperparameter tuning:

- **PPO** as the base RL algorithm (robust, requires little tuning).
- **Reward normalization:** divide by running std of discounted reward sum.
- **Advantage normalization:** zero mean, unit std within each batch.
- **Observation normalization:** computed from 10,000 random-agent steps before training.
- **128 parallel actors** (32 for Unity/Roboschool; 2048 for large-scale Mario).
- **Batch normalization** in the feature embedding network.
- **Infinite horizon:** remove the `done` signal so death is just another transition.

### Architecture

- Policy and embedding networks: standard CNN (3 conv layers → 512-dim features), working on 84×84 grayscale pixel stacks $[x_{t-3}, x_{t-2}, x_{t-1}, x_t]$ (partial observability via frame stacking).
- Forward dynamics and IDF networks: heads on top of the embedding network with additional fully-connected layers (512-dim).
- Learning rate: 0.0001 for all networks.

---

## Experimental Results

### 3.1 Curiosity-Driven Learning Without Extrinsic Rewards (54 environments)

**A) 48 Atari Games**
- Purely curiosity-driven agents (no reward, no done signal) obtain increasing extrinsic reward in the majority of games — the curves mostly go up.
- **Pixels:** performs poorly across all environments; prediction error dominated by irrelevant visual details.
- **RF vs. IDF:** RF does better than a random agent in ~67% of Atari games; IDF in ~71%. IDF outperforms RF in 55% of games.
- **VAE:** competitive but unstable; excluded from main comparison.
- Notable example: in **Breakout**, the agent learns to survive (avoiding death = avoiding predictable reset states) and break bricks as a byproduct of curiosity.
- Failure cases: games like 'Atlantis', 'IceHockey' where exploration does not correlate with game objectives.

**B) Super Mario Bros**
- With 128 parallel envs: agent explores and improves.
- With 2048 parallel envs: agent discovers **11 different game levels**, finds secret rooms, defeats bosses — purely curiosity-driven.

**C) Roboschool Juggling**
- Continuous 2D action space with two balls. Agent learns to intercept and strike balls without any reward — curiosity about ball dynamics drives interaction.

**D) Roboschool Ant Robot**
- 8-joint ant trained on pixel observations. A walking-like behavior emerges purely from curiosity, demonstrating emergent locomotion.

**E) Multi-Agent Two-Player Pong**
- Both paddles controlled by curiosity-driven agents (no teacher). Rally length grows over time as agents learn increasingly consistent behavior — rallies become so long they **break the Atari emulator** (background color cycling crashes the policy).

### 3.2 Generalization Across Novel Mario Levels

Pre-train on Level 1-1 with curiosity only, transfer to Level 1-2 or 1-3:

- **Level 1-1 → 1-2** (similar visual style): both RF and IDF transfer well.
- **Level 1-1 → 1-3** (day→night color shift): IDF transfers, RF does **not** — random features fail to bridge the domain gap.
- **Conclusion:** learned features (IDF) generalize better across visual distribution shifts; random features are sufficient for in-distribution performance but not transfer.

### 3.3 Curiosity with Sparse External Reward

- **Terminal reward (Unity 9-room maze):** Extrinsic-only (vanilla PPO) never finds the goal. IDF+curiosity and RF+curiosity both converge to reaching the goal consistently.
- **Sparse reward (5 Atari games):** In 4/5 games, adding curiosity bonus (coefficient 0.01, no tuning) improves over extrinsic-only.

### Noisy-TV Problem (Section 5)

Adding a TV with randomly changing channels (controllable by the agent) to the Unity maze:

- IDF and RF agents are strongly attracted to the TV; exploration of the maze collapses.
- With enough training time, agents can sometimes recover and find the goal — but convergence is drastically slower.
- Conclusion: prediction-error curiosity is fundamentally vulnerable to **stochastic attractors** in the environment.

---

## Limitations & Open Questions

- **Noisy-TV / stochastic dynamics:** Any irreducible source of entropy in the environment generates maximum curiosity reward without real progress. This applies even to partial observability and poor model class — not just true randomness.
- **Non-stationarity of intrinsic rewards:** Both the embedding network and dynamics model evolve during training, making the reward signal non-stationary and potentially destabilizing learning.
- **No sample efficiency claims:** Scaling to 2048 parallel envs improves performance per gradient step, not per environment step — this is a compute tradeoff, not a data efficiency result.
- **Feature sufficiency uncertain:** Whether random features remain sufficient for visually complex real-world environments (beyond game pixels) is an open question.
- **Optimal extrinsic + intrinsic combining:** The paper notes that combining intrinsic (coefficient 0.01) and extrinsic (coefficient 1.0) rewards without tuning leaves performance on the table; optimal combination is left as future work.
- **Open question:** How should curiosity be modified to be robust to stochastic dynamics? Subsequent work (e.g., RND — Random Network Distillation) addresses this by using a fixed target network to avoid stochastic attractors.

---

## See Also

- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — broader concept article covering prediction-error, count-based, and other intrinsic reward families
