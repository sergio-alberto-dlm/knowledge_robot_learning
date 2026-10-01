---
title: "Deep Reinforcement Learning with Double Q-learning"
type: paper
tags: [reinforcement-learning, q-learning, dqn, double-dqn, overestimation, value-based, atari, temporal-difference]
related: [concepts/value-based-rl.md, concepts/policy-gradient-methods.md, concepts/intrinsic-motivation.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/DDQN.pdf]
---

# Deep Reinforcement Learning with Double Q-learning

**Authors:** Hado van Hasselt, Arthur Guez, David Silver  
**Affiliation:** Google DeepMind  
**arXiv:** 1509.06461v3 — 8 Dec 2015  
**Published:** AAAI 2016

---

## Abstract

Q-learning is known to overestimate action values under certain conditions. It was previously unknown whether these overestimations are common, whether they harm performance, and whether they can be prevented. This paper answers all three questions affirmatively. First, it shows that DQN suffers from substantial overestimations on some Atari 2600 games. Second, it shows that the Double Q-learning idea — originally proposed for tabular settings — generalizes naturally to work with large-scale function approximation. Third, a specific adaptation to DQN (called Double DQN) reduces the observed overestimations and achieves much better performance on several games, obtaining new state-of-the-art results on the Atari 2600 domain with minimal implementation overhead.

---

## Key Contributions

- **Theoretical characterization of overestimation (Theorem 1):** If $Q_t(s,a)$ are unbiased on average but imperfect, the max operator introduces an upward bias of at least $\sqrt{C/(m-1)}$ (where $C$ is mean squared estimation error and $m$ is the number of actions). Double Q-learning's equivalent bound is exactly zero.
- **Overestimation is common and harmful in DQN:** Empirically confirmed on all 49 tested Atari games; overestimation is not a fluke of specific games but a systematic artifact of the max operator under function approximation.
- **Double DQN: a minimal change to DQN:** Decouple action *selection* (online network $\theta_t$) from action *evaluation* (target network $\theta_t^-$) in the DQN target. The only difference from DQN is one line: the argmax uses the online weights instead of the target weights.
- **New state-of-the-art on Atari (2015):** Double DQN improves over DQN on the majority of 49 games; median normalized score rises from 93.5% to 114.7% (no-op) and 47.5% to 116.7% (human starts, tuned); mean rises from 241% to 475% (tuned, human starts).
- **More stable training:** Overestimation in DQN correlates with sudden score collapses (Asterix, Wizard of Wor); Double DQN learning curves are much smoother.

---

## Methodology

### Background: Q-learning

Q-learning (Watkins 1989) learns estimates of $Q_*(s,a) = \mathbb{E}[R_1 + \gamma R_2 + \ldots \mid S_0=s, A_0=a, \pi^*]$ via temporal-difference updates:

$$\theta_{t+1} = \theta_t + \alpha\left(Y_t^Q - Q(S_t, A_t; \theta_t)\right)\nabla_{\theta_t}Q(S_t, A_t; \theta_t)$$

with the Q-learning target:

$$Y_t^Q \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t)$$

### Background: DQN

DQN (Mnih et al. 2015) adds two key ingredients:
- **Experience replay:** transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ are stored in a replay buffer and sampled uniformly to break temporal correlations.
- **Target network:** a separate network $\theta^-$ (periodically copied from $\theta$, every $\tau$ steps) provides stable regression targets:

$$Y_t^{DQN} \equiv R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t^-)$$

The target network stabilizes training but does not address overestimation: both selection and evaluation of $a^*$ still use the same values (from $\theta^-$).

### The Overestimation Problem

**Theorem 1 (lower bound on overestimation):** Consider state $s$ where all true optimal action values are equal ($Q_*(s,a) = V_*(s)$ for all $a$). Let $Q_t(s,a)$ be estimates that are unbiased in aggregate ($\sum_a (Q_t(s,a) - V_*(s)) = 0$) but imperfect ($\frac{1}{m}\sum_a(Q_t(s,a)-V_*(s))^2 = C > 0$). Then:

$$\max_a Q_t(s,a) \geq V_*(s) + \sqrt{\frac{C}{m-1}}$$

The Double Q-learning estimate's absolute error lower bound is zero under the same conditions. Key implications:
- Overestimation grows with estimation error $C$ and decreases with the number of actions $m$ (but not fast enough to be negligible).
- Overestimation occurs regardless of whether estimation errors are due to noise, function approximation, non-stationarity, or any other source.
- Overestimated values propagate via bootstrapping: if $\max_a Q$ is higher than $V_*$, TD updates push nearby state values up as well.

### Double Q-learning Generalized

Original Double Q-learning (van Hasselt 2010) maintains two separate estimators $\theta$, $\theta'$, using one for action selection and the other for value evaluation:

$$Y_t^{DoubleQ} \equiv R_{t+1} + \gamma Q(S_{t+1}, \text{argmax}_a Q(S_{t+1}, a; \theta_t); \theta_t')$$

### Double DQN

The target network in DQN provides a natural second estimator — no additional networks needed. **The only change from DQN** is in the target computation:

$$Y_t^{DoubleDQN} \equiv R_{t+1} + \gamma Q(S_{t+1}, \underbrace{\text{argmax}_a Q(S_{t+1}, a; \theta_t)}_{\text{select with online }\theta_t}; \underbrace{\theta_t^-}_{\text{evaluate with target }\theta^-})$$

Compared to DQN (Eq. 3), the only difference is that the $\text{argmax}$ uses $\theta_t$ instead of $\theta_t^-$. The target network update schedule and all other DQN components remain unchanged.

### Network Architecture (same as DQN)

- **Input:** 84×84×4 grayscale frame stack
- **Conv 1:** 32 filters, 8×8, stride 4
- **Conv 2:** 64 filters, 4×4, stride 2
- **Conv 3:** 64 filters, 3×3, stride 1
- **FC:** 512 units, ReLU
- **Output:** $|A|$ Q-values (one per action)
- **Optimizer:** RMSProp, momentum 0.95

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Discount γ | 0.99 |
| Learning rate α | 0.00025 |
| Target update frequency τ | 10,000 steps |
| Training | 50M steps (200M frames) |
| Replay buffer | 1M tuples |
| Minibatch size | 32 |
| ε-greedy (training) | 1.0 → 0.1 over 1M steps |
| Evaluation ε | 0.05 |

Tuned Double DQN additionally uses τ = 30,000, ε = 0.001 during evaluation, and a shared bias layer across all action values.

---

## Experimental Results

All experiments use the Atari 2600 benchmark (Arcade Learning Environment). Scores normalized as:

$$\text{score}_\text{normalized} = \frac{\text{score}_\text{agent} - \text{score}_\text{random}}{\text{score}_\text{human} - \text{score}_\text{random}}$$

### Overestimation Analysis (6 representative games)

DQN's learned Q-value estimates are consistently and substantially higher than the true discounted returns of the learned policies. On Asterix and Wizard of Wor, DQN's overestimation is so severe that scores collapse as soon as overestimation becomes large. Double DQN estimates track true values much more closely.

### No-Op Evaluation (5 min / 18,000 frames, Table 1 & 3)

| Metric | DQN | Double DQN |
|--------|-----|------------|
| Median normalized | 93.5% | 114.7% |
| Mean normalized | 241.1% | 330.3% |

Notable per-game improvements (raw scores):
- Road Runner: 18,256 → 48,377 (+165%)
- Asterix: 6,011 → 15,150 (+152%)
- Zaxxon: 4,976 → 10,182 (+105%)
- Kangaroo: 6,740 → 13,651 (+103%)
- Double Dunk: −18.1 → −6.3 (substantially improved)

### Human-Start Evaluation (30 min / 108,000 frames, Table 2 & 5 & 6)

Agents initialized from 100 human expert trajectories (tests generalization):

| Metric | DQN | Double DQN | Double DQN (tuned) |
|--------|-----|------------|-------------------|
| Median normalized | 47.5% | 88.4% | 116.7% |
| Mean normalized | 122.0% | 273.1% | 475.2% |

Double DQN (tuned) substantially outperforms on human starts, suggesting better generalization. Notable wins: Video Pinball (20,228 → 367,824 for tuned), Atlantis (+1884%), Demon Attack (+2152%).

### Games Where DQN Still Beats DDQN

Some games see slight regressions (Private Eye, Gravitar, Frostbite) — Double DQN is not universally better at the fixed DQN hyperparameter setting.

---

## Limitations & Open Questions

- **Does not eliminate overestimation:** Double DQN reduces overestimation but does not guarantee unbiased value estimates under function approximation. The target network still uses stale weights and off-policy data.
- **Under-estimation possible:** By decoupling selection and evaluation, Double Q-learning can theoretically underestimate values in some configurations — though this appears less harmful in practice than overestimation.
- **Hyperparameters tuned for DQN:** The default evaluation is somewhat adversarial for Double DQN, as all hyperparameters were originally tuned for DQN. Tuning for Double DQN gives substantially better results (116.7% median vs. 88.4%).
- **Montezuma's Revenge remains unsolved:** Both DQN and Double DQN score 0; the sparse reward and exploration challenge are orthogonal to the overestimation fix.
- **No convergence guarantee with function approximation:** The deadly triad (off-policy + function approximation + bootstrapping) means neither DQN nor DDQN has a formal convergence guarantee.
- **Open question:** How does Double DQN interact with other DQN extensions (Dueling networks, Prioritized Experience Replay, n-step returns, distributional RL)? Rainbow (Hessel et al. 2018) later combines all six extensions and shows Double DQN is among the most important components.

---

## See Also

- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — concept article covering Q-learning, DQN, Double DQN, experience replay, and target networks
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — contrasting on-policy approach; PPO vs. DQN tradeoffs
- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — curiosity methods built on top of value-based RL (DQN-based in some implementations)
