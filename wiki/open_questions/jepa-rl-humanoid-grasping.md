---
title: "JEPA World Models as Physics Priors for RL: Replacing VLAs in the RL Token Stack"
type: open_question
tags: [robot-learning, jepa, world-models, reinforcement-learning, humanoid, dexterous-manipulation, visual-grasping, representation-learning, research-hypothesis]
related: [papers/xu_2025_rlt.md, concepts/joint-embedding-predictive-architecture.md, concepts/latent-world-models.md, concepts/vision-language-action-models.md, concepts/dexterous-manipulation.md, papers/assran_2025_vjepa2.md, papers/terver_2026_jepa_wm.md, papers/agarwal_2023_dex_func_grasp.md, papers/amin_2025_pi06_experience.md]
created: 2026-05-19
updated: 2026-09-25
sources: [raw/papers/pdf/rlt.pdf, raw/papers/pdf/V-JEPA2.pdf, raw/papers/pdf/V-JEPA-AC.pdf, raw/papers/pdf/dex_func_grasp.pdf, raw/papers/pdf/pi_06_experience.pdf]
---

# JEPA World Models as Physics Priors for RL: Replacing VLAs in the RL Token Stack

## 1. Motivation and Research Gap

A converging consensus is forming around a three-tier architecture for robot learning:

$$\underbrace{\text{Massive self-supervision}}_{\text{tier 1}} \;\to\; \underbrace{\text{Supervised policy}}_{\text{tier 2}} \;\to\; \underbrace{\text{RL refinement}}_{\text{tier 3}}$$

The **RLT paper** (Xu et al., 2025) instantiates this stack with a Vision-Language-Action model (π₀.₆) as the Tier-1 prior. The method achieves meaningful RL improvement (20%→65% on screw installation) in only hours of real-robot practice. The RL Token ablation is decisive: replacing the VLA-derived RL token with a frozen ResNet-10 cuts performance by 50%, confirming that the quality of the state representation — not just the RL algorithm — is the central bottleneck.

**This observation motivates the central hypothesis:**

> **The VLA is the wrong prior for RL.** VLMs are trained on image-text pairs — a static, semantic modality with no temporal structure. The RL value function, however, requires a state representation that encodes *causal-temporal* structure: "does this state lead to reward?" A world model pretrained on video via JEPA explicitly learns this structure. Replacing the VLA with a JEPA world model should provide a better-aligned state representation for RL, with potentially greater sample efficiency in tasks involving complex physical dynamics.

---

## 2. The Structural Mismatch in RLT

### What the RL Token captures (VLA-based)

In RLT, the RL token $\mathbf{z}_\text{rl}$ is obtained by compressing the VLA's final-layer token embeddings:

$$\mathbf{z}_\text{rl} = g_\phi\!\left([\mathbf{z}_{1:M},\, \mathbf{e}_\text{rl}]\right)_{M+1}, \qquad \mathbf{z}_i = f_\text{VLA}(o_t, \ell;\, \theta_\text{vla})_i$$

The reconstruction loss ensures the token retains enough information to regenerate VLA internal features:

$$\mathcal{L}_\text{ro} = \mathbb{E}_\mathcal{D}\!\left[\sum_{i=1}^M \bigl\|h_\phi(d_\phi([\mathbf{z}_\text{rl},\hat{\mathbf{z}}_{1:i-1}]))_i - \hat{\mathbf{z}}_i\bigr\|^2\right]$$

This representation encodes *"what the VLA believes the current scene to be"* — rich in semantic and instruction-grounding information, but trained without any temporal prediction objective.

### What RL actually needs from a state

For the TD value function to generalize, the state $s_t$ must satisfy an implicit predictivity condition:

$$\text{good } s_t \;\Longleftrightarrow\; V^\pi(s_t) \approx \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \,\Big|\, s_t\right] \text{ is learnable with little data}$$

This requires $s_t$ to encode *future consequences under the policy*, not just instantaneous appearance. A state representation trained on a forward prediction objective — predicting $z_{t+1}$ from $z_t$ and $a_t$ — is by construction aligned with this requirement. A semantic VLM representation is not.

---

## 3. The JEPA Prior: Alignment by Construction

### JEPA pretraining objective

V-JEPA (Bardes et al., 2024) and V-JEPA 2 (Assran et al., 2025) train a video encoder $E_\theta$ via:

$$\min_{\theta, \phi}\;\mathbb{E}_{(x,y)\sim\mathcal{D}}\!\left[\bigl\|P_\phi(E_\theta(x)) - \operatorname{sg}(E_{\bar\theta}(y))\bigr\|_1\right]$$

where $x$ is a spatiotemporally masked video context, $y$ is the masked-out target region, $P_\phi$ is a narrow transformer predictor, and $\bar\theta$ is an EMA teacher. The model learns to predict *future latent states* from context — the same forward-model structure that RL value functions require.

### Action-conditioned world model (V-JEPA 2-AC)

Post-training on robot data adds an action-conditioned predictor $P_\phi^a$:

$$\hat{z}_{t+1} = P_\phi^a\!\left(z_{t-W:t},\, a_{t-W:t}\right), \qquad z_t = E_\theta(o_t)$$

trained with a combined teacher-forcing + rollout loss. This gives a **differentiable simulator in latent space**: given the current visual observation and an action, the model predicts the future latent state without generating pixels.

**The encoder-predictor pair $(E_\theta, P_\phi^a)$ is a physics oracle** — it encodes causal structure over how robot actions transform visual observations, distilled from millions of hours of internet video plus targeted robot interaction data.

---

## 4. Proposed Architecture: JEPA-RL

### Three-stage pipeline

**Stage 0 — Frozen physics prior (pretrained once)**

$$E_\theta: o_t \mapsto z_t \in \mathbb{R}^d \quad \text{(V-JEPA 2 encoder, 1B params, frozen)}$$
$$P_\phi^a: (z_{t-W:t}, a_{t-W:t}) \mapsto \hat{z}_{t+1} \quad \text{(action-conditioned predictor, 300M params)}$$

The predictor may be fine-tuned on a small humanoid-specific video dataset (analogously to V-JEPA 2-AC's 62h of Franka data).

---

**Stage 1 — Task-specific adapter (light fine-tuning on demonstrations)**

A small trainable projector $J_\psi$ combines the frozen encoder output with proprioception and task context:

$$\mathbf{z}_\text{rl} = J_\psi\!\left(z_t,\, s_t^\text{prop},\, \ell_\text{task}\right) \in \mathbb{R}^{512}$$

Unlike the RLT adapter — which must extract information through a bottleneck from a 4B-parameter VLA — this projector operates on a representation already structured for prediction. It is trained on task demonstrations with a goal-prediction auxiliary loss:

$$\mathcal{L}_\text{adapt} = \underbrace{\|\hat{z}_{t+1} - z_{t+1}\|_2^2}_{\text{dynamics consistency}} + \underbrace{\|\mathbf{z}_\text{rl}^\text{goal} - z_\text{goal}\|_2^2}_{\text{goal grounding}}$$

A lightweight reference policy $\pi_\text{ref}$ (behavior cloning on demonstrations) is trained on $\mathbf{z}_\text{rl}$ to provide reference actions during online RL.

---

**Stage 2 — Online RL on real robot**

**RL state** (richer than RLT):

$$\mathbf{x}_t = \bigl(\mathbf{z}_\text{rl}(s_t),\;\; P_\phi^a(z_{t-W:t}, \bar{a}_{t:t+C}),\;\; s_t^\text{prop}\bigr)$$

The state includes not just the current observation but also the predictor's lookahead — "where will I be if I take the reference action?" This directly conditions the critic on consequential structure.

**Actor** (Gaussian over action chunks, BC-regularized):

$$\pi_\theta(\mathbf{a}_{1:C} \mid \mathbf{x}_t,\, \bar{\mathbf{a}}_{1:C}) = \mathcal{N}\!\left(\mu_\theta(\mathbf{x}_t, \bar{\mathbf{a}}_{1:C}),\, \sigma^2 I\right)$$

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_\mathcal{B}\!\left[-Q_\psi(\mathbf{x}_t, \mathbf{a}_{1:C}) + \beta\|\mathbf{a}_{1:C} - \bar{\mathbf{a}}_{1:C}\|_2^2\right]$$

**Critic** with model-based $n$-step returns:

Define imagined future states via the predictor:

$$\hat{z}_{t+k} = P_\phi^a\!\left(\hat{z}_{t+k-1}, a_{t+k-1}\right), \qquad k = 1,\ldots,n$$

The $n$-step TD target then becomes:

$$\hat{Q}_n = \sum_{k=0}^{n-1} \gamma^k r_{t+k} \;+\; \gamma^n Q_{\psi'}\!\left(\hat{\mathbf{x}}_{t+n}, \mathbf{a}_{t+n}\right)$$

This reduces variance of the TD target without requiring additional real-robot rollouts — a direct sample efficiency gain from the world model.

**Reward** (hybrid sparse + dense):

$$r_t = \underbrace{r_t^\text{sparse}}_{\text{binary success}} + \lambda \underbrace{\Bigl(-\bigl\|P_\phi^a(z_t, a_t) - z_\text{goal}\bigr\|_2\Bigr)}_{\text{dense: rep-space goal distance}}$$

The dense term provides a gradient signal throughout the reaching and pre-grasp phases, without reward engineering. Goal images $o_g$ are encoded once via the frozen $E_\theta$.

---

### Comparison to RLT

| Dimension | RLT (Xu et al. 2025) | JEPA-RL (this hypothesis) |
|---|---|---|
| **Tier-1 prior** | VLA (semantic, language-grounded) | JEPA world model (temporal-causal, video-grounded) |
| **State signal** | Bottleneck of VLA semantic tokens | Encoder output + predictor lookahead |
| **Reference policy** | VLA's own sampled actions | BC policy from demonstrations |
| **Value backup** | Pure TD on real transitions | TD + imagined $n$-step rollouts |
| **Reward** | Sparse binary (human labeled) | Sparse + dense representation-space |
| **Prior alignment with RL** | Indirect (semantic features) | Direct (forward prediction objective) |
| **Primary cost reduced** | Manual precision improvement | Sample efficiency of RL |

---

## 5. Application: Visual Grasping in Humanoids

Humanoid grasping is a particularly appropriate testbed for this hypothesis for three reasons:

1. **The imitation ceiling is low.** Dexterous grasping demonstrations are expensive to collect and rarely cover the full range of object geometries, contact configurations, and failure recovery strategies. RL is necessary to surpass the demonstration quality.

2. **The physics are complex.** Contact-rich 20-DOF hand dynamics, occlusion of the object during finger closure, and force-dependent grasp stability cannot be reliably learned from semantic image representations.

3. **Video pretraining is aligned.** V-JEPA 2's training set contains millions of hours of egocentric and third-person human hand-object interactions — directly relevant priors for grasping affordances and contact dynamics.

### Task decomposition

The grasping pipeline decomposes naturally into two phases, each exploiting different parts of the JEPA representation:

**Phase 1 — Reach and pre-grasp shaping** (vision-dominated)

- State: $\mathbf{z}_\text{rl}$ from wrist + head cameras + joint proprioception
- Action space: end-effector delta + **eigengrasp coefficients** (9-dim PCA over hand joint DOFs, following Agarwal et al. 2023, who use a 16-DOF LEAP hand) — restricts RL to physically realizable poses
- Reward: dense representation-space distance to goal + sparse success
- Reference policy: JEPA 2-AC MPC plan (expensive but provides a physically grounded anchor)

The eigengrasp parameterization is essential here: it reduces the dexterous action space to 9 dimensions (from 16 for the LEAP hand used by Agarwal et al.), making RL tractable without sacrificing physical realism.

> **Verify:** §5 above assumes a ~20-DOF humanoid hand, whereas the 9-component figure is Agarwal et al.'s result for a 16-DOF LEAP hand. How many eigengrasp components a 20-DOF humanoid hand needs is not established — it requires its own PCA on that hand's pose data, and 9 should not be carried over unexamined.

**Phase 2 — Contact and lift** (proprioception-dominated)

- State: $\mathbf{z}_\text{rl}$ + force/torque sensor readings (JEPA encoder does not capture contact forces)
- Action space: finger joint torques (direct, not eigengrasp — finer control for grasp stabilization)
- Reward: sparse success + proprioceptive shaping (grip force stability, no-slip condition)

This decomposition is modular and honest: JEPA priors are leveraged where they apply (visual dynamics of reaching), and domain-specific sensing is added where they do not (contact forces).

### Affordance detection

Visual affordance detection — identifying *where* to grasp — can be handled by the frozen JEPA encoder:

$$c^* = \arg\max_c \;\operatorname{sim}\!\bigl(E_\theta(o_t)[c],\; E_\theta(o_\text{ref})[c_\text{ref}]\bigr)$$

This zero-shot one-shot affordance matching (analogous to Agarwal et al.'s DINOv2 approach) exploits the JEPA encoder's part-level correspondences to identify grasp contact points on novel objects without retraining.

---

## 6. Predicted Advantages

**Sample efficiency.** The world model predictor enables $n$-step model-based TD backups in latent space, reducing the number of real robot transitions required to train a reliable critic. V-JEPA 2-AC achieves zero-shot manipulation without any online RL — the JEPA prior already encodes enough dynamics. RL on top of this prior should require far fewer episodes than RL on top of a semantic VLM prior.

**Representation-space reward.** The dense reward $r = -\|P_\phi^a(z_t, a_t) - z_\text{goal}\|_2$ requires no reward engineering and eliminates the need for human labeling of episode success during the reaching phase. Only the final grasp-success signal requires human (or gripper-force-sensor) supervision.

**Generalization.** The JEPA encoder's representations are invariant to appearance variations (lighting, object texture) by virtue of SSL pretraining on diverse internet video. This may provide better out-of-distribution generalization than VLA-derived tokens that are entangled with language grounding.

---

## 7. Key Risk and the Terver et al. Counterevidence

Terver et al. (2026) find, in a systematic ablation of JEPA-WM design choices, that **DINOv3 (image SSL) outperforms V-JEPA and V-JEPA 2 (video SSL)** as the encoder for manipulation:

> "DINOv3 ViT-L outperforms V-JEPA encoders for manipulation."

This is the primary challenge to the hypothesis. The proposed explanation:

- The Terver finding holds for **CEM-based MPC**, where the predictor already provides temporal structure via multi-step rollouts. In that setting the encoder only needs rich visual features; temporal information in the encoder is redundant.
- **RL is different**: the value function must compress expected future returns into the state. A state representation from a temporally-trained encoder may provide this compression more naturally, reducing the data needed to learn $Q(s, a)$.
- The Terver benchmark focuses on pick-and-place; fine-grained dexterous grasping involves more temporally complex dynamics where video-trained encoders may have a comparative advantage.

> **The central empirical bet:** Does V-JEPA outperform DINOv3 *as the RL state encoder* (not as the MPC planning encoder) on contact-rich manipulation tasks? This is the pivotal experiment for this hypothesis.

---

## 8. Testable Predictions and Proposed Experiments

| Experiment | What it tests |
|---|---|
| **E1:** JEPA-RL vs. RLT on same precision task (e.g., screw installation) with matched real-robot budget | Does the physics prior improve final success rate or sample efficiency? |
| **E2:** DINOv3-WM+RL vs. V-JEPA 2-WM+RL (frozen encoder, identical predictor and RL algorithm) | Is video SSL or image SSL the better RL state encoder? (Directly addresses Terver counterevidence) |
| **E3:** JEPA-RL with $n$-step model-based returns vs. JEPA-RL with 1-step TD only | Does imagined rollout reduce required real transitions? |
| **E4:** Dense representation-space reward vs. sparse binary reward only | Does rep-space reward accelerate or destabilize training? |
| **E5:** JEPA-RL on humanoid visual grasping with eigengrasp action space | End-to-end test on the target application |

---

## 9. Relationship to LeCun's Architecture

LeCun (2022) argues that the correct architecture for autonomous agents has four levels:

| Level | Function | Realized by |
|---|---|---|
| Perception | Compress raw observation to state | V-JEPA encoder $E_\theta$ |
| World model | Predict future states under actions | JEPA predictor $P_\phi^a$ |
| Actor | Select actions to maximize objective | RL policy $\pi_\theta$ |
| Critic | Evaluate consequences of actions | Value function $Q_\psi$ |

The VLA-based RL Token stack conflates Levels 1 and 2 (perception + world model) inside a VLM that was never trained to model dynamics. The JEPA-RL stack separates them cleanly: $E_\theta$ does perception, $P_\phi^a$ does world modeling, and RL does reward optimization on top. This is architecturally cleaner and each component is trained on data best suited to its function.

---

## 10. Open Questions

1. **How much humanoid video is needed to fine-tune $P_\phi^a$?** V-JEPA 2-AC used 62h of Franka data. Humanoid data is scarcer; can the predictor transfer from human hand-object interactions without robot-specific fine-tuning?

2. **Can the dense reward replace human labeling entirely?** The sparse binary reward in RLT requires a human operator. A fully automated pipeline using representation-space reward + force-sensor grasp detection would be significantly more practical.

3. **Does the predictor's lookahead improve or hurt the actor when the predictor makes errors?** Conditioning the actor on $P_\phi^a(z_t, \bar{a}_t)$ assumes the predictor is accurate. In early training on a new task, model errors could introduce noise.

4. **Is the eigengrasp space compatible with force control in Phase 2?** The eigengrasp parameterization constrains RL to realistic poses but may not produce the precise torque profiles needed for stable lifting. A hierarchical approach (eigengrasp for pre-grasp, direct torque for contact) may be needed.

---

## See Also

- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md) — the SSL framework underlying the proposed prior
- [Latent World Models](../concepts/latent-world-models.md) — JEPA-WM family and design choices; Terver et al. ablation
- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md) — the prior used in RLT; its limitations for RL
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — eigengrasp action space, affordance detection, sim2real
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md) — CEM planning in JEPA-WM; contrasts with the proposed RL approach
- [RLT / RL Token (Xu et al. 2025)](../papers/xu_2025_rlt.md) — the method this hypothesis is derived from and contrasted with
- [V-JEPA 2 (Assran et al. 2025)](../papers/assran_2025_vjepa2.md) — the proposed Tier-1 prior
- [JEPA World Models (Terver et al. 2026)](../papers/terver_2026_jepa_wm.md) — key counterevidence and design choices for JEPA-WM
- [Dexterous Functional Grasping (Agarwal et al. 2023)](../papers/agarwal_2023_dex_func_grasp.md) — eigengrasp space and one-shot affordance detection
- [RECAP — π₀.₆ (Amin et al. 2025)](../papers/amin_2025_pi06_experience.md) — alternative full-model VLA RL approach
