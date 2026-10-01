---
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
type: paper
tags: [self-supervised-learning, video-representation, world-models, robot-learning, jepa, model-predictive-control, vision-transformer]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/latent-world-models.md, concepts/model-predictive-control.md, concepts/video-ssl.md, papers/zhou_2024_dino_wm.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/V-JEPA2.pdf]
---

# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

**Authors:** Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas
**Affiliations:** FAIR at Meta; Mila – Quebec AI Institute and Polytechnique Montréal
**Date:** June 13, 2025
**ArXiv:** 2506.09985

---

## Abstract

A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories) to develop models capable of understanding, predicting, and planning in the physical world. V-JEPA 2 is first pre-trained action-free on a dataset comprising over 1 million hours of internet video, achieving strong performance on motion understanding (77.3 top-1 on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100). After aligning with a large language model, V-JEPA 2 achieves state-of-the-art on multiple video question-answering tasks at the 8B parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, post-training a latent action-conditioned world model (V-JEPA 2-AC) using less than 62 hours of unlabeled robot videos from the Droid dataset enables zero-shot pick-and-place on Franka arms in two different labs using image goals.

---

## Key Contributions

- **Scaled self-supervised video pretraining:** V-JEPA 2 scales the V-JEPA pretraining recipe to 1B parameters (ViT-g) and 1M+ hours of video (VideoMix22M / VM22M), identifying four key scaling ingredients: data size, model size, longer training, and higher resolution.
- **Progressive resolution training strategy:** A warmup-constant-decay schedule that starts at low resolution/short clips and progressively increases both during cooldown, achieving 8× GPU-time reduction versus full-resolution training throughout.
- **3D-RoPE position embeddings:** Replacing sincos embeddings with 3D Rotary Position Embedding (partitioning feature dimensions into temporal, height, and width components) stabilizes training at larger scales.
- **Action-conditioned world model (V-JEPA 2-AC):** A ~300M-parameter block-causal transformer predictor trained on top of the frozen V-JEPA 2 encoder using only 62 hours of unlabeled Droid robot data, enabling latent-space planning.
- **Zero-shot robot manipulation via MPC:** V-JEPA 2-AC performs zero-shot prehensile manipulation (grasp, reach-with-object, pick-and-place) on Franka arms in novel environments using image goals and Cross-Entropy Method planning—without any task-specific training or reward.
- **First demonstration that a video encoder pretrained without language supervision can achieve state-of-the-art video QA performance** when aligned with an LLM, contrary to conventional wisdom.

---

## Methodology

### Stage 1: V-JEPA 2 Pretraining (Action-Free)

**Objective — Mask-Denoising in Representation Space:**

The model learns to predict the representation of masked video patches from unmasked context patches:

$$\min_{\theta, \phi, \Delta_y} \| P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\bar{\theta}}(y)) \|_1$$

where:
- $E_\theta(\cdot)$ is the encoder (ViT), $P_\phi(\cdot)$ is the predictor
- $x$ = masked video frames, $y$ = unmasked video frames
- $\Delta_y$ = learnable mask tokens indicating dropped patch positions
- $\bar{\theta}$ = exponential moving average (EMA) of encoder weights (stop-gradient target)
- Loss is applied only to masked patch predictions; EMA prevents representation collapse

**Architecture:**
- Encoder $E_\theta$: Vision Transformer (ViT-L: 300M, ViT-H: 600M, ViT-g: 1B params)
- Predictor $P_\phi$: ViT-small (~22M params), shared across all encoder sizes
- Video patchified as tubelets of size $2 \times 16 \times 16$ (T×H×W)
- **3D-RoPE:** Feature dimension split into three equal segments for temporal, height, width rotary embeddings (replaces absolute sincos positional encoding from original V-JEPA)
- Multiblock masking strategy (same as Bardes et al. 2024)

**Key Scaling Ingredients:**

| Ingredient | Change | Avg. Accuracy Gain |
|---|---|---|
| Data scaling | VM2M → VM22M (2M → 22M videos) | +1.0 pt |
| Model scaling | ViT-L → ViT-g (300M → 1B params) | +1.5 pt |
| Longer training | 90K → 252K iterations | +0.8 pt |
| Higher resolution | 256px/16fr → 384px/64fr | +1.0 pt |
| **Total** | | **+4.0 pt over ViT-L/16 baseline** |

**Training Schedule:** Warmup-constant-decay (not cosine); constant phase until plateau on IN1K/COIN/SSv2; cooldown phase ramps up resolution 256→384 and duration 16→64 frames over 12K steps.

**Progressive Resolution Training:** Primary phase: 16 frames, 256×256; Cooldown: 64 frames, 384×384. This achieves 8.4× GPU reduction vs. full-resolution training.

**Pretraining Dataset — VideoMix22M (VM22M):**

| Source | Samples | Hours | Weight |
|---|---|---|---|
| SSv2 | 168K | 168h | 0.056 |
| Kinetics | 733K | 614h | 0.188 |
| HowTo100M | 1.1M | 134K h | 0.318 |
| YT-Temporal-1B (curated) | 19M | 1.6M h | 0.188 |
| ImageNet | 1M | n/a | 0.250 |

YT-1B curation: scene extraction (PySceneDetect), DINOv2 scene embeddings, cluster-based retrieval matching target distribution (K710, SSv2, COIN, EpicKitchen) → 210K clusters, 115M retained scenes.

---

### Stage 2: V-JEPA 2-AC — Action-Conditioned World Model

**Architecture:** 300M-parameter transformer with block-causal attention:
- 24 layers, 16 heads, 1024 hidden dim, GELU activation
- Inputs: action tokens $a_k$ (7D Δend-effector), pose tokens $s_k$ (7D end-effector state), spatial feature maps $z_k = E(x_k) \in \mathbb{R}^{H \times W \times D}$ (16×16×1408 for ViT-g)
- Tokens are temporally interleaved: $(a_k, s_k, z_k)_{k \leq t}$ fed to causal predictor
- Block-causal attention: each patch at time $t$ attends to all patches, actions, and poses from $t$ and earlier time steps
- Separate learnable affine projections for action/pose/feature → hidden dim, and output → encoder embedding dim
- 3D-RoPE for spatiotemporal position of video patches; standard temporal rotary for action/pose tokens

**Training data:** ~62 hours of Droid dataset (Franka Emika Panda, 7-DoF, fixed exocentric camera, 256×256, 4fps → 16 frames per clip)

**Loss function:**

Teacher-forcing loss:
$$\mathcal{L}_{\text{teacher-forcing}}(\phi) := \frac{1}{T} \sum_{k=1}^{T} \left\| P_\phi\left((a_t, s_t, E(x_t))_{t \leq k}\right) - E(x_{k+1}) \right\|_1$$

Rollout loss (T=2 in practice):
$$\mathcal{L}_{\text{rollout}}(\phi) := \| P_\phi(\hat{a}_{1:T}, s_1, z_1) - z_{T+1} \|_1$$

Combined: $L(\phi) = \mathcal{L}_{\text{teacher-forcing}}(\phi) + \mathcal{L}_{\text{rollout}}(\phi)$

Only predictor weights $\phi$ are optimized; encoder is frozen.

---

### Stage 3: Planning via Model Predictive Control

**Energy function for goal-conditioned planning:**

$$\mathcal{E}(\hat{a}_{1:T}; z_k, s_k, z_g) := \| P(\hat{a}_{1:T}; s_k, z_k) - z_g \|_1$$

where $z_g = E(x_g)$ is the encoding of the goal image.

**Optimization:** Cross-Entropy Method (CEM / Rubinstein 1997):
- Sample $T$-step action trajectories from Gaussian distributions (zero mean, unit variance)
- Evaluate energy for each trajectory; update distribution from top-k statistics
- Repeat for several iterations; return mean of final Gaussian
- Actions constrained to L1-ball of radius 0.075 (~13 cm max displacement per step)

**Receding horizon control:** Execute only first action, observe new state, re-plan.

**Multi-goal (pick-and-place):** Sequence of sub-goals; optimize against each sub-goal for a fixed number of time steps before switching.

---

## Experimental Results

### Understanding: Probe-based Classification (Table 4)

Evaluation protocol: freeze encoder, train 4-layer attentive probe on top.

| Model | Params | Avg | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K |
|---|---|---|---|---|---|---|---|---|
| V-JEPA ViT-H (Bardes 2024) | 600M | 85.2 | 74.3 | 87.9 | 97.7 | 84.5 | 87.1 | 80.0 |
| **V-JEPA 2 ViT-g** | 1B | **87.5** | 75.3 | 90.1 | 97.7 | 86.6 | 90.7 | 84.6 |
| **V-JEPA 2 ViT-g₃₈₄** | 1B | **88.2** | **77.3** | **90.2** | **97.8** | 87.3 | 91.1 | 85.1 |

V-JEPA 2 ViT-g significantly outperforms all video encoders on motion understanding (SSv2: 77.3 vs. 74.3 for prior best video encoder).

### Prediction: Action Anticipation — EK100 (Table 5)

Metric: mean-class recall-at-5.

| Method | Params | Recall@5 (Action) |
|---|---|---|
| PlausiVL (Mittal et al. 2024) | 8B | 27.6 |
| V-JEPA 2 ViT-L | 300M | 32.7 |
| V-JEPA 2 ViT-g | 1B | 38.0 |
| **V-JEPA 2 ViT-g₃₈₄** | **1B** | **39.7** |

**+44% relative improvement** over prior SOTA (PlausiVL, 8B params) using only 1B params.

### Understanding: Video QA — 8B Model Class (Table 8)

| Method | PerceptionTest | MVP | TempCompass | TemporalBench | TOMATO |
|---|---|---|---|---|---|
| PLM 8B (Cho et al. 2025) | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 |
| **V-JEPA 2 ViT-g₃₈₄ Llama 3.1 8B** | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** |

State-of-the-art on PerceptionTest (+1.3), MVP (+4.2), TempCompass (+4.2), TemporalBench (+8.4), TOMATO (+7.1) relative to prior best 8B model (PLM 8B).

### Planning: Zero-Shot Robot Manipulation (Table 2)

Zero-shot deployment on two Franka Panda arms with RobotiQ grippers in two different labs (neither in Droid training data). Success rates over 10 trials.

| Method | Reach | Grasp (Cup/Box) | Reach w/ Obj (Cup/Box) | Pick-&-Place (Cup/Box) |
|---|---|---|---|---|
| Octo (fine-tuned on full Droid) | 100% | 15% / 0% | 15% / 0% | 15% / 10% |
| **V-JEPA 2-AC** | **100%** | **65% / 25%** | **75% / 75%** | **80% / 65%** |

V-JEPA 2-AC substantially outperforms Octo (which was fine-tuned with behavior cloning on full Droid) across all object interaction tasks.

**Planning speed:** V-JEPA 2-AC: 16 seconds/action (800 samples, 10 refinement steps). Cosmos (latent diffusion baseline): 4 minutes/action (80 samples).

---

## Limitations & Open Questions

### Limitations
- **Camera sensitivity:** V-JEPA 2-AC implicitly infers the action coordinate axis from monocular RGB without explicit camera calibration; manual camera positioning tuning was required before deployment. Robot base often not visible in frame.
- **Long-horizon planning:** Autoregressive prediction error accumulates; search space grows exponentially with horizon. Multi-step tasks requiring long-horizon planning (e.g., pick-and-place without sub-goals) remain challenging.
- **Image-only goals:** Goal specification requires visual images; language-conditioned planning is not yet supported.
- **Scale ceiling at 1B:** Consistent gains observed up to ViT-g (1B); benefits of further scaling (20B+) not yet demonstrated for this SSL objective.
- **EK100 benchmark constraints:** Closed vocabulary of kitchen actions; does not test generalization to open-ended categories.
- **Action anticipation degrades at longer horizons** beyond 1 second.

### Open Questions
- Can V-JEPA 2-AC representations support **language-based goal specification** (embed text goals into V-JEPA representation space)?
- What is the benefit of scaling to **20B+ parameter** video encoders for this SSL objective?
- Can **hierarchical world models** enable reliable long-horizon planning without sub-goals?
- How much does V-JEPA 2 generalize to **non-kitchen action anticipation** environments?
- Can the planning time be further reduced with **gradient-based planning** or learned policy initialization in the world model?

---

## See Also

- [Joint Embedding Predictive Architecture](../concepts/joint-embedding-predictive-architecture.md)
- [Latent World Models](../concepts/latent-world-models.md)
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md)
- [Video Self-Supervised Learning](../concepts/video-ssl.md)
- [Zhou et al. (2024) — DINO-WM](zhou_2024_dino_wm.md) — image-SSL (DINOv2) counterpart of V-JEPA 2-AC's action-conditioned latent planning, simulation only
