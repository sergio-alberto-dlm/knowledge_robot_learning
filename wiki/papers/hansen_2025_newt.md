---
title: "Newt: Learning Massively Multitask World Models for Continuous Control"
type: paper
tags: [world-models, model-based-rl, multitask-rl, continuous-control, self-predictive, td-mpc2, language-conditioning, benchmark, online-rl, planning]
related: [concepts/latent-world-models.md, concepts/model-predictive-control.md, concepts/intrinsic-motivation.md, papers/terver_2026_jepa_wm.md, papers/assran_2025_vjepa2.md, papers/zhou_2024_dino_wm.md]
created: 2026-05-18
updated: 2026-09-29
sources: [raw/papers/pdf/newt.pdf]
---

# Newt: Learning Massively Multitask World Models for Continuous Control

**Authors:** Nicklas Hansen\*, Hao Su†, Xiaolong Wang† (\*first author, †equal advising)  
**Affiliation:** University of California San Diego  
**Venue:** Preprint  
**Project / Code:** https://www.nicklashansen.com/NewtWM

---

## Abstract

Can a single RL agent be trained on hundreds of continuous control tasks simultaneously via online interaction? This work answers yes. The paper introduces **MMBench**, the first benchmark for massively multitask RL (200 tasks, 10 domains, language instructions, demonstrations, optionally image observations), and **Newt**, a language-conditioned multitask world model extending TD-MPC2 to the massively multitask online RL setting. Newt is first pretrained on demonstrations, then jointly optimized with online interaction across all tasks. It outperforms strong baselines (BC, PPO, FastTD3) on average across 200 tasks, transfers rapidly to unseen tasks/embodiments, and supports open-loop planning over horizons 16× longer than its training horizon.

---

## Key Contributions

- **MMBench**: first benchmark purpose-built for massively multitask online RL — 200 continuous control tasks, 10 diverse domains, language instructions per task, 10–40 demonstrations per task, and 20 test tasks for transfer evaluation; all code, data, and 200+ model checkpoints released
- **MiniArcade**: 19 new arcade-style tasks spanning 14 environments with well-defined reward functions, contributed as part of MMBench
- **Newt**: language-conditioned self-predictive world model for massively multitask continuous control, extending TD-MPC2 with language/image conditioning, discrete reward/value regression, and a four-pronged demonstration strategy
- **Key findings**: model and batch size scaling is beneficial in multitask RL (unlike single-task where it is marginal); language conditioning raises scores from 0.371 → 0.438; any single demonstration strategy helps but all four together are best; visual inputs benefit manipulation and hurt locomotion

---

## MMBench

### Task Suite

200 continuous control tasks across 10 domains:

| Domain | Tasks | Notes |
|--------|-------|-------|
| DMControl | 21 | Locomotion, classic continuous control |
| DMControl Extended | 16 | Harder locomotion variants |
| Meta-World | 49 | Tabletop robot manipulation |
| ManiSkill3 | 36 | Dexterous manipulation |
| MuJoCo | 6 | Classic physics tasks |
| MiniArcade | 19 | New: arcade-style 2D game tasks (this work) |
| Box2D | 8 | 2D physics |
| RoboDesk | 6 | Desktop manipulation |
| OGBench | 12 | Diverse offline-to-online |
| Atari | 27 | Discrete arcade games |

All tasks include:
- Low-dimensional state observations (128-dim)
- Language instructions (embodiment + action space + task description)
- 10–40 demonstrations collected by single-task TD-MPC2 agents
- Optionally: 224×224 RGB image observations
- Task-specific discount factors γ (domain defaults)

### Language Instructions

Each task receives a natural language instruction describing the embodiment (e.g., "Quadruped (ant) with 8 controllable joints (4 legs)") and the objective (e.g., "Push the soccer ball to the goal location"). Language is embedded with frozen **CLIP-ViT/B** (512-dim). Language is necessary because many tasks (e.g., RoboDesk manipulations) share the same observation space and cannot be distinguished without task identity. Language generalizes to unseen tasks; task indices do not.

### Infrastructure

- Docker image + async environment wrappers (stepping, rendering, batched frame-stacking, cached language embeddings, auto-reset)
- Reduces wall-time dramatically; supports 200 asynchronous environments across 10 simulators/engines
- 200+ single-task TD-MPC2 checkpoints released as behavior policies and baselines

---

## Newt Architecture

Newt is a **self-predictive control-centric world model** based on TD-MPC2, extended with language and optional image conditioning. All learnable components are MLPs (except the Gaussian policy prior); frozen pretrained backbones handle language and image encoding.

$$\begin{aligned}
\mathbf{g} &= \text{CLIP}_\text{text}(\mathbf{s}_\text{lang}) &&\triangleright\text{ language (frozen, 512-dim)} \\
\mathbf{x} &= \text{DINOv2/B}(\mathbf{s}_\text{img}) &&\triangleright\text{ image (optional, frozen, 768-dim)} \\
\mathbf{z} &= h(\mathbf{s}_\text{state}, \mathbf{x}, \mathbf{g}) &&\triangleright\text{ latent state encoder (MLP)} \\
\mathbf{z}' &= d(\mathbf{z}, \mathbf{a}, \mathbf{g}) &&\triangleright\text{ latent dynamics (MLP)} \\
\hat{r} &= R(\mathbf{z}, \mathbf{a}, \mathbf{g}) &&\triangleright\text{ reward head (cross-entropy on log-transformed bins)} \\
\hat{q} &= Q(\mathbf{z}, p(\mathbf{z},\mathbf{g}), \mathbf{g}) &&\triangleright\text{ terminal value (cross-entropy on log-transformed bins)} \\
\hat{\mathbf{a}} &= p(\mathbf{z}, \mathbf{g}) &&\triangleright\text{ policy prior (stochastic max-entropy Gaussian)}
\end{aligned}$$

**Discrete reward/value regression**: using cross-entropy on a log-transformed bin space rather than MSE allows modeling the wide range of reward distributions across tasks and domains with a single shared prediction head.

### World Model Training Objective

$$\mathcal{L}(\theta) \doteq \mathbb{E}_{\tau \sim \mathcal{B}}\!\left[\sum_{t=0}^H \lambda^t \!\left(\underbrace{\|\mathbf{z}'_t - \text{sg}(h(s'_{\text{state},t}, \mathbf{x}'_t, \mathbf{g}))\|_2^2}_\text{self-prediction} + \underbrace{\ell_\text{CE}(\hat{r}_t, r_t)}_\text{reward} + \underbrace{\ell_\text{CE}(\hat{q}_t, q_t)}_\text{value}\right)\right]$$

where λ ∈ [0,1] exponentially downweights temporally distant samples, and sg is stop-gradient (prevents representational collapse without a decoder).

### Policy Objective

$$\mathcal{L}_p(\theta) \doteq \mathbb{E}_{\tau \sim \mathcal{B}}\!\left[\sum_{t=0}^H \lambda^t \Big[\underbrace{\|p(\mathbf{z}_t,\mathbf{g}) - \mathbf{a}_t\|_2^2}_\text{model-based BC} - \underbrace{Q(\mathbf{z}_t, p(\mathbf{z}_t,\mathbf{g}), \mathbf{g})}_\text{Q-value} - \underbrace{\mathcal{H}(p(\cdot|\mathbf{z}_t,\mathbf{g}))}_\text{entropy}\Big]\right]$$

Three terms: (1) behavior cloning toward expert actions, (2) Q-value maximization, (3) entropy maximization. The BC term (1) directly leverages demonstrations and regularizes the policy when Q-value estimation is inaccurate.

**Planning**: at inference time, TD-MPC2 selects actions by running CEM in latent space, warm-started by the policy prior p. Per-task discount factors γ are used; the target Q-value uses an EMA of online Q-networks.

---

## Leveraging Demonstrations (Four Strategies)

Exploration is a key challenge in massively multitask online RL. Newt uses demonstrations in four complementary ways:

1. **Model-based pretraining**: before any online RL, jointly minimize $\mathcal{L}(\theta) + \mathcal{L}_p(\theta)$ on demonstration data (Q-value term temporarily disabled so BC dominates). This provides task-aware representations and action priors.

2. **Constrained planning**: at the start of online RL, planner is biased toward the pretrained policy prior (not pure planning). This bias is linearly annealed to zero during the first 12% of training to avoid using an inaccurate value function early on.

3. **Oversampling of demonstrations**: separate replay buffers for demos and online interactions; 50% of each training batch sampled from each, ensuring demonstration data remains available regardless of online buffer capacity.

4. **Action supervision in policy updates**: the model-based BC term in $\mathcal{L}_p$ uses demonstrations as direct action supervision throughout training, not only during pretraining.

All four strategies together yield the best performance; any individual strategy provides benefit, but the combination is strictly best.

---

## Experimental Results

**Training**: 100M environment steps total across 200 tasks (state observations); 20M learnable parameters (default); replay buffer 10M; 2× RTX 5090, ~4.6 days.

### Q1: Multitask Online RL Performance

Newt outperforms all baselines (BC, PPO, FastTD3) on average across 200 tasks and is more data-efficient:

| Domain | Newt vs. baselines |
|--------|-------------------|
| DMControl, DMControl Ext., ManiSkill, MiniArcade | Clear Newt advantage |
| Meta-World | Newt best |
| MuJoCo, Box2D, Atari | Newt ≈ BC baseline |

MuJoCo, Box2D, Atari tasks have little in common beyond action space, limiting cross-task transfer. This is an open challenge.

### Q2: Demonstrations

All four demonstration strategies help. Ablating any one strategy degrades performance. Combined ("All") achieves highest average score on 200 tasks.

### Q3: Model Capabilities

**Scaling**:
- Model size: 2M → 5M → 20M → 80M parameters — clear benefit up to a point (20M is compute-optimal for 200 tasks)
- Batch size: 128 → 256 → 512 → **1024** — consistent improvement
- Both scale beneficially in multitask (unlike single-task RL where scaling is marginal); larger tasks require proportionally larger models and batches

**Language**: conditioning on language raises average score from 0.371 → **0.438** (normalized), with largest gains where observations alone cannot distinguish tasks (RoboDesk, Meta-World). Matches performance of task-index conditioning while enabling zero-shot generalization.

**Task transfer (20 unseen tasks)**:
- Zero-shot (pretrained Newt, no online RL): score **0.192** vs. 0.013 from scratch
- Fine-tuned (100k env steps): score **0.868** vs. 0.480 from scratch
- Unseen language instructions can inhibit zero-shot generalization (Table 1: "Push <unseen>" 0.3% vs. "Push <cube>" 21.0%), so transfer experiments use unseen instructions to measure true capability

**Open-loop control** (planning without environment feedback, horizons up to 48 steps — 16× training horizon of 3):
- Newt successfully plans over long horizons on most tasks
- Failure modes: drifting dynamics (Walker Walk, DMControl), inability to decelerate after reaching target (Lunar Lander, Box2D), stochastic elements (Assault, Atari)

**Visual RL** (state + 224×224 RGB, 30M steps fine-tuning):
- Overall: +0.004 (marginal)
- RoboDesk: +0.125 (significant boost for manipulation)
- Meta-World: +0.069
- DMControl: −0.029 (locomotion worse from vision)

### Q4: Ablations Summary

| Ablation | Effect |
|----------|--------|
| No language (task index instead) | −0.067 avg score |
| No language (none) | −0.067 avg score |
| Model size 2M (vs. 20M) | −0.12 avg score |
| Batch size 128 (vs. 1024) | −0.10 avg score |
| No pretraining | significant drop |
| Only one demo strategy | worse than all four combined |

---

## Limitations & Open Questions

- **Atari / Box2D ceiling**: tasks with few shared structures across the domain benefit little from multitask training; performance ≈ single-task BC
- **Language bottleneck**: CLIP embeddings from a fixed instruction per task do not generalize well to paraphrases or unseen concepts; token-level language learning would help
- **Visual RL cost**: storing a large replay buffer for 200 visual tasks requires substantial GPU memory; practical visual multitask RL remains challenging
- **Simple architecture**: all components are deterministic MLPs; Transformer or Diffusion Policy action heads may improve expressiveness
- **Pretraining scope**: Newt pretrains only on task demonstrations; large-scale pretraining across datasets before any RL is a promising direction
- **Open question**: Can massively multitask world models serve as foundation models for zero-shot generalization to entirely new task families, analogous to LLMs?

**Future directions recommended by authors**: visual RL improvements, token-level language conditioning, improved learning curricula (balance per-task progress), non-uniform task/subtrajectory sampling, Transformer/Diffusion architectures, and further environment/dataset scaling.

---

## Implementation Details

- Language encoder: CLIP-ViT/B (frozen, 512-dim)
- Image encoder: DINOv2/B (frozen, 768-dim)
- State obs: 128-dim; actions: 16-dim
- Learnable components: MLPs, 20M params default
- Replay buffer: 10M transitions, separate demo and online buffers (50/50 batch split)
- Training accelerated via `torch.compile` and async multi-GPU multi-process rollouts
- Baselines: PPO (cleanRL + language conditioning + per-task γ), FastTD3 (n-step=8 + language conditioning), BC (language-conditioned), single-task TD-MPC2 (5M steps each, 200 agents)
- Hardware: 2× RTX 5090 (state, 4.6 days), 2× RTX PRO 6000 (state+RGB, 4.7 days)

---

## See Also

- [Latent World Models](../concepts/latent-world-models.md) — concept covering self-predictive models (TD-MPC2 family) and JEPA-WMs
- [Model Predictive Control for Robot Learning](../concepts/model-predictive-control.md) — CEM planning used by Newt at inference
- [JEPA World Models (Terver et al. 2026)](terver_2026_jepa_wm.md) — contrasting approach: frozen SSL encoder + action-conditioned predictor
- [V-JEPA 2 (Assran et al. 2025)](assran_2025_vjepa2.md) — large-scale video SSL world model for zero-shot manipulation
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — PPO baseline used in comparisons
- [Zhou et al. (2024) — DINO-WM](zhou_2024_dino_wm.md) — same frozen DINOv2 features but reward-free; its TD-MPC2 baseline scores 0 without rewards
