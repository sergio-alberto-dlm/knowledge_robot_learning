---
title: Vision-Language-Action Models (VLAs)
type: concept
tags: [robot-learning, vla, foundation-models, imitation-learning, manipulation, flow-matching, behavior-cloning, online-rl, rl-finetuning]
related: [papers/amin_2025_pi06_experience.md, papers/xu_2025_rlt.md, concepts/sim-to-real.md, concepts/dexterous-manipulation.md, concepts/policy-gradient-methods.md, concepts/latent-world-models.md, concepts/video-action-models.md, papers/pai_2025_mimic_video.md, open_questions/jepa-rl-humanoid-grasping.md]
created: 2026-05-17
updated: 2026-09-25
sources: [raw/papers/pdf/pi_06_experience.pdf, raw/papers/pdf/rlt.pdf, raw/papers/pdf/mimic_video.pdf]
---

# Vision-Language-Action Models (VLAs)

## Definition

A **Vision-Language-Action (VLA) model** is a robot policy that maps visual observations (camera images) and natural language task instructions directly to robot actions, by leveraging large pretrained vision-language model (VLM) backbones. VLAs bring the semantic understanding and generalization of internet-scale pretraining into robot control.

## Architecture

A typical VLA has two major components:

**1. VLM backbone**: A pretrained vision-language model (e.g., SigLIP + Gemma) encodes images and language into shared semantic representations. Provides object recognition, scene understanding, instruction following, and generalization to novel configurations.

**2. Action expert / policy head**: Translates VLM representations into low-level robot commands. Modern implementations use **flow matching** for continuous action chunks, producing expressive multi-modal distributions over joint trajectories at 50 Hz. Discrete outputs (sub-task tokens, predicted next step) are generated autoregressively via a tokenizer (e.g., FAST).

The π₀ family (Black et al., 2024 onward) uses:
- SigLIP (400M) + Gemma 3 (4B) backbone
- 860M-parameter flow-matching action expert with **stop-gradient** between backbone and action expert (Knowledge Insulation / KI recipe)
- Predicts both continuous action chunks and tokenized sub-task descriptions for high-level planning

## Training Paradigm

VLAs are trained primarily with **behavior cloning** from large heterogeneous offline demonstration datasets:

1. **Pre-training**: Imitation learning on tens of thousands of hours of multi-task, multi-robot demonstrations → broad capabilities, strong generalization
2. **Supervised Fine-Tuning (SFT)**: Fine-tune on target task demonstrations → task-specific precision
3. **RL improvement (optional)**: Collect autonomous rollouts, label with sparse reward, improve via advantage conditioning (RECAP) or PPO variants → surpass imitation ceiling

The SFT stage with `I_t = True` (all demonstrations treated as optimal) effectively means the model specializes to a task while retaining the RL-compatible conditioning interface.

## Key Architectural Trade-offs

| Choice | Option A | Option B |
|--------|----------|----------|
| Action distribution | Flow matching (expressive, no closed-form likelihood) | Diffusion (similar, slower inference) |
| Policy head coupling | Stop-gradient (KI) — stable pretraining | End-to-end — potentially more expressive |
| High-level reasoning | Discrete sub-task tokens | Pure reactive (no intermediate language) |
| Action representation | Chunked (H steps at once) | Step-by-step |

## RL Fine-tuning of VLAs

A growing body of work improves VLAs beyond the imitation ceiling via online RL. Methods differ on *what* is updated and *how* the RL signal is incorporated:

| Method | What is Updated | RL Signal | Key Idea |
|--------|----------------|-----------|----------|
| RECAP (Amin et al., 2025) | Full VLA (offline) | Advantage conditioning (I_t token) | Large-scale offline RL; binarized advantage token injected into VLA; distributional value function |
| GR-RL (Li et al., 2025) | Full VLA (online PPO then SFT) | PPO + offline BC | Multi-stage: filtered BC → online RL → distill back via SFT; long-horizon shoe lacing |
| **RLT (Xu et al., 2025)** | Lightweight actor-critic only; VLA frozen | Sparse binary + BC regularizer | RL token compresses VLA internals → small MLP actor-critic; ~hours of real data; up to 3× speedup |
| ConRFT (Chen et al., 2025) | Action head only; VLA encoder frozen | Consistency-based binary reward classifier | Fine-tunes action head on single-step actions with learned reward |
| DSRL (Wagenmaker et al., 2025) | Latent noise policy; VLA frozen | Sparse reward | RL in diffusion noise space; constrained to VLA's action modes |
| PLD (Xiao et al., 2025) | Residual policy (offline → distill) | Cal-QL offline pre-training | Single-step residual distilled back into VLA via SFT |

**Key design axes** (RLT vs. RECAP comparison):
- *Representation*: RLT compresses VLA internals to a task-specific RL token; RECAP uses the full VLA feature stream
- *Update scope*: RLT freezes VLA → lightweight and fast; RECAP updates the full 5B+ model → more expressive but slower
- *Action interface*: RLT uses chunked actions aligned with VLA's native interface; RECAP uses advantage-conditioned VLA decoding
- *Data regime*: RLT targets hours of real-robot data; RECAP targets tens of hours of large-scale offline data

## Limitations

- **Compounding errors**: Imitation learning degrades when encountering out-of-distribution states; RL and DAgger-style corrections are required to surpass the demonstration ceiling.
- **Reward sparsity**: Real-world RL requires scalable reward labeling; current practice relies on human binary success annotations.
- **Exploration**: Policy stochasticity alone is insufficient for discovering new behaviors; human interventions or curiosity bonuses are needed.
- **Flow matching + RL**: No closed-form action likelihood means policy gradient requires a lower-bound surrogate (e.g., single-step diffusion approximation).
- **Sample complexity**: Long-horizon real-world tasks require many iterations of data collection before significant RL improvement.
- **Critical-phase identification**: lightweight RL methods (RLT) require manual identification of the precision-critical task segment per task.

## Relationship to Other Paradigms

| Paradigm | Distinction |
|----------|------------|
| Latent world models (JEPA-WM) | Planning-based: optimize actions against a world model; VLAs are reactive policies |
| Sim-to-real RL | VLAs train directly on real robot data; sim-to-real bridges the gap via domain randomization |
| Dexterous RL | VLAs operate at the policy level end-to-end; dexterous methods often use modular affordance + control pipelines |
| **Video-Action Models (VAMs)** | Generative video backbone (encodes dynamics) + lightweight IDM decoder; 10× fewer robot demos needed vs. VLAs |

## See Also

- [RECAP — π*₀.₆](../papers/amin_2025_pi06_experience.md) — RL improvement method for VLAs via advantage conditioning (full-model offline RL)
- [RLT — RL Token](../papers/xu_2025_rlt.md) — lightweight online RL fine-tuning via compact VLA readout representation
- [Policy Gradient Methods](policy-gradient-methods.md) — RL algorithms applicable to VLA fine-tuning (PPO, advantage conditioning)
- [Latent World Models](latent-world-models.md) — complementary paradigm: plan in latent space rather than learn a reactive policy
- [Sim-to-Real Transfer](sim-to-real.md) — alternative training environment for robot policies
- [Dexterous Manipulation](dexterous-manipulation.md) — dexterous tasks where VLA-style systems are evaluated
- [Video-Action Models (VAMs)](video-action-models.md) — contrasting paradigm using generative video backbone instead of VLM; eliminates the dynamics bottleneck
- [mimic-video (Pai et al., 2025)](../papers/pai_2025_mimic_video.md) — first VAM; demonstrates 10× sample efficiency and 2× faster convergence over VLAs
- [JEPA World Models as Physics Priors for RL](../open_questions/jepa-rl-humanoid-grasping.md) — open research hypothesis arguing a JEPA world model is a better-aligned RL prior than a VLA
