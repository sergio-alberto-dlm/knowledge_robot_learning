---
title: "mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs"
type: paper
tags: [robot-learning, video-action-model, flow-matching, imitation-learning, video-generation, dexterous-manipulation, manipulation, inverse-dynamics, sample-efficiency]
related: [concepts/video-action-models.md, concepts/vision-language-action-models.md, concepts/latent-world-models.md, concepts/video-ssl.md, concepts/dexterous-manipulation.md, papers/amin_2025_pi06_experience.md, papers/xu_2025_rlt.md, papers/assran_2025_vjepa2.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/mimic_video.pdf]
---

# mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

**Authors:** Jonas Pai\*, Liam Achenbach\*, Victoriano Montesinos, Benedek Forrai, Oier Mees†, Elvis Nava†
**Affiliations:** mimic robotics, Microsoft Zurich, ETH Zurich, ETH AI Center, UC Berkeley
**arXiv:** 2512.15692v2 [cs.RO], December 19, 2025

---

## Abstract

Prevailing VLA models for robotic manipulation are built on VLM backbones pretrained on large-scale static image-text data. Despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories — creating a data burden requiring continuous large-scale expert teleoperation. mimic-video contends that video jointly captures semantics and **visual dynamics** during pretraining, isolating the remaining task of low-level control. It introduces a **Video-Action Model (VAM)**: a pretrained Internet-scale video model paired with a flow matching-based action decoder conditioned on intermediate video latent representations. The decoder acts as an **Inverse Dynamics Model (IDM)**, generating low-level motor commands from video-space action plans. The result: state-of-the-art manipulation performance with **10× greater sample efficiency** and **2× faster convergence** compared to VLA architectures.

---

## Key Contributions

- **Video-Action Model (VAM)** — a new class of robot policy that decouples visual dynamics learning (handled by video pretraining) from low-level control (handled by a lightweight action decoder), eliminating the bottleneck that forces VLAs to learn physics from scarce robot data.
- **Partial denoising strategy** — the video backbone follows the flow to an intermediate time τ_v (not full reconstruction), extracting rich intermediate latent representations that condition the action decoder; this is faster and empirically more accurate than full video generation.
- **Counterintuitive finding**: τ_v = 1 (pure noise input to video backbone — a single forward pass, no actual denoising) achieves the highest average autonomous policy performance, providing both maximum speed and best action decoder accuracy.
- **10× sample efficiency**: mimic-video with 10% of training data matches VLA performance with 100%.
- **SOTA on bimanual dexterous manipulation**: 72%/93% success on real-world packing and package-handover tasks, versus 11%/30% for the best single-task DiT-Block baseline.

---

## Methodology

### Motivation: Video Representations Carry Physical Dynamics

The paper first establishes (§III) that video model representations are fundamentally superior conditioning signals for robot control compared to VLM representations. Using an "oracle" case study: an action decoder conditioned on **ground-truth future video latents** achieves near-perfect success rates regardless of whether the underlying video backbone was finetuned to the target domain. This confirms that:
- Policy performance scales directly with video model quality
- The action decoder's role is reduced to a simple "visual plan → motor command" translation
- The burden of learning shifts from expensive robot action data to cheap video data

### Architecture

mimic-video couples two Conditional Flow Matching (CFM) components with **independent flow schedules**:

**Video Model** — Cosmos-Predict2 (2B latent Diffusion Transformer):
- Open-source; operates on video frames encoded by a pretrained 3D-tokenizer
- Input: concatenation of clean latent patches from a **5-frame context prefix** and "noisy" future latent patches
- Each transformer layer: (1) self-attention over full video sequence, (2) cross-attention to T5-encoded language instruction, (3) 2-layer MLP
- Finetuned with **LoRA** on a 200-hour robot video corpus to capture domain-specific visual dynamics while preserving temporal reasoning

**Action Decoder** — lightweight DiT with frozen video backbone:
- Cross-attends to intermediate video representations h^{τ_v} (activations from layer k of the video model)
- Self-attends over the predicted action sequence
- Encodes proprioceptive state q_t and action chunks A_t via separate MLPs
- **AdaLN** modulation conditioned on both τ_v (video noise level) and τ_a (action noise level)
- Trained from scratch on task-specific robot data (as few as 480–512 episodes)

### Flow Matching Background

Both components are trained with the CFM objective:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\mathcal{T}, p_0(x^0), p_\tau(x^\tau|x^0)} \left\| v_\theta(\hat{x}^\tau, \tau) - u_\tau(x^\tau \mid x^0) \right\|^2$$

where the conditional optimal transport path is $x^\tau = (1-\tau)x^0 + \tau\varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$, and $u_\tau(x^\tau|x^0) := \frac{d}{d\tau}x^\tau = \varepsilon - x^0$ is the conditional generating vector field.

### Partial Denoising and Inference (Algorithm 1)

The key insight is **partial denoising**: instead of integrating τ from 1→0 (full generation), stop at an intermediate τ_v > 0:

1. Initialize: z^0_past (5 context frames), q_t (proprioception), l (language), z^1_future ∼ N(0,I)
2. Partially denoise video: $\mathbf{z}^{\tau_v}_{\text{future}} = \mathbf{z}^1_{\text{future}} + \int_1^{\tau_v} v_\phi(\mathbf{z}^0_{\text{past}}, \mathbf{z}^{\tau'_v}_{\text{future}}, l, \tau'_v)\,d\tau'_v$
3. Extract intermediate representations: $\mathbf{h}^{\tau_v} = v_\phi^{(k)}(\mathbf{z}^0_{\text{past}}, \mathbf{z}^{\tau_v}_{\text{future}}, l, \tau_v)$
4. Decode actions: $\mathbf{A}^0_t \leftarrow \mathbf{A}^1_t + \int_1^0 \pi_\theta(\mathbf{A}^{\tau_a}_t, \mathbf{q}_t, \mathbf{h}^{\tau_v}, \tau_a, \tau_v)\,d\tau_a$

At τ_v = 1 (pure noise, no actual denoising), step 2 is skipped and line 3 of Algorithm 1 becomes redundant, so a **single forward pass** through the video backbone suffices — fastest inference, highest average performance.

### Training Protocol

**Phase 1** — Video backbone finetuning:
- LoRA adaptation of Cosmos-Predict2 on 200-hour multi-robot video corpus
- Flow matching objective on video sequences; τ_v ∼ logit-normal distribution
- Preserves temporal reasoning; captures domain-specific visual dynamics

**Phase 2** — Action decoder training (frozen video backbone):
- Flow matching on action chunks; τ_a distribution ∝ √(τ_a − 0.001) following π₀ convention
- Randomly masks proprioceptive state token (masked token) to prevent overfitting on low-dimensional observation
- Independent from Phase 1 — decoupled training enables data-efficient specialization

### Baselines

- **π₀.₅-style VLA**: PaliGemma 3B backbone + identical action decoder (fair comparison to isolate video vs. VLM backbone); trained with Knowledge-Insulation (KI) protocol
- **DiT-Block Policy**: ViT-S DINO backbone + 8-block 8-head diffusion policy; ~155M params; single/multi-view variants
- **State-of-the-art published**: Octo, ThinkAct, FLOWER, OpenVLA, OpenVLA-OFT, Diffusion Policy

---

## Experimental Results

### SIMPLER-Bridge (Table I)

Evaluates cross-task generalization on BridgeDataV2 (Widow-X robot) via SIMPLER visual matching simulator.

| Model | Put Carrot | Put Spoon | Stack Blocks | Eggplant | **Avg SR (%)** |
|---|---|---|---|---|---|
| OpenVLA (finetuned) | 4.2 | 8.3 | 0.0 | 45.8 | 14.6 |
| FLOWER (finetuned) | 13.0 | **71.0** | 8.0 | 88.0 | 45.0 |
| π₀.₅-style VLA (scratch) | 25.0 | 29.2 | **20.8** | 66.7 | 35.4 |
| **mimic-video (scratch)** | **37.5** | 37.5 | 12.5 | **100.0** | **46.9** |
| mimic-video (τ_v tuned) | 54.2 | 41.7 | 29.2 | 100.0 | **56.3** |

mimic-video trained from scratch surpasses all finetuned baselines except FLOWER on the spoon task, and inference-time τ_v optimization provides an additional +9.4%.

### LIBERO (Table II)

Multi-task tabletop manipulation (10 tasks × 3 suites, 50 expert demos each).

| Model | Spatial (%) | Object (%) | Goal (%) | **Avg (%)** |
|---|---|---|---|---|
| OpenVLA-OFT (finetuned) | **96.2** | **98.3** | **96.2** | **96.9** |
| π₀.₅-style VLA (scratch) | 79.2 | 94.0 | 84.0 | 85.9 |
| **mimic-video (scratch)** | 94.2 | 96.8 | 90.6 | **93.9** |

mimic-video trained from scratch outperforms all scratch baselines and most finetuned methods; only lags finetuned OpenVLA-OFT.

### Real-World Bimanual Dexterous (Table III)

Franka Emika Panda arms + mimic 16-DoF dexterous humanoid hands. Two long-horizon tasks (Package Sorting, Tape Stowing).

| Model | Packing (%) | Package Handover (%) |
|---|---|---|
| DiT-Block Policy | 11.0 | 30.0 |
| DiT-Block Policy (+ wrist cams) | 42.6 | 74.1 |
| **mimic-video** | **72.0** | **93.0** |

mimic-video significantly outperforms the single-task baseline even from a single workspace camera, demonstrating robustness to occlusion via video dynamics priors.

### Sample Efficiency (Fig. 5–6)

- At **10% of LIBERO data**, mimic-video matches the peak success rate the VLA baseline achieves at **100%** data → **10× sample efficiency**
- mimic-video's action decoder **converges 2× faster** to a higher asymptotic performance than the π₀.₅-style VLA
- At 1 episode per task (2% of data), mimic-video still achieves **77% average success**

### Video Fidelity vs. Action Performance (Fig. 7–8)

Sweeping τ_v reveals a counterintuitive result:
- **Action reconstruction MSE** is minimized at τ_v ≈ 0.4 (intermediate noise) when conditioned on ground-truth video latents
- **Autonomous policy performance** peaks at τ_v = 1 (pure noise, single forward pass) on SIMPLER
- Fully reconstructed video (τ_v → 0) introduces distribution shift artifacts that hurt performance
- The intermediate representations h^{τ_v} at high noise levels are actually information-richer for action decoding than fully denoised latents

---

## Limitations & Open Questions

- **Single camera view**: mimic-video currently relies on one workspace camera, limiting spatial reasoning and occlusion handling; multi-view video architectures are a natural extension.
- **No unified cross-embodiment model**: The VAM recipe has not yet been applied to train a single model across diverse robots; this step is expected to unlock VLA-level generalization.
- **Limited real-world task diversity**: Real evaluations cover only 2 bimanual tasks; broader manipulation behavior coverage remains future work.
- **Backbone dependency**: Performance is tied to Cosmos-Predict2 quality; improvements in open-source video models should directly and automatically improve VAM policies.
- **τ_v hyperparameter**: Optimal τ_v is task-dependent (though τ_v=1 is a strong default); tuning it per task adds modest inference cost but yields further gains.

---

## See Also

- [Video-Action Models (VAMs)](../concepts/video-action-models.md) — the new paradigm class this paper introduces
- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md) — the contrasted paradigm; VLAs use static VLM backbones, must learn dynamics from robot data
- [Video Self-Supervised Learning](../concepts/video-ssl.md) — pretraining paradigm for video backbones; mimic-video leverages generative video pretraining rather than discriminative SSL
- [Latent World Models](../concepts/latent-world-models.md) — related paradigm of latent video prediction for planning; mimic-video uses video latents reactively (IDM) rather than for planning
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — bimanual dexterous evaluation setting
- [Amin et al. (2025) — RECAP](amin_2025_pi06_experience.md) — RL fine-tuning of VLAs; contrasting improvement direction from mimic-video's pretraining approach
- [Assran et al. (2025) — V-JEPA 2](assran_2025_vjepa2.md) — action-conditioned discriminative video model for robot control; mimic-video uses generative video (Cosmos-Predict2) instead
