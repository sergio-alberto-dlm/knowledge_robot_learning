---
title: Video-Action Models (VAMs)
type: concept
tags: [robot-learning, video-action-model, imitation-learning, flow-matching, video-generation, inverse-dynamics, sample-efficiency, manipulation]
related: [concepts/vision-language-action-models.md, concepts/video-ssl.md, concepts/latent-world-models.md, papers/pai_2025_mimic_video.md, papers/assran_2025_vjepa2.md]
created: 2026-05-18
updated: 2026-05-18
sources: [raw/papers/pdf/mimic_video.pdf]
---

# Video-Action Models (VAMs)

## Core Idea

A **Video-Action Model (VAM)** is a robot policy that grounds control in a pretrained **generative video model backbone** rather than a vision-language model (VLM). The key hypothesis: internet-scale video inherently encodes physical dynamics, object deformation, temporal causality, and procedural structure — information that static image-text data is blind to. By leveraging this prior, the action decoder is reduced to a simple **Inverse Dynamics Model (IDM)**: given a visual plan (future video latents), recover the motor commands that would produce it.

This decouples the two hardest problems in robot learning:
1. **What should happen next?** → Handled by video pretraining (cheap, internet-scale)
2. **How do I move to make it happen?** → Handled by action decoder (lightweight, few hours of robot data)

## Contrast with VLAs

| Property | VLA | VAM |
|---|---|---|
| Backbone pretraining data | Static image-text pairs | Video (temporal, causal, dynamic) |
| Physical dynamics in backbone | ✗ Must be inferred from robot data | ✓ Encoded during video pretraining |
| Action decoder data requirement | Large-scale robot demonstrations | Small-scale (10× fewer demos) |
| Convergence speed | Slower (dynamics learned from scratch) | 2× faster |
| Backbone modality | Discriminative (image features) | Generative (latent video prediction) |
| Key limitation | Dynamics bottleneck; data-hungry | Single-view video; no cross-embodiment model yet |

## Architecture Pattern

A VAM has two stages trained with independent flow schedules:

**Stage 1 — Video Backbone** (pretrained, lightly finetuned):
- Generative video model (e.g., latent DiT like Cosmos-Predict2) finetuned with LoRA on a robot video corpus
- Language-conditioned via T5 cross-attention
- Given context frames + language instruction, predicts a future trajectory in latent video space
- The backbone is kept **frozen** during action decoder training

**Stage 2 — Action Decoder / IDM** (trained from scratch, lightweight):
- Flow matching DiT conditioned on intermediate video model representations h^{τ_v} (activations from a specific layer at video noise level τ_v)
- Also conditioned on proprioceptive state q_t
- Produces chunked action trajectories A_t via denoising
- Independently trained on task-specific robot data (as few as ~500 episodes)

## Partial Denoising Trick

A key design choice is the **video flow time** τ_v ∈ [0,1], which controls how much the video backbone denoises the future:
- τ_v = 0: full video reconstruction (highest fidelity, slowest, worst for policy due to distribution shift)
- τ_v = 1: no denoising — pure noise input, single forward pass, fastest
- τ_v ∈ (0,1): intermediate denoising

**Empirical finding** (mimic-video): τ_v = 1 (single forward pass with Gaussian noise as "future") achieves the highest autonomous policy performance. The intermediate representations h^{τ_v} at high noise levels are information-richer for action decoding than fully denoised latents, because fully reconstructed video can diverge from the training distribution and introduce artifacts.

This means: in practice a VAM needs only one forward pass through the video backbone per action chunk — fast enough for real-time control.

## Why Video Representations Outperform VLM Representations

The mimic-video case study (§III) demonstrates this directly: an action decoder conditioned on **ground-truth future video latents** achieves near-perfect success regardless of whether the video backbone was finetuned to the target domain. This confirms that:
- A high-quality video backbone provides all the information needed for action decoding
- The domain gap can be closed efficiently via LoRA finetuning on a modest video corpus
- The policy's performance scales **directly** with video model quality (not robot data quantity)

## Sample Efficiency Advantage

Because the video backbone handles all dynamics learning:
- The action decoder is a much simpler function to learn (IDM: visual plan → motor command)
- **10× fewer robot demonstrations** are needed to match VLA performance
- The decoder can be trained from scratch on task-specific data in hours, not days
- At 1 episode per task (extremely low-data regime), VAMs still achieve ~77% success (vs. VLA's much lower performance at the same data level)

## Relationship to Other Paradigms

**vs. Latent World Models (V-JEPA 2-AC, JEPA-WM)**: Both use video/latent predictions for robot control, but differently. World models predict future states for **planning** (CEM search over actions); VAMs use video latents **reactively** as conditioning for an IDM — no explicit planning at test time. VAMs are faster at inference; world models allow longer-horizon optimization.

**vs. Video Policy Learning (Video Prediction + IDM, UniPi, CoT-VLA)**: Prior work generating pixel-level future video requires full video synthesis per action step — computationally prohibitive. VAMs use intermediate latent representations (partial denoising), bypassing full pixel reconstruction entirely.

**vs. VLAs with auxiliary video signals**: Some VLAs use video-derived signals (affordances, keypoints, language plans) as auxiliary conditioning. VAMs differ by making the video backbone the **primary** computational backbone, not an auxiliary signal.

## Open Questions

- Can a single VAM be trained cross-embodiment (diverse robots, tasks) to achieve VLA-level semantic generalization?
- How does multi-view video backbone conditioning improve occlusion robustness?
- Does VAM performance scale with video foundation model size (analogous to LLM scaling)?
- Can VAMs be improved via RL fine-tuning of the action decoder (analogous to RECAP/RLT for VLAs)?

## See Also

- [mimic-video (Pai et al., 2025)](../papers/pai_2025_mimic_video.md) — first VAM system; Cosmos-Predict2 backbone + flow-matching IDM; SOTA on SIMPLER and bimanual dexterous
- [Vision-Language-Action Models (VLAs)](vision-language-action-models.md) — the contrasted paradigm; static image-text backbone; strong semantic generalization but dynamics bottleneck
- [Latent World Models](latent-world-models.md) — video-latent prediction for planning; complementary approach to VAM's reactive IDM
- [Video Self-Supervised Learning](video-ssl.md) — discriminative video pretraining (V-JEPA); VAMs use generative video pretraining (Cosmos-Predict2)
- [V-JEPA 2 (Assran et al., 2025)](../papers/assran_2025_vjepa2.md) — discriminative video model for robot control via latent planning; contrasts with mimic-video's generative backbone
