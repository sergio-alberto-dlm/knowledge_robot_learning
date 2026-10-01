---
title: "π*₀.₆: a VLA That Learns From Experience (RECAP)"
type: paper
tags: [robot-learning, vla, reinforcement-learning, advantage-conditioning, imitation-learning, real-world-rl, flow-matching, value-function]
related: [concepts/vision-language-action-models.md, concepts/policy-gradient-methods.md, concepts/value-based-rl.md, concepts/sim-to-real.md, concepts/dexterous-manipulation.md, concepts/intrinsic-motivation.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/pi_06_experience.pdf]
---

# π*₀.₆: a VLA That Learns From Experience (RECAP)

**Authors:** Ali Amin et al. (Physical Intelligence)  
**Venue:** arXiv 2511.14759v2 [cs.LG], 19 Nov 2025  
**Blog:** https://pi.website/blog/pistar06

## Abstract

This paper presents **RECAP** (RL with Experience and Corrections via Advantage-conditioned Policies), a general-purpose method for improving vision-language-action (VLA) models through real-world deployments via reinforcement learning. RECAP incorporates heterogeneous data: offline demonstrations, on-policy autonomous rollouts, and expert teleoperated interventions that correct mistakes. Starting from a pre-trained generalist VLA (π*₀.₆, an RL-ready extension of π₀.₆), RECAP specializes the model to high-performance on downstream tasks. The trained model more than doubles task throughput and roughly halves failure rates on the hardest tasks compared to the pre-trained π₀.₆ baseline — enabling 13 hours of continuous espresso making and 2+ hours of diverse laundry folding without interruption.

## Key Contributions

- **RECAP framework**: An iterated offline RL method for VLA training via advantage conditioning — scalable, off-policy, and compatible with flow-matching action distributions.
- **Distributional value function**: A large multi-task value function (670M-param VLM backbone, 201 bins) that predicts time-to-completion and detects failures, trained jointly on robot data and web data.
- **Advantage-conditioned policy extraction**: Conditions the policy on a binarized improvement indicator I_t ("Advantage: positive/negative" text token) instead of policy gradient ratios. Provably equivalent to regularized RL improvement; enables CFG-style sharpening at test time.
- **π*₀.₆ model**: Extension of π₀.₆ that additionally conditions on the binarized advantage indicator, making it suitable as both a pre-trained generalist and a base for downstream RL fine-tuning.
- **Heterogeneous data integration**: Seamlessly combines demonstrations, autonomous rollouts (good and bad), and human corrections in a single supervised objective.
- **Realistic long-horizon evaluation**: Tasks lasting 5–15 minutes — diverse laundry (11 item types), espresso making (6 sub-tasks), factory box assembly (6 sub-tasks) — rather than toy pick-and-place.

## Methodology

### Problem Setting

Standard RL with trajectory ρ_π(τ) = p(o₀) ∏ π(aₜ|oₜ) p(oₜ₊₁|oₜ,aₜ). Undiscounted return R(τ) = Σ rₜ. N-step advantage:

A^π(oₜ, aₜ) = E[Σ_{t'=t}^{t+N-1} r_{t'} + V^π(o_{t+N})] − V^π(oₜ)

**Sparse reward** (normalized by max episode length per task):

```
rₜ = 0         if t = T and success
     -C_fail   if t = T and failure
     -1         otherwise
```

The value function thus approximates the (negative) fraction of episode steps remaining to success.

### Model Architecture

**π*₀.₆ VLA:**
- VLM backbone: SigLIP (400M) + Gemma 3 4B
- Action expert: 860M-param flow matching network with stop-gradient (KI recipe, Springenberg et al.)
- Produces continuous action chunks at 50 Hz (joint positions + gripper) **and** discrete sub-task tokens (via FAST tokenizer)
- Advantage conditioning: "Advantage: positive" or "Advantage: negative" injected as text input after task prompt ℓ and before action prediction; only the action log-likelihoods are affected

**Value function (V^πref):**
- Same architecture, smaller VLM backbone: 670M-param Gemma 3
- Distributional: V_φ(V|oₜ, ℓ) ∈ Δ_B with B = 201 bins
- Training loss: cross-entropy H(R_t^B(τ), p_φ(V|oₜ, ℓ)) over discretized Monte Carlo returns
- Co-trained on a small mixture of multi-modal web data to prevent overfitting

### RECAP Algorithm

```
Require: multi-task demonstration dataset D_demo
1.  Train V_pre    on D_demo                          (Eq. 1)
2.  Train π_pre    on D_demo with V_pre               (Eq. 3)
3.  Initialize D_ℓ ← demonstrations for task ℓ
4.  Train V⁰_ℓ    from V_pre on D_ℓ
5.  Train π⁰_ℓ    from π_pre on D_ℓ, Iₜ=True        (SFT stage)
For k = 1 … K:
6.  Collect rollouts with π^{k-1}_ℓ → add to D_ℓ
7.  Train V^k_ℓ   from V_pre on D_ℓ
8.  Train π^k_ℓ   from π_pre on D_ℓ using V^k_ℓ
```

Key design choice: always fine-tune from the **pre-trained checkpoint** (not from the last iteration), preventing representational drift over multiple RL iterations.

### Policy Extraction via Advantage Conditioning

From regularized RL theory, the optimal policy given a value function satisfies:

π̂(a, o, ℓ) ∝ π_ref(a|o, ℓ) · p(I | A^πref(o, a, ℓ))^β

For β = 1 and binary I_t = 𝟙(A^πref(oₜ, aₜ, ℓ) > ε_ℓ), this reduces to π̂(a|o, ℓ) = π_ref(a|I, o, ℓ). No ratio computation needed; just train the policy to model both distributions.

**Training objective** (Eq. 3):

min_θ E_{D_πref} [ −log π_θ(aₜ|oₜ, ℓ) − α log π_θ(aₜ|Iₜ, oₜ, ℓ) ]

where Iₜ = 𝟙(A^πref(oₜ, aₜ, ℓ) > ε_ℓ) and α trades off unconditional and conditional losses.

**Advantage threshold ε_ℓ**: 30th percentile of predicted values during pre-training; 40th percentile during fine-tuning. (For high-quality but slow demonstration data, set to 10th percentile to emphasize speed.)

**CFG at test time**: Setting β > 1 sharpens the policy at inference without additional training, by re-weighting actions proportionally to how much better conditional > unconditional log-likelihood.

**Flow matching lower bound**: Since continuous actions use flow matching (no closed-form likelihood), the loss combines discrete action cross-entropy with the diffusion/flow lower bound (Lipman et al., Kingma & Gao):

log π_θ(aₜ:H | Iₜ, oₜ, ℓ, ℓ̂) ≥ E_{η,ω}[−w(η) ‖ω − aₜ:H − f_θ(a^{η,ω}_{1:H}, Iₜ, oₜ, ℓ, ℓ̂)‖²] + c

**Advantage conditioning dropout**: 30% of training steps randomly omit Iₜ, replacing it with unconditional mode — this enables CFG-style inference and is equivalent to treating α as stochastic.

### Human Interventions (DAgger-style)

Expert teleoperators monitor autonomous rollouts and intervene to correct mistakes (catastrophic failures, exploration traps). Correction episodes are appended to D_ℓ with I_t=True forced. Human corrections and autonomous RL are complementary: RL improves speed and robustness; corrections fix large mistakes that RL alone cannot explore past.

## Experimental Results

### Tasks

| Task | Duration | Items / Variants | Success Criteria |
|------|----------|-----------------|-----------------|
| Laundry T&S | ≤200s | T-shirts and shorts (2 types) | Folded, on stack, roughly rectangular |
| Laundry Diverse | ≤500s | 11 item types, eval on button-up shirts | Folded, on stack |
| Laundry Ablation | ≤200s | 1 orange T-shirt, fixed start | Strict: collar facing up, folded |
| Cafe (espresso) | ≤200s | Double shot, 6 sub-tasks | No drop, locked, dispensed, no spill |
| Box Assembly | ≤600s | 6 sub-tasks, factory scenario | Full build, stacked |

**Robot:** Static bimanual system, 2×6 DoF arms, parallel jaw grippers, 50 Hz, 3 cameras.

**Two metrics:** *Success rate* (proportion of successful episodes) and *throughput* (successes/hour) — throughput captures both success and speed in one metric.

### Main Quantitative Results (Figures 7–8)

| Task | π₀.₅ | π₀.₆ | OfflineRL+SFT | RECAP (ours) |
|------|------|------|--------------|-------------|
| Laundry T&S (throughput, /hr) | ~15 | ~15 | ~30 | **~60** |
| Laundry Diverse (throughput, /hr) | ~2 | ~3 | ~4 | **~10** |
| Espresso (throughput, /hr) | ~5 | ~7 | ~12 | **~25** |
| Box Assembly (throughput, /hr) | ~0 | ~1 | ~10 | **~14** |

- On all tasks except simple laundry, RECAP roughly doubles throughput from offline RL + SFT baseline
- Final π*₀.₆ achieves **>90% success rate** on all tasks except diverse laundry
- Failure rate reduces by approximately **2×** on hardest tasks

### Iterative Improvement (Figures 9–10)

- **Laundry T&S** (2 iterations, autonomous only, 300 trajectories/iteration/4 robots):
  - Iteration 1: success rate hits 90%+; throughput improves ~30%
  - Iteration 2: mainly further throughput gain; cumulative ~50% improvement
- **Box Assembly** (2 iterations, autonomous + corrections, 600 + 360 trajectories each):
  - Iteration 1: success improves but still some failures
  - Iteration 2: ~2× throughput improvement; ~90% success on all 4 sub-tasks

### Policy Extraction Comparison (Figure 11)

RECAP (advantage conditioning) > AWR > PPO on throughput for laundry T&S:
- **AWR**: achieves reasonable success rate but much slower policies (lower throughput)
- **PPO** (SPO variant, η=0.01): training stabilizes but policy performance is poor; off-policy setting is problematic even with small trust region

### Failure Mode Removal (Figure 12)

- Targeted laundry ablation (strict criterion: collar facing up)
- RL only, no corrections; 2 iterations × 600 trajectories
- Achieves **97% success rate** after 2 iterations (baseline offline RL + SFT: ~30%)
- Confirms RECAP can surgically remove failure modes even without additional demonstrations

## Limitations & Open Questions

- **Partial autonomy**: Requires human labeling of episode success, human corrections, and episode resets; automating these is a key open problem.
- **Greedy exploration**: Relies on policy stochasticity and human interventions; lacks principled exploration for discovering entirely new behaviors.
- **Batch (offline) updates**: Collects data in batches and retrains — not a fully concurrent online RL loop; extending to streaming updates is future work.
- **Sample complexity for long-horizon tasks**: Box assembly required 2 full iterations before significant improvement; harder tasks may require more.
- **Reward sparsity**: Binary episode success may be insufficient for tasks requiring nuanced progress signals.
- **Open question**: Can automated reward labeling (VLM judges, state detectors) replace human annotation at scale?
- **Open question**: How do more sophisticated exploration strategies (curiosity, RND, go-explore) interact with advantage conditioning in VLAs?

## See Also

- [Vision-Language-Action Models](../concepts/vision-language-action-models.md) — the VLA model family π*₀.₆ belongs to
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — PPO, advantage estimation, regularized RL foundations
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — value function training, distributional RL
- [Intrinsic Motivation & Curiosity-Driven RL](../concepts/intrinsic-motivation.md) — exploration methods that could complement RECAP
- [Sim-to-Real Transfer](../concepts/sim-to-real.md) — alternative approach: train in sim, deploy on real hardware
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md) — related manipulation tasks and challenges
