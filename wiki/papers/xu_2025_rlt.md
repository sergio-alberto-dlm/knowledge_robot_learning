---
title: "RLT: RL Token — Bootstrapping Online RL with Vision-Language-Action Models"
type: paper
tags: [robot-learning, vla, reinforcement-learning, online-rl, actor-critic, chunked-actions, real-world-rl, representation-learning, sample-efficiency]
related: [concepts/vision-language-action-models.md, papers/amin_2025_pi06_experience.md, concepts/policy-gradient-methods.md, concepts/value-based-rl.md, open_questions/jepa-rl-humanoid-grasping.md]
created: 2026-05-18
updated: 2026-09-25
sources: [raw/papers/pdf/rlt.pdf]
---

# RLT: RL Token — Bootstrapping Online RL with Vision-Language-Action Models

**Authors:** Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke  
**Affiliation:** Physical Intelligence  
**Venue:** arXiv, 2025  
**Project:** https://pi.website/research/rlt

---

## Abstract

VLA models can perform diverse manipulation skills out of the box but struggle with the last-millimeter precision that real-world tasks demand. RLT introduces a sample-efficient online RL fine-tuning method for pretrained VLAs that uses just a few hours of real-world practice. The method (1) adapts the VLA to expose an **RL token**, a compact readout representation that preserves task-relevant knowledge while serving as an efficient interface for online RL, and (2) trains a small actor-critic head on this token to refine actions. Across four real-robot tasks, RLT improves speed up to 3× and raises success rates significantly within minutes to a few hours of practice, surpassing human teleoperation speed on some tasks.

---

## Key Contributions

- **RL Token interface**: an encoder-decoder transformer added to a frozen VLA that compresses VLA internal features into a 1×2048 compact state representation for lightweight RL
- **Reference action conditioning**: the RL actor is conditioned on both the RL token *and* the VLA's sampled action chunk, turning online RL into local refinement around the VLA's prior rather than unconstrained search
- **Reference action dropout**: randomly zeros the reference chunk during training to prevent the actor from collapsing to VLA copy-paste, maintaining an independent action-generation pathway
- **Chunked off-policy actor-critic**: operates over action chunks (C=10 steps), shortening the effective TD horizon for sparse rewards at 50 Hz control; works with all data sources in a shared replay buffer
- Achieves 20%→65% success on screw installation, up to 3× faster execution on critical phases, and emerges strategies not present in demonstration data

---

## Method

### Problem Setup

- MDP: (S, A, p, r, γ) with continuous action space
- Reward: sparse binary — human labels end of episode as success (r=1) or failure (r=0)
- **Action chunks**: both policies and critics operate over chunks **a**_{t:t+C−1} ∈ ℝ^{C×d} (C=10, d=14); the chunk-level value function is:
  $$Q^\pi(\mathbf{s}_t, \mathbf{a}_{t:t+C-1}) = \sum_{\ell'=t}^{t+C-1} \gamma^{\ell'-t} r_{\ell'} + \gamma^C \mathbb{E}_{\mathbf{a}' \sim \pi|s_{t+C}}\bigl[Q^\pi(s_{t+C}, \mathbf{a}')\bigr]$$

### Stage 1: Adapting the VLA to Expose an RL Interface

The VLA (π₀.₆: SigLIP 400M + Gemma 4B + 860M action expert) is frozen after a light task-specific fine-tune. An encoder-decoder transformer is trained on task demonstrations:

- **RL token extraction**: let **z** = f(s, ℓ; θ_vla) be the final-layer token embeddings of the VLA (M tokens). Append a learned embedding **e**_{rl} = **e**_φ(<rl>) and process with encoder g_φ:
  $$\mathbf{z}_\text{rl} = g_\phi([\mathbf{z}_{1:M}, \mathbf{e}_\text{rl}])_{M+1}$$
  The RL token **z**_rl is the output at the special-token position — a 1×2048 vector.

- **Decoder reconstruction objective**: decoder d_φ autoregressively reconstructs the original VLA token embeddings from **z**_rl:
  $$\mathcal{L}_\text{ro} = \mathbb{E}_\mathcal{D}\left[\sum_{i=1}^M \bigl\|h_\phi(d_\phi([\mathbf{z}_\text{rl}, \hat{\mathbf{z}}_{1:i-1}]))_i - \hat{\mathbf{z}}_i\bigr\|^2\right]$$
  The bottleneck forces **z**_rl to retain enough information to regenerate VLA features — preserving task-relevant priors compactly.

- Optional: jointly fine-tune θ_vla on task demos with weight α (supervised VLA loss).

### Stage 2: Lightweight Online Actor-Critic

After Stage 1, both the VLA and RL token module are **frozen**. A small MLP actor π_θ and critic Q_ψ are trained online:

**RL state**: x = (**z**_rl(s), **s**^p) — RL token concatenated with proprioceptive state (joint positions, end-effector pose, etc.)

**Actor**: Gaussian over action chunks conditioned on RL state *and* VLA reference action chunk **ā**_{1:C}:
$$\pi_\theta(\mathbf{a}_{1:C} \mid \mathbf{x}, \bar{\mathbf{a}}_{1:C}) = \mathcal{N}\!\left(\mu_\theta(\mathbf{x}, \bar{\mathbf{a}}_{1:C}),\; \sigma^2 \mathbf{I}\right)$$

**Actor loss** (BC-regularized toward VLA reference):
$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\mathbf{s} \sim \mathcal{B}}\!\left[-Q_\psi(\mathbf{x}, \mathbf{a}_{1:C}) + \beta \|\mathbf{a}_{1:C} - \bar{\mathbf{a}}_{1:C}\|_2^2\right]$$

**Critic loss** (standard TD backup on action chunks):
$$\mathcal{L}_Q = \mathbb{E}_{(\mathbf{x}, \mathbf{a}_{1:C}, \mathbf{x}') \sim \mathcal{B}}\!\left[\!\left(\hat{Q} - Q_\psi(\mathbf{x}, \mathbf{a}_{1:C})\right)^2\right],\quad \hat{Q} = \sum_{\ell'=1}^C \gamma^{\ell'-1}r_{\ell'} + \gamma^C \mathbb{E}_{\mathbf{a}' \sim \pi_\theta}\!\left[Q_{\psi'}(\mathbf{x}', \mathbf{a}')\right]$$

**Reference action dropout**: with probability 0.5, the reference chunk **ā** is zeroed before being passed to the actor. This prevents the actor from degenerating to VLA copy-paste before the critic is reliable.

### Complete Training Loop (Algorithm 1)

1. **Warmup**: roll out base VLA policy for N_warm steps to pre-fill replay buffer B
2. **Rollout**: at each action chunk boundary, frozen VLA produces reference **ā**_{1:H}; actor outputs refined chunk **a**_{1:C} ~ π_θ(·|x, **ā**)
3. **Human interventions**: operator may provide teleoperated corrections **a**^human, overwriting actor for the intervention duration; stored in B with the corresponding VLA reference
4. **Off-policy updates** (G times per rollout step): sample batch from B, compute TD targets, update critic (×2) then actor (×1); update-to-data ratio = 5
5. **Subsampling**: stride-2 action chunks stored as transitions <x₀, a_{0:C}>, <x₂, a_{2:C+2}>, ... yielding ~25 samples per control step

### Key Design Choices and Why They Matter

| Component | Role | Ablation result |
|-----------|------|-----------------|
| RL token | Compact task-relevant state for RL | −50% throughput with ResNet-10 replacement |
| Chunked actions (C=10) | Shorter TD horizon → effective sparse-reward learning | Single-step variant cannot match base VLA |
| BC regularizer (β>0) | Anchors actor to VLA prior, stabilizes early learning | Removing β causes largest single performance drop |
| Reference pass-through | Guided exploration; VLA action distribution recovered by actor | Slower learning, more early failures w/o it |

---

## Experimental Results

### Tasks (4 real-robot manipulation tasks requiring sub-millimeter precision)

| Task | Challenge |
|------|-----------|
| **Screw installation** | M3 screw into threaded receptacle; rotation amplified by 10 cm lever arm; visual cues on opposite wrist camera |
| **Zip tie fastening** | Thread tail through locking slot; bimanual with deformable object; infer slot position from wrist cameras only |
| **Ethernet insertion** | Accurate positional + angular alignment then firm decisive insertion motion |
| **Charger insertion** | Centimeter-level alignment without direct visibility of prongs/socket |

**Setup**: π₀.₆ base VLA; 1–10 h teleoperated demos; 400–1000 RL episodes; actor = 2-layer MLP (hidden 256) for 3 tasks, 3-layer MLP (hidden 512) for screw; critic = ensemble of 2 Q-functions (TD3 style); 50 Hz control, 14-DOF actions, C=10 chunk.

### Q1: Does RLT improve over the base VLA?

- **Critical-phase controlled evaluation**: RLT consistently improves all 4 tasks; up to **3× faster** on Ethernet and Charger (already reliable); **20%→65% success** on screw installation
- **Full-task evaluation**: +40% success on screwdriver, +60% on zip tie; maintains high success on Ethernet while **halving mean steps to completion**

### Q2: Comparison to alternative RL methods (Ethernet task)

| Method | Success Rate | Throughput gain |
|--------|-------------|-----------------|
| DAgger | ≈ VLA | modest |
| HIL-SERL | low | low (no action chunking) |
| PLD | low | low (no action chunking) |
| DSRL | ≈ VLA | low (constrained to VLA modes) |
| **RLT (ours)** | ≈ VLA | **2× over base VLA** |

HIL-SERL and PLD fail because single-step actions face a very long TD horizon for sparse rewards. DSRL stays close to VLA, providing stability but less improvement room.

### Q3: Ablation contributions

All four components — RL token, chunked actions, BC regularizer, reference pass-through — contribute positively. Removing the BC regularizer (β=0) causes the largest single drop (forces the actor to explore the full action space with only Q-function gradients). Replacing the RL token with a frozen ResNet-10 reduces throughput by ~50%.

### Q4: Emergent strategies

On the Ethernet task, the base VLA exhibits a "probing" behavior: approach, retreat, readjust, retry (sometimes cycling multiple times). RLT learns a fluid one-shot insertion motion. Even after first-attempt failures, RLT applies pressure and wiggles the connector slightly — a strategy **not present in demonstration data**, arising purely from RL exploration.

---

## Implementation Details

- Base model: π₀.₆ (arXiv:2410.24164) — SigLIP 400M + Gemma 4B backbone + 860M flow-matching action expert
- RL token: encoder-decoder transformer trained for 2000–10000 gradient steps on task demos
- VLA fine-tuned jointly with RL token if α > 0
- Actor architecture: 2-layer MLP (hidden 256) for zip tie, Ethernet, charger; 3-layer MLP (hidden 512) for screw
- Critic: TD3-style ensemble of 2 Q-functions; target network updated periodically
- Reference action dropout probability: 0.5
- Proprioceptive auxiliary state: joint positions + end-effector pose (task-specific: screw also uses joint position; zip tie/Ethernet/charger use end-effector pose)
- Training begins shortly after warmup; 2 critic updates per actor update; UTD ratio = 5
- Experiment duration: ~15 minutes to 5 hours of actual robot data per task

---

## Limitations & Open Questions

- **Human supervision still required**: sparse reward labeling, intervention corrections, and policy-switching decisions all require a human operator during training
- **Two-phase training** (critical phase first, then full task) requires manual identification of the critical segment per task
- **No autonomous reward model**: a learned reward model or progress predictor could automate the supervision pipeline
- **Fixed frozen VLA**: the VLA's perceptual backbone cannot adapt online; tasks with large perceptual distribution shift would require VLA fine-tuning
- **Sparse binary reward**: richer reward signals (e.g., progress-based shaping) could accelerate learning further
- **Open question**: Can RLT generalize across a family of related tasks without per-task RL token training?

---

## See Also

- [Vision-Language-Action Models (VLAs)](../concepts/vision-language-action-models.md) — base model class (π₀) and training paradigm
- [RECAP — π*₀.₆](amin_2025_pi06_experience.md) — alternative VLA RL method: full-model offline advantage conditioning
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) — RL algorithms; RLT uses off-policy actor-critic (TD3 style)
- [Value-Based Reinforcement Learning](../concepts/value-based-rl.md) — Q-learning foundations for the critic
- [Sim-to-Real Transfer](../concepts/sim-to-real.md) — contrasting paradigm: train in sim; RLT trains directly on real robot
- [JEPA World Models as Physics Priors for RL](../open_questions/jepa-rl-humanoid-grasping.md) — open research hypothesis that swaps this paper's frozen VLA prior for a JEPA world model
