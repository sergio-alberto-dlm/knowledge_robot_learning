---
title: In-Context Adaptation & Cross-Embodiment Policies
type: concept
tags: [in-context-adaptation, meta-rl, cross-embodiment, history-conditioned-policy, transformer-xl, domain-randomization, legged-locomotion, system-identification]
related: [papers/liu_2025_locoformer.md, concepts/legged-locomotion.md, concepts/sim-to-real.md, concepts/continual-learning-and-tracking.md, concepts/policy-gradient-methods.md, concepts/step-size-adaptation.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/locoformer.pdf]
---

# In-Context Adaptation & Cross-Embodiment Policies

## What It Is

**In-context adaptation** means a policy with **fixed weights** changes its behaviour at test time by conditioning on its recent history of observations, actions (and, implicitly, outcomes). Adaptation happens in the *activations*, not through gradient updates. The history acts as an implicit system identification: from how the body and environment respond, the policy infers what it is controlling and acts accordingly.

A **cross-embodiment (omni-bodied) policy** is one network that controls many robot bodies. The two ideas meet when the policy is **not told** which body it controls and has to infer it in context, as in LocoFormer (Liu et al. 2025).

## Where It Sits Among Adaptation Methods

| Approach | How it adapts | Test-time cost | Typical horizon |
|---|---|---|---|
| Robust policy (plain domain randomization, no memory) | Doesn't adapt; it acts well on average | None | n/a |
| Explicit conditioning on robot description / privileged params | Kinematics or dynamics fed as input | Needs system ID or a URDF | n/a |
| Short-history adaptation module (RMA-style, teacher-student) | Infers latent env params from ~0.1–0.5 s of history | Forward pass | Hundreds of ms |
| **Long-context in-context adaptation** (RL², LocoFormer) | Attends over seconds of history, **across trial resets** | Forward pass (KV-cache) | Seconds to tens of seconds, multiple trials |
| Gradient-based meta-RL (MAML) / online fine-tuning | Weight updates on new data | Backprop on robot | Minutes and longer |

Short-history methods are the default in [legged locomotion](legged-locomotion.md) (e.g. teacher-student distillation). LocoFormer argues they are **myopic**. They work for a narrow task distribution around one robot, but a few hundred ms does not carry enough information to identify a body drawn from a very wide distribution.

## Ingredients (from LocoFormer)

1. **A task distribution wide enough to force adaptation.** Procedurally generated bodies (bipeds, quadrupeds, wheeled variants) plus aggressive dynamics randomization. If a single robust behaviour works everywhere, the policy has no incentive to identify the task. Width is what makes in-context learning emerge, analogous to web-scale LLM pretraining.
2. **A multi-trial (RL²-style) objective.** An episode holds several trials in the *same* MDP, memory persists across resets, and return is summed across trials. The policy is then rewarded for *using* failures to do better next time (exploit later what it explored earlier).
3. **An architecture with long, cheap memory.** A GRU trained the same way collapses (0.37 vs 0.96 normalized displacement). Transformer-XL segment recurrence gives $O(\text{layers} \times \text{segment})$ context (~18 s at 50 Hz). A KV-cache keeps per-step inference linear.
4. **A unified embodiment interface.** A superset joint space for observations and actions, so one network fits every body.

## What It Buys

- **Zero-shot transfer to unseen real robots** (Unitree G1/H1/Go2/Go2-W, even Go2 walking bipedally on its rear legs).
- **Graceful response to failures** never seen in training: locked knees, locked wheels, cut lower legs, stilts, payloads. Recovery takes 2–8 s, and in some cases tens of seconds.
- **Emergent system identification:** internal representations of different humanoids start identical and separate into embodiment-specific clusters within ~5 s.
- **A sim-to-real argument:** a policy that adapts can be trained under *wider* randomization, so the real robot is more likely to fall inside the training distribution (see [Sim-to-Real Transfer](sim-to-real.md)).

## Costs and Open Issues

- **Compute:** ~500× a specialist policy (amortised over ~100k training bodies). Large-GPU PPO for tens of hours.
- **Hand-designed task space:** the procedural generator encodes design assumptions, and real robots can violate them (e.g. G1's offset joint axes). How to build such spaces for manipulation or other skills is open.
- **How much memory is actually used** at deployment vs. the theoretical context length has not been measured.
- **Relation to continual learning:** in-context adaptation handles non-stationarity (a joint locks mid-run) without weight updates, which complements the weight-space tracking methods in [Continual Learning & Tracking](continual-learning-and-tracking.md) and [Step-Size Adaptation](step-size-adaptation.md). It cannot, however, accumulate knowledge beyond its context window.

> **Verify:** The descriptions of RMA, RL², MAML and the Adaptive Agent here come only from LocoFormer's related-work section. None of those papers is compiled in the wiki yet.

## See Also

- [Liu et al. (2025) — LocoFormer](../papers/liu_2025_locoformer.md)
- [Legged Locomotion & Perceptive Control](legged-locomotion.md)
- [Sim-to-Real Transfer](sim-to-real.md)
- [Policy Gradient Methods](policy-gradient-methods.md)
- [Continual Learning & Tracking in Non-Stationary Tasks](continual-learning-and-tracking.md)
- [Step-Size Adaptation & Meta-Learning of Learning Rates](step-size-adaptation.md): weight-space meta-learning, as opposed to activation-space adaptation
