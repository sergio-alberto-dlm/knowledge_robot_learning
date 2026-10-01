---
title: Sim-to-Real Transfer
type: concept
tags: [robot-learning, sim2real, domain-randomization, reality-gap, simulation, transfer-learning, dexterous-manipulation]
related: [papers/agarwal_2023_dex_func_grasp.md, papers/cheng_2023_extreme_parkour.md, concepts/dexterous-manipulation.md, concepts/legged-locomotion.md, concepts/policy-gradient-methods.md, concepts/model-predictive-control.md, papers/liu_2025_locoformer.md, concepts/in-context-adaptation.md]
created: 2026-05-17
updated: 2026-09-29
sources: [raw/papers/pdf/dex_func_grasp.pdf, raw/papers/pdf/locoformer.pdf]
---

# Sim-to-Real Transfer

## What It Is

Sim-to-real transfer refers to the practice of training a robot policy in a physics simulation and deploying it on real hardware — without collecting real-world training data (or with minimal real data for fine-tuning). The motivation is:
- **Data efficiency:** Simulators allow parallelized training at 1000s × real time; collecting real robot data is slow, expensive, and potentially dangerous.
- **Safe exploration:** Policies can explore failure modes freely in simulation.
- **Reset efficiency:** Simulators can reset to any state instantly; real environments need manual resets.

## The Reality Gap

The **reality gap** (or sim-to-real gap) refers to the discrepancy between simulated and real-world physics, appearance, and sensor noise. Key sources:

| Source | Description |
|--------|-------------|
| **Physics modeling errors** | Friction, stiffness, damping are approximated; contact dynamics are particularly hard to simulate accurately |
| **Unmodeled dynamics** | Motor lag, joint flex, backlash, cable stretch — all absent from rigid-body simulators |
| **Sensor noise** | Real sensors (cameras, IMUs, joint encoders) have measurement noise absent in clean simulation |
| **Visual appearance gap** | Textures, lighting, reflections differ between sim and real |
| **Object properties** | Real objects have variable mass, center-of-mass, surface texture |

The gap is particularly severe for **contact-rich manipulation** (dexterous grasping) because small errors in friction or stiffness dramatically change grasp stability.

## Domain Randomization

**Domain randomization** (DR) is the primary technique for bridging the reality gap without real data. The idea: train on a wide distribution of simulated environments so that the real world is just another sample from that distribution.

### Physics Randomization
Randomize physical parameters at the start of each episode:
- Object scale, mass, center of mass
- Friction coefficient, restitution
- Joint stiffness, damping
- Motor strength, action delays

In Agarwal et al. (2023), domain randomization for dexterous grasping:

| Parameter | Range |
|-----------|-------|
| Object scale | [0.8, 1.2] |
| Object mass | [0.5, 1.5] × nominal |
| Friction coefficient | [0.7, 1.3] |
| Stiffness | [0.75, 1.5] × nominal |
| Damping | [0.3, 3.0] × nominal |

Gaussian noise is additionally added to observations and actions.

### Randomization Width vs. Adaptation

Wider randomization makes it more likely that the real world lies inside the training distribution, but a memoryless policy gets more conservative as the ranges grow. **Adaptive** policies loosen this trade-off. LocoFormer (Liu et al. 2025) randomizes *morphology itself* (procedurally generated bodies) plus mass, CoM, inertia, PD gains and joint limits over ranges much wider than standard locomotion setups. Because its long-context Transformer can identify the body online, it transfers zero-shot to unseen real robots. Under 2× its training randomization ranges, 5 s of in-context adaptation lets 15.4% more robots exceed the reward threshold. See [In-Context Adaptation](in-context-adaptation.md).

### Visual Randomization (for vision-based policies)
Randomize textures, lighting, camera position, and background. Less relevant for blind proprioceptive policies.

## Action Space Design for Sim2Real

A critical but underappreciated factor in sim2real transfer is **action space design**. Policies trained in full joint space often discover motion patterns that are kinematically possible in simulation but dynamically unfeasible on real hardware (e.g., finger gaiting in dexterous hands).

**Eigengrasp action spaces** (Agarwal et al. 2023) restrict the RL policy to a low-dimensional subspace of physically realistic hand poses derived from PCA of human demonstrations:
- Reduces search space (e.g., 16 → 9 dimensions for LEAP hand)
- Guarantees all output poses are approximately realistic
- Dramatically improves training stability and sim2real transferability
- Allows discovering behaviors not in the demo data (unlike offline RL or imitation learning)

This approach contrasts with:
- **Unconstrained training:** Full joint space, unstable and unrealistic
- **VAE latent space:** Generative model of poses — cannot extrapolate beyond demo distribution
- **Offline RL / behavior cloning:** Cannot improve beyond demo quality

## Sim2Real Success Stories

| Domain | Method | Key Technique |
|--------|--------|---------------|
| Legged locomotion | RMA (Kumar et al. 2021), Extreme Parkour | Adaptive policies, teacher-student distillation |
| Cross-embodiment locomotion | LocoFormer (Liu et al. 2025) | Procedural robots + extreme DR + long-context (Transformer-XL) in-context adaptation |
| In-hand reorientation | OpenAI Five, Dextreme | Large-scale DR, asymmetric actor-critic |
| Dexterous grasping | Agarwal et al. (2023) | Eigengrasp action space + DR |
| Robot arm manipulation | V-JEPA 2-AC (Assran et al. 2025) | Latent world model + MPC (only 62h robot data) |

## Challenges Specific to Dexterous Manipulation

Dexterous manipulation is among the hardest sim2real problems because:
1. **Contact richness:** Stable grasps require accurate multi-point contact modeling — each contact point introduces nonlinear friction dynamics.
2. **High force requirements:** Tool use (hammering, drilling) requires exerting large forces through a stable grasp, amplifying any contact modeling error.
3. **High dimensionality:** 16+ DOF hands have large, poorly conditioned action spaces where simple reward functions lead to degenerate local minima.
4. **Sensor limitations:** Real hand hardware often lacks velocity sensing — policies must infer velocity from position history (motivating recurrent architectures).

Approaches that work for locomotion (simple reward, full joint space DR) often produce unrealistic finger-gaiting behaviors in dexterous manipulation that fail on real hardware.

## Relation to Other Approaches

- **Model-based RL / MPC:** Rather than transferring a fixed policy, learn a world model in simulation and plan with it at test time. V-JEPA 2-AC (Assran et al. 2025) uses a latent world model + CEM planning; this avoids direct policy transfer but requires the world model to generalize. See [[model-predictive-control]].
- **Real-to-sim-to-real:** Scan real objects into simulation for more accurate object models; reduces the visual and physical reality gap.
- **Domain adaptation:** Use real images to adapt the visual encoder while keeping the policy fixed.
- **Meta-learning / RMA:** Train a policy that quickly adapts its internal state to new dynamics (few-step online adaptation from real interactions).

## Practical Recommendations

- **Start with physics DR** before visual DR — physics gaps are typically the bigger obstacle for contact-rich manipulation.
- **Design the action space carefully:** Eigengrasp or joint-velocity parameterizations can make a larger difference than increasing DR range.
- **Use recurrent policies** for contact-rich tasks: hidden states implicitly model velocity and adapt to dynamics changes.
- **Domain-randomize simulation parameters broadly:** Better to randomize too much (policy is more conservative) than too little (overfit to nominal sim).
- **Validate sim2real transfer early:** Train in sim with progressively decreasing randomization to identify which parameters the policy is most sensitive to.

## See Also

- [Agarwal et al. (2023) — Dexterous Functional Grasping](../papers/agarwal_2023_dex_func_grasp.md) — concrete application with eigengrasp action space and domain randomization
- [Cheng et al. (2023) — Extreme Parkour](../papers/cheng_2023_extreme_parkour.md) — teacher-student distillation for legged parkour; dual distillation of motor commands and heading direction from depth
- [Dexterous Manipulation](dexterous-manipulation.md) — covers the specific challenges and techniques for multi-fingered robot hands
- [Legged Locomotion & Perceptive Control](legged-locomotion.md) — paradigm overview for RL-trained quadruped locomotion with depth cameras
- [Policy Gradient Methods](policy-gradient-methods.md) — PPO is the standard RL algorithm for sim2real training (parallel actors naturally exploit simulation parallelism)
- [Model Predictive Control for Robot Learning](model-predictive-control.md) — alternative to sim2real policy transfer; plan with a learned world model instead
- [Liu et al. (2025) — LocoFormer](../papers/liu_2025_locoformer.md) — extreme (morphology-level) randomization made workable by long-context adaptation
