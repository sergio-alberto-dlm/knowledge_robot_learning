---
title: "Liu et al. (2025) — LocoFormer: Generalist Locomotion via Long-context Adaptation"
type: paper
tags: [legged-locomotion, cross-embodiment, in-context-adaptation, meta-rl, transformer-xl, sim2real, domain-randomization, ppo, procedural-generation, humanoid, quadruped, wheeled-legged]
related: [concepts/in-context-adaptation.md, concepts/legged-locomotion.md, concepts/sim-to-real.md, concepts/policy-gradient-methods.md, papers/cheng_2023_extreme_parkour.md, papers/schulman_2017_ppo.md, concepts/continual-learning-and-tracking.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/locoformer.pdf]
---

# Liu et al. (2025) — LocoFormer: Generalist Locomotion via Long-context Adaptation

**Venue:** CoRL 2025 (Seoul); arXiv 2509.23745v1 (28 Sep 2025, cs.RO)
**Authors:** Min Liu, Deepak Pathak, Ananye Agarwal (Skild AI)
**Project page:** generalist-locomotion.github.io

## Abstract

LocoFormer is a single **omni-bodied locomotion policy**. It controls legged and wheeled robots it has never seen (bipeds, quadrupeds, and their wheeled variants), zero-shot, without being told their kinematics. It also adapts at test time to large changes in morphology and dynamics: locked joints, cut legs, stilts, added payloads, locked wheels. Two choices make this work:
1. **Massive-scale RL on procedurally generated robots with aggressive domain randomization.** The task distribution is wide enough that the policy has to learn to identify the task and then adapt.
2. **Context lengths orders of magnitude longer** than standard locomotion policies (seconds instead of a few hundred milliseconds), and the context **spans episode/trial boundaries**.

With this recipe, adaptation *emerges*: the policy uses failed early trials to improve later ones, much like in-context learning in LLMs.

## Motivation

- Standard locomotion controllers are trained **tabula rasa for one embodiment** with narrow domain randomization. They adapt only over a **few hundred ms of history** (e.g. RMA-style adaptation modules), so they are *myopic*.
- That design favours in-distribution performance but **fails catastrophically** out of distribution (burnt-out motors, broken legs, modeling errors). Sim-to-real then needs painstaking system identification for each robot.
- Hypothesis: training on a very wide task distribution forces the policy to learn *general* strategies for task identification and adaptation. This mirrors how web-scale pretraining produces in-context learning in LLMs (Brown et al. 2020) and how open-ended training produced human-timescale adaptation in agents (Adaptive Agent Team 2023).

## Key Contributions

1. **A generalist cross-embodiment locomotion controller** trained purely in simulation on procedurally generated robots, with no real robot models in the training set.
2. **Long-context, multi-trial RL objective** (RL²-style). Memory persists across trials inside an episode, and return is maximised over the whole episode.
3. **Transformer-XL policy with a KV-cache** that gives ~18 s of effective memory at 50 Hz while staying real-time.
4. **Real-world zero-shot deployment** on Unitree G1, H1, Go2 and Go2-W (the last two also in a *bipedal* mode on their rear legs), plus **emergent adaptation** to severe unseen perturbations.

## Methodology

### Multi-trial objective

A **trial** is one trajectory $\tau^M = [s_0, a_0, \dots]$ in an MDP $M$. An **episode** $\mathcal{E}^{M,k} = [\tau_1^M, \dots, \tau_k^M]$ is $k$ trials in the *same* MDP, with $k \sim \mathrm{Uniform}(1, K)$. The policy $\pi_\theta(a_t \mid s_{0:t}, a_{0:t-1})$ conditions on the **whole in-episode history, across trial resets**. With $H_i$ the cumulative length up to trial $i$:

$$J(\pi_\theta) = \mathbb{E}_{M\sim\rho_\mathcal{M},\,k\sim\rho_K,\,\mathcal{E}^{M,k}\sim\pi_\theta}\left[\sum_{i=1}^{k}\sum_{t=0}^{T}\gamma^{t+H_{i-1}}\,\mathcal{R}^M(s_t,a_t)\right]$$

In practice the trial count is replaced by an **adaptation time budget** $u \sim \mathrm{Uniform}(0, U)$ seconds. The policy may run any number of trials within $u$, followed by a final trial. Training has **two phases**. It first uses short trials and small $U$ to prioritise learning adaptive behaviour. It then lengthens trials and adaptation time to prepare for real-world deployment. Optimiser: **PPO** at scale in physics simulation.

### Task generation (procedural robots)

- Four morphology families: **quadrupeds, bipeds, wheeled quadrupeds, wheeled bipeds** (App. Fig. 6). They follow common design principles of existing robots but use **no real robot's exact parameters**.
- **Kinematics randomised:** joints, joint sequence, link lengths, and so on.
- **Dynamics randomised:** mass, centre of mass, inertia, control (PD) gains, joint limits, and so on. The standard parameters from Agarwal et al. (2023) are used, over a much wider range.
- About **100k robots** in total (App. C).
- The space is *not* exhaustive. Berkeley Humanoid has anhedral hip-roll *and* hip-yaw angles (generated bipeds have at most one), and G1 has joint axes offset along $x$. Both are therefore genuinely out of distribution.

### Unified observation/action/reward space

- **Unified joint space:** a superset of joints that covers the motor count of most current legged robots. The policy outputs **target joint positions** in this space, and each robot uses only its own subset.
- **Reward** (App. A.2), inspired by Agarwal et al. (2023):
  - linear and angular velocity command tracking, as exp-kernels on $\|v_{xy}-v^{cmd}_{xy}\|^2$ and $\|\omega_z-\omega_z^{cmd}\|^2$;
  - penalties on base $v_z$, base $\omega_{xy}$, deviation from nominal base height, joint acceleration and torque;
  - an alive bonus of $+1$.

> **Verify:** App. A.2 prints the tracking terms as $\exp(+\|\cdot\|^2/s)$ and gives only the torque penalty a minus sign. These look like sign typos. The standard form is $\exp(-\|\cdot\|^2/s)$, with every penalty term negative.

### Policy architecture: Transformer-XL with long-term memory

- Observations → MLP → token $x_t$ → **Transformer-XL** backbone → separate **actor and critic MLP heads**.
- **Segment-level recurrence:** context is split into segments of length $L$. When processing segment $z+1$, each layer attends to its own tokens plus the **cached hidden states of segment $z$ from the layer below, with gradients stopped**:
  $\tilde{\mathbf{h}}^{n-1}_{z+1} = [\mathrm{SG}(\mathbf{h}^{n-1}_z) \circ \mathbf{h}^{n-1}_{z+1}]$, keys and values come from $\tilde{\mathbf{h}}$, and queries come from $\mathbf{h}_{z+1}$.
- The effective context grows as **$O(NL)$**. With 6 layers and $L=128$, the paper quotes a maximum memory of **896 timesteps ≈ 18 s at 50 Hz**. Segments may span trial boundaries, but the attention mask **blocks attention across *episode* boundaries**.

> **Verify:** $N \cdot L = 6 \times 128 = 768$. The quoted 896 equals $(N+1)\cdot L$, i.e. it apparently counts the current segment as well.

- **Inference acceleration:** a **KV-cache** turns per-step inference from quadratic to linear in $t$. In training rollouts, the cache is initialised from the previous segment's keys and values. At deployment it keeps only the most recent **$2L-1$ timesteps**, the longest directly-conditioned history seen in training.

> **Verify:** $2L-1 = 255$ steps ≈ 5 s of *direct* attention. Older information can still enter through the cached hidden states, which already summarise earlier segments. The paper does not quantify how much of the ~18 s theoretical memory is actually used at deployment. The real-world adaptations that take 7–8 s and "tens of seconds" suggest information persists beyond 5 s.

### Deployment modes

- **Zero-shot ($u=0$):** the policy adapts within seconds to an unseen body, often walking stably immediately.
- **Few-shot:** when early exploration fails (falls), the stored failed trajectories stay in context and later trials improve.

## Results

### Simulation: 10 unseen robots (Table 1)

1,000 randomised environments per robot, rough terrain, intensified domain randomization. The metric is displacement toward a random goal, normalised to [0, 1].

| Method | G1 | H1 | GR1 | TRON1 | BK-H | A1 | Spot | AnyMal C | TRON1-W | Go2-W | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LocoFormer (zero-shot) | 0.96 | 0.98 | 0.95 | 0.87 | 0.98 | 0.92 | 1.00 | 0.95 | 0.99 | 0.97 | **0.96** |
| LocoFormer (few-shot, 5 s) | 0.98 | 0.99 | 0.99 | 0.96 | 0.99 | 0.94 | 1.00 | 0.96 | 1.00 | 0.98 | **0.98** |
| GRU policy (same training) | 0.09 | 0.08 | 0.09 | 0.03 | 0.47 | 0.46 | 0.68 | 0.42 | 0.66 | 0.74 | 0.37 |
| Conditioning (privileged kinematics, ctx 64) | 0.94 | 0.94 | 0.81 | 0.07 | 0.62 | 0.93 | 0.99 | 0.89 | 0.72 | 0.92 | 0.78 |
| Expert (trained on target robot) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 | 0.98 | 0.99 |

- LocoFormer comes close to the per-robot **expert upper bound** (0.98 vs 0.99) with no access to the test robots.
- **The architecture matters a lot.** A GRU trained identically collapses (0.37), especially on bipeds.
- **Explicit morphology conditioning with short context** (0.78) is worse than long-context adaptation *without* morphology input. It fails on **TRON1** (a biped with no ankle joints, 0.07), where 5 s of few-shot adaptation lifts LocoFormer from 0.87 to 0.96 (+10%). In simulation TRON1 falls in early trials and walks by trial 4 (App. Fig. 8).
- Standard deviations are large (e.g. ±0.31 on TRON1 zero-shot), and results are reported as mean ± std over the 1,000 environments.

### Adaptation over time (Fig. 2, Fig. 4a)

- With **2× the training domain-randomization ranges** (OOD), multiple seconds of history keep improving performance. Going from 0 to 5 s of adaptation lets **15.4% more robots reach reward > 6** and raises the **25th-percentile reward by 3.6**. Prior myopic policies stop improving after a few hundred ms.
- Average survival over 6 s rollouts (50k episodes per setting) rises with adaptation budget (0–6 s) for all four morphology families. Most of the gain comes in the first 1–2 s.

### Representation dynamics (Fig. 4b)

- The mean output of the 2nd Transformer layer over 4,096 zero-shot rollouts of four humanoid variants (GR2, GR1, H1-2, H1) starts nearly identical at 0 s and **separates into distinct embodiment-specific clusters by ~5 s**. The policy builds an implicit system-identification representation online, without explicit robot descriptors.

> The text cites "Fig. 6c/6d" for these results, but they are Fig. 4a/4b (Fig. 6 shows the procedural robots). This is a figure-reference typo.

### Real world (Fig. 5), all from one model, deployed zero-shot

| Scenario | Robot | Behaviour |
|---|---|---|
| **A. Knee locked** (PD setpoint overridden) | Go2 | Initially tips forward, then shifts weight back onto 3 legs and walks after **2–3 s**. The 3-legged body is outside the design space. |
| **B. Adaptation across trials** | Go2 as a biped (no ankle motor, single support point) | Falls in trial 1. The failed trajectory stays in the TXL cache, and by **trial 3** it walks stably and resists pushes and weights. |
| **C. Wheel locking / payload** | Go2-W | Wheel gains are set to 0 mid-roll. It detects that wheel commands no longer have an effect and switches to a walking gait. When the wheels are unlocked it switches back to the lower-energy rolling gait. It also adjusts gait under added mass. |
| **D. Stilts** | Go2 | Leg-to-body ratio far beyond training. A few unstable steps, then retuned step timing and foot placement. |
| **E. One/two legs locked** | Go2-W | Redistributes load to keep balance and mobility. |
| **F. Lower legs cut off** (4 DoF removed) | Go2 | Steps in place at first. After **7–8 s** it discovers large-amplitude thigh swings and walks on its "knees". |

### Compute (App. C, Fig. 9)

- It takes **~500× the compute of one specialist policy** while covering **~100k robots**. That is about 0.005 vs 1 day of compute amortised per robot.
- Scaling runs used 32 / 64 / 128 GPUs and 3- or 6-layer TXL. **4K iterations take ~28 h for the 6-layer model.** More GPUs and more layers give better reward and OOD displacement, but smaller configurations (e.g. 32 GPUs, 6 layers) already adapt and generalise across embodiments.
- LLM-style efficiency techniques (mixed precision, ZeRO) are suggested as ways to cut cost further.

## Limitations

- **Resource-intensive** compared to a specialist policy. The authors point to algorithmic gains for massively parallel RL, such as SAPG (Singla et al. 2024).
- The **procedural task space is hand-crafted** and may be hard to design for other skills. Automated task generation from LLMs or web-scale sources is suggested.
- Only **blind locomotion with velocity commands**: no perception, manipulation, or other skills yet, although the recipe is claimed to generalise.
- Real-world results are **qualitative** (videos and figure panels). There are no quantitative real-robot success rates or baselines on hardware.
- Only **mean ± std in simulation**, no seeds or statistical tests. Training details (PPO hyperparameters, exact randomization ranges, observation contents, command sampling) are sparse.

## Relation to Other Work

- **Single-robot sim-to-real locomotion** (RMA, ANYmal, DreamWaQ, Walk These Ways, [Extreme Parkour](cheng_2023_extreme_parkour.md), humanoid RL): each is tuned to one robot with narrow randomization and ~100 ms adaptation context. LocoFormer keeps the sim-to-real + PPO recipe but widens both the task distribution and the context.
- **Cross-embodiment learning:**
  - morphology-aware structure: GNN message passing (Huang et al. 2020, *One Policy to Control Them All*) and attention masks (Body Transformer);
  - co-evolution (Gupta et al. 2021);
  - per-morphology readout heads trained with imitation (CrossFormer, Doshi et al. 2024);
  - latent observation space with explicit kinematics (Bohlinger et al. 2024, "Conditioning"-style);
  - next-token prediction (Gato);
  - procedurally generated quadrupeds (GenLoco, Feng et al. 2023), the closest prior work.

  LocoFormer's claimed difference is **massive scale + long context + a broader space** (wheeled/legged quadrupeds *and* humanoids) with no morphology input.
- **Adaptation / meta-RL / in-context learning:** MAML, memory-augmented meta-RL, evolutionary meta-learning, and real-world fine-tuning (*A Walk in the Park*). The closest in spirit are **RL²** (context spanning episodes; Duan et al. 2016) and the **Adaptive Agent** (human-timescale adaptation from open-ended training). LocoFormer carries these ideas into **high-frequency, closed-loop control**. See [In-Context Adaptation & Cross-Embodiment Policies](../concepts/in-context-adaptation.md).

## See Also

- [In-Context Adaptation & Cross-Embodiment Policies](../concepts/in-context-adaptation.md)
- [Legged Locomotion & Perceptive Control](../concepts/legged-locomotion.md)
- [Sim-to-Real Transfer](../concepts/sim-to-real.md)
- [Policy Gradient Methods](../concepts/policy-gradient-methods.md) and [Schulman et al. (2017) — PPO](schulman_2017_ppo.md): the training algorithm
- [Cheng et al. (2023) — Extreme Parkour](cheng_2023_extreme_parkour.md): specialist single-robot counterpart from overlapping authors (Agarwal, Pathak)
- [Continual Learning & Tracking in Non-Stationary Tasks](../concepts/continual-learning-and-tracking.md): mid-episode morphology changes (locked joints/wheels) make a non-stationary tracking problem solved in-context, without weight updates
