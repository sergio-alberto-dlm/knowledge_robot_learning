---
title: Wiki Index
updated: 2026-09-29
---

# Robot Learning Knowledge Base — Master Index

> Maintained automatically by the LLM. Do not edit manually.

## Concepts
<!-- format: - [Title](concepts/file.md) — one-line description -->
- [Joint Embedding Predictive Architecture (JEPA)](concepts/joint-embedding-predictive-architecture.md) — SSL framework that predicts in latent representation space rather than pixel space
- [Latent World Models](concepts/latent-world-models.md) — dynamics models that predict future latent states conditioned on actions, enabling efficient planning
- [Model Predictive Control for Robot Learning](concepts/model-predictive-control.md) — receding-horizon planning with a world model and CEM action optimization
- [Video Self-Supervised Learning](concepts/video-ssl.md) — learning visual representations from unlabeled video via pretext tasks (masking, prediction)
- [SIGReg (Sketched Isotropic Gaussian Regularization)](concepts/sigreg.md) — distribution-matching regularizer that enforces isotropic Gaussian embeddings via random 1D projections and the Epps-Pulley CF test; linear complexity, bounded gradients
- [Intrinsic Motivation & Curiosity-Driven RL](concepts/intrinsic-motivation.md) — reward functions intrinsic to the agent (prediction error, count-based, information-theoretic) that drive exploration without extrinsic reward
- [Policy Gradient Methods](concepts/policy-gradient-methods.md) — family of RL algorithms that directly optimize a stochastic policy; covers REINFORCE, TRPO, PPO, GAE, and actor-critic architectures
- [Value-Based Reinforcement Learning](concepts/value-based-rl.md) — Q-learning, DQN, and Double DQN; covers experience replay, target networks, overestimation bias, and the deadly triad
- [Dexterous Manipulation](concepts/dexterous-manipulation.md) — multi-fingered robot hands for functional grasping; covers affordances, eigengrasp action spaces, blind proprioceptive policies, and LEAP hand
- [Sim-to-Real Transfer](concepts/sim-to-real.md) — training in simulation and deploying on real hardware; covers domain randomization, the reality gap, and action space design
- [Vision-Language-Action Models (VLAs)](concepts/vision-language-action-models.md) — robot policies combining VLM backbones with flow-matching action experts; trained via imitation learning and RL
- [Legged Locomotion & Perceptive Control](concepts/legged-locomotion.md) — RL-trained quadruped locomotion with depth cameras; unified reward design, teacher-student distillation, automatic curriculum
- [Discrete-Time Gaussian Processes for Imitation Learning](concepts/discrete-time-gaussian-processes.md) — DiGaP/MiDiGaP: per-timestep Gaussian sequences for few-shot multimodal robot policy learning; supports inference-time Bayesian updating and cross-embodiment VAPOR transfer
- [Gaussian Belief Propagation & Factor Graphs](concepts/gaussian-belief-propagation.md) — distributed probabilistic inference via local message passing on factor graphs; connects to solving Ax=b; local, probabilistic, iterative, asynchronous
- [Sparse Gaussian Process Approximations](concepts/sparse-gaussian-processes.md) — pseudo-datapoint methods (FITC, VFE, PIC, Tree) for scalable GP regression; tree-structured variant achieves O(N) inference via GBP
- [Learning Robot Manipulation from Human Videos](concepts/learning-from-human-videos.md) — extracting priors from third-person human video to bootstrap robot policies; agent-agnostic alignment bridges embodiment gap
- [Visual Affordances for Robotics](concepts/visual-affordances-robotics.md) — (contact point, post-contact trajectory) representation learned from egocentric video; versatile interface for IL, exploration, goal-conditioned RL, and action spaces
- [Dynamic Movement Primitives & Trajectory-Space Policies](concepts/dynamic-movement-primitives.md) — classical DMP representation (2nd-order ODE + weighted RBFs) embedded in neural policies (NDP); design space for trajectory-space policy parameterization
- [Successor Features & Successor Measures](concepts/successor-features.md) — decouples RL dynamics from reward via vector-valued Bellman targets; enables zero-shot transfer to any linear reward at test time
- [Zero-Shot / Unsupervised RL](concepts/zero-shot-unsupervised-rl.md) — pre-train task-agnostic representations and policies from reward-free data; recover optimal policy for any new task via linear regression at test time
- [Video-Action Models (VAMs)](concepts/video-action-models.md) — new paradigm grounding robot control in generative video model backbones (vs. VLMs); separates dynamics learning (video pretraining) from low-level control (IDM); 10× sample efficiency over VLAs
- [In-Context Adaptation & Cross-Embodiment Policies](concepts/in-context-adaptation.md) — fixed-weight policies that adapt by conditioning on long histories (across trials); one network for many robot bodies
- [Superquadric & Primitive-Based Scene Representations](concepts/superquadric-scene-representations.md) — compact, editable 3D scenes as sets of 11-parameter superquadrics; analytic distances for planning and grasping
- [Pretrained Visual Representations for Robotics](concepts/pretrained-visual-representations.md) — frozen SSL encoders (DINOv2, R3M, V-JEPA) as perception for world models/policies; patch tokens ≫ global vectors for spatial control
- [RL Evaluation Methodology](concepts/rl-evaluation-methodology.md) — frozen protocols, equal per-env quotas, reset determinism, det vs stoch scoring, seed-derived detection thresholds
- [Step-Size Adaptation & Meta-Learning of Learning Rates](concepts/step-size-adaptation.md) — learning the learning rate online (IDBD, Delta-Bar-Delta); per-input step sizes as bias and as a feature-relevance signal
- [Continual Learning & Tracking in Non-Stationary Tasks](concepts/continual-learning-and-tracking.md) — drifting-target tracking tasks as the right testbed for bias learning; why train-once benchmarks hide meta-learning effects
- [TD(λ) & Eligibility Traces](concepts/td-lambda-and-eligibility-traces.md) — λ-returns, forward/backward views, True Online TD(λ)'s exact equivalence, and the correction ratio as a stability device

## Papers
<!-- format: - [Author et al. (Year) — Short Title](papers/file.md) — venue, key contribution -->
- [Bardes et al. (2024) — V-JEPA](papers/bardes_2024_vjepa.md) — arXiv Feb 2024; establishes feature prediction as stand-alone video SSL objective; introduces multi-block masking and attentive probing
- [Assran et al. (2025) — V-JEPA 2](papers/assran_2025_vjepa2.md) — arXiv 2506.09985; scales JEPA video pretraining to 1B params / 1M+ hours; adds action-conditioned world model for zero-shot robot manipulation
- [Zhou et al. (2024) — DINO-WM](papers/zhou_2024_dino_wm.md) — arXiv 2411.04983; frozen DINOv2 patch features + frame-causal ViT predictor trained reward-free offline; zero-shot CEM-MPC goal reaching; beats DreamerV3/TD-MPC2/IRIS on Push-T, rope, granular
- [Terver et al. (2026) — JEPA World Models](papers/terver_2026_jepa_wm.md) — arXiv 2512.24497; systematic ablation of JEPA-WM design choices; introduces NeverGrad planner; best config outperforms V-JEPA 2-AC and DINO-WM
- [Balestriero & LeCun (2025) — LeJEPA](papers/balestriero_2025_lejepa.md) — arXiv 2511.08544; proves isotropic Gaussian is the optimal JEPA embedding distribution; introduces SIGReg for heuristic-free collapse prevention; 60+ architectures, in-domain pretraining beats frontier transfer learning
- [Burda et al. (2018) — Large-Scale Curiosity](papers/burda_2018_curiosity_largescale.md) — arXiv 1808.04355; first large-scale study of pure curiosity-driven learning across 54 environments; random features surprisingly effective; demonstrates noisy-TV limitation
- [Schulman et al. (2017) — PPO](papers/schulman_2017_ppo.md) — arXiv 1707.06347; introduces clipped surrogate objective enabling multiple epochs of minibatch SGD; outperforms TRPO and A2C on continuous control; competitive with ACER on Atari
- [van Hasselt et al. (2016) — Double DQN](papers/van_hasselt_2016_ddqn.md) — AAAI 2016, arXiv 1509.06461; proves Q-learning overestimation; decouples action selection and evaluation; new SOTA on Atari 2600
- [Agarwal et al. (2023) — Dexterous Functional Grasping](papers/agarwal_2023_dex_func_grasp.md) — CoRL 2023, arXiv 2312.02975; one-shot DINOv2 affordance matching + eigengrasp action space + sim2real PPO; beats teleop oracle on 5/7 objects
- [Amin et al. (2025) — π*₀.₆ / RECAP](papers/amin_2025_pi06_experience.md) — arXiv 2511.14759; RL with Experience and Corrections via Advantage-conditioned Policies; doubles throughput on laundry/espresso/box assembly vs. imitation baseline
- [Xu et al. (2025) — RLT / RL Token](papers/xu_2025_rlt.md) — Physical Intelligence; bootstraps online RL on frozen VLA via compact RL token readout; up to 3× speedup, 20%→65% success on screw installation; surpasses human teleoperation speed
- [Hansen et al. (2025) — Newt / MMBench](papers/hansen_2025_newt.md) — UCSD preprint; first massively multitask online RL benchmark (200 tasks, 10 domains); Newt world model extends TD-MPC2 with language/image conditioning; outperforms PPO and FastTD3 on average, rapid few-shot transfer to unseen tasks
- [Bahl et al. (2020) — Neural Dynamic Policies](papers/bahl_2020_ndp.md) — NeurIPS 2020; embeds DMP as differentiable layer in neural policy; end-to-end trainable in both IL and RL (PPO); multi-action critic; outperforms raw-action PPO on throwing, picking, pushing
- [von Hartz et al. (2025) — MiDiGaP](papers/vonhartz_2025_midigap.md) — arXiv 2505.03296; Mixture of Discrete-time Gaussian Processes for robot policy learning; state-of-the-art on RLBench from 5 demos, CPU training, VAPOR cross-embodiment transfer
- [Ortiz et al. (2021) — Visual Introduction to GBP](papers/ortiz_2021_gbp.md) — arXiv 2107.02308; tutorial on Gaussian Belief Propagation as a local, probabilistic, asynchronous inference framework for distributed/heterogeneous hardware
- [Bui & Turner (2014) — Tree-structured GP Approximations](papers/bui_2014_tree_gp.md) — NeurIPS 2014; tree/chain-structured pseudo-datapoint GP approximation with GBP inference; O(N) complexity; dominates speed-accuracy frontier on audio and terrain tasks
- [Nabarro et al. (2024) — GBP Learning](papers/nabarro_2024_gbp_learning.md) — ICML 2024, arXiv 2311.14649; trains NN-shaped Gaussian factor graphs (params as variables) with GBP; local/asynchronous training, continual learning via Bayesian filtering; 98.16% single-epoch MNIST, beats Lucibello et al. by 11.8% on CIFAR10
- [Liu et al. (2025) — LocoFormer](papers/liu_2025_locoformer.md) — CoRL 2025, arXiv 2509.23745; omni-bodied Transformer-XL locomotion policy trained with PPO on ~100k procedural robots; zero-shot to unseen real robots, emergent multi-trial adaptation to locked/cut limbs
- [Fedele et al. (2025) — SuperDec](papers/fedele_2025_superdec.md) — arXiv 2504.00992; class-agnostic Transformer + LM decomposes point clouds into superquadrics; ~6× lower L2 than prior learned methods with half the primitives; ShapeNet→ScanNet++/Replica; planning, Spot grasping, ControlNet editing
- [Cheng et al. (2023) — Extreme Parkour](papers/cheng_2023_extreme_parkour.md) — arXiv 2309.14341; single depth-camera end-to-end RL policy on Unitree A1; 2× height jumps, 2× length gaps, handstand via unified inner-product reward and dual distillation
- [Bahl et al. (2022) — WHIRL](papers/bahl_2022_whirl.md) — arXiv 2207.09450; one-shot in-the-wild manipulation from a single human video via prior extraction + agent-agnostic video alignment + real-world CEM; 83–92% success on 20 tasks
- [Bahl et al. (2023) — VRB](papers/bahl_2023_vrb.md) — arXiv 2304.08488; learns (contact point, trajectory) affordances from egocentric video; versatile across 4 robot paradigms; 57% avg IL success, 3–10× exploration gain over random on 10 real-world tasks
- [Bagatella et al. (2025) — TD-JEPA](papers/bagatella_2025_tdjepa.md) — arXiv 2510.00739; TD-based latent-predictive JEPA for zero-shot unsupervised RL; trains state+task encoders and policy-conditioned predictor from offline data; SOTA on ExoRL+OGBench RGB settings
- [Pai et al. (2025) — mimic-video](papers/pai_2025_mimic_video.md) — arXiv 2512.15692; Video-Action Model (VAM) pairing Cosmos-Predict2 generative backbone with flow-matching IDM; 10× sample efficiency and 2× faster convergence vs. VLAs; SOTA on SIMPLER and bimanual dexterous manipulation
- [Sutton (1992) — IDBD](papers/sutton_1992_idbd.md) — AAAI-92, pp. 171–176; Incremental Delta-Bar-Delta adapts one learning rate per input by meta-gradient descent; fully incremental, single free parameter; halves tracking error vs. best fixed-α LMS and recovers the optimal rates
- [Javed et al. (2024) — SwiftTD](papers/javed_2024_swifttd.md) — RLC 2024, pp. 840–863; True Online TD(λ) + step-size optimization + overshoot bound + step-size decay; lower lifetime error than True Online TD(λ) on every Atari Prediction Benchmark game, never diverges across its hyperparameter sweep

## Codebase
<!-- format: - [System / Component](codebase/system/file.md) — what it does -->
- [rl_lab / Overview](codebase/rl_lab/_overview.md) — personal repo: from-scratch PPO on ManiSkill PickCubeSO100 + frozen eval protocol; staged plan toward a SwiftTD linear critic on frozen DINOv2 + depth features
- [rl_lab / PPO Agent](codebase/rl_lab/ppo-agent.md) — Gaussian actor with log-std floor −1.2, MLP critic, clipped surrogate/value loss, GAE with truncation bootstrap
- [rl_lab / ManiSkill Training Pipeline](codebase/rl_lab/maniskill-training-pipeline.md) — 1024-env GPU rollouts, ignore-terminations fix, dual-axis logging, γ pilot (0.8 → 0.95)
- [rl_lab / Evaluation Protocol](codebase/rl_lab/evaluation-protocol.md) — frozen evaluate(): 128 eps, equal quota, reconfigure reset, det + stoch; 3-seed baseline 0.909 ± 0.030
- [rl_lab / Research Roadmap](codebase/rl_lab/research-roadmap.md) — stages 0–4: observation contract, DINOv2 3×3 pooling + depth (3560-d), compute budget, teacher–student Plan B, SwiftTD critic design
- [rl_lab / Legacy MuJoCo & Auxiliary](codebase/rl_lab/legacy-mujoco-and-auxiliary.md) — custom MuJoCo SO-100 env with staged reward, unfinished SAC, Genesis Franka grasp env, SIGReg CartPole identifiability

## Comparisons
<!-- format: - [Topic](comparisons/file.md) — what is being compared -->

## Open Questions
<!-- format: - [Question](open_questions/file.md) — why it matters -->
- [JEPA-RL for Humanoid Grasping (research hypothesis)](open_questions/jepa-rl-humanoid-grasping.md) — can a JEPA world model replace the VLA as the RL prior? proposed experiments for humanoid visual grasping
- [Object-Centric Persistent Slots + SwiftTD Critic (research design)](open_questions/object-centric-swifttd-critic.md) — how to combine SuperDec-style persistent object slots, DINO embeddings and SwiftTD feature selection in rl_lab; offline de-risking steps and pre-committed arms
