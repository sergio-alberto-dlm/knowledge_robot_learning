---
title: Article Summaries
updated: 2026-09-29
---

# Article Summaries

> One entry per wiki article, 3–5 sentences each.
> The LLM loads this file first on every Q&A query to decide which full articles to read.
> Maintained automatically — do not edit manually.

<!-- Template:
## [Title](relative/path.md)
**Type:** concept | paper | codebase | comparison | open_question
**Tags:** tag1, tag2
Summary text here (3-5 sentences).
-->

---

## [Joint Embedding Predictive Architecture (JEPA)](concepts/joint-embedding-predictive-architecture.md)
**Type:** concept
**Tags:** self-supervised-learning, representation-learning, world-models, yann-lecun
JEPA is a self-supervised learning framework (LeCun 2022) in which a predictor learns to predict the latent representation of a target view from a context view, rather than reconstructing pixels. Predictions are made in a learned representation space, letting the model focus on semantically predictable structure while ignoring unpredictable visual details. An EMA teacher encoder and stop-gradient operation prevent representation collapse. Key variants include I-JEPA (images), V-JEPA (video, Bardes et al. 2024), and V-JEPA 2 (scaled video + robot adaptation, Assran et al. 2025).

## [Latent World Models](concepts/latent-world-models.md)
**Type:** concept
**Tags:** world-models, representation-learning, planning, model-based-rl, robot-learning
A latent world model learns to predict future latent states conditioned on actions, using a frozen pretrained encoder to map observations to representations. This enables efficient planning in representation space rather than pixel space, which is orders of magnitude faster than video-generation-based world models. Key design choices include whether to use teacher-forcing vs. rollout losses, autoregressive rollout length, and action conditioning architecture. V-JEPA 2-AC (Assran et al. 2025) is a concrete instance: a 300M-param causal transformer trained on 62 hours of robot data on top of a frozen 1B-param video encoder.

## [Model Predictive Control for Robot Learning](concepts/model-predictive-control.md)
**Type:** concept
**Tags:** planning, control, model-based-rl, robot-learning, optimization
MPC plans actions by simulating future states with a world model, optimizing a goal-conditioned cost function, executing only the first planned action, and re-planning after each observation (receding horizon). In latent world models, the cost is typically an L1 distance between the predicted future representation and the goal image encoding. The Cross-Entropy Method (CEM) is a common derivative-free optimizer for this setting: it iteratively samples action trajectories, selects the top-k, and refines the sampling distribution. V-JEPA 2-AC achieves zero-shot robot manipulation this way, requiring 16 seconds per action on a single GPU.

## [Video Self-Supervised Learning](concepts/video-ssl.md)
**Type:** concept
**Tags:** self-supervised-learning, video-representation, vision-transformer, pretraining
Video SSL learns visual representations from unlabeled video via pretext tasks such as masked patch prediction, contrastive temporal alignment, or next-frame prediction. Spatiotemporal multiblock masking (used in V-JEPA and V-JEPA 2) forces the model to reason about motion rather than just appearance. Scaling laws are consistent: more data, larger models, longer training, and higher resolution all improve downstream task accuracy. 3D-RoPE (Rotary Position Embeddings applied separately to temporal, height, and width dimensions) improves training stability at 1B+ parameter scale compared to sincos embeddings.

## [Dexterous Manipulation](concepts/dexterous-manipulation.md)
**Type:** concept
**Tags:** robot-learning, dexterous-manipulation, functional-grasping, eigengrasp, affordances, sim2real
Dexterous manipulation with multi-fingered hands enables picking up and functionally using tools — a capability impossible with two-fingered grippers — but requires both semantic understanding of affordances (where to grasp) and precise low-level control (how to grasp). Functional affordances identify the correct region of an object for its intended use (e.g., hammer handle vs. head), and can be detected one-shot using DINOv2 feature matching against a single annotated exemplar per category. The eigengrasp action space compresses 16-DOF hand poses to 9 principal components via PCA on human VR demonstrations, restricting RL exploration to physically realistic poses and dramatically reducing training variance. A blind proprioceptive RNN policy handles the reactive grasping motion; decoupling visual affordance detection from contact control makes the pipeline modular and generalizable across object categories.

---

## [Sim-to-Real Transfer](concepts/sim-to-real.md)
**Type:** concept
**Tags:** robot-learning, sim2real, domain-randomization, reality-gap, simulation
Sim-to-real transfer trains robot policies in physics simulation (exploiting massive parallelism and safe exploration) and deploys on real hardware, bridging the reality gap caused by inaccurate friction, unmodeled dynamics, and sensor noise. Domain randomization — randomizing physical parameters (mass, friction, stiffness, damping) across episodes — is the primary technique to ensure the real world is within the training distribution. Action space design is a critical but underappreciated factor: eigengrasp parameterizations (PCA over realistic hand poses) substantially improve transfer for dexterous manipulation by eliminating physically inconsistent finger configurations. Dexterous manipulation is among the hardest sim2real problems due to its contact-rich dynamics, high-dimensional action spaces, and the need to apply large forces through stable grasps.

---

## [Value-Based Reinforcement Learning](concepts/value-based-rl.md)
**Type:** concept
**Tags:** reinforcement-learning, q-learning, dqn, double-dqn, temporal-difference, experience-replay, overestimation
Value-based RL learns the optimal action-value function $Q_*(s,a)$ via temporal-difference updates, deriving a greedy policy implicitly — primarily effective for discrete action spaces. DQN (Mnih et al. 2015) scales Q-learning to visual inputs using experience replay (breaking temporal correlation) and a periodic target network (stabilizing regression targets). Q-learning's max operator causes systematic overestimation bias: van Hasselt et al. (2016) prove the bias lower-bound is $\sqrt{C/(m-1)}$ for $m$ actions with estimation error $C$, and show DQN overestimates substantially on all 49 tested Atari games. Double DQN fixes this with a single-line change — use the online network to select actions, the target network to evaluate them — improving Atari median from 93.5% to 114.7% normalized.

---

## [Policy Gradient Methods](concepts/policy-gradient-methods.md)
**Type:** concept
**Tags:** reinforcement-learning, policy-gradient, ppo, trpo, actor-critic, advantage-estimation, gae
Policy gradient methods directly optimize a stochastic policy $\pi_\theta$ by gradient ascent on expected reward, working naturally with continuous action spaces unlike value-based methods. The core stability problem — catastrophic collapse from large updates — is solved by TRPO (KL-constrained second-order optimization) or PPO (clipped probability ratio, first-order). PPO's clipped surrogate objective $L^{CLIP}$ takes the minimum of the unclipped and clipped ratio objectives, forming a pessimistic lower bound that prevents large updates while enabling multiple epochs of minibatch SGD on the same data. Generalized Advantage Estimation (GAE, λ=0.95) balances bias and variance in the advantage signal; the actor-critic architecture shares parameters between policy and value function for efficiency.

---

## [Burda et al. (2018) — Large-Scale Curiosity](papers/burda_2018_curiosity_largescale.md)
**Type:** paper
**Tags:** intrinsic-motivation, curiosity, exploration, reinforcement-learning, prediction-error, forward-dynamics, feature-learning, atari, mario
This paper performs the first large-scale empirical study of purely curiosity-driven learning — no extrinsic reward, no end-of-episode signal — across 54 diverse environments (48 Atari, Super Mario Bros, Roboschool, Unity mazes, Two-player Pong). The central method is dynamics-based curiosity: intrinsic reward equals the MSE prediction error of a forward dynamics model in a learned feature space, with four feature spaces compared (pixels, random features RF, inverse dynamics features IDF, VAE). A key surprising finding is that frozen random CNN features (RF) are competitive with or better than learned features in ~67% of Atari games due to their stationarity, while IDF features transfer better to novel Mario levels across visual distribution shifts. The paper also introduces the infinite-horizon "death is not the end" formulation and empirically demonstrates the noisy-TV problem: stochastic environment elements (a TV with random channels) are irresistible attractors for prediction-error curiosity, severely hampering exploration.

---

## [Bardes et al. (2024) — V-JEPA](papers/bardes_2024_vjepa.md)
**Type:** paper
**Tags:** self-supervised-learning, video-representation, jepa, feature-prediction, vision-transformer, masked-modeling
V-JEPA introduces feature prediction as a stand-alone self-supervised objective for video — training a ViT encoder to predict masked spatiotemporal patch representations (rather than pixels) using a narrow predictor and an EMA teacher, with an L1 loss and stop-gradient to prevent collapse. The key design choices are 3D multi-block masking (~90% masking ratio, full temporal coverage) and attentive cross-attention probing for frozen evaluation (outperforming linear probing by +17 pts K400). Trained on only 2M videos (VideoMix2M, 90K iterations, 270M samples seen), V-JEPA ViT-H/16 achieves 82.0 K400 and 71.4 SSv2 with frozen backbone — outperforming VideoMAEv2 (1B params) by +5 pts SSv2 while using 6× fewer training samples. V-JEPA is also more label-efficient than pixel prediction models, with the performance gap widening as labeled data decreases. This paper establishes the core V-JEPA pretraining recipe subsequently scaled in V-JEPA 2 (Assran et al. 2025).

---

## [Terver et al. (2026) — JEPA World Models](papers/terver_2026_jepa_wm.md)
**Type:** paper
**Tags:** world-models, robot-learning, planning, model-predictive-control, jepa, ablation
This paper formalizes the JEPA-WM family (encoder + action-conditioned predictor on frozen SSL encoder) and systematically ablates six design axes: encoder, action conditioning, rollout depth, proprioception, planner, and cost metric. Key findings: CEM with L2 cost is the best planner; NeverGrad is a competitive tuning-free alternative on real-world data; gradient-based planners fail on non-smooth tasks; 2-step rollout is optimal; DINOv3 ViT-L outperforms V-JEPA encoders for manipulation; AdaLN is the best action conditioning scheme on average; proprioception consistently helps. The best configuration (DINOv3 + ViT-L depth-12 + AdaLN + RoPE + 2-step rollout + CEM L2) outperforms both V-JEPA 2-AC and DINO-WM. The paper also discovers and corrects a bug in the official V-JEPA 2-AC rollout loss implementation.

---

## [Intrinsic Motivation & Curiosity-Driven RL](concepts/intrinsic-motivation.md)
**Type:** concept
**Tags:** intrinsic-motivation, curiosity, exploration, reinforcement-learning, prediction-error, forward-dynamics
Intrinsic motivation provides reward signals generated by the agent itself — independent of task-specific extrinsic rewards — to drive exploration and skill acquisition without hand-designed reward functions. The main families are: (1) prediction-error / dynamics-based curiosity (reward for hard-to-predict transitions, as in ICM and RND); (2) count-based exploration (reward for rarely-visited states via pseudo-counts from density models); (3) information-theoretic methods (VIME rewards information gain; empowerment rewards control over future states). A core limitation of prediction-error curiosity is the "noisy-TV problem": stochastic environment elements generate high prediction error without real progress, attracting and trapping the agent. Random Network Distillation (RND) sidesteps this with a fixed deterministic target, while Random Features (RF) are surprisingly effective for in-distribution tasks though they fail to generalize across visual distribution shifts.

---

## [SIGReg (Sketched Isotropic Gaussian Regularization)](concepts/sigreg.md)
**Type:** concept
**Tags:** self-supervised-learning, distribution-matching, isotropic-gaussian, characteristic-function, regularization, theory
SIGReg is a regularization objective (Balestriero & LeCun, 2025) that enforces embedding distributions to be isotropic Gaussian by testing that all 1D random projections of the embeddings match a standard Gaussian — a consequence of the Cramér-Wold theorem. It uses the Epps-Pulley (1983) empirical characteristic function test, which has bounded gradients and O(N) time/memory complexity — unlike moment-based (unstable gradients) or CDF-based (non-differentiable, O(N log N)) alternatives. Balestriero & LeCun prove (Theorem 1) that isotropic Gaussian is the unique distribution minimizing worst-case downstream risk for linear, k-NN, and kernel probes; SIGReg is the tractable mechanism to achieve it. Only ~30 lines of PyTorch using `all_reduce` on complex exponentials for DDP support; a single hyperparameter (num_slices ≥ 512) is sufficient for reliable training.

---

## [Balestriero & LeCun (2025) — LeJEPA](papers/balestriero_2025_lejepa.md)
**Type:** paper
**Tags:** self-supervised-learning, representation-learning, jepa, theory, distribution-matching, isotropic-gaussian, sigreg, vision-transformer, in-domain-pretraining
LeJEPA (Latent-Euclidean JEPA) provides the first principled theoretical framework for JEPA-based SSL, proving that isotropic Gaussian embeddings uniquely minimize worst-case downstream prediction risk (linear and nonlinear probes). The paper introduces SIGReg, a sketched characteristic-function regularizer enforcing this distribution in O(N) time with bounded gradients and no curse of dimensionality, eliminating all heuristics (stop-gradients, teacher-student, whitening, schedulers) through a single trade-off hyperparameter λ. Empirically, LeJEPA trains stably across 60+ architectures from 8 families without modification, and its training loss exhibits ~85–99% Spearman correlation with downstream accuracy — enabling label-free model selection, a first for JEPA methods. In-domain pretraining on Galaxy10 (small dataset, specialized domain) outperforms frontier DINOv2/v3 transfer learning in both 1-shot and full-supervision regimes, validating that principled SSL beats generic transfer when the framework scales to any domain.

---

## [Cheng et al. (2023) — Extreme Parkour](papers/cheng_2023_extreme_parkour.md)
**Type:** paper
**Tags:** robot-learning, legged-locomotion, parkour, sim2real, teacher-student, perceptive-locomotion, depth-camera, curriculum-rl, ppo
This paper trains a single neural network policy to perform extreme parkour on the low-cost Unitree A1 quadruped using a single front-facing depth camera (10 Hz, jittery), with no elevation maps or external sensors. The core methodological contribution is a two-phase dual distillation: Phase 1 trains a teacher via PPO with privileged scandots (height map) and oracle heading from waypoints; Phase 2 distills both motor commands (scandots → depth via convnet-GRU + DAgger) and heading direction (oracle → predicted from depth) into a deployable student, using Mixture of Teacher and Student (MTS) to prevent distribution shift in the heading channel. A unified inner-product reward formulation (world-frame velocity toward waypoints + foot clearance penalty + optional style term) produces all diverse behaviors from a single training run without per-skill reward engineering. Results set a new SOTA for learning-based parkour: high jump 2× hip height (0.5m), long jump 2× body length (0.8m), tilted ramps at 37°, and quadruped-to-biped handstand transition — achieving near-oracle performance (MXD 0.92 vs 0.94) on real hardware.

---

## [Amin et al. (2025) — π*₀.₆ / RECAP](papers/amin_2025_pi06_experience.md)
**Type:** paper
**Tags:** robot-learning, vla, reinforcement-learning, advantage-conditioning, imitation-learning, real-world-rl, flow-matching, value-function
RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) is a general-purpose method for improving VLA models through real-world deployments, combining offline demonstrations, autonomous rollouts, and expert teleoperated corrections. The core mechanism is advantage conditioning: a binarized improvement indicator I_t ("Advantage: positive/negative") is injected as a text token into the VLA, allowing the policy to model both conditional and unconditional distributions — enabling policy extraction equivalent to regularized RL and classifier-free guidance sharpening at test time. A large multi-task distributional value function (670M-param, 201 bins) predicts time-to-completion, providing advantage estimates for the conditioning signal. On realistic long-horizon tasks (diverse laundry folding, espresso making, factory box assembly), RECAP roughly doubles throughput and halves failure rates compared to the offline RL + SFT baseline, achieving >90% success rate on most tasks.

---

## [Agarwal et al. (2023) — Dexterous Functional Grasping](papers/agarwal_2023_dex_func_grasp.md)
**Type:** paper
**Tags:** robot-learning, dexterous-manipulation, functional-grasping, sim2real, affordances, eigengrasp, dinov2
This paper tackles functional grasping of tools (hammers, drills, staplers, screwdrivers) with the LEAP dexterous hand via a modular three-stage pipeline: DINOv2 feature matching for one-shot affordance localization, eigengrasp-space sim2real RL for blind grasping, and MoCap-driven post-grasp trajectories. The key innovation is the eigengrasp action space — PCA of 16-DOF hand poses from a small VR demo dataset yields 9 principal components that constrain RL to physically realistic poses, reducing training variance ~15× and enabling perfect success rates on all training seeds. DINOv2 affordance matching substantially outperforms CLIP-based methods (CLIPPort, CLIPSeg) on unseen object categories because it captures part-level correspondences rather than whole-object semantics. On real hardware, the method beats a hardcoded baseline on all 7 objects and matches or exceeds a trained teleop oracle on 5/7, including surpassing human teleoperation on heavy hammer, stapler, and screwdriver where swift forceful motion is required.

---

## [van Hasselt et al. (2016) — Double DQN](papers/van_hasselt_2016_ddqn.md)
**Type:** paper
**Tags:** reinforcement-learning, q-learning, dqn, double-dqn, overestimation, value-based, atari
This paper identifies, theoretically characterizes, and fixes Q-learning's overestimation bias: the max operator uses the same values for both selecting and evaluating the best action, inducing an upward bias of at least $\sqrt{C/(m-1)}$ that the Double Q-learning estimate avoids entirely. The fix (Double DQN) changes a single line of DQN: action selection uses the online network $\theta_t$ while action evaluation uses the target network $\theta_t^-$, requiring no additional networks or parameters. Empirically, DQN overestimates on all 49 Atari games tested; Double DQN's value curves track true policy returns much more closely and produce more stable training (no score collapses). The median normalized Atari score improves from 93.5% to 114.7% (no-op) and from 47.5% to 116.7% (human starts, tuned), setting a new SOTA at the time.

---

## [Schulman et al. (2017) — PPO](papers/schulman_2017_ppo.md)
**Type:** paper
**Tags:** reinforcement-learning, policy-gradient, ppo, trust-region, actor-critic, continuous-control, atari, mujoco
PPO introduces a clipped surrogate objective $L^{CLIP}$ that clips the probability ratio $r_t(\theta)$ to $[1-\epsilon, 1+\epsilon]$ and takes the pessimistic minimum with the unclipped objective, preventing large policy updates with first-order optimization only. This enables multiple epochs of minibatch SGD on the same trajectory data — impossible to do safely with vanilla policy gradient — while being far simpler to implement than TRPO (no conjugate gradient, no second-order methods, compatible with shared architectures and dropout). A systematic comparison across 7 MuJoCo tasks confirms clipping with ε=0.2 (score 0.82) beats adaptive KL penalty (0.74) and fixed KL penalty (0.71); PPO outperforms A2C, TRPO, and CEM on continuous control and wins 30/49 Atari games on average training reward vs. 18/49 for ACER. PPO has become the standard base RL algorithm for curiosity and intrinsic motivation methods, robot learning, and LLM fine-tuning (RLHF).

---

## [Legged Locomotion & Perceptive Control](concepts/legged-locomotion.md)
**Type:** concept
**Tags:** robot-learning, legged-locomotion, quadruped, sim2real, teacher-student, perceptive-locomotion, curriculum-rl, depth-camera
Legged locomotion with RL follows a teacher-student paradigm: a teacher policy is trained in simulation with privileged information (dense height maps / scandots, oracle heading) via PPO and an automatic terrain curriculum; a student policy distills the teacher into a deployable network operating from depth images and proprioception via DAgger. A key challenge is heading direction distillation — the Mixture of Teacher and Student (MTS) strategy blends oracle and predicted headings during training to prevent catastrophic distribution shift. The unified inner-product reward (world-frame velocity toward waypoints, foot clearance penalty, optional style term) elicits diverse emergent behaviors — high jump, long jump, handstand — without per-skill reward engineering. Extreme Parkour (Cheng et al., 2023) demonstrates this paradigm on the low-cost Unitree A1, achieving 2× height jumps, 2× length gaps, and handstand from a single 10 Hz depth camera.

---

## [Vision-Language-Action Models (VLAs)](concepts/vision-language-action-models.md)
**Type:** concept
**Tags:** robot-learning, vla, foundation-models, imitation-learning, manipulation, flow-matching, behavior-cloning
A Vision-Language-Action (VLA) model is a robot policy combining a large pretrained VLM backbone (SigLIP + Gemma) with a dedicated flow-matching action expert, trained end-to-end via behavior cloning on heterogeneous multi-robot demonstration data. The action expert uses a stop-gradient (KI recipe) to prevent the backbone from being distorted by action gradients, and produces both continuous action chunks at 50 Hz and discrete sub-task tokens for high-level planning. VLAs are trained first via pre-training on tens of thousands of hours of demonstrations, then SFT on task-specific data, and optionally improved further via RL (e.g., RECAP). The core limitation is the imitation ceiling: compounding errors and distributional shift require RL or DAgger-style interventions to surpass the quality of the training demonstrations.

---

## [Sparse Gaussian Process Approximations](concepts/sparse-gaussian-processes.md)
**Type:** concept
**Tags:** gaussian-processes, sparse-gp, pseudo-datapoints, variational-inference, kl-divergence, scalable-inference, time-series, spatial-inference
Sparse GP methods reduce the $\mathcal{O}(N^3)$ cost of exact GP regression to $\mathcal{O}(NM^2)$ by summarizing the data with $M \ll N$ pseudo-datapoints, but standard methods (FITC, VFE, PIC) still require $M \propto N$ for large time-series or spatial datasets because each pseudo-datapoint is locally effective over a region of radius $\sim l_d$. The tree-structured GP approximation (Bui & Turner, 2014) breaks this by arranging pseudo-datapoints in a tree of $K$ blocks, where each block conditions on its parent; KL minimization against the true GP prior yields closed-form optimal conditionals $q(\mathbf{u}_{B_k}|\mathbf{u}_{\mathrm{par}}) = \mathcal{N}(\mathbf{A}_k\mathbf{u}_\mathrm{par}, \mathbf{Q}_k)$. Inference via the GBP up-down algorithm reduces to $\mathcal{O}(KD^3) = \mathcal{O}(N)$ for fixed block size $D$; for chain-structured temporal data, this is equivalent to the Kalman smoother. The tree-GP subsumes FITC, VFE, PIC, and the full GP as special cases and dominates the speed-accuracy frontier on audio missing data imputation and 2D terrain reconstruction.

---

## [Bui & Turner (2014) — Tree-structured GP Approximations](papers/bui_2014_tree_gp.md)
**Type:** paper
**Tags:** gaussian-processes, sparse-gp, pseudo-datapoints, belief-propagation, variational-inference, time-series, spatial-inference, kl-divergence
This paper introduces a tree-structured prior approximation for GP regression that achieves $\mathcal{O}(N)$ inference by partitioning $M$ pseudo-datapoints into $K$ tree-arranged blocks, where each block conditions on its parent via KL-optimal Gaussian conditionals. The resulting model is a tree-structured linear Gaussian system solved exactly by two GBP passes (up-down algorithm), equivalent to a Kalman smoother for chain/temporal structures. The approach subsumes FITC, VFE, and PIC as special cases (by setting $\mathbf{A}_k = \mathbf{0}$) and enables learning of hyperparameters via factored marginal likelihood gradients that involve only local GBP posteriors. On three challenging real-world tasks — audio sub-band imputation, audio spectral mixture imputation, and 2D terrain reconstruction (all with $N \approx 50$K–240K) — the tree-structured method dominates the speed-accuracy frontier of all compared methods (FITC, VFE, SSGP, SDE, local PIC) across every training and test time budget.

---

## [Gaussian Belief Propagation & Factor Graphs](concepts/gaussian-belief-propagation.md)
**Type:** concept
**Tags:** bayesian-inference, gaussian-belief-propagation, factor-graphs, probabilistic-graphical-models, distributed-inference, slam, robot-perception, message-passing
Gaussian Belief Propagation (GBP) is an iterative algorithm for marginal inference in Gaussian graphical models that operates by passing $(\eta, \Lambda)$ message pairs between variable and factor nodes of a factor graph — guaranteeing exact marginal means on convergence and connecting directly to solving the linear system $\Lambda\mu = \eta$. GBP's four key properties — local (no global coordination), probabilistic (full uncertainty estimates), iterative (runs continuously, tolerates new data), and asynchronous (message-order-independent) — make it uniquely suited for emerging distributed and heterogeneous computing hardware. Non-linear measurement functions are handled by first-order Taylor linearization with per-factor just-in-time relinearization; non-Gaussian distributions (outliers) are handled by covariance scaling (Huber energy). Message damping ($\tilde{\eta}_t = \beta\eta_t + (1-\beta)\tilde{\eta}_{t-1}$) improves convergence in loopy graphs, and Residual Belief Propagation prioritizes high-information messages; multiscale/coarse-to-fine acceleration addresses slow long-range propagation on grids.

---

## [Ortiz et al. (2021) — Visual Introduction to GBP](papers/ortiz_2021_gbp.md)
**Type:** paper
**Tags:** bayesian-inference, gaussian-belief-propagation, factor-graphs, distributed-inference, probabilistic-graphical-models, slam, robot-perception
This tutorial paper argues that Gaussian Belief Propagation is the right inference algorithm for future distributed ML systems because it is local, probabilistic, iterative, and asynchronous — matching the properties of emerging heterogeneous hardware (graph processors, neuromorphic chips, edge devices). The paper derives GBP from factor graphs and Gaussian models, showing that inference reduces to solving $\Lambda\mu = \eta$ distributed across nodes with only local communication. Four practical extensions are covered: linearization of non-linear factors (with Jacobians), covariance scaling for robust/non-Gaussian distributions (Huber energy), message scheduling strategies (synchronous, random, sweep, residual BP), and multiscale/coarse-to-fine acceleration for grid graphs. Applications demonstrated include geometric pose estimation, image denoising with Huber loss, and robot SLAM pose graphs; a referenced result shows 24× speedup for bundle adjustment on a graph processor.

---

## [Nabarro et al. (2024) — GBP Learning](papers/nabarro_2024_gbp_learning.md)
**Type:** paper
**Tags:** gaussian-belief-propagation, factor-graphs, learning-as-inference, bayesian-deep-learning, continual-learning, distributed-training, energy-based-models, predictive-coding, image-classification, denoising
Nabarro, van der Wilk & Davison (ICML 2024) build Gaussian factor graphs that mirror NN layers (conv, transposed conv, max-pool, upsample, dense, softmax observation), with inputs, activations, outputs *and parameters* all as variable nodes. Training and prediction are the same GBP inference with different variables observed: there is no backprop and no learning rate, only local Jacobians of relinearized non-linear factors, stabilised by message damping and message dropout. Woodbury-based low-rank updates bring the cost of a full message sweep to $O(BLC^2)$, the same as a backprop pass. Continual learning and minibatching are Bayesian filtering: each batch's parameter posterior becomes the next prior, and data is seen once. Results: learnable 5-layer factor graphs beat a hand-designed pairwise GBP smoother on unsupervised video denoising (continual > per-frame learning). Single-epoch MNIST reaches 98.16% (≈ CNN + 6k-example replay buffer, 98.11% with random asynchronous layer schedules), and the method beats Lucibello et al. on CIFAR10 (53.1 vs 41.3%). Limitations: ~3 h on an RTX 3090 for MNIST, scalar variable nodes (no weight correlations), no convergence guarantees, small-scale experiments only.

---

## [Liu et al. (2025) — LocoFormer](papers/liu_2025_locoformer.md)
**Type:** paper
**Tags:** legged-locomotion, cross-embodiment, in-context-adaptation, meta-rl, transformer-xl, sim2real, domain-randomization, ppo, procedural-generation, humanoid, quadruped, wheeled-legged
LocoFormer (Skild AI, CoRL 2025) is a single omni-bodied locomotion policy that controls unseen legged and wheeled robots zero-shot, without being given their kinematics. Two choices make it work: massive-scale PPO on ~100k procedurally generated bipeds/quadrupeds (and wheeled variants) with aggressive dynamics randomization in a unified superset joint space, and a Transformer-XL policy with ~18 s of memory at 50 Hz trained on an RL²-style multi-trial objective, so memory persists across falls and resets. On 10 unseen simulated robots it reaches 0.96 normalized displacement zero-shot and 0.98 after 5 s of adaptation (per-robot expert 0.99; GRU 0.37; short-context morphology-conditioned Transformer 0.78). Internal representations of different humanoids separate into embodiment clusters within ~5 s. On real Unitree G1/H1/Go2/Go2-W it shows emergent adaptation to locked knees (2–3 s), cut lower legs (7–8 s), stilts, locked wheels (switches to walking) and a no-ankle biped mode learned across trials. Limitations: ~500× the compute of a specialist (amortised over 100k robots), a hand-crafted procedural task space, blind velocity-tracking only, and qualitative real-world results.

---

## [In-Context Adaptation & Cross-Embodiment Policies](concepts/in-context-adaptation.md)
**Type:** concept
**Tags:** in-context-adaptation, meta-rl, cross-embodiment, history-conditioned-policy, transformer-xl, domain-randomization, legged-locomotion, system-identification
In-context adaptation means a fixed-weight policy changes its behaviour at test time by conditioning on its history, which amounts to implicit system identification in the activations rather than via gradient updates. A cross-embodiment policy controls many robot bodies with one network, and combining the two lets the policy infer which body it controls. The article places this among robust (memoryless) policies, explicit morphology conditioning, short-history RMA-style adaptation modules (~100s of ms, called myopic), long-context RL²/LocoFormer adaptation (seconds, across trial resets), and gradient-based meta-RL or fine-tuning. The ingredients drawn from LocoFormer are a task distribution wide enough to force identification, a multi-trial objective that rewards using failures, a long-memory architecture (Transformer-XL + KV-cache; a GRU fails), and a unified joint interface. Open issues: compute cost, hand-designed task spaces, unmeasured effective memory use, and the fact that it cannot accumulate knowledge beyond its context window, unlike weight-space continual learning.

---

## [Fedele et al. (2025) — SuperDec](papers/fedele_2025_superdec.md)
**Type:** paper
**Tags:** 3d-scene-representation, superquadrics, shape-abstraction, primitive-decomposition, point-clouds, unsupervised-segmentation, transformer, levenberg-marquardt, path-planning, grasping, controllable-generation
SuperDec (ETH/Stanford/Microsoft) builds compact 3D scene representations by decomposing point clouds into superquadrics (11 parameters each, with a closed-form radial distance). A class-agnostic, self-supervised model (PVCNN point features, Transformer decoder with 16 primitive queries, soft point-to-primitive segmentation, existence head) is trained with bidirectional Chamfer + normal + parsimony + existence losses, then refined by Levenberg–Marquardt. Full scenes are handled per instance via Mask3D. Trained only on 13 ShapeNet classes, it reaches L2 Chamfer 0.047 (×10²) vs 0.279 for the best learned baseline with ~half the primitives, stays strong out of category (0.061), and generalises to real ScanNet++/Replica objects (L2 0.11/0.19 vs 0.41/0.70 for CSA). Downstream it supports RRT* planning at ~0.04 MB per scene (91.7% success vs 89.6% for dense point clouds, below voxels), analytic superquadric grasping on a Spot arm (qualitative), and ControlNet-based spatial/semantic image editing. Caveats: dependence on the instance segmenter, a thin robotics evaluation, a λ_par inconsistency (0.06 vs 0.6), and an ε_exist threshold that does not match its definition.

---

## [Superquadric & Primitive-Based Scene Representations](concepts/superquadric-scene-representations.md)
**Type:** concept
**Tags:** 3d-scene-representation, superquadrics, shape-abstraction, primitive-decomposition, point-clouds, path-planning, grasping
Primitive-based shape abstraction represents objects and scenes as a few simple parametric volumes, trading photorealism (NeRF, 3DGS) for compactness, interpretability and editability; a room costs tens of KB. Superquadrics (Barr 1981) use 3 scales + 2 shape exponents + 6-DoF pose and span boxes, ellipsoids, cylinders and octahedra, with an inside/outside test, a closed-form radial distance, and explicit surface sampling. Generalised ellipsoids were rejected in SuperDec because they lack a closed-form distance. Decomposition methods range from learned global-code models (category-specific) through optimisation (EMS, Marching-Primitives, DBW: accurate but slow and heuristic) to SuperDec's local-feature Transformer + LM refinement, which is class-agnostic. The shared recipe is self-supervised Chamfer, a parsimony penalty and existence probabilities. Robotics uses include memory-light collision checking, analytic geometric grasping, and editable scene proxies for generative models. Open issues are segmentation dependence, thin/concave geometry, and small-scale robot evaluations.

---

## [Zhou et al. (2024) — DINO-WM](papers/zhou_2024_dino_wm.md)
**Type:** paper
**Tags:** latent-world-models, world-models, dinov2, pretrained-visual-representations, model-predictive-control, cem, zero-shot-planning, offline-learning, goal-conditioned, manipulation, deformable-objects
DINO-WM (NYU/Meta) is a task-agnostic world model that predicts future frozen DINOv2 patch features (14×14×384) with a ~19M-parameter ViT using a frame-level causal mask. Action and proprioception embeddings are concatenated to every patch, and it is trained with a teacher-forced latent MSE on reward-free offline trajectories, with no pixel reconstruction (a decoder is trained separately for visualisation only). Tasks are solved zero-shot as image-goal reaching via CEM-MPC on terminal latent distance. Across PointMaze, Wall, Reacher, Push-T, Rope and Granular it matches DreamerV3 on navigation and clearly wins on manipulation (Push-T 0.90 SR vs 0.32; Granular CD 0.26 vs 0.37). TD-MPC2 scores 0 without rewards. It also generalises best to unseen wall layouts, shapes and particle counts. Ablations: patch tokens ≫ CLS/ResNet/R3M (Push-T 0.90 vs ≤0.44); the causal mask is required for longer history; a decoder loss hurts (0.92 → 0.80); MPC > open-loop CEM ≫ gradient descent; success scales with data (0.08 → 0.92 from 200 to 18.5k trajectories). Limitations: simulation only, needs action labels and data coverage, ~53 s per CEM plan, and Push-T data derived from noisy expert replays. It conflicts with Terver et al.'s reported 32% Push-T for DINO-WM (likely a protocol difference).

---

## [Pretrained Visual Representations for Robotics](concepts/pretrained-visual-representations.md)
**Type:** concept
**Tags:** pretrained-visual-representations, dinov2, self-supervised-learning, patch-features, world-models, manipulation, representation-learning
Robot systems can reuse frozen encoders pretrained on large image or video corpora (ImageNet ResNet, R3M, MVP, DINO/DINOv2/DINOv3, I-/V-JEPA) and spend scarce robot data only on dynamics, policy or value. Key wiki evidence: in DINO-WM, swapping encoders under the same world-model recipe shows that all encoders solve easy navigation, but global-vector encoders (ResNet, R3M, DINOv2 CLS) collapse on spatial manipulation, while DINOv2 patch tokens reach Push-T 0.90 vs ≤0.44. Terver et al. find image SSL (DINOv2/v3) beats video SSL (V-JEPA/V-JEPA 2) for manipulation world models, and Agarwal et al. use dense DINOv2 features for one-shot affordance correspondence. Design considerations: freezing keeps priors and cuts cost but leaves nuisance variation in latent costs; reconstruction coupling hurts; patch grids make planning heavy (~53 s per CEM plan in DINO-WM).

---

## [rl_lab / Overview](codebase/rl_lab/_overview.md)
**Type:** codebase
**Tags:** rl_lab, ppo, maniskill, so100, pick-and-place, evaluation-protocol, swifttd, dinov2, research-roadmap
rl_lab is the user's personal research repo (Jun–Sep 2026). It trains a from-scratch PPO on ManiSkill 3's PickCubeSO100-v1 (1,024 GPU envs, 36-d privileged state, pd_joint_delta_pos) and wraps it in a frozen evaluation protocol that serves as the instrument for a staged plan. The research question: can a SwiftTD-trained linear critic on frozen DINOv2 + depth features match a deep MLP critic with less compute and no replay? Stage 0 is closed (privileged baseline 0.909 ± 0.030 deterministic success over 3 seeds at γ = 0.95). Stage 1 (visual observations via an env wrapper, 3,560-d features) is in planning, and stage 2 (SwiftTD critic) has not started. The overview has a Mermaid architecture diagram and component map. Other notes: an empty README, planning docs in Spanish, unlisted dependencies (genesis, sklearn, tensordict), and legacy MuJoCo/SAC/Genesis/JEPA code outside the main path.

---

## [rl_lab / PPO Agent](codebase/rl_lab/ppo-agent.md)
**Type:** codebase
**Tags:** rl_lab, ppo, gae, actor-critic, gaussian-policy, entropy, value-clipping, rollout-buffer
The PPO has separate MLP actor (Gaussian, state-independent log_std) and critic networks, one Adam optimiser, clipped surrogate and clipped value losses, joint gradient clipping with separately logged actor/critic gradient norms, and a KL-adaptive LR (pinned constant in the ManiSkill config). Its key design is a hard log-std floor of −1.2 (σ = 0.301), applied after each step. Without it σ collapsed within ~150 iterations and the agent never learned to place, and an entropy bonus cannot fix this because ∂H/∂log σ is constant while the surrogate's push grows with advantages. VecRolloutBuffer computes GAE with V(s_final) bootstrapping on auto-reset truncations and rollout-wide advantage normalisation. A flagged potential issue: absolute value-clip ε = 0.2 against value targets near 20 at γ = 0.95.

---

## [rl_lab / ManiSkill Training Pipeline](codebase/rl_lab/maniskill-training-pipeline.md)
**Type:** codebase
**Tags:** rl_lab, maniskill, ppo, gpu-parallel-simulation, reward-design, terminations, discount-factor, wandb, so100
The training script runs 50-step rollouts on 1,024 envs, then GAE, a PPO update (8 epochs × 32 minibatches), frozen-protocol eval every 100 iterations, and logging keyed by env-steps with cumulative wall-clock (TensorBoard + optional W&B), at ~19.8k steps/s on an RTX A6000. Training ignores terminations because terminate-on-success made grasp-and-stall (0.53/step forever) worth 2.65× more than finishing; letting success keep paying 1.0/step fixes it at any γ (~30% vs ~1% success for the official bootstrap approach). The γ pilot showed that at γ = 0.8, V ≈ r_t (reward-only R² 0.55) with a 2-step trace, so γ = 0.95 was adopted: it doubled success (0.45 → 0.91 det), roughly tripled sample efficiency, and made value learning non-trivial for the SwiftTD study. Logging is hardened after a mid-run directory deletion killed a 2,362-iteration run, and one config comment misstates the official γ.

---

## [rl_lab / Evaluation Protocol](codebase/rl_lab/evaluation-protocol.md)
**Type:** codebase
**Tags:** rl_lab, evaluation-protocol, reproducibility, seeds, success-rate, baseline, maniskill
rl/evaluation.py is frozen: 128 episodes, reserved eval seeds 90,000/90,017, and deterministic (headline) plus stochastic passes. Three measured properties are load-bearing: equal per-env quotas (first-N-to-finish over-samples successes: 0.470 vs 0.375), reconfigure-on-reset (a noise floor of ≈ 0 vs a 0.38–0.47 spread without it), and dual mode (σ floor makes mean and sampled policies differ). Metrics are success, steps-to-success over successes only, "final" as the mean of the last 5 evals, and a sustained 80% crossing; mean return is not a quality axis because better policies end earlier. The frozen 3-seed baseline is 0.909 ± 0.030 det (0.944 ± 0.005 stoch), steps-to-success 28.18 ± 0.15, and 56.3 ± 8.9M env-steps to 80%, implying a ~0.06 detection threshold and a Stage-1 target of 0.64–0.73. Seed-44 anomalies (high-KL phase, still climbing) are recorded, and the eval docstring's calibration numbers disagree with PROTOCOL.md.

---

## [rl_lab / Research Roadmap](codebase/rl_lab/research-roadmap.md)
**Type:** codebase
**Tags:** rl_lab, research-plan, swifttd, dinov2, depth, linear-critic, teacher-student, distillation, observation-contract, compute-budget
The staged plan: Stage 0 (instrument, closed), Stage 1 (perception), Stage 2 (SwiftTDNonSparse linear critic replacing the PPO critic, one trace stream per env, compared on success, critic compute, lifetime error and a per-feature credit heatmap over DINO patches), Stage 3 (write-up) and Stage 4 (actor step sizes, Genesis). The Stage-1 contract drops the 13 cube-dependent dimensions and keeps 23 (goal_pos stays because the goal is invisible to cameras, so the honest claim is "no privileged object information"; is_grasped is a documented concession). Perception will be a vec-env wrapper so the agent and eval stay byte-identical: 126² renders give 9×9 DINOv2 ViT-S/14 patches, mean-pooled 3×3 to 3,456-d, plus 9×9 depth, for 3,560-d with a frozen normaliser. The measured cost is 3.5× per sample (bf16 DINO 4.84 s/iteration), budgeted to ~26 GPU-hours. A pre-committed Plan B (<0.30 at iteration 1,500) switches to teacher–student BC/DAgger then PPO, and a modality ablation anticipates that depth alone may suffice.

---

## [rl_lab / Legacy MuJoCo & Auxiliary](codebase/rl_lab/legacy-mujoco-and-auxiliary.md)
**Type:** codebase
**Tags:** rl_lab, mujoco, so100, reward-shaping, sac, genesis, sigreg, jepa, identifiability
This article covers code outside the main pipeline. The custom MuJoCo SO-100 pick-and-place env used absolute joint-position actions, a 26-d base-frame observation, contact-based grasp scoring, and a staged reward with strict dominance (reach 0.1 → grasp 0.4 → lift 0.9 → carry ~1.9 → placed ~5.2) trained by a CPU PPO script. The SAC agent is an unfinished exercise with placeholder methods, so its scripts cannot run. The Genesis files are a render demo and an unreferenced batched Franka grasp env with stereo cameras, DLS IK and a keypoint reward. utils/train_jepa.py trains a CartPole encoder with alignment + SIGReg and measures linear identifiability (R² to the true state); it has hardcoded foreign dataset paths and relates to the uncompiled IdentJEPA.pdf.

---

## [RL Evaluation Methodology](concepts/rl-evaluation-methodology.md)
**Type:** concept
**Tags:** evaluation-methodology, reproducibility, seeds, statistical-significance, success-rate, gpu-simulation, rl-benchmarking
Staged RL research needs a measuring instrument frozen before the first comparison, because small scoring choices can exceed the effects studied. Measured pitfalls from rl_lab: duration-biased sampling in vectorised eval (fixed by equal per-env quotas), residual GPU-sim state (fixed by scene reconfiguration plus a measured noise floor), deterministic vs stochastic policy gaps (score both, compare within mode), eval-seed leakage, last-checkpoint noise (report the mean of the last k evals), first-touch threshold crossings (require sustained crossings), return misleading under terminate-on-success, and wall-clock contaminated by GPU sharing. Seed spreads should set detection thresholds (≈ 2× std), different metrics differ widely in tightness, and decision rules should be pre-committed. It contrasts with lifetime error in online learning and warns that cross-paper protocol differences can dominate comparisons.

---

## [Object-Centric Persistent Slots + SwiftTD Critic (research design)](open_questions/object-centric-swifttd-critic.md)
**Type:** open_question
**Tags:** object-centric-representation, superquadrics, dinov2, swifttd, step-size-adaptation, linear-critic, tile-coding, ppo, maniskill, rl_lab, research-design
This proposal organises rl_lab's observation into persistent object slots. Shape (SuperDec superquadrics) and identity/semantics (a DINOv2 ROI-crop embedding, PCA-reduced) are computed once per episode, while pose is re-fit every step with the shape frozen, rigidly attached to the TCP while grasped, and held with decaying confidence when occluded. The actor gets a ~80-d continuous slot vector. The SwiftTD critic gets a sparse binary tile coding of relations (TCP↔object, object↔goal, height, grasp state), which matches SwiftTD's validated sparse-binary linear regime, addresses the R² ≈ 0.45 limit of linear-on-state value, and turns step-size credit into object × relation feature selection. Caveats: SuperDec is overkill for a single cube (the pipeline mostly reconstructs obj_pose from depth), segmentation must be non-privileged (table-plane removal; GT segmentation only as a labelled oracle arm), and SwiftTD over 1,024 parallel streams is unsolved (recommended: train on a subset of streams, predict for all). The plan: three offline de-risking checks (tile-feature R², SwiftTD in prediction mode on logged data, per-phase tracking error), then pre-committed arms (P, G, O-oracle, O, O-S, G-S) under the frozen protocol. Compute is estimated at ~6 s/iteration vs ~10 s for per-step DINO.

---

## [Discrete-Time Gaussian Processes for Imitation Learning](concepts/discrete-time-gaussian-processes.md)
**Type:** concept
**Tags:** imitation-learning, gaussian-processes, policy-representation, robot-manipulation, multimodal, few-shot, riemannian-geometry, inference-time-adaptation, cross-embodiment
A Discrete-time Gaussian Process (DiGaP) is a finite sequence of per-timestep Gaussians fitted independently to trajectory demonstrations on Riemannian manifolds; diagonal covariance prevents spurious inter-dimensional correlations and the model scales linearly with data. Unlike continuous GPs (kernel-constrained) or GMMs (locally linear), DiGaP models oscillatory, piecewise, non-stationary, discontinuous, and chaotic trajectories. Its mixture variant MiDiGaP clusters demonstrations into trajectory modes (using Riemannian k-means/DBSCAN/GMM) and supports skill sequencing for long-horizon tasks via TAPAS. Inference-time adaptation is achieved through Constrained Gaussian Updating (moment-matching to incorporate collision/reachability evidence) and VAPOR (Variance-Aware Path Optimization), which converts probabilistic end-effector trajectories to kinematically feasible joint trajectories and enables cross-embodiment transfer.

---

## [von Hartz et al. (2025) — MiDiGaP](papers/vonhartz_2025_midigap.md)
**Type:** paper
**Tags:** imitation-learning, gaussian-processes, robot-manipulation, policy-representation, multimodal, few-shot, task-parameterization, cross-embodiment, inference-time-adaptation
MiDiGaP introduces Discrete-time Gaussian Processes (DiGaP) and their mixtures as a flexible, sample-efficient policy representation for robot manipulation imitation learning — learning from as few as 5 demonstrations on CPU in under 1 minute. The core insight is that modeling trajectories as per-timestep Gaussian distributions (with diagonal covariance on Riemannian manifolds) avoids both the kernel-function limitations of continuous GPs and the local-linearity and spurious-correlation failures of GMMs, enabling modeling of highly constrained, dynamic, and multimodal behaviors. Constrained Gaussian Updating provides inference-time adaptation to novel obstacles and reachability constraints via moment matching, substantially outperforming Diffusion Policy + ITPS (0.98 vs. 0.14 avg). VAPOR (Variance-Aware Path Optimization) enables effective cross-embodiment transfer (Franka→UR5, 0.68–0.73 avg success) and reduces trajectory cost 67% vs. TAPAS-GMM; on a real Franka robot, MiDiGaP achieves 1.00 unimodal and 0.99 multimodal success vs. 0.00 for all deep learning baselines.

---

## [Visual Affordances for Robotics](concepts/visual-affordances-robotics.md)
**Type:** concept
**Tags:** robot-learning, affordances, human-video, contact-points, trajectory, representation-learning, manipulation
A visual affordance for robotics is defined robot-first as a (contact point c, post-contact trajectory τ) pair: c is the pixel location where the robot should make initial contact, and τ is the sequence of relative end-effector displacements after contact — together encoding *where* and *how* to interact, independently of human morphology. This representation is learned from egocentric video using automated supervision: a hand-object detector finds contact frames, skin-color segmentation extracts contact candidates, a GMM captures multi-modality, and homography compensation handles camera ego-motion; affordances are projected onto the human-less first frame to resolve the visual domain shift at deployment. A single affordance model f_θ (ResNet encoder + K deconvolutional heatmap heads + Transformer trajectory network) plugs into four distinct robot learning paradigms: offline IL data collection (k-NN / BC), reward-free exploration (biasing toward contact regions, ranking by environment change), goal-conditioned RL (sampling toward goal-image-minimizing trajectories), and DQN action-space parameterization. VRB (Bahl et al. 2023) demonstrates this versatility across 10 real-world tasks on 2 robot platforms, achieving 57% average IL success and 3–10× exploration improvement over random baselines.

## [Learning Robot Manipulation from Human Videos](concepts/learning-from-human-videos.md)
**Type:** concept
**Tags:** robot-learning, imitation-learning, human-video, visual-imitation, embodiment-gap, one-shot, in-the-wild
Learning robot manipulation from third-person human video is appealing because human demonstrations are abundant and free, but faces three core challenges: embodiment mismatch (different morphologies), no action labels, and no reward signal. The standard pipeline extracts a structured *prior* from video (hand positions, contact events, wrist orientation via 100DOH/MANO detectors) and maps it to robot waypoints; a residual policy then learns corrections to this prior via real-world interaction. The embodiment gap is bridged by an *agent-agnostic representation*: both human and robot videos are inpainted to remove the agent, then embedded with an action recognition model (SlowFast 3D ResNets) whose output captures task semantics rather than agent appearance. A CEM-style zeroth-order optimizer samples action residuals, executes them in the real world, ranks them by agent-agnostic cost, and fits the policy (a CVAE) to the elite set; an exploration policy simultaneously maximizes frame-wise visual change to avoid local minima.

## [Bahl et al. (2023) — VRB](papers/bahl_2023_vrb.md)
**Type:** paper
**Tags:** robot-learning, affordances, human-video, egocentric-video, imitation-learning, exploration, goal-conditioned-rl, action-space, contact-points
VRB (Vision-Robotics Bridge) learns robot-centric visual affordances — (contact point c, post-contact trajectory τ) — from large-scale egocentric human video using automated supervision (100DOH hand-object detector, GMM over contact candidates, homography ego-motion compensation) and a ResNet + deconv + Transformer architecture trained on human-less frames to avoid domain shift. Unlike prior affordance methods (Hotspots, HAP, HOI), VRB explicitly handles multi-modality and camera motion, and its representation is directly transferable to any robot regardless of morphology. The same trained model f_θ seamlessly supports four robot learning paradigms without re-training: offline IL data collection (best on 7/8 tasks), reward-free exploration (3–10× over random), goal-conditioned RL (faster convergence on 6 tasks), and DQN over discretized (c, τ) action spaces. Evaluated on 10 real-world tasks across 4 environments on 2 robot platforms (Franka + Hello Stretch), spanning hundreds of robot-hours — one of the largest real-world robot learning evaluations at its time.

## [Bahl et al. (2022) — WHIRL](papers/bahl_2022_whirl.md)
**Type:** paper
**Tags:** robot-learning, imitation-learning, human-video, visual-imitation, embodiment-gap, real-world-rl, one-shot, manipulation
WHIRL (In-the-Wild Human Imitating Robot Learning) is the first framework to learn robot manipulation from unstructured third-person human video at scale, demonstrated on 20 diverse tasks across 3 real-world environments. The method extracts a prior Ψ_k from a single human video (hand waypoints h_interaction/h_mid/h_end, wrist orientation, gripper state), initializes a residual CVAE policy to sample perturbations ΔΨ around the prior, and iteratively improves via real-world rollouts ranked by an agent-agnostic cost ||Φ(V_human) - Φ(R_robot)||_2 where Φ inpaints the agent and uses action recognition features. An exploration policy maximizes frame-wise visual change to escape local minima. After 3 iterations (~20 minutes each on a Hello Robot Stretch), WHIRL achieves 83% drawer success and 92% door success, versus 53% and 30% for behavior cloning — with no robot demonstrations, no reward engineering, and no simulation.

## [Hansen et al. (2025) — Newt / MMBench](papers/hansen_2025_newt.md)
**Type:** paper
**Tags:** world-models, model-based-rl, multitask-rl, continuous-control, self-predictive, td-mpc2, language-conditioning, benchmark, online-rl
Newt and MMBench address massively multitask online RL for continuous control: can a single agent train on 200 tasks simultaneously via online interaction? MMBench is the first benchmark of this kind, spanning 10 domains (DMControl, Meta-World, ManiSkill3, Atari, and more) with language instructions, 10–40 demonstrations per task, and optional image observations. Newt extends TD-MPC2 — a self-predictive control-centric world model — with frozen CLIP (language) and DINOv2 (image) encoders, discrete reward/value regression across log-transformed bins, and four complementary demonstration strategies (model pretraining, constrained planning, oversampling, action supervision). Key findings: language conditioning raises scores from 0.371 → 0.438; model and batch size scale beneficially in multitask RL (unlike single-task); Newt zero-shot transfers to unseen tasks (0.192 vs. 0.013 from scratch) and fine-tunes rapidly (0.868 vs. 0.480 at 100k steps); open-loop planning succeeds over 48-step horizons (16× training horizon).

---

## [Dynamic Movement Primitives & Trajectory-Space Policies](concepts/dynamic-movement-primitives.md)
**Type:** concept
**Tags:** robot-learning, policy-representation, dynamic-movement-primitives, trajectory-space, action-reparameterization, classical-robotics, imitation-learning, reinforcement-learning
A Dynamic Movement Primitive (DMP) represents a robot motion as a second-order ODE: ÿ = α(β(g−y) − ẏ) + f(x,g), where g is a goal attractor and f is a forcing function (weighted Gaussian RBFs over a decaying phase variable x) — giving smooth, goal-directed, time-invariant trajectories parameterized by (w, g). Neural Dynamic Policies (NDPs, Bahl et al. 2020) embed a DMP as a fully differentiable layer in a neural policy: the network predicts (w, g) from observations, the ODE forward integrator generates k actions, and gradients flow back through closed-form ∂f/∂w and ∂f/∂g — enabling end-to-end training in both IL and RL (PPO + multi-action critic). Classical DMP limitations (unimodal, no uncertainty, fixed kernel) motivated later trajectory-space approaches: DiGaP models per-timestep Gaussians on Riemannian manifolds, and MiDiGaP extends this to mixtures for multi-modal few-shot IL. The design space spans ODE-based (compact, physically interpretable, RL-compatible) vs. per-timestep Gaussian (expressive, multi-modal, Riemannian) vs. denoising/flow (maximally expressive, IL-oriented).

---

## [Successor Features & Successor Measures](concepts/successor-features.md)
**Type:** concept
**Tags:** reinforcement-learning, zero-shot-rl, successor-features, representation-learning, linear-reward, temporal-difference
Successor features (SFs) decouple the dynamics of an MDP from the reward function: for a feature function ψ and policy π, the successor feature F_ψ^π(s,a) = E[Σ γ^t ψ(s_{t+1})] satisfies a vector Bellman equation and yields Q-values for any linear reward r(s) = ψ(s)^T z_r via Q_r^π = F_ψ^π^T z_r. This enables zero-shot transfer: learn SFs once during reward-free pre-training, then solve any new task at test time by projecting its reward onto ψ via linear regression and reading off the pre-trained policy. The successor measure M^π(·|s,a) is the unnormalized discounted future-state distribution from which SFs are the expectation under ψ; minimizing a successor measure approximation loss directly bounds zero-shot policy evaluation error. TD-JEPA (Bagatella et al. 2025) shows that a JEPA predictor trained with TD bootstrapping recovers successor features in the latent space of the encoder.

## [Zero-Shot / Unsupervised RL](concepts/zero-shot-unsupervised-rl.md)
**Type:** concept
**Tags:** zero-shot-rl, unsupervised-rl, successor-features, representation-learning, task-agnostic, offline-rl
Zero-shot (unsupervised) RL separates learning into reward-free pre-training (exploration + representation learning from an offline dataset) and test-time task solving (recover optimal policy from a tiny reward dataset, no further environment interaction). The dominant paradigm trains a task encoder ψ spanning rewards of interest and policies {π_z}_{z∈Z}; at test time z_r = argmin E[(r-ψ(s)^T z)²] is solved in closed form and π_{z_r} is executed immediately. Key design choices are: how to train ψ (contrastive/Laplacian/latent-predictive), whether to maintain a separate state encoder φ, and what dynamics target to predict (one-step behavioral vs. multi-step zero-shot policy successor measures). TD-JEPA (Bagatella et al. 2025) is the current SOTA on ExoRL+OGBench; it uses TD bootstrapping to avoid on-policy data and trains both φ and ψ via symmetric latent-predictive losses.

## [Bahl et al. (2020) — Neural Dynamic Policies](papers/bahl_2020_ndp.md)
**Type:** paper
**Tags:** robot-learning, policy-representation, dynamic-movement-primitives, trajectory-space, action-reparameterization, imitation-learning, reinforcement-learning, end-to-end, ppo
NDPs bridge classical robotics and deep learning by reparameterizing the action space of a deep policy via second-order ODEs: the network Φ predicts DMP parameters (basis weights w, goal g) from state/image, a forward integrator produces k actions per forward pass, and the system is trained end-to-end via closed-form gradients through the ODE. A multi-action critic (k heads for k rollout steps) enables effective PPO training under sparse rewards and high control frequency. NDP outperforms raw-action PPO and PPO-multi on throwing and picking (dynamic tasks) and faucet/pushing (quasi-static), and outperforms CNN baselines on digit writing by ~14× in test loss; inference runs at 0.5–5 kHz vs. 100 Hz environment. Key limitations: unimodal (one trajectory per forward pass), MT50 shows no efficiency gains, and contact-rich precision tasks (picking, IL) are harder for NDP than for raw-action baselines.

---

## [Xu et al. (2025) — RLT / RL Token](papers/xu_2025_rlt.md)
**Type:** paper
**Tags:** robot-learning, vla, reinforcement-learning, online-rl, actor-critic, chunked-actions, sample-efficiency, real-world-rl, representation-learning
RLT (RL Token) is a sample-efficient method for online RL fine-tuning of pretrained VLA models, requiring only a few hours of real-robot practice. The core idea is an encoder-decoder transformer added to a frozen VLA that compresses its internal features into a compact 1×2048 RL token, which then serves as the state for a lightweight MLP actor-critic trained off-policy with chunked actions (C=10 steps) and a BC regularizer toward VLA reference actions. Reference action dropout (50% of transitions) prevents the actor from copying VLA actions before the critic becomes informative. Evaluated on 4 real-robot precision tasks (screw installation, zip tie fastening, Ethernet insertion, charger insertion), RLT improves critical-phase execution speed up to 3× and success rates substantially (e.g., 20%→65% on screw), even discovering emergent insertion strategies absent from demonstration data.

---

## [Video-Action Models (VAMs)](concepts/video-action-models.md)
**Type:** concept
**Tags:** robot-learning, video-action-model, imitation-learning, flow-matching, video-generation, inverse-dynamics, sample-efficiency
A Video-Action Model (VAM) is a robot policy paradigm that pairs a pretrained generative video backbone (e.g., a latent diffusion transformer) with a lightweight Inverse Dynamics Model (IDM) action decoder, bypassing the fundamental limitation of VLAs — their VLM backbones are trained on static image-text data and must infer physical dynamics from scarce robot trajectories. By grounding the policy in Internet-scale video pretraining, VAMs inherit rich priors about how objects move, deform, and react to forces, reducing the action decoder's task to a simple visual-plan-to-motor-command translation. The key architectural trick is *partial denoising*: rather than fully reconstructing future video frames, the backbone extracts intermediate latent representations at a high noise level (τ_v ≈ 1) that are computationally cheap (single forward pass) and empirically superior conditioning signals for the action decoder. mimic-video (Pai et al. 2025) demonstrates 10× sample efficiency and 2× faster convergence over comparable VLA baselines.

## [Bagatella et al. (2025) — TD-JEPA](papers/bagatella_2025_tdjepa.md)
**Type:** paper
**Tags:** zero-shot-rl, unsupervised-rl, successor-features, latent-predictive, jepa, temporal-difference, representation-learning
TD-JEPA introduces a novel zero-shot unsupervised RL method that trains a state encoder φ, task encoder ψ, policy-conditioned multi-step predictor T_φ, and a family of parameterized policies {π_z} entirely from offline, reward-free transitions, using a TD bootstrapping JEPA loss: L = E[||T_φ(φ(s),a,z) - ψ̄(s') - γT̄_φ(φ̄(s'),a',z)||²]. The key insight is that the predictor trained this way approximates successor features F_ψ^{π_z}(s,a), enabling zero-shot policy extraction for any new reward by linear regression onto ψ followed by policy retrieval — no additional environment interaction needed. Three theoretical results support the design: (1) non-collapse guaranteed by covariance preservation; (2) TD-JEPA gradients match the forward-backward successor measure objective; (3) zero-shot policy evaluation error is bounded above by 2× the successor measure approximation loss. Empirically TD-JEPA achieves SOTA on DMControl (628.8 avg RGB, +8% over best baseline) and competitive OGBench performance across 65 tasks, 13 datasets, and is particularly dominant in challenging pixel-based settings.

## [Pai et al. (2025) — mimic-video](papers/pai_2025_mimic_video.md)
**Type:** paper
**Tags:** robot-learning, video-action-model, flow-matching, imitation-learning, dexterous-manipulation, sample-efficiency, inverse-dynamics
mimic-video introduces Video-Action Models (VAMs), pairing a pretrained Cosmos-Predict2 (2B latent DiT) generative video backbone finetuned with LoRA on a 200-hour robot video corpus, with a lightweight flow-matching action decoder (IDM) that cross-attends to intermediate video model representations at a chosen noise level τ_v. The core insight is that generative video pretraining inherently encodes physical dynamics, so the action decoder needs only to learn the simpler IDM problem (visual plan → motor command), achieving 10× sample efficiency and 2× faster convergence compared to π₀.₅-style VLA baselines. A counterintuitive finding is that τ_v = 1 (single video backbone forward pass with pure Gaussian noise as "future") achieves the highest autonomous policy performance — intermediate representations at high noise are richer conditioning signals than fully denoised video latents. Evaluated on SIMPLER-Bridge (46.9% scratch, SOTA), LIBERO (93.9% scratch), and real-world bimanual dexterous manipulation (72%/93% on packing/handover vs. DiT-Block Policy's 11%/30%).

## [Assran et al. (2025) — V-JEPA 2](papers/assran_2025_vjepa2.md)
**Type:** paper
**Tags:** self-supervised-learning, video-representation, world-models, robot-learning, jepa, model-predictive-control, vision-transformer
V-JEPA 2 scales the V-JEPA self-supervised video pretraining recipe to 1B parameters (ViT-g) and over 1 million hours of internet video (VideoMix22M), achieving state-of-the-art on motion understanding (77.3 top-1 on SSv2) and action anticipation (39.7 recall-at-5 on EK100, +44% over prior SOTA). After alignment with a Llama 3.1 8B LLM, it achieves state-of-the-art video QA (84.0 PerceptionTest, 76.9 TempCompass). A latent action-conditioned world model (V-JEPA 2-AC) is post-trained on only 62 hours of Droid robot data, enabling zero-shot pick-and-place on Franka arms in novel environments via Cross-Entropy Method planning with image goals. Key technical contributions include 3D-RoPE position embeddings, progressive resolution training (8× compute speedup), and a teacher-forcing + rollout combined loss for the action-conditioned predictor.

---

## [JEPA-RL for Humanoid Grasping (research hypothesis)](open_questions/jepa-rl-humanoid-grasping.md)
**Type:** open_question
**Tags:** robot-learning, jepa, world-models, reinforcement-learning, humanoid, dexterous-manipulation, visual-grasping, research-hypothesis
This article proposes replacing the VLA prior in the RLT (RL Token) stack with a JEPA-style video world model (V-JEPA 2 + action-conditioned predictor), arguing that the JEPA forward-prediction objective is structurally aligned with what RL value functions require, whereas VLM semantic representations are not. The core hypothesis is that the encoder-predictor pair $(E_\theta, P_\phi^a)$ acts as a physics oracle enabling model-based $n$-step TD backups, dense representation-space rewards, and a richer RL state that includes predictor lookahead — yielding better sample efficiency on contact-rich tasks. The proposal is applied concretely to humanoid visual grasping, decomposing the task into a vision-dominated reaching phase (JEPA representations + eigengrasp action space) and a proprioception-dominated contact phase (force/torque augmented). The central empirical bet — that V-JEPA outperforms DINOv3 as the RL state encoder (reversing Terver et al.'s finding for MPC) — is identified as the pivotal experiment, along with four other controlled comparisons.

---

## [Step-Size Adaptation & Meta-Learning of Learning Rates](concepts/step-size-adaptation.md)
**Type:** concept
**Tags:** meta-learning, step-size-adaptation, idbd, lms, delta-rule, online-learning, optimization, feature-relevance
Step-size adaptation treats the learning rate as a parameter learned online rather than a hyperparameter tuned by hand, on the argument (Sutton 1992) that per-input learning rates *are* a form of bias and the only automatic source of good bias is previous learning experience. The Delta-Bar-Delta heuristic increases a step size when the current weight change correlates positively with recent weight changes and decreases it when the correlation is negative; IDBD makes this fully incremental via an exponential parameterization $\alpha_i = e^{\beta_i}$ (geometric steps, guaranteed positivity) and a trace $h_i = \partial w_i/\partial\beta_i$ whose decay is gated by $x_i^2$ and tied to the current learning rate, leaving one free parameter $\theta$. IDBD is exactly stochastic gradient descent in $\beta$-space under the approximation $\partial w_j/\partial\beta_i \approx 0$ for $i \neq j$, and multiplying its meta-increment by any $\alpha_i^p$ gives a family of still-valid descent algorithms. The key contrast with Adam/RMSProp/AdaGrad: those are gradient-magnitude *normalizers* and cannot zero out a step size for being useless, whereas a meta-gradient method uses a sign-sensitive correlation signal and therefore doubles as an online relevance detector — a difference that matters under non-stationarity and is second-order on stationary train-once problems.

## [Continual Learning & Tracking in Non-Stationary Tasks](concepts/continual-learning-and-tracking.md)
**Type:** concept
**Tags:** continual-learning, non-stationarity, tracking, online-learning, meta-learning, evaluation-methodology
A tracking task is a learning problem whose target function drifts over time, seen as a one-pass stream of examples that are processed and discarded, and measured by ongoing (asymptotic) error rather than final convergence — so the right thing to do with a step size is to hold it at a useful standing value per feature, not anneal it to zero. This setting matters because bias learning is only *measurable* when the learner meets a series of related problems sharing the same bias; on a single train-once task meta-learning effects are second-order, while on Sutton's drifting task learned biases cut squared error ~60%. The canonical testbed (Sutton 1992) uses 20 i.i.d. Gaussian inputs where 5 are relevant with ±1 weights and one relevant weight flips sign every 20 examples, keeping the relevance structure stationary while the mapping drifts — cleanly separating learnable bias from what must be tracked. This article is a stub; it also flags the connection to plasticity loss in deep continual learning and to RL's intrinsic non-stationarity (a moving bootstrap target) as framing claims needing a compiled source.

## [Sutton (1992) — IDBD](papers/sutton_1992_idbd.md)
**Type:** paper
**Tags:** meta-learning, step-size-adaptation, idbd, lms, delta-rule, online-learning, non-stationarity, tracking, gradient-descent
Sutton introduces Incremental Delta-Bar-Delta (IDBD), a meta-learning algorithm that adapts one learning rate per input of an LMS learner online, using the update $\beta_i \leftarrow \beta_i + \theta\delta x_i h_i$ with $\alpha_i = e^{\beta_i}$ and trace $h_i \leftarrow h_i[1-\alpha_i x_i^2]^+ + \alpha_i \delta x_i$; it improves on Jacobs's batch Delta-Bar-Delta by being fully incremental and having one free parameter instead of three, at ~3× the memory and compute of plain LMS. The paper derives IDBD as stochastic gradient descent in learning-rate space (with $h_i \equiv \partial w_i/\partial\beta_i$ and the approximation that a rate change affects mainly its own weight), and notes that adding any factor $\alpha_i^p$ yields a family of descent — though no longer steepest-descent — algorithms, some of which appeared more efficient in unpublished experiments. On a drifting tracking task (20 Gaussian inputs, 5 relevant with a sign flipping every 20 examples) IDBD reaches ≈1.5 asymptotic MSE against ≈3.5 for the best fixed-α LMS (~60% error reduction) over a broad range of θ, and with θ=0.001 over 250k examples it drives the 15 irrelevant rates below 0.007 while converging the relevant rates to 0.13±0.015 — matching the optimum found by an exhaustive fixed-α sweep. Sutton also reinterprets IDBD as incremental hold-one-out cross validation (each new example is the held-out one, so generalization to it is optimized without using it) and argues that meta-learning issues only become first-order on sequences of related tasks. Limitations: linear base learner only, optimality shown empirically for one special case, a cross-input approximation in the derivation, and the meta step-size θ still hand-picked.

---

## [TD(λ) & Eligibility Traces](concepts/td-lambda-and-eligibility-traces.md)
**Type:** concept
**Tags:** temporal-difference, td-lambda, eligibility-traces, lambda-return, online-learning, prediction, credit-assignment
TD learning solves the delayed-feedback problem by updating a prediction towards a bootstrapped target built from the next prediction, so learning is online and constant-memory rather than storing experience until outcomes arrive. The λ-return $G^\lambda_t = (1-\lambda)\sum_n \lambda^{n-1}G_{t:t+n}$ interpolates all n-step returns ($\lambda=0$ is one-step TD, $\lambda=1$ approaches Monte Carlo), and its significance is that eligibility traces make learning from it $O(n)$ per step — the eligibility vector is the mechanism of temporal credit assignment, decaying by $\gamma\lambda$ and answering which recently-active features are responsible for a current surprise. Plain TD(λ) only *approximates* learning from λ-returns and only at small step sizes, which is the wrong failure mode for fast online learning; True Online TD(λ) (Van Seijen et al. 2016) is exactly equivalent to the Online λ-return algorithm and wins at large step sizes. That exactness is load-bearing rather than cosmetic: SwiftTD's overshoot bound on the correction ratio $\tau_t = \sum_i \alpha_t[i]\phi_t[i]^2$ (where $\tau>1$ means overshooting the target) can only be derived and safely applied because True Online TD(λ) behaves like a learner with undelayed targets, and must be applied when incrementing the eligibility vector rather than at weight-update time.

## [Javed et al. (2024) — SwiftTD](papers/javed_2024_swifttd.md)
**Type:** paper
**Tags:** temporal-difference, td-lambda, eligibility-traces, step-size-adaptation, idbd, meta-learning, online-learning, prediction, atari
SwiftTD combines True Online TD(λ) with three mechanisms — IDBD-style step-size optimization over per-feature step sizes ($\alpha[i]=e^{\beta[i]}$, meta-step normalized by $e^{\beta[i]}$), an overshoot bound capping the correction ratio $\tau_t=\sum_i\alpha_t[i]\phi_t[i]^2$ applied when incrementing the eligibility vector, and a non-gradient step-size decay $\alpha_{t+1}[i]=\alpha_t[i]\epsilon^{\phi_t[i]^2}$ that fires when the bound triggers — with the explicit goal of a fourth option beyond slow-but-stable, fast-but-divergent, and replay-based learning: fast, stable, single-update online learning with no stored data. The ablations show all three are needed and that True Online TD(λ) (not TD(λ)) is required: step-size optimization alone diverges for $\theta > 10^{-2}$ or $\alpha_{init} > 10^{-4}$, TD(λ) with the bound still diverges on VideoPinball/DemonAttack/BattleZone, while full SwiftTD never diverged across a 3,025-configuration Pong sweep. On the Atari Prediction Benchmark (201,619 binary features from binned frames, cumulant ±1, γ=0.98, lifetime 210k steps, actions from frozen Rainbow-DQN policies) SwiftTD matched or beat True Online TD(λ) on every game, by up to an order of magnitude, and succeeded on Atlantis and Pooyan where the baseline failed for all hyperparameters; used in a conv net's last layer only it also won on almost all games. The paper also corrects the IDBD→TD record (Thill 2015 wrong; Kearney et al. 2018/TIDBD used the TD(0) objective under TD(λ); Young et al. 2019 correct for TD(λ)) and validates the semi-gradient meta-update over the full-gradient one. Limitations: prediction only under fixed policies (never control), derived for linear learners with neural nets handled only via the last layer, five added hyperparameters, per-game tuning, no convergence theory, unreported compute overhead (nine per-feature vectors vs. three), and an internal inconsistency where Appendix D's sweep grid excludes the defaults recommended in §6.
