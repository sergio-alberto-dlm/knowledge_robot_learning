---
title: "Nabarro et al. (2024) — Learning in Deep Factor Graphs with Gaussian Belief Propagation"
type: paper
tags: [gaussian-belief-propagation, factor-graphs, learning-as-inference, bayesian-deep-learning, continual-learning, distributed-training, energy-based-models, predictive-coding, image-classification, denoising]
related: [concepts/gaussian-belief-propagation.md, concepts/continual-learning-and-tracking.md, papers/ortiz_2021_gbp.md, papers/bui_2014_tree_gp.md, concepts/step-size-adaptation.md, concepts/sparse-gaussian-processes.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/GBP_learning.pdf]
---

# Nabarro et al. (2024) — Learning in Deep Factor Graphs with Gaussian Belief Propagation

**Venue:** ICML 2024 (PMLR 235); arXiv 2311.14649v3 (17 Jul 2024, cs.LG)
**Authors:** Seth Nabarro, Mark van der Wilk, Andrew J. Davison (Dyson Robotics Lab, Imperial College London; University of Oxford)
**Code:** github.com/sethnabarro/gbp_learning (TensorFlow)

## Abstract

The paper proposes **GBP Learning**: build Gaussian factor graphs whose structure mirrors neural-network architectures (conv, transposed conv, max-pool, upsample, dense, softmax) and treat *every* quantity (inputs, outputs, activations **and parameters**) as a random variable. Training (infer parameters given observed inputs/outputs) and prediction (infer outputs given inputs and parameters) are then the same computation, Gaussian Belief Propagation, with different variables observed. Because GBP updates are local, training can run distributed and asynchronously. Continual learning comes almost for free: the GBP posterior over parameters after one task becomes the prior for the next (Bayesian filtering). Experiments show learnable factor graphs beat a hand-designed pairwise smoother on video denoising, and single-epoch continual MNIST training matches a CNN with a 6×10³-example replay buffer.

## Motivation

- **Backward locking.** In backprop, processors that hold early layers sit idle waiting for the backward error signal. This gets worse with (i) models sharded across many devices, (ii) hardware with large local memory per core (Graphcore IPU, Cerebras), and (iii) distributed and embedded devices.
- **Fusion of signals.** Incremental learning (old vs. new data), hand-crafted vs. learnt components, and agreement between layers are all fusion problems. The Bayesian answer is to combine them by the rules of probability.
- GBP is already a proven distributed inference engine in spatial AI (bundle adjustment, multi-robot localisation and planning). See [Ortiz et al. (2021)](ortiz_2021_gbp.md).

## Key Contributions

1. **A general recipe for training deep factor graphs with GBP.** Any architecture works as long as its factor energies can be written down. Message updates do not need to be derived by hand, unlike Lucibello et al. (2022), which is limited to dense MLPs with binary weights and sign activations.
2. **Parameters as variable nodes.** Weights and biases are shared across all observations. Inputs, outputs and activations are copied once per observation (Fig. 1).
3. **Efficient factor-to-variable updates** via the Woodbury identity and low-rank structure. Updating all messages costs $O(BLC^2)$, the same as one forward or backward pass of backprop.
4. **Continual learning and minibatching by Bayesian filtering** over the parameter posterior. Each datapoint is seen once and then discarded.
5. **Experiments:** toy XOR and regression, video denoising, and MNIST / FashionMNIST / CIFAR10 classification, including asynchronous layer schedules.

## Methodology

### Factor energies (deep factor graphs)

Every factor is Gaussian in a (possibly non-linear) measurement function, $E(\mathbf{x}_\phi) = \tfrac12 (\mathbf{y} - \mathbf{h}(\mathbf{x}_\phi))^\top \Lambda_y (\mathbf{y} - \mathbf{h}(\mathbf{x}_\phi))$. Layers are tied together by **local consistency**: activations $\mathbf{x}_l$ should "predict" the previous layer $\mathbf{x}_{l-1}$ through a parametric non-linear map. This is written in the generative direction:

$$E(\mathbf{x}_l, \mathbf{x}_{l-1}, \Theta_l) = \frac{\|\mathbf{x}_{l-1} - f(\mathbf{x}_l, \Theta_l)\|_2^2}{2\sigma_l^2}$$

$\sigma_l$ is the **factor strength**, the main per-layer hyperparameter. The layer types are:

| Factor | Energy (per location / unit) | Notes |
|---|---|---|
| Convolution | $\big(x_l^{(a,b,c)} - g(\mathrm{vec}(\mathbf{X}_{l-1}^{(a,b)})\cdot \mathrm{vec}(\theta_l^{(c)}) + b_l^{(c)})\big)^2 / 2\sigma_l^2$ | Filters $\theta$ and bias $b$ are variables shared across the layer |
| Transposed conv | $\big(x_{l-1}^{(a,b,c)} - r(\mathbf{X}_l^{(a,b)}, \theta_l^{(c)}, b_l^{(c)})\big)^2 / 2\sigma_l^2$ | Overlapping receptive fields sum to reconstruct the input |
| Dense | $\|\mathbf{x}_L - g(\mathbf{W}_L^\top \mathrm{vec}(\mathbf{X}_{L-1}) + \mathbf{d}_L)\|^2 / 2\sigma_L^2$ | Decomposed into one factor per output unit for efficiency |
| Max-pool | $(\max(\mathbf{X}_{l-1}^{(a,b,c)}) - x_l^{(a,b,c)})^2 / 2\sigma_l^2$ | App. A |
| Upsample | $\sum_{i,j}(x_{l-1}^{(a-i,b-j,c)} - x_l^{(a,b,c)})^2 / 2\sigma_l^2$ | App. A |
| Softmax observation | $\|\mathrm{softmax}(\mathbf{x}_L) - \mathbb{1}_y\|^2 / 2\sigma_\text{softmax}^2$ | Class supervision, attached only at train time |

$g(\cdot)$ is an elementwise non-linearity (Leaky ReLU, sigmoid, or identity). Non-linear factors are **relinearized** around the current estimates (§2.3 of the paper; see [GBP concept](../concepts/gaussian-belief-propagation.md#non-linear-factors-linearization)). The Jacobian $\mathbf{J}(\mathbf{x}_0)$ depends on the current activations, so the *effective strength* of each factor changes with the input. The authors argue this gives a "soft switching" of connections similar to non-linear NNs.

### Training and prediction are the same inference

- **Training:** observe inputs (and outputs, when supervised), then infer the posterior over parameters and activations.
- **Prediction:** fix the parameters, remove the output observation factor, run GBP, and read the last-layer beliefs as logits or predictions.
- **Unsupervised variant:** drop the output observation factor. The model then learns to reconstruct its inputs (used in video denoising).
- **Schedule:** sweep forward then backward through the layers, repeated for a fixed number of iterations. Within a layer, all factor→variable updates run in parallel, then all variable→factor updates.
- **Stabilisation:** **message damping plus dropout on factor→variable messages** was sufficient for stable GBP. Typical values are damping 0.7–0.9 and dropout 0.5–0.6.

### Efficient GBP (§3.3, App. B)

The naive factor→variable update inverts a $(V_j-1)\times(V_j-1)$ matrix, which costs $O((V_j-1)^3)$ per message. The paper exploits two facts: (i) the linearized precision $\mathbf{J}^\top\Lambda_y\mathbf{J}$ is low-rank when the observation dimension $M < V$, and (ii) the incoming-message precision $\mathbf{D}$ is diagonal for scalar variable nodes. With the **Woodbury identity**, and by reusing intermediates across outgoing messages, updating all $V$ messages of a factor drops from $O(V(V-1)^3)$ to $O(VM^3)$, and memory from $O(V^2)$ to $O(VM + M^2)$. Dense factors with many outputs are split into $M$ single-output factors, so each update is $M^2$ cheaper while the model stays the same. For $L$ dense layers of width $C$ with batch $B$, the total cost is **$O(BLC^2)$**, the same as a backprop pass.

### Continual learning and minibatching via Bayesian filtering (§3.4)

Start with parameter priors $p_1(\psi_{l,i}) = \mathcal{N}(0,\sigma)$. For each task or batch $\mathbf{z}_t$:
1. Build a copy of the graph for $\mathbf{z}_t$.
2. Attach a unary prior factor to each parameter, equal to the previous marginal posterior $p(\psi_{l,i}\mid \mathbf{z}_{1:t-1})$.
3. Run GBP to convergence, giving the new fully factorised posterior $\prod p(\psi_{l,i}\mid\mathbf{z}_{1:t})$.

This equals message passing on the combined graph of all tasks with messages passed only **forward in $t$**. Past data can be discarded, and the same routine is used for **memory-efficient minibatching** (a single epoch, each datum visited once). Unlike Lucibello et al., which revisits each datum over many epochs with few GBP iterations and needs ad-hoc "forgetting" factors to avoid double-counting, GBP Learning runs to convergence on each batch and visits each datum only once.

**Controlled forgetting (App. D.1.1).** The single-layer denoiser overfit to salt-and-pepper noise under continual learning. The fix sets each frame's prior to an **interpolation between the previous posterior and the original prior**: $\mu = \alpha\mu_0 + (1-\alpha)\mu^{(t-1)}$ and $\sigma = \alpha\sigma_0 + (1-\alpha)\sigma^{(t-1)}$, with $\alpha = 0.5$. The standard deviation is interpolated, not the variance. The five-layer model did **not** need this.

## Results

### Toy tasks (§5.1)
- One-hidden-layer MLP-like factor graphs solve **XOR** (8 hidden units, Leaky ReLU) and **1-D non-linear regression** (90 points, 16 hidden units, sigmoid) with sensible predictive uncertainty. This confirms that iterative linearization captures non-linear dependencies.

### Video denoising, DAVIS "bear" (§5.2)
- 82 frames downsampled to 258×454. 10% of pixels are replaced by $U(0,1)$ noise. The model learns **from noisy frames only, without supervision**. Pixel and reconstruction factors use robust (Tukey-like) energies.
- Methods: pairwise smoother with no learning (classic GBP denoiser) vs. a 1-layer transposed-conv model (4 filters, 3×3) vs. a 5-layer model (transposed conv / upsample / transposed conv / upsample / transposed conv). Parameters are learned either per frame or continually across frames.
- **PSNR ranking:** continual 5-layer > per-frame 5-layer > continual 1-layer ≈ per-frame 1-layer ≫ pairwise smoother. Example on frame 5: 34.3 / 33.4 / 32.9 / 32.6 / 31.1 dB. Learned models keep more high-frequency detail.
- Continual learning improves on per-frame learning, and depth gives clearly higher PSNR, so the deep model learns aligned multi-layer representations using only local updates.
- Compute on an RTX 3090: ~8 min (1 layer), ~27 min (5 layers), ~2 min (pairwise) for the whole video.

### Image classification (§5.3)
Architecture: conv (16 filters, 5×5, Leaky ReLU) → max-pool 2×2 → dense (2304→10) → softmax observation. Batch size 50, 500 GBP iterations per batch at train time, 300 per test batch. **Single epoch**, trained by filtering.

| Dataset | Lucibello et al. (2022) | GBP Learning |
|---|---|---|
| MNIST | 97.40 ± 0.04 | **98.16 ± 0.03** |
| FashionMNIST | 88.2 ± 0.1 | 88.2 ± 0.1 |
| CIFAR10 | 41.3 ± 0.1 | **53.1 ± 0.3** |

- The MNIST hyperparameters were reused for FashionMNIST and CIFAR10 with no task-specific tuning. Lucibello et al. train for 100 epochs, GBP Learning for one.
- **Low-data regime:** GBP Learning beats every baseline (linear classifier + Adam over many epochs; single-epoch CNN + Adam with or without a FIFO replay buffer). The authors attribute this to regularising priors and marginalising over uncertain activations.
- **Full MNIST:** it beats the CNN + Adam variants except those with large replay buffers (3,600 or 6,000 examples, i.e. 6–10% of the training set), which perform about the same.
- **Asynchronous training (§5.3.1):** at each iteration, 4 layers are sampled uniformly *with replacement* for the update order, reaching **98.11 ± 0.04%** vs. 98.16 ± 0.03% for synchronous sweeps. The learning curves overlap throughout (Fig. D3).
- **Iteration budget (App. F.5):** ~200 train iterations per batch is enough *if* test-time inference runs ≥200 iterations. The best result uses 1,600 train iterations per batch. Too few test iterations (<~100) collapses accuracy.
- **Cost:** ~3 h on an RTX 3090 for full MNIST, vs. minutes for a CNN on CPU. The authors attribute the gap to hardware and software optimized for backprop.

## Hyperparameters (what replaces the learning rate)

GBP Learning has **no learning rate and no global gradient**. The knobs are:
- **Factor strengths** $\sigma$ per layer and factor type (recon σ, input/pixel obs σ, class obs σ).
- **Prior widths** for weights and for activations.
- **Robust thresholds** $N_\text{rob}$ (Mahalanobis distance beyond which the energy is flat).
- **Message damping** (0.7–0.9) and **message dropout** (0.5–0.6).
- **Number of GBP iterations** per batch at train and test time.
- **Batch size**, and for continual denoising the **prior interpolation $\alpha$**.

These were tuned on held-out validation sets. MNIST used 9,000 examples for architecture and hyperparameter selection, and the video used the first five frames.

> **Verify:** The paper does not say how the validation search was done (grid, random, or manual). Nothing suggests any gradient-based outer loop, but this is inferred from what the paper leaves out.

## Limitations

- **Scale and speed:** only small convnets on small images, and much slower than backprop on current GPU stacks. The claimed efficiency depends on future bespoke hardware (on-chip-memory processors such as the IPU, Poplar-level primitives).
- **Scalar variable nodes only:** the fully factorised marginals cannot capture correlations between weights or activations. The authors propose multivariate nodes for strongly correlated groups (e.g. the weights of one layer), which could also reduce loops and help stability.
- **No convergence guarantees:** loopy graphs with non-linear factors; damping and dropout are heuristic stabilisers.
- **Message schedules** are admittedly suboptimal, and exploring better ones is left open.
- **Overfitting under continual learning** in the shallow model needed an ad-hoc prior-interpolation fix.
- **FashionMNIST shows no gain** over Lucibello et al. The authors conjecture that CIFAR10's gain comes from the translation equivariance of the conv layer.
- The layer catalogue is non-exhaustive (no attention, normalization, or residual connections).

## Relation to Other Work

- **Energy-based models:** the model is an EBM whose energy is a sum of quadratic factor energies, so it normalises in closed form (no MCMC). Unlike most EBMs, its parameters are variables, which is why learning and prediction are one procedure.
- **RBMs / DBNs:** these look like single fully-connected layers, but deep RBM stacks are trained greedily layer by layer. The authors report GBP Learning trains multi-layer models jointly "without issue".
- **Bayesian deep learning** (VI, Laplace, HMC, SG-MCMC): these rely on backprop and keep point activations. GBP Learning also treats activations as random, so it can model disagreement between bottom-up and top-down signals.
- **Predictive coding:** the inter-layer consistency factors resemble PC's layer-wise Gaussian prediction errors. GBP is proposed as an alternative to gradient descent on variational free energy for training PC models.
- **Lucibello et al. (2022):** the closest prior work (GBP-trained MLPs with binary weights, analytically derived messages, multi-epoch training).

## Relevance to Robot Learning

- It makes the "open direction" from [Ortiz et al. (2021)](ortiz_2021_gbp.md) concrete: GBP for learning in over-parameterised networks, not only for SLAM-style estimation.
- **Learned factors can plug directly into classical robotics factor graphs** (SLAM, bundle adjustment, sensor fusion) and be trained with the same inference engine, instead of pairing a backprop-trained network with a GBP back end.
- The replay-free, single-pass Bayesian filtering fits the on-device [continual learning](../concepts/continual-learning-and-tracking.md) setting, and asynchronous layer updates suit distributed or multi-robot compute.

## See Also

- [Gaussian Belief Propagation & Factor Graphs](../concepts/gaussian-belief-propagation.md)
- [Ortiz et al. (2021) — A Visual Introduction to GBP](ortiz_2021_gbp.md)
- [Bui & Turner (2014) — Tree-structured GP Approximations](bui_2014_tree_gp.md), another use of GBP inside a learning pipeline (hyperparameters still fit by BFGS)
- [Continual Learning & Tracking in Non-Stationary Tasks](../concepts/continual-learning-and-tracking.md)
- [Step-Size Adaptation & Meta-Learning of Learning Rates](../concepts/step-size-adaptation.md), a contrast: per-weight step sizes meta-learned vs. implied by posterior precision
- [Sparse Gaussian Process Approximations](../concepts/sparse-gaussian-processes.md)
- Report: [Learning with GBP — gradients and hyperparameters](../../outputs/reports/2026-09-25_gbp-learning-gradients-hyperparameters.md)
