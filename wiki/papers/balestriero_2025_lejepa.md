---
title: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
type: paper
tags: [self-supervised-learning, representation-learning, jepa, theory, distribution-matching, isotropic-gaussian, sigreg, vision-transformer, in-domain-pretraining]
related: [concepts/joint-embedding-predictive-architecture.md, concepts/sigreg.md, concepts/video-ssl.md, papers/bardes_2024_vjepa.md, papers/assran_2025_vjepa2.md, papers/terver_2026_jepa_wm.md]
created: 2026-05-17
updated: 2026-05-17
sources: [raw/papers/pdf/LeJEPA.pdf]
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**Authors:** Randall Balestriero (Brown / Meta-FAIR) and Yann LeCun (NYU / Meta-FAIR) — equal contribution  
**arXiv:** 2511.08544v3 — 14 Nov 2025

---

## Abstract

Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. The paper presents a comprehensive theory of JEPAs and instantiates it in **LeJEPA** (Latent-Euclidean JEPA), a lean, scalable, and theoretically grounded training objective. Two key results drive the design: (i) the isotropic Gaussian is the **optimal embedding distribution** that uniquely minimizes downstream prediction risk across broad task families; (ii) a novel regularizer — **Sketched Isotropic Gaussian Regularization (SIGReg)** — enforces this distribution in linear time and memory with bounded gradients. Combining the JEPA predictive loss with SIGReg yields LeJEPA, which eliminates stop-gradients, teacher-student networks, and all other heuristics, using a single hyperparameter λ and approximately 50 lines of PyTorch. Empirical validation spans 10+ datasets, 60+ architectures, all with varying scales and domains.

---

## Key Contributions

- **Contribution 1 — Optimal embedding distribution (Theorem 1):** Proves that the isotropic Gaussian uniquely minimizes worst-case downstream risk for both linear probes (OLS with Tikhonov regularization) and nonlinear probes (radius-based k-NN and kernel regression). Anisotropy amplifies both bias (Lemma 1) and variance (Lemma 2) of downstream estimators.

- **Contribution 2 — SIGReg:** Introduces *Sketched Isotropic Gaussian Regularization*, a distribution-matching objective with (i) provable statistical correctness, (ii) linear O(N) time and memory complexity, (iii) bounded loss gradients and curvature (Theorem 4), and (iv) defeat of the curse of dimensionality (Theorem 5). Uses the Epps-Pulley (1983) characteristic-function test applied to random 1D projections of embeddings.

- **Contribution 3 — LeJEPA design:** Combines SIGReg with the standard JEPA prediction loss into a single objective. No stop-gradient, no EMA teacher-student, no whitening/normalization layers, no negative samples, no hyperparameter schedulers. Eliminates collapsed shortcut solutions by construction.

- **Contribution 4 — Empirical validation at scale:** LeJEPA trains stably on 50+ architectures (ViTs, ResNets, ConvNeXts, MaxViTs, Swin Transformers) up to 1.8B parameters. In-domain pretraining on Galaxy10 outperforms frontier DINOv2/v3 transfer learning. Training loss reliably predicts downstream accuracy (Spearman ρₛ ≈ 85–99%), enabling label-free model selection.

---

## Methodology

### Problem Setup

Data has shape $(N, V, D) \in \mathbb{N}^{\times 3}$: $N$ samples, $V$ views per sample, $D$ observation dimension. The encoder $f_\theta : \mathbb{R}^D \to \mathbb{R}^K$ maps observations to $K$-dimensional embeddings. JEPA requires:

$$\text{JEPA}(x) \iff \text{Enc}(x_{n,t+1,\cdot}) \text{ is predictable from } \text{Enc}(x_{n,t,\cdot})\ \forall n,t \text{ and Enc}(x_{\cdot,\cdot,\cdot}) \text{ is not degenerate.} \tag{1}$$

### Why Isotropic Gaussian? (Section 3)

For linear probing (OLS), the optimal probe parameters are $\hat{\beta} = \arg\min_\beta \|y - Z\beta\|_2^2 + \lambda\|\beta\|_2^2$. Analysis of integrated square bias (ISB) for k-NN and kernel probes yields:

$$\text{ISB}_{k\text{-NN}} = \frac{r_0^4}{(K+2)^2} \tau_S^2 J(p) + O(r_0^4), \qquad \text{ISB}_\text{kernel} \le \left(\frac{h^2 \mu_2(K)}{2}\right)^2 \left(2B^2 + 8L^2 J(p)\right) + o(h^4)$$

**Theorem 1 (Isotropic Gaussian Optimality):** Among distributions with a scalar-based covariance constraint, the isotropic Gaussian is the unique minimizer of the integrated square bias for both k-NN and kernel probes.

### SIGReg (Section 4)

SIGReg matches the embedding distribution $P_\theta$ to an isotropic Gaussian $Q$ via the **Cramér-Wold sketching** principle: testing that all 1D projections match their Gaussian counterparts suffices to test the full multivariate distribution (Lemma 3: Hyperspherical Cramér-Wold).

For a set of unit-norm directions $\mathbb{A} = \{a_1,\ldots,a_M\}$:

$$\text{SIGReg}_T(\mathbb{A}, \{f_\theta(x_n)\}_{n=1}^N) \triangleq \frac{1}{|\mathbb{A}|} \sum_{a \in \mathbb{A}} T(\{a^\top f_\theta(x_n)\}_{n=1}^N) \tag{SIGReg}$$

where $T$ is a statistical test targeting the standard Gaussian. The **Epps-Pulley (EP)** test is chosen because it: (i) is differentiable, (ii) has bounded gradients (Theorem 4), (iii) has O(N) time/memory complexity, and (iv) is DDP-friendly via `all_reduce` on complex exponentials.

The EP statistic compares the empirical characteristic function $\hat{\phi}_X(t) = \frac{1}{n}\sum e^{it X_j}$ to the standard Gaussian CF $e^{-t^2/2}$:

$$EP = N \int_{-\infty}^{\infty} |\hat{\phi}_X(t) - \phi(t)|^2 w(t) \, dt, \quad w(t) = e^{-t^2/2}$$

**Theorem 5 (Unified Error Bounds):** $O(K)$ directions suffice for ε-approximation when the embedding distribution has Sobolev smoothness $\alpha \ge 1$, defeating the curse of dimensionality.

**Theorem 6 (Vanishing Gradient Bias):** The EP gradient has bias $O(1/N)$ — negligible even for minibatches of size 16.

### LeJEPA Loss (Section 5)

The prediction loss follows the DINO setup with $V_g$ global and $V_l$ local views:

$$\mathcal{L}_\text{pred}(\{z_{n,v}\}_{v=1}^V) = \frac{1}{V} \sum_{v'=1}^{V} \left\| \frac{1}{V_g} \sum_{v=1}^{V_g} z_{n,v} - z_{n,v'} \right\|_2^2 = \frac{1}{V} \sum_{v'=1}^{V} \|\mu_n - z_{n,v'}\|_2^2 \tag{7}$$

The total LeJEPA loss is:

$$\mathcal{L}_\text{LeJEPA}(\{x_{n,v}\}_{n,v=1}^{B,V}) = \frac{\lambda}{V} \sum_{v=1}^{V} \text{SIGReg}(\{z_{n,v}^B\}_{n=1}^B) + \frac{1-\lambda}{B} \sum_{n=1}^{B} \mathcal{L}_\text{pred}^{(V_g)}(\{z_{n,v}\}_{v=1}^V) \tag{LeJEPA}$$

**Single hyperparameter:** $\lambda \in [0,1]$ balances isotropic Gaussian enforcement vs. prediction. Recommended default: $\lambda = 0.05$.

### Implementation

The entire method is ~50 lines of PyTorch. No prototypes, stop-gradients, or teacher-student networks. Key recipe defaults:
- 8 views ($V=8$): 2 global at 224×224, 6 local at 96×96
- AdamW optimizer, lr ∈ {5e−3, 3e−4}, wd ∈ {1e−1, 2e−1, 2e−5}
- Linear warm-up + cosine-annealing
- SIGReg: 1024 slices, integration domain [−5, 5], 17 quadrature points
- Optional SWA for ViTs; register tokens not required

---

## Experimental Results

### 6.1 Stability Across Hyper-Parameters and Architectures

- **Hyper-parameter stability (Table 1):** On ImageNet-1K with ViT-L/14, all SIGReg-specific hyper-parameters (number of slices, integration domain, quadrature points) have negligible impact on performance. No catastrophic collapse observed under any setting.
- **Architecture breadth (Figure 9):** 50 architectures from 8 families pretrained on ImageNet-10 all achieve 91.5%–95% top-1 accuracy with frozen backbone linear probing — without any method modification.
- **Removal of heuristics:** Removing the predictor head and teacher-student architecture does not cause collapse thanks to SIGReg. A teacher-student setup gives a small boost for ViTs but is not required.

### 6.2 Training Loss Predicts Downstream Performance

- The combined LeJEPA training loss exhibits **Spearman correlation ρₛ ≈ 85–99%** with downstream test accuracy across models and datasets (Figure 10, 11).
- A scaled correlation $C^{(\alpha)} = \rho_s(\text{train\_loss}/\lambda^\alpha, \text{test\_accuracy})$ reaches ~99% at $\alpha \approx 0.4$.
- This enables **label-free SSL model selection and cross-validation** — a property absent in other JEPA methods (VICReg, DINO, I-JEPA all exhibit weak or non-monotonic training loss / accuracy correlation).

### 6.3 In-Domain LeJEPA Outperforms Frontier Transfer Learning

- **Galaxy10 dataset** (11,000 training samples, 10 galaxy morphology classes): LeJEPA in-domain pretraining with small architectures (ResNet-18, ResNet-34, ConvNeXtV2-Nano) outperforms DINOv2-S/16 (1LVD142M) and DINOv3-S/16 (1LVD1.7B) on both linear probing and full finetuning across all label regimes (1-shot to full supervision) — Figure 12, Table in Figure 1.
- **Few-shot classification (Table 2):** LeJEPA ViT-L (304M, IN-1K, 100 epochs) achieves avg 78.30% (all shots) across DTD/aircr/cars/cifar10/cifar100/flowers102/food/pets, vs I-JEPA ViT-H (632M, IN-1K, 300 epochs) at 73.32% — a smaller model with 3× fewer epochs.

### 6.4 Scaling Results

| Method | Backbone | Dataset | Frozen Top-1 |
|--------|----------|---------|--------------|
| LeJEPA | ViT-L/14 | ImageNet-1K | 77.1% |
| LeJEPA | ConvNeXtV2-Huge | ImageNet-1K | 78.5% |
| I-JEPA | ViT-H (0.6B) | ImageNet-1K | ~75.67% |
| LeJEPA | ViT-g (1.8B) | ImageNet-1K | Stable training (Figure 1) |

### 6.5 Emergent Semantic Structure

- PCA of LeJEPA (ViT-L, 100 epochs on ImageNet-1K) features reveals semantically meaningful structure: warm colors (red/magenta) capture foreground objects; cool colors (cyan/green/yellow) capture backgrounds — without any supervision (Figure 14).
- Thresholding [CLS] token self-attention maps produces binary segmentation masks that **track objects across video frames with temporal coherence** (Figure 13), without any segmentation labels during training.

---

## Limitations & Open Questions

- **Image-focused:** This work focuses on image SSL. Extending to video (as a drop-in replacement for V-JEPA pretraining) and robot/action-conditioned settings is an open direction.
- **Gradient bias:** The Epps-Pulley statistic introduces a bias of O(1/N) in gradients; it is negligible in practice but could matter for extremely small batch sizes (<16).
- **Batch size:** Recommended minimum batch size of 128 for reliable SIGReg estimation; smaller batches increase the stochasticity of the sketch.
- **Optimal λ search:** While λ = 0.05 is a robust default, slightly tuning λ proportionally to the number of views can improve performance (Figure 8).
- **Relation to VICReg:** In the limit of a particular statistical test (T = mean² + (std−1)²), LeJEPA reduces to VICReg, but this setting leads to shortcut solutions — highlighting why the Epps-Pulley test is preferred.
- **Open question:** Can SIGReg be extended to enforce structured (non-isotropic) priors for domains where isotropy is suboptimal (e.g., time-series with strong temporal correlations)?

---

## See Also

- [Joint Embedding Predictive Architecture (JEPA)](../concepts/joint-embedding-predictive-architecture.md) — the framework LeJEPA extends and theoretically grounds
- [SIGReg (Sketched Isotropic Gaussian Regularization)](../concepts/sigreg.md) — the core regularizer introduced in this work
- [Video Self-Supervised Learning](../concepts/video-ssl.md) — context for image/video SSL methods
- [V-JEPA (Bardes et al. 2024)](bardes_2024_vjepa.md) — feature-prediction video SSL that LeJEPA generalizes
- [V-JEPA 2 (Assran et al. 2025)](assran_2025_vjepa2.md) — scaled JEPA with action conditioning
- [JEPA World Models (Terver et al. 2026)](terver_2026_jepa_wm.md) — ablation study of JEPA-WM design choices
