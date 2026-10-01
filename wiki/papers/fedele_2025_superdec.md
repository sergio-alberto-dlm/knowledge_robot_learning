---
title: "Fedele et al. (2025) — SuperDec: 3D Scene Decomposition with Superquadric Primitives"
type: paper
tags: [3d-scene-representation, superquadrics, shape-abstraction, primitive-decomposition, point-clouds, unsupervised-segmentation, transformer, levenberg-marquardt, path-planning, grasping, controllable-generation]
related: [concepts/superquadric-scene-representations.md, concepts/dexterous-manipulation.md, concepts/visual-affordances-robotics.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/SuperDec.pdf]
---

# Fedele et al. (2025) — SuperDec: 3D Scene Decomposition with Superquadric Primitives

**arXiv:** 2504.00992v2 (first posted Apr 2025; v2 dated 19 Mar 2026, cs.CV). The PDF does not name a venue.
**Authors:** Elisabetta Fedele, Boyang Sun, Leonidas Guibas, Marc Pollefeys, Francis Engelmann (ETH Zurich, Stanford, Microsoft)
**Project page:** super-dec.github.io

## Abstract

SuperDec builds **compact, geometrically faithful 3D scene representations** by decomposing point clouds into a small set of **superquadrics**. A class-agnostic, self-supervised Transformer predicts superquadric parameters and a soft point-to-primitive segmentation for each object. A short Levenberg–Marquardt (LM) refinement follows. Full scenes are handled by running the model on every instance from an off-the-shelf 3D instance segmenter (Mask3D). The model is trained only on ShapeNet. It generalises to real ScanNet++ objects and whole Replica scenes, and on ShapeNet its L2 Chamfer error is ~6× lower than the best learned baseline while using about half the primitives. The representation supports **robot path planning, grasping (demonstrated on a Spot with an arm)** and **spatially and semantically controllable image generation**.

## Motivation

- NeRF and 3D Gaussian Splatting aim for **photorealism**. They produce large, non-modular encodings that are poor for explicit spatial reasoning and memory-constrained robotics.
- Primitive decomposition (cuboids, superquadrics) is compact and interpretable, but prior methods fall into two camps:
  - **Learning-based** (Tulsiani 2017 cuboids; SQ / Paschalidou 2019; CSA / Yang & Chen 2021): fast, but rely on a **global shape code** and **category-specific training**, so they fail out of category.
  - **Optimization-based** (EMS / Liu 2022; Marching-Primitives 2023; DBW 2023): more accurate, but slow, heuristic (EMS assumes hierarchical structure), and sometimes need SDFs or multi-view images. DBW is limited to fewer than 10 primitives and takes ~3 h per DTU scene.
- SuperDec's thesis: **local point features plus learned shape priors** in a Mask2Former-style query decoder give accuracy, compactness and cross-category generalisation together.

## Why Superquadrics

A superquadric in canonical pose has 5 parameters, three scales $(s_x,s_y,s_z)$ and two shape exponents $(\epsilon_1,\epsilon_2)$:

$$f(\mathbf{x}) = \left(\left(\tfrac{x}{s_x}\right)^{2/\epsilon_2} + \left(\tfrac{y}{s_y}\right)^{2/\epsilon_2}\right)^{\epsilon_2/\epsilon_1} + \left(\tfrac{z}{s_z}\right)^{2/\epsilon_1} = 1$$

A 6-DoF pose brings the total to **11 parameters**, vs. 9 for a cuboid. The two extra parameters let one primitive cover cuboids, ellipsoids, cylinders and octahedra. The key property is a closed-form **radial distance** from any point to the surface, $d_r = |\mathbf{x}|\cdot|1 - f(\mathbf{x})^{-\epsilon_1/2}|$. There is also an explicit surface parameterisation for sampling, $s(\eta,\omega)=[s_x\cos^{\epsilon_1}\!\eta\cos^{\epsilon_2}\!\omega,\ s_y\cos^{\epsilon_1}\!\eta\sin^{\epsilon_2}\!\omega,\ s_z\sin^{\epsilon_1}\!\eta]$. The supplement explains why ellipsoids (too inexpressive) and generalised ellipsoids (no closed-form surface distance) were rejected. See [Superquadric & Primitive-Based Scene Representations](../concepts/superquadric-scene-representations.md).

## Methodology

### 1. Feed-forward network (single object)

- **Input:** point cloud $\mathcal{P}\in\mathbb{R}^{N\times3}$ ($N=4096$ via farthest-point sampling). A **PVCNN** encoder produces per-point features $\mathcal{F}_{PC}\in\mathbb{R}^{N\times H}$.
- **Queries:** $P=16$ superquadric queries $\mathcal{F}_{SQ}\in\mathbb{R}^{P\times H}$ with sinusoidal positional encodings, refined by a **Transformer decoder** ($D=3$ layers of self-attention, cross-attention to point features, and FFN; $H=128$).
- **Segmentation head:** soft assignment $M\in\mathbb{R}^{N\times P}$, $m_{ij} = \mathrm{softmax}(\phi(\mathcal{F}_{PC})\cdot\mathcal{F}_{SQ})$.
- **Superquadric head:** 12 numbers per query: scale (3), shape $\epsilon$ (2), rotation (3), translation (3), and an **existence probability** $\alpha_j$. The number of primitives per object can therefore vary.

### 2. Self-supervised losses (no ground-truth primitives or parts)

$$\mathcal{L} = \underbrace{\mathcal{L}_{\mathcal{P}\to SQ} + \mathcal{L}_{SQ\to\mathcal{P}} + \mathcal{L}_N}_{\mathcal{L}_{rec}} + \lambda_{par}\mathcal{L}_{par} + \lambda_{exist}\mathcal{L}_{exist}$$

- **Bidirectional Chamfer**, with $S=4096$ surface samples per superquadric using Pilu & Fisher's equal-distance sampling:
  - points→primitives is weighted by the soft assignment $m_{ij}$;
  - primitives→points is weighted by existence $\alpha_j$.
- **Normal loss** $\mathcal{L}_N$ from CSA (Yang & Chen 2021), which speeds up convergence.
- **Parsimony loss:** $\mathcal{L}_{par} = \big(\tfrac1P\sum_j\sqrt{\bar m_j/P}\big)^2$ with $\bar m_j = \sum_i m_{ij}/N$. This is a 0.5-"norm" on primitive usage that pushes toward fewer primitives.
- **Existence loss:** BCE between $\alpha_j$ and a pseudo-label $\hat\alpha_j = [\bar m_j > \epsilon_{exist}]$. The segmentation acts as a teacher for the existence head.

> **Verify:** The paper sets $\epsilon_{exist}=24$, yet $\bar m_j$ as defined is a fraction in [0, 1], so the threshold could never be reached. It is probably applied to the unnormalised point count $\sum_i m_{ij}$ (24 of 4096 points).

### 3. Levenberg–Marquardt refinement

Starting from the network output, LM minimises two sets of residuals:
- **Point-side:** $r_{ij} = m_{ij}\,\tilde d_j(\mathbf{x}_i)$, the segmentation-weighted radial distance.
- **Primitive-side:** $K=25$ surface samples per superquadric, each scored by its distance to the nearest input point. This normalisation term stops primitives from growing to cover empty space.

The network is the **initialiser**. Learned priors let LM avoid the local minima that trap pure optimisation (EMS).

### 4. Full scenes

Mask3D produces class-agnostic instance masks. Each instance is centred, rescaled to a sphere of radius 0.5, and decomposed independently. There is **no fine-tuning on scene data**.

### Training details (supplement)

- Trained jointly on all **13 ShapeNet classes**, class-agnostic.
- Adam with a one-cycle schedule (max LR 4e-4) on 4×A100 with batch 128.
- Augmentations: z-rotation 0–180°, x/y tilt 0–7.5°, translation within radius 0.05.
- 500 epochs with $\lambda_{par}=0.1$, then 500 epochs with $\lambda_{par}=0.6$.

> **Conflict:** The main text lists $\lambda_{par}=0.06$, but the supplement uses **0.6** for both the training schedule and the chosen accuracy/compactness operating point (Fig. 12). 0.06 is most likely a typo.

## Results

### ShapeNet (Table 1; Chamfer ×10², lower is better)

| Model | Primitive | Seg. | In-cat L1 | In-cat L2 | In-cat #Prim | Out-cat L1 | Out-cat L2 | Out-cat #Prim |
|---|---|---|---|---|---|---|---|---|
| EMS (optimisation) | SQ | ✗ | 5.771 | 1.345 | 5.68 | 5.410 | 1.211 | 5.68 |
| CSA (learned) | Cuboid | ✓ | 5.157 | 0.527 | 9.21 | 4.897 | 0.427 | 11.75 |
| SQ / Paschalidou (learned) | SQ | ✗ | 3.668 | 0.279 | 10 | 4.193 | 0.354 | 9 |
| **SuperDec** | SQ | ✓ | **1.698** | **0.047** | 5.8 | **1.847** | **0.061** | **5.26** |

- **Out-of-category protocol:** train on airplane, bench, chair, lamp, rifle and table; test on car, sofa, loudspeaker, cabinet, display, telephone and watercraft.
- L2 is **~6× lower than SQ** with about half the primitives, and **~20–29× lower than EMS** with a similar primitive count.

### Real-world objects (Table 2; ×10²)

| Model | ScanNet++ L1 | L2 | #Prim | Replica L1 | L2 | #Prim |
|---|---|---|---|---|---|---|
| SQ | 10.45 | 4.26 | 10.0 | 11.84 | 5.91 | 10 |
| EMS | 5.51 | 2.11 | **4.25** | 5.40 | 2.12 | **3.61** |
| CSA | 2.91 | 0.41 | 11.64 | 3.68 | 0.70 | 9.63 |
| **SuperDec** | **1.70** | **0.11** | 5.18 | **1.79** | **0.19** | 6.58 |

The objects are noisy, partially observed, and in arbitrary poses. SuperDec is trained only on ShapeNet, yet it wins on accuracy by a wide margin. EMS uses fewer primitives but is ~10–20× worse in L2.

### Robotics: path planning (Table 3, mean over 15 ScanNet++ scenes)

| Representation | Validity-check time (ms) | Success (%) | Memory (MB) |
|---|---|---|---|
| Occupancy grid (10 cm) | 0.056 | 100.00 | 0.873 |
| Dense point cloud | 0.063 | 89.57 | 19.286 |
| Voxels (10 cm) | **0.030** | 98.78 | 0.101 |
| Cuboids (MonteBoxFinder) | 0.120 | 61.23 | **0.024** |
| **SuperDec** | 0.150 | 91.71 | 0.042 |

- **Setup:** OMPL RRT* in a 3D state space with a 2 s budget per start–goal pair. Start and goal heights are 0.4–0.6 m, and the collision radius is 25 cm. A path fails if more than 10% of its 5 cm waypoints violate occupancy-grid clearance.
- SuperDec needs about **460× less memory than the dense point cloud** and ~20× less than the occupancy grid, with higher success than the point cloud. It still trails voxels and occupancy, and it has the slowest collision check.

> **Caveat:** success is validated *against the occupancy grid*, so Occupancy's 100% holds by construction and is a reference, not a competitor. Per-scene results (supplement Table 6) vary widely: SuperDec reaches 57% on two scenes and 100% on others. Cuboids run out of memory on one large scene and find no valid path on another.

### Robotics: grasping and real robot

- Grasp poses come from a **superquadric-based geometric grasp planner** (Vezzani et al. 2017, via `superquadric-library`) applied to the selected primitive. There is **no data-driven grasp network**, and it works on objects like bottles, flowers, side tables and plants.
- **Real-world:** an iPad scan → dense point cloud → SuperDec on a **Boston Dynamics Spot with arm**. RRT* uses a 60 cm collision radius, the robot navigates, and it grasps a **milk bottle** with its built-in IK. The demonstration is qualitative and involves a single target.
- The authors suggest combining SuperDec with **open-vocabulary 3D segmentation** (OpenMask3D) for language-specified navigate-and-grasp.

### Controllable generation

Depth maps rendered from the superquadrics condition **ControlNet** (Stable Diffusion):
- **Spatial control:** moving, duplicating or deleting primitives edits the generated image coherently.
- **Semantic control:** the text prompt changes style while the layout is kept. Plausible object semantics emerge from primitive arrangement alone, e.g. pillows appear on couches.

### Analyses

- **Unsupervised part segmentation:** the segmentation matrix gives sharp part masks, especially in-category. The authors suggest it could be used as pretraining for semantic part segmentation.
- **Hierarchical decomposition:** running SuperDec on a *whole* simple scene coarsely separates grass, table and chairs. Re-running on each segment gives finer instances, without any instance segmenter.
- **Learned embedding:** a BERT-style [CLS] token over the primitive tokens, visualised with t-SNE, clusters by category (chairs, airplanes, cars) without labels. High-variance classes such as watercraft are diffuse.
- **Speed (RTX 4090):** up to 256 objects in parallel. The forward pass takes ~0.13 s per full Replica scene and Mask3D ~0.3 s. Each LM step takes under 1 s.

### Ablations (supplement)

- **LM rounds (Fig. 11):** these help **out-of-category** more than in-category, which narrows the generalisation gap. Gains are modest overall, so the network's solutions sit in local minima that LM can't escape.
- **Existence head (Table 4):** its effect is negligible. Using $\alpha_j$ or the segmentation-derived $\hat\alpha_j$ at train and inference time changes L2 by ≤0.002.
- **$\lambda_{par}$ sweep (Fig. 12):** it trades the number of primitives (~9 → ~3.5) against L2 (~0.047 → ~0.09) smoothly. 0.6 sits near the crossover.
- **FPS vs. random sampling (Table 5):** there is almost no difference, so random sampling is enough in practice.

> **Verify:** The L2 values in Fig. 11 (~0.052 in-category and ~0.076 out-of-category after 10 LM rounds) do not match Table 1 (0.047 / 0.061). The figure may use a different setting or checkpoint, but the paper does not say.

## Limitations

- **Scene pipeline depends on Mask3D:** segmentation errors propagate, and objects are decomposed independently, so there is no scene-level reasoning about support or contact between objects.
- **Geometry only:** no appearance or colour. It cannot replace GS/NeRF where photorealism matters.
- **Trained on 13 ShapeNet classes only.** Scaling the training data is left open.
- **LM refinement is local** and gives limited gains, and the authors call for a different optimiser.
- **The robotics evaluation is thin:** planning success is below voxels and occupancy, collision checks are the slowest, and the grasping and real-robot results are qualitative single demonstrations. There are no grasp success rates.
- Primitives are **convex-ish, symmetric** shapes. Thin structures and concavities need many primitives or are approximated poorly.

## Relevance to Robot Learning

- It provides a **compact object-centric world representation** (about 12 numbers per part, ~40 KB per room) that suits memory-constrained onboard planning and could serve as a structured observation or state for policies.
- **Analytic geometry enables classical grasp planners** on real scans without learned grasp networks. This complements the learned/functional grasping view in [Dexterous Manipulation](../concepts/dexterous-manipulation.md), since primitives supply geometric but not functional affordances.
- **Editable scene proxies** for generative models suggest a route to data augmentation and sim-scene generation for robot learning.

## See Also

- [Superquadric & Primitive-Based Scene Representations](../concepts/superquadric-scene-representations.md)
- [Dexterous Manipulation](../concepts/dexterous-manipulation.md): geometric vs. functional grasping
- [Visual Affordances for Robotics](../concepts/visual-affordances-robotics.md): an alternative, learned, interaction-centric scene representation
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md) — using SuperDec shapes as persistent RL state (fit at reset, pose tracked per step)
