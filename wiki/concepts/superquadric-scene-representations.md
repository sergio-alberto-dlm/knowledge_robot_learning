---
title: Superquadric & Primitive-Based Scene Representations
type: concept
tags: [3d-scene-representation, superquadrics, shape-abstraction, primitive-decomposition, point-clouds, path-planning, grasping]
related: [papers/fedele_2025_superdec.md, concepts/dexterous-manipulation.md, concepts/visual-affordances-robotics.md, open_questions/object-centric-swifttd-critic.md]
created: 2026-09-29
updated: 2026-09-29
sources: [raw/papers/pdf/SuperDec.pdf]
---

# Superquadric & Primitive-Based Scene Representations

## What It Is

**Primitive-based shape abstraction** represents an object or scene as a small set of simple parametric volumes (cuboids, ellipsoids, superquadrics, convexes) instead of dense points, meshes, voxels, SDFs, NeRFs or Gaussian splats. The aim is **compactness and interpretability** rather than photorealism. A room becomes a few hundred primitives of about a dozen numbers each, and each primitive can be read, edited and reasoned about directly.

## Where It Sits Among 3D Representations

| Representation | Strength | Weakness for robotics |
|---|---|---|
| Point cloud / mesh | Faithful raw geometry | Large (SuperDec: ~19 MB per ScanNet++ scene), unstructured |
| Voxel / occupancy grid | Fast collision checks | Resolution–memory trade-off, no object structure |
| SDF / NeRF | Smooth, continuous geometry or appearance | Implicit, expensive to query and edit, per-scene optimisation |
| 3D Gaussian Splatting | Photorealistic, explicit | Memory-heavy, many primitives, no semantic grouping |
| **Superquadrics / cuboids** | Tiny (tens of KB per room), object- and part-structured, editable, analytic distances | Coarse, misses thin and concave detail, needs a decomposition method |

## Superquadrics

Introduced by Barr (1981). The canonical implicit surface is

$$f(\mathbf{x}) = \left(\left|\tfrac{x}{s_x}\right|^{2/\epsilon_2} + \left|\tfrac{y}{s_y}\right|^{2/\epsilon_2}\right)^{\epsilon_2/\epsilon_1} + \left|\tfrac{z}{s_z}\right|^{2/\epsilon_1} = 1$$

- **Parameters:** 3 scales plus 2 shape exponents, with a 6-DoF pose on top, for **11** in total (a cuboid has 9). Varying $\epsilon_1,\epsilon_2$ morphs the shape between an ellipsoid ($\epsilon=1$), a box ($\epsilon\to0$), a cylinder and an octahedron ($\epsilon=2$).
- $f<1$ inside, $f>1$ outside: an **inside/outside test** comes for free.
- **Closed-form radial distance** $d_r = |\mathbf{x}|\,|1-f(\mathbf{x})^{-\epsilon_1/2}|$. This is useful for fitting (as an LM residual) and for collision checks.
- **Explicit surface parameterisation** $s(\eta,\omega)$ gives direct surface sampling for Chamfer losses.
- **Design choice:** generalised ellipsoids (three separate exponents) are slightly more expressive but have **no closed-form surface distance**. Superquadrics give up one DoF (shared x/y roundness) to keep it (Fedele et al. 2025).

## How Decompositions Are Obtained

| Family | Examples | Trade-off |
|---|---|---|
| **Learned, global code** | Tulsiani 2017 (cuboids), SQ / Paschalidou 2019, CSA / Yang & Chen 2021 | Fast, but category-specific, and generalises poorly |
| **Optimisation** | EMS (Liu 2022, hierarchical probabilistic fitting), Marching-Primitives (needs SDF), DBW (differentiable rendering, <10 primitives, hours per scene) | Accurate per object, but slow, heuristic, and no learned priors |
| **Learned local features + refinement** | **SuperDec** (Fedele 2025): PVCNN point features, Transformer queries as primitives, soft segmentation, then LM refinement | Class-agnostic, generalises from ShapeNet to real scans; ~6× lower L2 than SQ with half the primitives |

The common recipe in the learned methods is:
1. Predict $P$ primitives, each with an existence probability.
2. Train self-supervised with a bidirectional **Chamfer** loss between input points and sampled primitive surfaces.
3. Add a **parsimony** penalty to trade accuracy against primitive count.

Accuracy and compactness trade off smoothly along a single weight ($\lambda_{par}$).

## Uses in Robotics

- **Motion planning:** collision checking against primitives. In SuperDec it uses ~460× less memory than a dense point cloud and gets higher RRT* success, though it is below voxel and occupancy grids and has slower checks.
- **Geometric grasping:** analytic superquadric grasp planners (e.g. Vezzani et al. 2017; SuperQ-GRASP) compute grasp poses from primitive parameters without learned grasp networks. This gives geometric rather than functional affordances (see [Dexterous Manipulation](dexterous-manipulation.md)).
- **Scene proxies for generation and editing:** depth rendered from primitives can condition diffusion models (ControlNet) for spatially editable scene synthesis. This is a possible route to generating varied training scenes.
- **Object-centric state:** a set of about 12-number tokens per part is a natural structured observation for policies or scene graphs. This is suggested, not yet demonstrated, in the compiled sources.

## Open Issues

- Pipelines depend on **instance segmentation quality** (e.g. Mask3D). Joint scene-level decomposition is only coarsely demonstrated, via hierarchical re-application.
- **Thin and concave structures** need many primitives.
- **Robotic evaluations so far are small** (single real-robot demonstration, no grasp success rates).
- The relation to **open-vocabulary 3D scene graphs** (OpenMask3D, ConceptGraphs) is proposed but not yet built.

## See Also

- [Fedele et al. (2025) — SuperDec](../papers/fedele_2025_superdec.md)
- [Dexterous Manipulation](dexterous-manipulation.md): geometric vs. functional grasping
- [Visual Affordances for Robotics](visual-affordances-robotics.md): interaction-centric rather than geometry-centric scene representation
- [Open question: Object-Centric Persistent Slots + SwiftTD Critic](../open_questions/object-centric-swifttd-critic.md) — superquadric slots as the object-centric state for a policy and critic
