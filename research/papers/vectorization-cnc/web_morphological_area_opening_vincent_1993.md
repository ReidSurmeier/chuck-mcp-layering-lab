---
title: "Morphological Area Openings and Closings for Grey-scale Images"
authors: ["Luc Vincent"]
year: 1993
venue: "EURASIP Workshop on Mathematical Morphology and its Applications to Signal Processing / book chapter in Shape in Picture, Springer 1994"
source: "https://link.springer.com/chapter/10.1007/978-3-662-03039-4_13"
also: "Soille, P. (2003). Morphological Image Analysis: Principles and Applications, 2nd ed. Springer."
relevance: "MUST-READ — area-opening is the operation chuck-mcp's S6.c already runs (island pressure). Vincent's formulation generalizes to grayscale and provides the efficient union-find / priority-queue implementation. Pairing area-opening *before* Potrace eliminates spurs Potrace would otherwise delete unaccountably."
tags: [morphology, area-opening, attribute-filter, binary-mask, cleanup]
---

# Morphological Area Openings — Vincent (1993)

## Why this matters for chuck-mcp S6.c

The S6.c stage reports "island pressure" — connected components below
a minimum area get killed. This is **exactly area opening**:

> γ_λ(X) = ⋃ { C : C connected component of X, area(C) ≥ λ }

But chuck-mcp currently does this *after* the plate solver, then hands
the cleaned mask to the SVG exporter (Potrace). The cleaner design:

1. Solver outputs raw plate mask
2. S6.c runs **area-opening at λ = min_feature_px²** as the first step
3. S6.c measures ΔE between cleaned mask and pre-clean mask
4. If ΔE ≤ budget: emit; else escalate
5. Potrace traces the cleaned mask — no spurs to remove

## Formal definition

For a binary image X, area opening with parameter λ:

  γ_λ(X) = ⋃_C {C : C is a 4-connected (or 8-connected) component of X with area(C) ≥ λ}

Equivalently:

  γ_λ(X) = X \ {pixels in components of area < λ}

Grayscale generalization (Vincent's contribution): apply binary area
opening to each threshold set X_h = {x : f(x) ≥ h} and reconstruct.
Implemented efficiently with a min-priority queue and union-find:

  - Sort pixels by intensity (descending)
  - Process pixels in order, merging into a component graph
  - When a component's area exceeds λ, mark it "alive"
  - At end, output = max intensity of alive component containing each pixel

Complexity: O(n log n) with priority queue, or O(n α(n)) with union-find
(near-linear).

## Properties

- **Idempotent**: γ_λ(γ_λ(X)) = γ_λ(X)
- **Anti-extensive**: γ_λ(X) ⊆ X
- **Increasing**: X ⊆ Y → γ_λ(X) ⊆ γ_λ(Y)
- **Shape-agnostic**: unlike structuring-element opening, area opening
  doesn't bias the surviving components toward any particular shape.
  A thin spur of area ≥ λ survives; an isotropic blob of area < λ dies.
  This is the right semantic for "machinable / not machinable".

## Compare to structuring-element opening

| Filter                    | Param | Topology | Shape bias        |
|---------------------------|-------|----------|-------------------|
| Open with disk SE radius r| `r`   | preserved| Smooths features  |
| Open with square SE       | `r`   | preserved| Adds axis bias    |
| Area opening              | `λ`   | preserved| None              |
| Opening by reconstruction | `r`   | preserved| Restores survivors|

For CNC printability, the **right filter is area opening (kill below
λ pixels)** combined with **opening by reconstruction at radius
r = end-mill_radius_px** (kill features narrower than the end mill).
Doing both is critical:

- Area opening kills the isolated spurs (too small to print).
- Opening by reconstruction kills the thin necks (too narrow for the
  end mill to enter).

Doing only area opening misses thin necks (they have enough area).
Doing only structuring-element opening either kills too much (small
isotropic blobs that are perfectly machinable get erased by the disk
erosion) or too little.

## Recommended chuck-mcp S6.c step

```python
import skimage.morphology as morph
import numpy as np

# 1. Area opening — kill below min_feature_area
mask_clean = morph.area_opening(mask, area_threshold=min_feature_px**2)

# 2. Opening by reconstruction — kill necks narrower than end-mill
eroded = morph.binary_erosion(mask_clean, morph.disk(endmill_radius_px))
mask_clean = morph.reconstruction(eroded, mask_clean)

# 3. Area closing + closing by reconstruction symmetrically
#    (kill speckle HOLES below min_feature)
mask_clean = morph.area_closing(mask_clean, area_threshold=min_feature_px**2)
dilated = morph.binary_dilation(mask_clean, morph.disk(endmill_radius_px))
mask_clean = morph.reconstruction(dilated, mask_clean, method='erosion')
```

## Citation

- Vincent, L. "Morphological Area Openings and Closings for Grey-scale
  Images." Proc. NATO Shape in Picture Workshop, 1992, published 1994.
  Springer. doi:10.1007/978-3-662-03039-4_13
- Vincent, L. "Grayscale area openings and closings, their efficient
  implementation and applications." EURASIP Workshop on Mathematical
  Morphology, 1993.
- Soille, P. Morphological Image Analysis: Principles and Applications,
  2nd ed. Springer, 2003. Chapter 6.3.9 (opening by reconstruction),
  Chapter 6.4 (area opening).
- scikit-image: `skimage.morphology.area_opening`,
  `skimage.morphology.area_closing`,
  `skimage.morphology.reconstruction`.
- MATLAB: `bwareaopen`, `imreconstruct`.
