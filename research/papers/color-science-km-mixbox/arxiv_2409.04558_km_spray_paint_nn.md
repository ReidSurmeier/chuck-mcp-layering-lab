# Solve paint color effect prediction problem in trajectory optimization of spray painting robot using artificial neural network inspired by the Kubelka Munk model

authors: Hexiang Wang, Zhiyuan Bi, Zhen Cheng, Xinru Li, Jiake Zhu, Liyuan Jiang, Hao Li, Shizhou Lu
arxiv: 2409.04558v1 [cs.RO] 6 Sep 2024
relevance: Hybrid K-M-physics + MLP-with-gating-and-residual that learns the color-rendering surrogate from real measurements of multi-color overlapped paint films. This is the t2 empirical-LUT tier of chuck-mcp expressed as a neural surrogate: K-M provides the inductive bias, the network corrects for what K-M alone can't capture (drying gradients, edge effects, anisotropic film thickness).

## Abstract

The spray-painting robot trajectory planning technology aiming at spray painting quality mainly applies to single-color spraying. Conventional methods of optimizing the spray gun trajectory based on simulated thickness can only qualitatively reflect the color distribution, and can not simulate the color effect of spray painting at the pixel level. Therefore, it is not possible to accurately control the area covered by the color and the gradation of the edges of the area, and it is also difficult to deal with the situation where multiple colors of paint are sprayed in combination. To solve the above problems, this paper is inspired by the Kubelka-Munk model and combines the 3D machine vision method and artificial neural network to propose a spray painting color effect prediction method. The method is enabled to predict the execution effect of the spray gun trajectory with pixel-level accuracy from the dimension of the surface color of the workpiece after spray painting. On this basis, the method can be used to replace the traditional thickness simulation method to establish the objective function of the spray gun trajectory optimization problem, and thus solve the difficult problem of spray gun trajectory optimization for multi-color paint combination spraying.

## Key technical contributions

1. **K-M model derivation as mathematical scaffold**: the paint-film color rendering equations from Kubelka-Munk are formalized first, then used to define the input/output space for the MLP.
2. **Dataset construction**: depth-camera + point-cloud capture of real sprayed workpieces yields a paired (thickness profile, measured surface color) dataset — exactly analogous to the chuck-mcp need to scan/photograph Pace Editions proofs to train an empirical LUT.
3. **MLP with gating + residual connections**: predicts color outcome at pixel level, replacing the closed-form K-M evaluation where the simple model is insufficient (edges, multi-color overlap).
4. **Multi-color combination spraying**: the network handles the case of N paint colors overlapping — the precise mokuhanga overprint case (113 colors stacked across 132 pulls for the Close Emma).

## Why this matters for chuck-mcp

- Validates the tier-2 empirical LUT plan. A neural surrogate trained on measured plate proofs is a legitimate intermediate between pure analytic K-M (t3) and Mixbox lerp (t1).
- Demonstrates that K-M is a useful *inductive bias* (not the final renderer) — feature engineering inputs as concentration vectors, not raw RGB, helps the network generalize across pigment combinations.
- The gating-and-residual MLP architecture is a reasonable starting point for our t2 implementation.

## Limitations

- Spray paint is opaque + thick; mokuhanga washi pigment films are translucent + thin. The forward-scatter regime is different.
- No mention of Saunderson surface correction or paper sizing effects.
