# Chuck MCP Role-Based Solver Plan

This plan is for the role-based S5 solver branch. It starts from saved baseline
`chuck-mcp-speckle-m10-20260512` and keeps this repo separate from the
`emma-mokuhanga-mcp` experiment.

## Research Basis

- Burt and Adelson's Laplacian pyramid motivates splitting reconstruction into
  coarse tonal structure and higher-frequency detail bands:
  https://vision.csee.wvu.edu/classes/cs674-f17/reading/TheLaplacianPyramidAsACompactImageCode_COM1983.pdf
- SLIC superpixels are a practical next step for forcing printable brushed
  zones instead of arbitrary pixel islands:
  https://www.cs.jhu.edu/~ayuille1/JHUcourses/VisionAsBayesianInference2022/4/Achanta_SLIC_PAMI2012.pdf
- Rudin, Osher, and Fatemi's total-variation model supports using TV as a
  smoothness pressure against noisy masks:
  https://web.eecs.utk.edu/~hqi/ece692/references/noise-TV-PhysicaD92.pdf
- JAX `image.resize` supports differentiable control-grid upsampling for
  lower-resolution underlayer parameters:
  https://docs.jax.dev/en/latest/jax.image.html

## Implemented Build

1. Role layout is assigned after light-to-dark print-order sorting.
   - `under`: first 3 pulls when enough pigments exist.
   - `mid`: regional color/shadow pulls.
   - `detail`: last 2 key/detail pulls.

2. Different pull parameterization is in S5.
   - Underlayers optimize on a 4x coarser control grid.
   - Mid layers optimize on a 2x coarser control grid.
   - Detail layers remain full solve-grid.
   - All groups expand through differentiable `jax.image.resize`.

3. Frequency-banded loss is in S5.
   - Under prefix compares to a partial-strength, heavily low-pass target.
   - Mid prefix compares to a partial-strength mid-pass target.
   - Full composite still carries edge-weighted RGB, low-pass, TV, speckle,
     and dark-on-bright terms.

4. Staged solve is supported but not the default.
   - `WOODBLOCK_ROLE_WARMUP=1` enables short under/mid/detail warm-ups before
     the final joint solve.
   - Validation showed warm-up stages over-shaped prefixes on the Emma input,
     so default behavior is joint optimization over role-parameterized groups.

5. Machinability moved upstream in practice.
   - Under/mid groups have fewer degrees of freedom before SVG/vectorization.
   - Soft local-support and TV terms remain in the optimizer.
   - Hard post-solve topology repair remains rejected unless it passes the ΔE
     guard, because earlier tests showed it damaged color match.

## Validation Snapshot

Input:
`/srv/woodblock-share/input-images/close_emma_2002_2048.jpg`

Saved baseline:
`chuck-mcp-thorough-m10-main-20260512-150141`

Best role-parameterized ablation:

| Metric | Saved M10 | Role Param |
|---|---:|---:|
| mean ΔE76 | 6.941 | 7.277 |
| p95 ΔE76 | 17.772 | 18.258 |
| first 3 pull components | 2847 | 1364 |
| solver wall | 27.7s | 43.4s |

Warm-up stage runs were rejected as defaults:

| Run | mean ΔE76 | first 3 components | Decision |
|---|---:|---:|---|
| stage1 | 21.810 | 990 | rejected, underlayers overfit prefix |
| stage2 | 8.393 | 1887 | rejected, color still regressed |
| stage3 | 8.056 | 1805 | rejected, color still regressed |
| final warm-up | 8.211 | 1304 | rejected, worse than joint role solve |

## Remaining Work

- Replace mid-layer control grids with SLIC/superpixel brushed-zone parameters.
- Add explicit component-count/topology terms to score output, not just offline
  diagnostics.
- Add role-aware acceptance thresholds: early/mid component counts, ΔE budget,
  and clean preview inspection.
- Promote `WOODBLOCK_ROLE_WARMUP=1` only after it beats joint role optimization
  on the shared examples.
