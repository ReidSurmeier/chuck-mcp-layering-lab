# Mitsuba 3 — A Retargetable Forward and Inverse Renderer

authors: Wenzel Jakob, Sébastien Speierer, Nicolas Roussel, Merlin Nimier-David, Delio Vicini, Tizian Zeltner, Baptiste Nicolet, Miguel Crespo, Vincent Leroy, Ziyi Zhang
project: https://www.mitsuba-renderer.org/
docs: https://mitsuba.readthedocs.io/
relevance: Reference implementation of physics-based differentiable rendering. Built on Dr.Jit (a JIT compiler), Mitsuba 3 differentiates the entire light transport simulation with respect to camera pose, geometry, BSDFs, textures, volumes. The Mitsuba 3 inverse-rendering tutorials (https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering_tutorials.html) are the cleanest worked examples of staged inverse rendering loops with bounded parameter updates, multiscale optimization, and trust-region-like learning-rate schedules.

Not on arXiv as a single paper (predecessor "Mitsuba 2" by Jakob et al. is in ACM TOG 38(6) 2019). For chuck-mcp purposes the relevant artifacts are:
- The Mitsuba 3 inverse-rendering tutorials — concrete code for parameter-bounded gradient-based optimization with periodic checkpoint/rollback.
- The "Many-Worlds Inverse Rendering" paper (arXiv:2408.16005, already saved here) — Mitsuba-3-era SOTA for discontinuous-configuration gradients.
- "Path Replay Backpropagation" (Vicini, Speierer, Jakob, TOG 2021) — the canonical algorithm Mitsuba 3 uses for memory-efficient differentiable Monte Carlo rendering. Not on arxiv but cited heavily; pdf at https://rgl.epfl.ch/publications/Vicini2021PathReplay.

Why chuck-mcp should care:
- Mitsuba 3's `optim.Adam` + `dr.set_grad_enabled` pattern is essentially the same pattern chuck-mcp uses with JAX/JAXopt. Mitsuba's tutorials show how to STAGE the optimization: lock low-frequency parameters once they've converged, then unlock high-frequency parameters. This is the staged-solver pattern chuck-mcp wants.
- The Mitsuba team's "rebound" / multi-resolution rendering tutorial is a working example of coarse-to-fine optimization where each scale acts as an implicit trust region for the next finer scale.
- If chuck-mcp ever needs full path-traced gradients (e.g., for translucent washi paper sub-surface scattering during the layered overprint), Mitsuba 3 is the only mature engine that supports it.

Reference for further reading:
- Mitsuba 3 inverse rendering tutorials: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering_tutorials.html
- Survey: "A Survey on Physics-based Differentiable Rendering" (Zeng, Cai, Zhao, arXiv:2504.01402, 2025) is the canonical companion read.
