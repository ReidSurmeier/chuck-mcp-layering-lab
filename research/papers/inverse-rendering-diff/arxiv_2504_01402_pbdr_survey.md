# A Survey on Physics-based Differentiable Rendering

authors: Yunfan Zeng, Guangyan Cai, Shuang Zhao
arxiv: 2504.01402
venue: Eurographics State of the Art Report (STAR), 2025
relevance: Canonical 2025 survey of the entire physics-based differentiable rendering field — Mitsuba 3, Dr.Jit, Path Replay Backpropagation, edge-sampling, importance sampling for BRDF derivatives, neural inverse rendering. Use this as the bibliography map: every algorithm chuck-mcp might want to import is cited here with its strengths/weaknesses.

Why chuck-mcp should care:
- The survey explicitly tabulates which differentiable rendering algorithms preserve physical accuracy of discontinuities (silhouette edges, alpha hard edges) — exactly the property chuck-mcp needs when jigsaw plate boundaries become hard masks during S6/S7.
- Section on "Monte Carlo sampling strategies" maps directly onto chuck-mcp's pigment-slot sampling problem: same variance vs. bias tradeoff.
- Section on "computational efficiency" benchmarks JAX-based and Dr.Jit-based renderers — useful when chuck-mcp considers whether to stay on JAX or port to Dr.Jit.

Abstract:
"Physics-based differentiable rendering has emerged as a powerful technique in computer graphics and vision, with a broad range of applications in solving inverse rendering tasks. At its core, differentiable rendering enables the computation of gradients with respect to scene parameters, allowing optimization-based approaches to solve various problems. Over the past few years, significant advancements have been made in both the underlying theory and the practical implementations of differentiable rendering algorithms. In this report, we provide a comprehensive overview of the current state of the art in physics-based differentiable rendering, focusing on recent advances in general differentiable rendering theory, Monte Carlo sampling strategy, and computational efficiency."

Use as: master bibliography map. When a chuck-mcp design question comes up ("how do I handle X discontinuity?"), grep the survey for it and follow the citation graph.

Companion: "Physics Based Differentiable Rendering for Inverse Problems and Beyond" (Kakkar et al., arXiv:2412.08563, 2024) — a more compact tutorial-style overview of the same field.
