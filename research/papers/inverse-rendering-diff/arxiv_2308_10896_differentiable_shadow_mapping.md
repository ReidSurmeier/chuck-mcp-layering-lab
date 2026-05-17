# Differentiable Shadow Mapping for Efficient Inverse Graphics

authors: Markus Worchel, Marc Alexa
arxiv: 2308.10896
venue: ICCV 2023
relevance: Pre-filtered shadow mapping made differentiable. The take-away for chuck-mcp is the META-LESSON: a coarse, prefiltered approximation of a discontinuous operation (visibility / opaque-vs-not) gives differentiable gradients that are MORE useful than the exact-but-noisy Monte Carlo gradient. Same lesson applies to chuck-mcp's hard alpha-matte boundaries: keep the forward render exact, but compute gradients through a prefiltered version of the alpha stack.

Key idea:
- Pre-filtered shadow mapping (PCF, VSM, ESM) is an old computer-graphics trick: render shadow depth, filter it, sample to get a smooth visibility approximation. Worchel & Alexa observe that this smooth visibility is naturally differentiable with respect to scene parameters, and orders of magnitude faster than differentiable light-transport simulation (Mitsuba-style).
- For inverse-graphics problems where shadows are causing convergence failures, swapping in differentiable shadow maps fixes the gradients without needing the full Monte Carlo apparatus.
- Demonstrates that differentiable rasterization WITHOUT shadows often fails to converge — the shadow gradient is doing real work, but it doesn't need to be expensive.

Algorithmic moves to import for chuck-mcp:
1. For chuck-mcp's layered alpha mattes: build a "prefiltered alpha map" (running blur of the alpha stack at multiple scales) and route the gradient of the loss through the PREFILTERED stack, not the hard-rendered stack. Forward render uses the hard stack (correct visual output); gradient computation uses the prefiltered stack (smooth, informative).
2. This is exactly the trust-region knob: prefilter radius is the trust region size. Start with a large radius (coarse alpha gradients), shrink it as the solver converges.
3. Equivalent to chuck-mcp's existing "blurred underlayer envelope before logit parameterization" — Worchel & Alexa give the principled justification for it.

Abstract:
"We show how shadows can be efficiently generated in differentiable rendering of triangle meshes. Our central observation is that pre-filtered shadow mapping, a technique for approximating shadows based on rendering from the perspective of a light, can be combined with existing differentiable rasterizers to yield differentiable visibility information. We demonstrate at several inverse graphics problems that differentiable shadow maps are orders of magnitude faster than differentiable light transport simulation with similar accuracy — while differentiable rasterization without shadows often fails to converge."

Reference for further reading: project page on the paper has shadow-map code; chuck-mcp would adapt the prefilter idea to its layered-alpha stack rather than the depth buffer.
