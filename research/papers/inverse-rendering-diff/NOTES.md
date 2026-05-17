# NOTES — Inverse Rendering / Differentiable Layering for Chuck MCP

Research agent INVERSE-RENDERING, swarm-1778962124344-s4cm4l.
Goal: find published precedent for chuck-mcp's staged hierarchical solver (broad supports → color/depth → detail/key), with bounded trust-region feedback and discrete plate/pigment choices.

## Artifacts in this folder

| # | File | Status | Topic |
|---|------|--------|-------|
| 1 | `arxiv_2511_13191_birth_of_a_painting.md` | full text | Differentiable brushstroke reconstruction, coarse-to-fine |
| 2 | `arxiv_2511_20034_clair_obscur_illumination_aware_vectorization.md` | full text | Albedo / shade / light layer decomposition |
| 3 | `arxiv_2408_15741_segmentation_layerwise_vectorization.md` | full text | Segmentation-driven next-path placement |
| 4 | `arxiv_2406_05404_layered_vectorization_semantic_simplification.md` | full text | Semantic simplification → layered SVG |
| 5 | `arxiv_2408_16005_many_worlds_inverse_rendering.md` | full text (trimmed) | Mitsuba 3 SOTA for discontinuous-configuration gradients |
| 6 | `arxiv_2312_11334_optimize_and_reduce_topdown_vectorization.md` | full text | Top-down prune-from-overcomplete vectorization |
| 7 | `arxiv_2206_04655_LIVE_layerwise_vectorization.md` | reference + algorithm summary | Foundational layer-wise vectorization (CVPR 2022 Oral) |
| 8 | `arxiv_2308_09865_topological_derivatives_inverse_rendering.md` | reference + algorithm summary | Hole/phase nucleation gradients (ICCV 2023) |
| 9 | `arxiv_2110_09107_perturbed_optimizers.md` | reference + algorithm summary | Randomized-smoothing gradients (NeurIPS 2021) |
| 10 | `arxiv_2308_10896_differentiable_shadow_mapping.md` | reference + algorithm summary | Prefiltered-visibility gradient trick (ICCV 2023) |
| 11 | `arxiv_1611_01144_gumbel_softmax_jang.md` | reference + algorithm summary | Categorical reparameterization (ICLR 2017) |
| 12 | `arxiv_2504_01402_pbdr_survey.md` | reference | Survey of physics-based differentiable rendering |
| 13 | `ref_diffvg_li_2020.md` | reference | DiffVG rasterizer (foundation) |
| 14 | `ref_mitsuba3_jakob_2022.md` | reference | Mitsuba 3 inverse-renderer (foundation) |

## (a) Five papers most directly applicable to a STAGED HIERARCHICAL solver

In order of immediate impact on chuck-mcp's next sprint:

1. **Birth of a Painting: Differentiable Brushstroke Reconstruction (Jiang et al., 2025, arXiv:2511.13191)**
   - Coarse-to-fine differentiable stroke reconstruction with a parallel differentiable paint renderer.
   - Explicitly schedules large support strokes first, refines with smaller strokes later.
   - This is the most recent and most directly analogous published work — they share chuck-mcp's exact architectural insight that you need to *schedule* layer granularity, not let it emerge from a flat solver.

2. **Layered Image Vectorization via Semantic Simplification (Wang et al., 2024, arXiv:2406.05404)**
   - Iteratively SIMPLIFIES the target image, vectorizes each simplification level as a separate layer.
   - Direct algorithmic answer to chuck-mcp's question "how do we force batch 1 to take large supports without re-litigating in batch 3?": fit batch 1 to a coarsely-simplified target, fit batch 3 to the original.

3. **Clair Obscur: Illumination-Aware Image Vectorization (Lin et al., 2025, arXiv:2511.20034)**
   - Decomposes into ALBEDO / SHADE / LIGHT layers via differentiable rendering with semantic-guided initialization.
   - Validates chuck-mcp's existing role logic (light support / mid hue / detail key) as a published framework.

4. **Towards Layer-wise Image Vectorization (LIVE, Ma et al., 2022, arXiv:2206.04655)**
   - Foundational. Progressive add-paths + joint re-optimize, with UDF contour loss acting as a soft trust region.
   - The OUTER LOOP of chuck-mcp's staged solver should be LIVE-style; inner loop stays L-BFGS-B.

5. **Segmentation-guided Layer-wise Image Vectorization (Zhou et al., 2024, arXiv:2408.15741)**
   - LIVE's direct successor. Uses Laplacian-on-residual + Otsu + watershed to pick where to place the next path. Cleaner, more deterministic than LIVE's color-bucket approach.
   - Direct blueprint for chuck-mcp's "after batch 1, where should the next chroma/accent plate go?" decision.

**Honorable mention** (would be in the top-5 if "staged" is read more loosely):

- **A Theory of Topological Derivatives (Mehta et al., 2023, arXiv:2308.09865)** — formal answer to "where should a new plate be NUCLEATED?" Pixel-local "spawn a new region here" gradient. Pairs beautifully with #5 above.

## (b) Recommended algorithmic moves for chuck-mcp

Concrete, ordered by how much they will move the needle on the May-14 benchmark.

### Move 1 (highest impact): Make staged solving an OUTER LOOP, not a flat solve

Stop solving 8-12 alpha planes simultaneously. Adopt the LIVE pattern:

```
for batch_idx in [1=supports, 2=color_depth, 3=detail_key]:
    target_for_this_batch = simplify(original_target, scale=batch_idx)
    new_plates = segment_residual(target_for_this_batch - render(current_stack))
    initialize_new_plates_at_residual_centroids(new_plates)
    optimize_all_plates_jointly(
        active_plates = current_stack ++ new_plates,
        trust_region_on_old_plates = decreasing_with_batch_idx,
        objective = delta_e_against(target_for_this_batch),
    )
    freeze_or_soft_constrain(current_stack, trust_region)
    current_stack = current_stack ++ new_plates
```

The novel piece vs. LIVE: chuck-mcp solves against a *simplified* target per batch (Wang et al. 2024, arXiv:2406.05404), not the original. This is the principled answer to "how do I prevent batch 1 from chasing high-frequency residuals."

### Move 2: Trust region = prefiltered alpha map

Adopt Worchel & Alexa's prefiltering trick (arXiv:2308.10896) for chuck-mcp's alpha stack.

- Forward render uses the hard alpha stack (unchanged — keeps your forward render physically meaningful).
- Backward gradient routes through a PREFILTERED alpha stack (Gaussian blur with radius `r_batch`).
- `r_batch_1` = large (supports get smoothed gradients, can't chase pixel detail).
- `r_batch_3` = ~1 pixel (detail plates get crisp gradients).

This gives you a *continuous* trust-region knob instead of an on/off freeze. Empirically much better than freezing.

### Move 3: Replace "new plate" guess with topological-derivative nucleation

Use Mehta, Chandraker, Ramamoorthi (arXiv:2308.09865) topological derivatives to decide where a NEW plate gets spawned at the start of each batch.

- After batch_idx is solved, compute the topological-derivative map: at each pixel, "what's the change in DeltaE from inserting a new alpha-1 region here at color c?"
- The argmax over (pixel, color) is the next plate's location and pigment.
- This replaces the current heuristic of "guess m_prior=10, hope it's enough." chuck-mcp can grow plate count adaptively and STOP when the marginal topological derivative falls below a threshold.

This is the single biggest algorithmic improvement available, and the most novel relative to chuck-mcp's current pipeline.

### Move 4: Gumbel-softmax for pigment slot, perturbed-optimizer for batch assignment

Two discrete choices, two different tools:

- **Per-plate pigment slot** (~36 catalog entries): Gumbel-softmax (Jang et al., arXiv:1611.01144) with τ annealed from 1.0 to 0.1 over the inner L-BFGS-B run. Low-dim discrete choice, Gumbel works fine.
- **Per-plate batch index** (3 batches): perturbed optimizer (Le Lidec et al., arXiv:2110.09107) with noise scale annealed across outer-loop iterations. Higher-dim and the assignments interact — perturbed optimizer's variance-reduced randomized smoothing is the better tool.

### Move 5: Bounded feedback via prefilter, not via L-BFGS-B box constraints

chuck-mcp's current "bounded feedback from batch composites" plan in the build plan (item #2) is exactly what prefiltered alpha maps deliver. The L-BFGS-B box-constraint approach (clip param updates) is fragile; the prefilter approach is principled (gradient smoothness, not parameter smoothness).

### Move 6: Top-down sanity check via Optimize&Reduce

After staged solve completes, run the Hirschorn/Jevnisek/Avidan top-down sanity check (arXiv:2312.11334):
- Start with all m_max plates active.
- Rank by marginal DeltaE contribution.
- Prune the lowest-contributing plate, refit, repeat.
- Compare the pruned-down result with the staged-built-up result.

If they don't agree, the staged solver missed something. This is a cheap, mechanical validation step.

## (c) Reusable JAX libraries / tools to consider

In priority order:

1. **JAXopt** (already in chuck-mcp): keep using `jaxopt.LBFGSB` for the inner continuous solve. Verified to work well with bound constraints.

2. **Dr.Jit + Mitsuba 3** (https://github.com/mitsuba-renderer/mitsuba3): if chuck-mcp ever wants full path-traced gradients (e.g., translucent washi paper sub-surface scattering during overprint), this is the only mature open option. Has a Python frontend with JAX-compatible numpy semantics. Mitsuba's "rebound" / multi-resolution inverse-rendering tutorial is the cleanest staged-optimization example in the literature.

3. **diffvg / diffvg-jax** (https://github.com/BachiLi/diffvg, plus community JAX ports): if chuck-mcp wants to emit *vector* path geometry directly during the inverse solve (not just alpha mattes + post-hoc vectorization), this is the gradient backbone. The chuck-mcp staged solver could plug each batch directly into DiffVG: each batch's plates are sets of Bezier paths, and gradients flow through path control points + fill colors.

4. **optax** (DeepMind's JAX optimizer library): chuck-mcp may want Adam-based exploration alongside L-BFGS-B. Use Adam (with weight decay) for batch 1 (rough convergence, wide exploration), then switch to L-BFGS-B for batch 2-3 (sharp convergence near the optimum). Optax has the lr-scheduling / warmup / cosine-decay tooling chuck-mcp will need.

5. **equinox** (JAX neural-net library): if chuck-mcp wants to add a learned region-proposal network on top of the topological-derivative nucleation (predict "good plate seed locations" from a CNN), equinox is the cleanest way to keep everything in JAX.

6. **jax-md** or **JAX-LBFGSB**: alternative LBFGS-B implementations if `jaxopt`'s gets stuck. Mostly only relevant if profiling shows the inner solver is the bottleneck (currently it isn't — the bottleneck is the OUTER loop structure).

7. **scikit-image** (NOT JAX, but ubiquitous in this literature): for the Laplacian + Otsu + watershed segmentation step (segmentation-guided initialization, Zhou et al. 2024). Run on CPU between JAX inner-loop iterations.

8. **Mixbox** / **K-M libraries**: chuck-mcp already has these in the color science package. No change recommended.

## Single biggest algorithmic recommendation

**Switch from a flat 8-12-plane simultaneous L-BFGS-B solve to a 3-batch staged loop, where each batch fits against a different progressively-detailed target (Wang et al. 2024 pattern), uses a prefiltered alpha-map gradient (Worchel & Alexa 2023 trick) sized to its batch index for trust-region bounded feedback, and seeds new plates at topological-derivative nucleation maxima (Mehta et al. 2023). The inner L-BFGS-B per batch stays; the OUTER loop is what changes.**

This combines moves 1-3 from section (b). It is the smallest possible change that addresses all three of the build plan's failure modes (no support hierarchy, runaway pixel chasing in early batches, no principled "should we add another plate" gate) and every piece of it has a published reference.

## Report-back summary

- File count: 14 artifacts (12 arXiv-IDed papers + 2 foundational references).
- Top 3 must-reads:
  1. "Birth of a Painting: Differentiable Brushstroke Reconstruction" — arXiv 2511.13191 — Jiang et al. 2025.
  2. "Layered Image Vectorization via Semantic Simplification" — arXiv 2406.05404 — Wang et al. 2024.
  3. "A Theory of Topological Derivatives for Inverse Rendering of Geometry" — arXiv 2308.09865 — Mehta, Chandraker, Ramamoorthi 2023 (ICCV).
- Single biggest algorithmic recommendation: see section above. Three-batch staged loop, each batch against a progressively-detailed simplified target, with prefiltered-alpha-map gradient as trust region, and topological-derivative nucleation gating each new plate's spawn.
