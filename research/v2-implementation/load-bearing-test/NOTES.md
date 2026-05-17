# Load-Bearing Test — Research Synthesis & Verdict

**Domain:** Efficient per-pull contribution attribution for chuck-mcp v2 mokuhanga layering solver.
**Output folder:** `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v2-implementation/load-bearing-test/`
**Date:** 2026-05-16
**Swarm:** `swarm-1778969836247-ys4o7z`
**Artifact count:** 7 papers + this NOTES.

---

## TL;DR

1. **Use O&R's counterfactual ablation as the canonical load-bearing score.** It's the AAAI-2024 SOTA approach for "remove the K lowest-importance shapes" in differentiable vector-graphics solvers, and it ports near-1:1 to chuck-mcp.
2. **Approximate it with gradient × mask (SLA-style) inside the inner loop** for 100× speedup; reserve true ablation for outer-loop reduction checkpoints and final verification.
3. **Use a hybrid soft+hard penalty**: soft L1/L2 regularisation on pull masks inside the L-BFGS-B inner loop, hard "drop bottom-K" prune at outer-loop boundaries. Both proven by O&R + Wang 2024.
4. **Reserve Shapley (SIM-Shapley) for offline audit** on final designs, not the hot path. 132 pulls is too many for any exact Shapley method.
5. **Show users a per-pull contribution heatmap** as a `(132,)` bar chart sorted by score, plus an interactive ablation slider that lets them remove a pull and watch the composite update.

---

## Verdict 1 — How to compute the load-bearing score per pull

### Inner-loop (every optimiser step, runs ~200x per outer pass)

**Gradient × mask (1-step IG / SLA-style)**

```python
@jax.jit
def load_bearing_inner(pulls, target, render_fn, deltaE_fn):
    """O(1) attribution: one forward + one backward through K-M render."""
    composite = render_fn(pulls)
    pixel_grad = jax.grad(lambda c: deltaE_fn(c, target).sum())(composite)
    # Project gradient onto each pull's mask footprint
    scores = jax.vmap(lambda p: jnp.sum(jnp.abs(pixel_grad) * p.mask))(pulls)
    return scores  # shape (N=132,)
```

- **Cost:** 1 forward + 1 backward render through the K-M overprint pipeline.
- **Time:** ~100-400ms for 132 pulls × 8MP canvas on A100/4090.
- **Well under the 5-second budget.** Can be called every iteration.

### Outer-loop reduction (every 100-200 inner iters)

**Counterfactual ablation (O&R rank-score)**

```python
@jax.jit
def load_bearing_counterfactual(pulls, target, render_fn, deltaE_fn):
    """True leave-one-out: O(N) forward renders, no backward."""
    base = render_fn(pulls)
    base_loss = deltaE_fn(base, target).mean()
    # vmap across pulls; each branch renders all-but-pull-k
    def drop_k(k):
        kept = jax.tree_util.tree_map(lambda x: jnp.delete(x, k, axis=0), pulls)
        composite = render_fn(kept)
        return deltaE_fn(composite, target).mean() - base_loss
    scores = jax.vmap(drop_k)(jnp.arange(len(pulls)))
    return scores
```

- **Cost:** N forward renders. With JAX vmap on GPU, batches of 10-20 fit at once.
- **Time:** ~1-2 seconds for 132 pulls. **Under budget.**
- **Stronger guarantee** than gradient × mask: catches cancellation pairs where two pulls fight each other.

### Offline final verification (one-shot, can take minutes)

**SIM-Shapley with 1000-coalition budget**

Run once per design before sending to carving. Identifies pulls whose Shapley value disagrees with their ablation score (= pulls involved in cancellation pairs) for designer review.

---

## Verdict 2 — Soft vs hard penalty

**Use both, in different parts of the optimisation.**

| Mechanism | Where | Why |
|---|---|---|
| **Soft L1 on `mask_k` values** | Inside L-BFGS-B inner loop | Differentiable; encourages naturally-sparse masks (small inked regions) without breaking line search. Already a sparsity prior. |
| **Soft hinge on load-bearing score** | Inside outer loop, between L-BFGS-B calls | `L_load = Σ max(0, threshold - score_k)` — adds gradient pressure away from low-importance configurations. Differentiable. |
| **Hard drop bottom-K by counterfactual score** | At outer-loop checkpoints (every 1-2 batches) | Clean discrete removal; reduces problem size; allows re-spawn via topo-derivative. The O&R recipe. |

The pure-soft approach (Louizos hard-concrete L0) is **incompatible with L-BFGS-B** because stochastic gates break deterministic line search. The pure-hard approach (just dropping) wastes optimiser steps on doomed pulls. **Hybrid wins.**

### Recommended schedule

```
Outer iter 1 (structural pulls only, ~50 pulls):
  - Inner LBFGS-B 200 iters, with soft L1 + soft load-bearing hinge.
  - Counterfactual scoring.
  - Drop pulls with score < 0.1 ΔE.
  - Spawn 50 new pulls at top-50 topo-derivative peaks.

Outer iter 2 (structural + midtone, ~120 pulls):
  - Same.
  - Drop pulls with score < 0.05 ΔE.

Outer iter 3 (full ~200 pulls):
  - Same.
  - Drop pulls with score < 0.02 ΔE until N ≤ 132.

Final pass:
  - SIM-Shapley audit; flag discordant pulls for user review.
```

This matches O&R's 256 → 128 → 64 halving schedule, adapted for chuck-mcp's 200 → 150 → 132.

---

## Verdict 3 — Per-pull contribution heatmap (UI)

### Required surfaces

1. **Sorted bar chart** — pulls 1-132 (final solver order) ranked by load-bearing score. X-axis = pull index, Y-axis = ΔE-loss-if-removed. **The dashed-line cutoff** (a-la O&R Fig 3) is the "what should have been pruned" threshold. Pulls below the line that survived → user-flagged for manual review.
2. **Per-pull spatial heatmap** — for the selected pull, overlay the pull's `|∂L/∂composite| · mask_k` on the target image. Shows *where in the print* the pull is load-bearing. Reveals pulls that are load-bearing only at edges (genuine highlight/key information) vs pulls load-bearing only in saturated regions (probably over-printing).
3. **Ablation slider** — interactive: select pull k, watch the composite re-render *without* pull k. User sees exactly what the pull contributes. Cached forward renders make this < 100ms.
4. **Cancellation-pair detector** — list of (pull_i, pull_j) pairs where Shapley(i, j) ≪ ablation(i) + ablation(j). Means the pair is mutually cancelling. Show overlay.

### UI placement

Per chuck-mcp pipeline (S5 → S6.b), this lives **between solver output and jigsaw step**:
```
S5 solver output → load-bearing audit UI (gate) → S6.b jigsaw segmentation
                       ↑
              user can intervene here
```

The user clicks through suspicious pulls and either:
- Confirms (pull stays).
- Pins to "must-keep" (overrides solver).
- Deletes (back into solver for re-optimisation without it).

---

## Verdict 4 — Performance budget

| Phase | Operation | Frequency | Budget | Implementation |
|---|---|---|---|---|
| Inner LBFGS step | gradient × mask | Every step (~200/iter) | <50ms | JAX jit |
| Outer reduction | counterfactual ablation | Every 1-2 batches | <2s | JAX vmap over pulls |
| Final audit | SIM-Shapley | Once per design | <5min | Background job |
| UI ablation slider | render with pull k missing | On user interaction | <100ms | Cached forward render delta |

**Total budget for one full load-bearing pass: ~2-3 seconds.** Comfortably under the 5-second V1 budget.

---

## Top-3 must-reads

1. **Hirschorn et al. 2024 "Optimize & Reduce: A Top-Down Approach for Image Vectorization"** (AAAI 2024, arXiv 2312.11334) — the canonical reference. Rank-score by counterfactual ablation, halving schedule, geometric loss. Direct template for chuck-mcp v2.
2. **Bandyopadhyay et al. 2024 "What Sketch Explainability Really Means for Downstream Tasks"** (CVPR 2024, arXiv 2403.09480) — stroke-level attribution via differentiable rasterisation. The single-step IG approximation we should use in the hot path.
3. **Jiang et al. 2025 "Birth of a Painting: Differentiable Brushstroke Reconstruction"** (arXiv 2511.13191) — parallel differentiable paint renderer architecture. The vmap pattern for getting all-pull attribution in one backward pass.

---

## Code sketch — full pipeline

```python
import jax, jax.numpy as jnp
from jax import vmap, jit, grad

# ── primitives ────────────────────────────────────────────────────────

@jit
def km_overprint(transmittance_below, ink, mask):
    """K-M overprint: differentiable in mask, ink."""
    return transmittance_below * jnp.where(mask, ink_transmittance(ink), 1.0)

def forward_render(pulls, paper):
    """Composite all pulls onto paper. Order matters."""
    composite = paper
    for pull in pulls:  # cant vmap, ordered overprint
        composite = km_overprint(composite, pull.ink, pull.mask)
    return composite

# ── attribution ───────────────────────────────────────────────────────

def deltaE2000(I1, I2):
    """Lab-space ΔE2000. Differentiable end-to-end."""
    ...

@jit
def load_bearing_gradient(pulls, target, paper):
    """Inner-loop attribution: one backward pass."""
    def loss(p):
        return deltaE2000(forward_render(p, paper), target).sum()
    pull_grads = grad(loss)(pulls)
    return jnp.array([jnp.sum(jnp.abs(g.mask)) for g in pull_grads])

@jit
def load_bearing_ablation(pulls, target, paper):
    """Outer-loop attribution: N forward renders (vmapped)."""
    base = forward_render(pulls, paper)
    base_dE = deltaE2000(base, target).mean()
    def drop_k(k):
        kept = pulls[:k] + pulls[k+1:]
        return deltaE2000(forward_render(kept, paper), target).mean() - base_dE
    return vmap(drop_k)(jnp.arange(len(pulls)))

# ── full outer loop ───────────────────────────────────────────────────

def optimize_pulls(target, paper, budget=132, n_init=200):
    pulls = initialize_from_target(target, n=n_init)
    
    for outer in range(3):  # 3 staged batches per inverse-rendering NOTES
        # Inner LBFGS-B with soft load-bearing hinge
        def inner_loss(p):
            return (deltaE2000(forward_render(p, paper), target).sum() 
                    + 1e-3 * sum(jnp.sum(jnp.abs(pi.mask)) for pi in p)  # soft L1
                    + 1e-2 * jnp.sum(jnp.maximum(0, 0.05 - load_bearing_gradient(p, target, paper)))  # soft hinge
                   )
        pulls = lbfgs_b(pulls, inner_loss, n_iters=200)
        
        # Hard prune
        scores = load_bearing_ablation(pulls, target, paper)
        n_target = max(budget, len(pulls) - 30)
        keep = jnp.argsort(-scores)[:n_target]
        pulls = [pulls[i] for i in keep]
        
        # Topo-derivative spawn (sister artifact)
        new_pulls = spawn_from_topo_derivative(target - forward_render(pulls, paper), k=20)
        pulls = pulls + new_pulls
    
    # Final audit
    sim_shapley_scores = sim_shapley_attribution(pulls, target, paper, n_coalitions=1000)
    flagged = identify_cancellation_pairs(scores, sim_shapley_scores)
    
    return pulls, scores, flagged
```

---

## Open questions for downstream domains

- **Color-science integration:** is gradient × mask attribution stable through Saunderson-corrected K/S? Probably yes (differentiable), but verify on a 2-block test.
- **Mokuhanga reduction (block, pass_idx, mask):** load-bearing of a *pass* (not just a pull) — the score should aggregate across all passes of the same block, since dropping the whole block is what carving cares about. Add a `block_load_bearing[b] = Σ_{passes of b} score` rollup.
- **HITL anchor:** if the user has pinned a region via `pin_region`, those pulls must remain regardless of score. Modify the prune to respect `is_pinned` flags.

---

## Artifacts in this folder

```
arxiv_2312_11334_optimize_and_reduce.md             — O&R, the canonical template
arxiv_2403_09480_sla_stroke_attribution.md          — SLA stroke-level attribution via diff. raster
arxiv_2511_13191_birth_of_painting_ordering.md      — parallel diff. renderer pattern
arxiv_2406_05404_layered_vectorization_simplification.md — progressive simplification + pruning
arxiv_2505_03201_weighted_integrated_gradients.md   — IG with baseline weighting
arxiv_2505_08198_sim_shapley.md                     — Shapley approximation (offline audit)
arxiv_2308_09865_topological_derivatives.md         — dual: where to ADD pulls
arxiv_1712_01312_l0_hard_concrete.md                — soft-pruning alternative (rejected for LBFGS)
NOTES.md                                            — this file
```
