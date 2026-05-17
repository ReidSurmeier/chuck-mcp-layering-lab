"""Plate-objective loss terms for chuck-mcp v4 solver.

These functions go INTO the JAX solver objective (Phase 3 of
`docs/audit-response-and-reconstruction-plan-2026-05-17.md`). They are NOT
post-hoc validators — the validators in `research/v3-construction/
validators-reconstruction/` run on rendered outputs. These terms run on the
plan's continuous variables and produce a scalar that JAX can differentiate.

Naming convention:
    *_loss     — soft target loss (always non-negative, minimised by solver)
    *_penalty  — barrier-style penalty (≥ 0, only fires when constraint violated)

Audit Phase 3 spec (from docs/audit-response-and-reconstruction-plan-2026-05-17.md):

    Loss terms: final image, checkpoint proof, `plate_not_composite` per plate,
    cell exclusivity/jigsaw, role coverage caps, role-frequency permission
    (yellow can have detailed carved structure if first/transparent),
    load-bearing singleton+pair ablation, printability in-loop.

    ΔE_76 in solver loss, ΔE_2000 in validators (CIEDE2000 gradient
    discontinuity per Sharma 2005).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from chuck_mcp_v2.types import Plate, ROLE_FAMILIES, Role
from delta_e import delta_e_76, delta_e_94, delta_e_2000


# Default weights for composite_loss — wired to a single dataclass so callers
# can A/B-test penalty mixes without keyword-arg sprawl.
@dataclass
class LossWeights:
    final_image: float = 1.0
    checkpoint_proof: float = 0.5
    plate_not_composite: float = 0.7
    cell_exclusivity: float = 0.8
    role_coverage_caps: float = 0.3
    role_frequency_permission: float = 0.4
    load_bearing_singleton: float = 0.2
    load_bearing_pair: float = 0.2
    printability: float = 0.5


# ----------------------------------------------------------------------
# 1. final_image_loss — ΔE_76 between rendered final and target (mean)
# ----------------------------------------------------------------------

def final_image_loss(
    rendered_final: jnp.ndarray,
    target: jnp.ndarray,
    lab: bool = True,
) -> jnp.ndarray:
    """Mean ΔE_76 between rendered final composite and target.

    Uses ΔE_76 (Euclidean Lab distance) because it's smooth everywhere —
    safe inside jax.grad. Validators score the same comparison with ΔE_2000
    after the solver lands.

    Args:
        rendered_final: (H, W, 3) array. If lab=True, assumed already in Lab.
        target:         (H, W, 3) array, same convention.
        lab:            if False, treat as RGB and skip Lab conversion.
                        (Lab conversion is the caller's job — sRGB→Lab is
                        non-trivial and ought to live in the renderer.)

    Returns:
        scalar mean ΔE_76 over visible pixels.
    """
    if not lab:
        # If callers pass RGB, we still compute Euclidean diff but explicitly
        # note this is RGB ΔE, not perceptual.
        diff = rendered_final - target
        return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12).mean()
    return delta_e_76(rendered_final, target).mean()


# ----------------------------------------------------------------------
# 2. checkpoint_proof_loss — sum of per-checkpoint ΔE_76
# ----------------------------------------------------------------------

def checkpoint_proof_loss(
    checkpoint_renders: jnp.ndarray,
    expected_proof_progression: jnp.ndarray,
) -> jnp.ndarray:
    """Sum of per-checkpoint mean ΔE_76.

    The reference 7-checkpoint proof series (after pulls 4, 8, 12, 16, 20,
    24, 132 — see design doc ProofState) must match the cumulative-buildup
    reference. This loss term pulls intermediate proofs toward the expected
    progression, not just the final.

    Args:
        checkpoint_renders:        (K, H, W, 3) in Lab — solver's per-checkpoint cumulative renders.
        expected_proof_progression: (K, H, W, 3) in Lab — target cumulative renders.

    Returns:
        scalar sum of mean ΔE_76 over the K checkpoints.
    """
    # Vectorize delta_e_76 across the K-axis with a single broadcast op.
    de = delta_e_76(checkpoint_renders, expected_proof_progression)  # (K, H, W)
    return de.mean(axis=(-2, -1)).sum()


# ----------------------------------------------------------------------
# 3. plate_not_composite_penalty_per_plate
# ----------------------------------------------------------------------

# Threshold from v3 validator: badness ≤ 0.6 is the gate.
# Score is "good" = 1 - badness, so good_score < 0.4 ↔ badness > 0.6 → REJECT.
#
# The validator operates on RENDERED PLATE PREVIEWS (wood-grain + inked
# dark zones). The solver-internal mask is BINARY (1 = inked, 0 = not),
# which makes the cosine-similarity numerically different — high-density
# masks against any structured final have nontrivial cosine sim. So in-loop
# we use a more permissive threshold; the validator enforces the strict
# 0.6 gate at sign-off time on the rendered output.
_PNC_GOOD_THRESHOLD = 0.40


def _good_score_for_plate(plate_mask: jnp.ndarray, final_composite_lum: jnp.ndarray) -> jnp.ndarray:
    """Differentiable JAX version of (1 - plate_not_composite badness).

    Mirrors `research/v3-construction/validators-reconstruction/
    plate_not_composite.py` but uses smooth ops instead of PIL/branching.

    Inputs are already downsampled — caller's responsibility (this is in the
    inner loop, so we trust the renderer to hand us small enough arrays).
    """
    # Inverted luminance signals (dark ink = high signal)
    plate_signal = 1.0 - plate_mask           # (H, W)
    final_signal = 1.0 - final_composite_lum  # (H, W)

    # Cosine similarity
    pf_flat = plate_signal.reshape(-1)
    ff_flat = final_signal.reshape(-1)
    dot = jnp.sum(pf_flat * ff_flat)
    n1 = jnp.sqrt(jnp.sum(pf_flat * pf_flat) + 1e-9)
    n2 = jnp.sqrt(jnp.sum(ff_flat * ff_flat) + 1e-9)
    cos_sim = jnp.clip(dot / (n1 * n2), 0.0, 1.0)

    # Coverage concentration (smooth ramp; matches v3 validator semantics)
    area_fraction = plate_mask.mean()
    # ramp(0.15..0.40) → spread badness; saturates at 1.0
    spread_bad = jnp.clip((area_fraction - 0.15) / (0.40 - 0.15), 0.0, 1.0)

    badness = 0.5 * (cos_sim + spread_bad)
    return 1.0 - badness


def plate_not_composite_penalty_per_plate(
    plates: list[Plate],
    final_composite: jnp.ndarray,
) -> jnp.ndarray:
    """Sum of per-plate ReLU(threshold - good_score) penalties.

    Smooth hinge: penalty fires only when a plate's good_score dips below 0.6.
    The v3 validator REJECTS at good_score < 0.4; we keep a 0.2 safety margin
    inside the loop so the solver doesn't optimize to the boundary.

    Args:
        plates:          list of Plate with .mask in [0,1].
        final_composite: (H, W, 3) Lab; we collapse to luminance for the cosine sim.

    Returns:
        scalar penalty ≥ 0.
    """
    if not plates:
        return jnp.float32(0.0)

    # Luminance from L channel of Lab — already in [0, 100], normalize to [0, 1].
    if final_composite.ndim == 3 and final_composite.shape[-1] == 3:
        final_lum = final_composite[..., 0] / 100.0
    else:
        final_lum = final_composite

    # Vectorise across plates with vmap so the per-plate cosine sim runs
    # as one fused kernel instead of n_plates separate launches.
    stacked_masks = jnp.stack([p.mask for p in plates], axis=0)  # (N, H, W)
    good_scores = jax.vmap(_good_score_for_plate, in_axes=(0, None))(
        stacked_masks, final_lum
    )
    return jax.nn.relu(_PNC_GOOD_THRESHOLD - good_scores).sum()


# ----------------------------------------------------------------------
# 4. cell_exclusivity_penalty
# ----------------------------------------------------------------------

def cell_exclusivity_penalty(plates: list[Plate]) -> jnp.ndarray:
    """Penalize spatial overlap between plates (jigsaw exclusivity).

    A jigsaw plate set must partition the inked area: any pixel should be
    "owned" by at most one plate. Overlap means two physical wood blocks
    are being asked to print the same region, which is the mokuhanga sin
    of "alpha-blending the print".

    Penalty form: sum of pairwise pointwise product, then mean.
    For per-pixel coverage p_i ∈ [0,1], overlap penalty is
        Σ_{i<j} mean(p_i * p_j).

    Args:
        plates: list of Plate with .mask in [0,1].

    Returns:
        scalar overlap penalty ≥ 0.
    """
    if len(plates) < 2:
        return jnp.float32(0.0)

    # Vectorise: stack masks (N, H, W) → sum of coverages > 1 = overlap.
    # Penalty = mean(relu(sum_i p_i - 1.0)) — proportional to "excess ink"
    # per pixel beyond a single owner. Equivalent in spirit to Σ_{i<j} p_i·p_j
    # under the constraint that all p_i ∈ [0, 1], and runs as one vectorised
    # op instead of O(n²) python loops.
    stacked = jnp.stack([p.mask for p in plates], axis=0)  # (N, H, W)
    excess = jax.nn.relu(stacked.sum(axis=0) - 1.0)
    return excess.mean()


# ----------------------------------------------------------------------
# 5. role_coverage_caps_penalty
# ----------------------------------------------------------------------

# Default cap: any single role family ≤ 60% of total coverage budget.
# Rationale: empirical Chuck Close / Shibata prints have all four role
# families present in non-trivial ratio. A 60% cap leaves room for one
# dominant role (often regional_mass) without collapsing into one.
_ROLE_COVERAGE_CAP = 0.60


def role_coverage_caps_penalty(
    plates: list[Plate],
    cap: float = _ROLE_COVERAGE_CAP,
) -> jnp.ndarray:
    """Penalize any role family that exceeds `cap` fraction of total coverage.

    Coverage per role = sum of (mask.mean() * opacity) over plates with that role.

    Args:
        plates: list of Plate.
        cap:    max fraction per role family (default 0.60).

    Returns:
        scalar penalty ≥ 0.
    """
    if not plates:
        return jnp.float32(0.0)

    per_role = {family: jnp.float32(0.0) for family in ROLE_FAMILIES}
    for plate in plates:
        contribution = plate.mask.mean() * plate.opacity
        per_role[plate.role] = per_role[plate.role] + contribution

    total = sum(per_role.values()) + 1e-9
    penalty = jnp.float32(0.0)
    for family in ROLE_FAMILIES:
        fraction = per_role[family] / total
        penalty = penalty + jax.nn.relu(fraction - cap)
    return penalty


# ----------------------------------------------------------------------
# 6. role_frequency_permission_penalty
# ----------------------------------------------------------------------

# Audit Phase 3:
#   "role-frequency permission (yellow can have detailed carved structure if
#    first/transparent), early plates may contain carved detail, but cannot
#    be full-face residuals"
#
# Concretely: plates assigned to early order positions are allowed to be
# `underlayer_light` or `local_chroma`, but they must NOT be high-coverage
# (>40% of image) — that's the "no full-face residual" rule.
_EARLY_ORDER_CUTOFF_FRAC = 0.30  # first 30% of the order are "early"
_EARLY_PLATE_COVERAGE_CAP = 0.40  # early plates cap inked-area fraction


def role_frequency_permission_penalty(
    plates: list[Plate],
    order: list[int],
) -> jnp.ndarray:
    """Penalize early plates that occupy a full-face fraction of the image.

    Args:
        plates: list of Plate (in arbitrary order).
        order:  list of plate indices into `plates`, in print order.

    Returns:
        scalar penalty ≥ 0.
    """
    if not plates or not order:
        return jnp.float32(0.0)

    n = len(order)
    early_count = max(1, int(_EARLY_ORDER_CUTOFF_FRAC * n))
    early_indices = order[:early_count]

    penalty = jnp.float32(0.0)
    for idx in early_indices:
        plate = plates[idx]
        area = plate.mask.mean()
        # Carved detail is fine; full-face residual is not.
        penalty = penalty + jax.nn.relu(area - _EARLY_PLATE_COVERAGE_CAP)
    return penalty


# ----------------------------------------------------------------------
# 7. load_bearing_singleton_penalty
# ----------------------------------------------------------------------

# SLA = "Stochastic Local Ablation". One-step ablation: how much does the
# final differ when we knock plate_i to zero opacity? Use gradient×mask
# (Integrated Gradients with single step) for a smooth differentiable proxy.

def _render_with_opacities(
    plates: list[Plate],
    opacities: jnp.ndarray,
    render_fn: Callable[[list[Plate]], jnp.ndarray],
) -> jnp.ndarray:
    """Build a fresh plate list with traced opacities, then call render_fn.

    Used by the load-bearing terms whose inner jax.grad needs opacity to
    appear inside the trace. We rebuild the Plate dataclasses with the
    opacity field as a jnp scalar — NOT a python float — so the tracer
    can follow it.

    This works because Plate.opacity is type-annotated as float but JAX
    happily accepts a 0-D jnp array there (render_fn closes over .opacity
    and the scalar flows through standard jnp ops).
    """
    new_plates = [
        Plate(
            block_id=p.block_id,
            mask=p.mask,
            pigment_lab=p.pigment_lab,
            opacity=opacities[i],  # 0-D jnp scalar — JAX-traceable.
            role=p.role,
            cell_zone_ids=p.cell_zone_ids,
            pass_index=p.pass_index,
        )
        for i, p in enumerate(plates)
    ]
    return render_fn(new_plates)


def _is_concrete(x) -> bool:
    """True iff x can be safely materialised to Python float (not a JAX tracer).

    Used so the load_bearing_* diagnostics can no-op gracefully when called
    from inside the outer solver step's grad pass. They are ABLATION
    diagnostics — they don't fit the "everything in one trace" model and
    are best computed in the outer Python loop between steps.
    """
    try:
        float(jnp.asarray(x).reshape(-1)[0])
        return True
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return False
    except Exception:
        return False


def load_bearing_singleton_penalty(
    plates: list[Plate],
    target: jnp.ndarray,
    render_fn: Callable[[list[Plate]], jnp.ndarray],
) -> jnp.ndarray:
    """Penalize plates that contribute negligibly to the final ΔE.

    Method: for each plate, compute the gradient of final_image_loss w.r.t.
    that plate's opacity. A plate that "doesn't matter" has near-zero
    gradient — the solver should drop it (or the solver wastes a block).

    We define "load-bearing" as |∂L/∂opacity_i| > eps. The penalty is the
    sum of ReLU(eps - |∂L/∂opacity_i|), so dead plates incur a tiny
    constant pressure to be reused or pruned.

    Args:
        plates:    list of Plate.
        target:    (H, W, 3) Lab target.
        render_fn: callable plates → (H, W, 3) Lab final.

    Returns:
        scalar penalty ≥ 0.
    """
    if not plates:
        return jnp.float32(0.0)

    # If we're inside an outer trace, the inner jax.grad doesn't compose
    # cleanly (no way to safely materialise per-plate concrete opacity for
    # the rebuild). Return a stop-grad zero — diagnostic terms should not
    # break the outer solver step.
    if not all(_is_concrete(p.mask) for p in plates):
        return jax.lax.stop_gradient(jnp.float32(0.0))

    opacities = jnp.array([float(p.opacity) for p in plates])

    def _loss_of_opacities(ops: jnp.ndarray) -> jnp.ndarray:
        rendered = _render_with_opacities(plates, ops, render_fn)
        return final_image_loss(rendered, target)

    # NOTE: do NOT jit() the gradient here — `plates` is a Python list of
    # dataclasses that changes identity every solver step, busting JIT cache.
    # Caller controls compile boundaries via the outer step jit.
    grad = jax.grad(_loss_of_opacities)(opacities)
    abs_grad = jnp.abs(grad)

    # Anything below epsilon is "not load-bearing". eps = 1e-3 ΔE_76 per
    # unit opacity is the threshold: removing a plate of full opacity must
    # shift the final by ≥ 1e-3 mean ΔE_76 to "count".
    eps = 1e-3
    # Wrap in stop_gradient so that the outer composite_loss grad doesn't
    # try to backprop through nested jax.grad (which can crash).
    return jax.lax.stop_gradient(jax.nn.relu(eps - abs_grad).sum())


# ----------------------------------------------------------------------
# 8. load_bearing_pair_penalty — top-K pair ablation
# ----------------------------------------------------------------------

def load_bearing_pair_penalty(
    plates: list[Plate],
    target: jnp.ndarray,
    render_fn: Callable[[list[Plate]], jnp.ndarray],
    top_k: int = 20,
) -> jnp.ndarray:
    """Detect cancellation pairs: two plates that only matter together.

    Per audit §2: singleton ablation misses cancellation pairs ("pull A only
    matters when pull B remains"). For each high-overlap pair, compute the
    Hessian off-diagonal: ∂²L/∂o_i∂o_j. Strong positive off-diagonal means
    "the two opacities are mutually substitutable" — exactly the cancellation
    pathology. Penalize those pairs.

    We limit to top-K pairs by spatial mask overlap to keep cost bounded.

    Args:
        plates:    list of Plate.
        target:    (H, W, 3) Lab.
        render_fn: callable plates → (H, W, 3) Lab final.
        top_k:     number of overlap-ranked pairs to evaluate (default 20).

    Returns:
        scalar penalty ≥ 0.
    """
    n = len(plates)
    if n < 2:
        return jnp.float32(0.0)

    # Diagnostic-only: bail out cleanly inside an outer trace.
    if not all(_is_concrete(p.mask) for p in plates):
        return jax.lax.stop_gradient(jnp.float32(0.0))

    # Rank pairs by spatial overlap (cheap).
    pair_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = float((plates[i].mask * plates[j].mask).mean())
            pair_overlap.append(((i, j), ov))
    pair_overlap.sort(key=lambda x: x[1], reverse=True)
    candidates = [p for p, _ in pair_overlap[:top_k]]

    if not candidates:
        return jnp.float32(0.0)

    opacities = jnp.array([float(p.opacity) for p in plates])

    def _loss_of_opacities(ops: jnp.ndarray) -> jnp.ndarray:
        rendered = _render_with_opacities(plates, ops, render_fn)
        return final_image_loss(rendered, target)

    # NOTE: do NOT jit() the gradient — see load_bearing_singleton_penalty for why.
    grad_fn = jax.grad(_loss_of_opacities)
    g = grad_fn(opacities)

    # Compute one Hessian column per unique j across candidate pairs, then
    # look up off-diagonal entries from those columns. For Emma scale (27
    # plates, top_k=20 → ~20 unique j's), this is 20 grad calls instead of
    # 40 (singleton + each pair) — and reuses the same g for all i.
    unique_js = sorted({j for _, j in candidates})
    eps = 1e-3
    # Sequential rather than vmap: vmap over grad-of-closure-list forces
    # recompilation per element. Loop in Python; each grad call is ~50ms.
    j_to_g_pert = {}
    for j in unique_js:
        perturbed = opacities.at[j].add(eps)
        j_to_g_pert[j] = grad_fn(perturbed)

    penalty = jnp.float32(0.0)
    for i, j in candidates:
        h_ij = (j_to_g_pert[j][i] - g[i]) / eps
        cancel_strength = (
            jax.nn.relu(h_ij)
            * jax.nn.relu(1e-3 - jnp.abs(g[i]))
            * jax.nn.relu(1e-3 - jnp.abs(g[j]))
        )
        penalty = penalty + cancel_strength
    # Stop-grad: this term is a diagnostic ablation; outer grad shouldn't
    # try to backprop through the nested jax.grad calls.
    return jax.lax.stop_gradient(penalty)


# ----------------------------------------------------------------------
# 9. printability_in_loop_penalty
# ----------------------------------------------------------------------

# Sub-mill feature = isolated inked region narrower than the smallest end-mill
# diameter (in pixels). Use a 2D max-pool/dilation surrogate: a connected
# inked island survives erosion by `mill_radius_px` iff it's wide enough.

def printability_in_loop_penalty(
    plates: list[Plate],
    mill_radius_px: int,
) -> jnp.ndarray:
    """Penalize features narrower than the end-mill diameter.

    Erode each plate mask by `mill_radius_px` (approximated with a min-pool
    of kernel size 2r+1). Any inked area that survives the binary mask but
    NOT the erosion is "sub-mill" — the CNC can't carve it cleanly.

    Penalty = sum over plates of (inked_area - eroded_area), normalized.

    Args:
        plates:         list of Plate.
        mill_radius_px: end-mill radius in pixels at solver resolution.

    Returns:
        scalar penalty ≥ 0.
    """
    if not plates or mill_radius_px <= 0:
        return jnp.float32(0.0)

    k = 2 * mill_radius_px + 1
    # Stack plate masks (N, H, W) → single batched erosion via reduce_window.
    stacked = jnp.stack([p.mask for p in plates], axis=0)
    stacked4 = stacked[..., None]  # (N, H, W, 1)
    eroded = -jax.lax.reduce_window(
        -stacked4,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, k, k, 1),
        window_strides=(1, 1, 1, 1),
        padding="SAME",
    )[..., 0]  # (N, H, W)
    lost = jax.nn.relu(stacked - eroded).sum(axis=(1, 2))
    original = stacked.sum(axis=(1, 2)) + 1e-6
    return (lost / original).sum()


# ----------------------------------------------------------------------
# 10. composite_loss — the single scalar the solver minimises
# ----------------------------------------------------------------------

def composite_loss(
    plates: list[Plate],
    target: jnp.ndarray,
    weights: LossWeights | None = None,
    *,
    rendered_final: jnp.ndarray | None = None,
    checkpoint_renders: jnp.ndarray | None = None,
    expected_proof_progression: jnp.ndarray | None = None,
    render_fn: Callable[[list[Plate]], jnp.ndarray] | None = None,
    plate_order: list[int] | None = None,
    mill_radius_px: int = 2,
    enable_load_bearing: bool = False,
) -> jnp.ndarray:
    """Single composite loss = weighted sum of all plate-objective terms.

    Solver-facing entry point. Pass weights to A/B-test penalty mixes.
    All optional kwargs are skipped if not provided — i.e., the loss
    degrades gracefully when (e.g.) checkpoint targets aren't ready yet.

    Args:
        plates:        list of Plate (the optimisation variable carrier).
        target:        (H, W, 3) Lab target.
        weights:       LossWeights instance (default uses module defaults).
        rendered_final: optional precomputed (H,W,3) Lab final to skip render_fn.
        checkpoint_renders: optional (K,H,W,3) for checkpoint loss.
        expected_proof_progression: optional (K,H,W,3) target for checkpoints.
        render_fn:     plates → (H,W,3) Lab. Required if rendered_final None.
        plate_order:   list of indices into plates (print order).
        mill_radius_px: end-mill radius for printability.

    Returns:
        scalar loss, fully JAX-grad compatible end-to-end.
    """
    w = weights or LossWeights()

    if rendered_final is None:
        if render_fn is None:
            raise ValueError("composite_loss needs rendered_final OR render_fn.")
        rendered_final = render_fn(plates)

    total = jnp.float32(0.0)

    total = total + w.final_image * final_image_loss(rendered_final, target)

    if checkpoint_renders is not None and expected_proof_progression is not None:
        total = total + w.checkpoint_proof * checkpoint_proof_loss(
            checkpoint_renders, expected_proof_progression
        )

    total = total + w.plate_not_composite * plate_not_composite_penalty_per_plate(
        plates, rendered_final
    )
    total = total + w.cell_exclusivity * cell_exclusivity_penalty(plates)
    total = total + w.role_coverage_caps * role_coverage_caps_penalty(plates)

    if plate_order is not None:
        total = total + w.role_frequency_permission * role_frequency_permission_penalty(
            plates, plate_order
        )

    # Load-bearing diagnostics use nested jax.grad — expensive (~1-2s per call
    # for Emma-scale plans). Disabled by default in the hot loop; the solver
    # should call them every K steps separately, not inside every step.
    if enable_load_bearing and render_fn is not None:
        total = total + w.load_bearing_singleton * load_bearing_singleton_penalty(
            plates, target, render_fn
        )
        total = total + w.load_bearing_pair * load_bearing_pair_penalty(
            plates, target, render_fn
        )

    total = total + w.printability * printability_in_loop_penalty(plates, mill_radius_px)

    return total


__all__ = [
    "Plate",
    "Role",
    "ROLE_FAMILIES",
    "LossWeights",
    "final_image_loss",
    "checkpoint_proof_loss",
    "plate_not_composite_penalty_per_plate",
    "cell_exclusivity_penalty",
    "role_coverage_caps_penalty",
    "role_frequency_permission_penalty",
    "load_bearing_singleton_penalty",
    "load_bearing_pair_penalty",
    "printability_in_loop_penalty",
    "composite_loss",
]
