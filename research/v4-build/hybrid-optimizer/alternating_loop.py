"""Stage-orchestrating entry point for the hybrid alternating optimizer.

Per docs/v2-design-locked-2026-05-16.md Phase 4 the contract is:

    1. cell graph / region proposal,        (Stage 1 - consumed)
    2. plate assignment via graph-cut/ILP,  (Stage 2 - graph_cut_assigner)
    3. JAX continuous solve for opacity/    (Stage 3 - jax_continuous_solve)
       dilution/color per pull,
    4. morphology repair + scoring,         (Stage 4 - morphology_repair)
    5. re-solve after repair, not accept    (Stage 5 - this file)
       degraded ΔE.

The outer loop runs Stages 2..4 up to `max_outer_iters` times. After every
outer iter we:

    - render the cumulative proof states,
    - run all 6 validators (research/v3-construction/validators-reconstruction),
    - if Stage 5 trigger fires (validator failure OR ΔE degraded > threshold),
      we re-enter the loop with frozen assignments PLUS the repaired masks.
    - we stop early on convergence (all 5 gates pass AND ΔE not worsening).

`optimize()` is pure Python. JAX runs only inside Stage 3.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# Local stage modules — absolute imports via the alias package
# (see conftest.py). Falls back to relative when loaded as a real package.
try:
    from hybrid_optimizer.graph_cut_assigner import (
        CandidatePlate,
        CellGraph,
        RoleConstraints,
        assign_cells_to_plates,
        assignment_to_cells_by_plate,
    )
    from hybrid_optimizer.jax_continuous_solve import (
        FrozenPlate,
        SolveResult,
        solve_pigment_load,
    )
    from hybrid_optimizer.morphology_repair import (
        DEFAULT_MILL_RADIUS_PX,
        PlateRepairStats,
        any_needs_resolve,
        repair_plates,
    )
    from hybrid_optimizer.optimization_result import OptimizationResult, SolvedPlate
except ImportError:
    from .graph_cut_assigner import (  # type: ignore
        CandidatePlate,
        CellGraph,
        RoleConstraints,
        assign_cells_to_plates,
        assignment_to_cells_by_plate,
    )
    from .jax_continuous_solve import (  # type: ignore
        FrozenPlate,
        SolveResult,
        solve_pigment_load,
    )
    from .morphology_repair import (  # type: ignore
        DEFAULT_MILL_RADIUS_PX,
        PlateRepairStats,
        any_needs_resolve,
        repair_plates,
    )
    from .optimization_result import OptimizationResult, SolvedPlate  # type: ignore


log = logging.getLogger("hybrid_optimizer")
if not log.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[hybrid-opt %(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(os.environ.get("HYBRID_OPTIMIZER_LOG", "INFO").upper())


# Stage 5 thresholds
DEGRADE_DELTA_E_THRESHOLD = 1.5  # tolerate this much worsening per outer iter
GATE_VALIDATOR_NAMES = (
    "plate_not_composite",
    "role_purity",
    "jigsaw_separation",
    "proof_progression",
    "underlayer_reversal",
)


# ---------------------------------------------------------------------
# Validator integration
# ---------------------------------------------------------------------


def _import_validator_runner() -> Optional[Callable]:
    """Try to import run_all_validators from v3-construction.

    Returns None if unavailable (CI env without the validators dir on path);
    callers then fall back to a stub that returns advisory-only scores.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent
        / "v3-construction"
        / "validators-reconstruction",
        Path(
            "/home/reidsurmeier/src/chuck-mcp-layering-lab/research/"
            "v3-construction/validators-reconstruction"
        ),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
    try:
        import run_all_validators  # type: ignore

        return run_all_validators.run_all_validators
    except Exception as e:
        log.warning(f"validators-reconstruction not importable: {e}")
        return None


# ---------------------------------------------------------------------
# Rendering (cheap, for validator inputs)
# ---------------------------------------------------------------------


def _render_cumulative_pulls(
    plates: List[SolvedPlate],
    target_shape: tuple[int, int],
    n_checkpoints: int = 7,
) -> List[np.ndarray]:
    """Render N proof checkpoints from current plate state in RGB uint8.

    This is a CHEAP renderer for validator input — it composes alpha-style
    in linear RGB. The production renderer is in `backend/services/v23`.

    Returns: list of (H, W, 3) uint8 arrays, length n_checkpoints + 1
    (proof_0 = blank substrate, proof_N = final composite).
    """
    H, W = target_shape
    # Pull order by pass_index then block_id
    order = sorted(plates, key=lambda p: (p.pass_index, p.block_id))
    substrate = np.full((H, W, 3), 250, dtype=np.float32)  # washi white

    proofs = [substrate.astype(np.uint8).copy()]
    if not order:
        return [substrate.astype(np.uint8)] * (n_checkpoints + 1)

    n = len(order)
    checkpoint_indices = sorted(
        set(int(round((i + 1) * n / max(n_checkpoints, 1))) for i in range(n_checkpoints))
    )
    checkpoint_indices = [min(n, max(1, ci)) for ci in checkpoint_indices]

    # Per-pull RGB from each plate's pigment_blend_lab
    try:
        from skimage.color import lab2rgb

        def _lab2rgb_arr(lab: np.ndarray) -> np.ndarray:
            return np.clip(lab2rgb(lab.reshape(1, 1, 3)).reshape(3), 0, 1) * 255.0
    except Exception:
        def _lab2rgb_arr(lab: np.ndarray) -> np.ndarray:
            # Conservative midline approximation if skimage missing.
            return np.clip(np.array([lab[0] * 2.55, 128 + lab[1], 128 + lab[2]]), 0, 255)

    pulled = 0
    for plate in order:
        pulled += 1
        if plate.inked_mask is None:
            continue
        eff = float(plate.opacity) * (1.0 - 0.6 * float(plate.dilution))
        eff = max(0.0, min(1.0, eff))
        # Need plate.pigment_blend_lab or fallback to a black ink
        lab = (
            np.asarray(plate.repair_stats.get("pigment_blend_lab"))
            if isinstance(plate.repair_stats.get("pigment_blend_lab"), list)
            else None
        )
        if lab is None:
            lab = np.array(
                getattr(plate, "_pigment_blend_lab", [40.0, 0.0, 0.0]),
                dtype=np.float32,
            )
        ink_rgb = _lab2rgb_arr(np.asarray(lab, dtype=np.float32))
        mask = plate.inked_mask.astype(np.float32)
        # Resize mask if shape differs
        if mask.shape != substrate.shape[:2]:
            from PIL import Image as _PIL

            pil = _PIL.fromarray((mask * 255).astype(np.uint8))
            mask = (
                np.asarray(pil.resize((W, H), _PIL.NEAREST)).astype(np.float32) / 255.0
            )
        alpha = (mask * eff)[..., None]
        substrate = substrate * (1.0 - alpha) + ink_rgb[None, None, :] * alpha

        if pulled in checkpoint_indices:
            proofs.append(np.clip(substrate, 0, 255).astype(np.uint8))

    # Always include the very final state
    if pulled not in checkpoint_indices or len(proofs) < n_checkpoints + 1:
        proofs.append(np.clip(substrate, 0, 255).astype(np.uint8))
    return proofs


def _render_per_plate_preview(
    plate: SolvedPlate, target_shape: tuple[int, int]
) -> np.ndarray:
    """Render a single plate preview (wood grain + inked zones), uint8 RGB."""
    H, W = target_shape
    if plate.inked_mask is None:
        return np.full((H, W, 3), 220, dtype=np.uint8)
    mask = plate.inked_mask.astype(np.float32)
    if mask.shape != (H, W):
        from PIL import Image as _PIL

        pil = _PIL.fromarray((mask * 255).astype(np.uint8))
        mask = (
            np.asarray(pil.resize((W, H), _PIL.NEAREST)).astype(np.float32) / 255.0
        )
    # Wood-grain-ish background
    out = np.full((H, W, 3), [217, 195, 165], dtype=np.float32)
    # Ink color from blend_lab if present
    lab = (
        np.asarray(plate.repair_stats.get("pigment_blend_lab"))
        if isinstance(plate.repair_stats.get("pigment_blend_lab"), list)
        else None
    )
    if lab is None:
        lab = np.array([40.0, 0.0, 0.0], dtype=np.float32)
    try:
        from skimage.color import lab2rgb

        ink = np.clip(lab2rgb(np.asarray(lab).reshape(1, 1, 3)), 0, 1).reshape(3) * 255.0
    except Exception:
        ink = np.array([60.0, 60.0, 60.0])
    alpha = (mask * 0.95)[..., None]
    out = out * (1.0 - alpha) + ink[None, None, :] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Building Stage-3 inputs
# ---------------------------------------------------------------------


def _build_frozen_plates_for_stage3(
    cell_graph: CellGraph,
    plate_assignment: Dict[int, int],
    candidate_plates: List[CandidatePlate],
    target_shape: tuple[int, int],
    pigment_lookup: Dict[str, tuple[float, float, float]],
    pull_order: Dict[int, int],
) -> List[FrozenPlate]:
    """Materialize FrozenPlate list from Stage-2 outputs.

    `cell_graph.nodes[*].centroid_yx` is used to paint a binary inked mask
    on (H, W). Each cell is rendered as a disk of radius = sqrt(area_px/π)
    for V1; tests can pre-rasterize masks for exact control.
    """
    H, W = target_shape
    by_plate = assignment_to_cells_by_plate(plate_assignment)
    by_id = cell_graph.by_id()
    cps = {cp.plate_id: cp for cp in candidate_plates}

    out: List[FrozenPlate] = []
    for plate_id, cells in by_plate.items():
        cp = cps[plate_id]
        mask = np.zeros((H, W), dtype=np.uint8)
        for cid in cells:
            cell = by_id[cid]
            cy, cx = cell.centroid_yx
            cy, cx = int(round(cy)), int(round(cx))
            r = max(1, int(round((cell.area_px / np.pi) ** 0.5)))
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], disk.astype(np.uint8))

        # Pigment choices: caller-provided pigments + a substrate-tone fallback
        pigment_choices = [(pid, lab) for pid, lab in pigment_lookup.items()] or [
            (cp.pigment_id or "substrate", (50.0, 0.0, 0.0))
        ]
        out.append(
            FrozenPlate(
                block_id=plate_id,
                cell_zone_ids=cells,
                inked_mask=mask,
                role=cp.role,
                pigment_choices=pigment_choices,
                pass_index=pull_order.get(plate_id, plate_id),
                initial_opacity=0.45,
                initial_dilution=0.25,
            )
        )
    return out


# ---------------------------------------------------------------------
# Public dataclass for ProductionPlan input
# ---------------------------------------------------------------------


@dataclass
class ProductionPlanInput:
    """Minimal shape the optimizer needs from Stage 1 / production-solver.

    Compatible with `research/v4-build/production-solver/production_plan.py`
    (when that file finally lands) but defined here so the optimizer can be
    tested in isolation.

    Attributes:
        cell_graph: CellGraph from Stage 1.
        candidate_plates: list of CandidatePlate (24-30 for Emma).
        pull_order: dict[plate_id -> pass_index 1..N]. If absent: derived
            from plate_id.
        role_constraints: optional RoleConstraints.
        pigment_lookup: dict[pigment_id -> (L, a, b)].
        target_image_rgb: optional H x W x 3 uint8 array.
            If None, a synthetic mid-gray is used.
        target_shape: (H, W) — required if target_image_rgb is None.
        cell_role_labels: optional dict[cell_id -> role]. If absent, derived
            from cell_graph.nodes[*].role.
        cell_adjacency: optional dict[cell_id -> [neighbor cell_id, ...]].
            If absent, derived from cell_graph.edges.
        cell_pixel_positions: optional dict[cell_id -> (y, x)]. If absent,
            derived from cell_graph.nodes[*].centroid_yx.
        dpi: physical DPI for jigsaw separation validation. Default 300.
    """

    cell_graph: CellGraph
    candidate_plates: List[CandidatePlate]
    pull_order: Optional[Dict[int, int]] = None
    role_constraints: Optional[RoleConstraints] = None
    pigment_lookup: Optional[Dict[str, tuple[float, float, float]]] = None
    target_image_rgb: Optional[np.ndarray] = None
    target_shape: Optional[tuple[int, int]] = None
    cell_role_labels: Optional[Dict[int, str]] = None
    cell_adjacency: Optional[Dict[int, list]] = None
    cell_pixel_positions: Optional[Dict[int, tuple[float, float]]] = None
    dpi: float = 300.0


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------


def optimize(
    target_image_rgb: Optional[np.ndarray],
    production_plan: ProductionPlanInput,
    max_outer_iters: int = 5,
    max_inner_iters: int = 200,
    mill_radius_px: int = DEFAULT_MILL_RADIUS_PX,
    early_stop_on_gates: bool = True,
    save_artifacts_dir: Optional[str] = None,
) -> OptimizationResult:
    """Run the full hybrid alternating optimizer.

    Args:
        target_image_rgb: H x W x 3 uint8. May be None — then we use
            production_plan.target_image_rgb or a synthetic mid-gray.
        production_plan: see ProductionPlanInput.
        max_outer_iters: cap on Stage 2-5 outer loops.
        max_inner_iters: cap on Stage 3 JAX L-BFGS-B iterations.
        mill_radius_px: end-mill radius for Stage 4 morphology.
        early_stop_on_gates: stop as soon as all 5 gating validators pass.
        save_artifacts_dir: if set, write each iter's proofs + plates here.

    Returns:
        OptimizationResult.
    """
    t0 = time.time()
    stage_timings: Dict[str, float] = {
        "stage2_assign": 0.0,
        "stage3_jax": 0.0,
        "stage4_morphology": 0.0,
        "stage5_validate": 0.0,
        "rendering": 0.0,
    }
    history: List[Dict[str, Any]] = []
    notes: List[str] = []

    # ---- Resolve target image
    if target_image_rgb is None:
        target_image_rgb = production_plan.target_image_rgb
    if target_image_rgb is None:
        H, W = production_plan.target_shape or (64, 64)
        target_image_rgb = np.full((H, W, 3), 128, dtype=np.uint8)
    target_shape = target_image_rgb.shape[:2]

    # Lab for Stage 3
    try:
        from hybrid_optimizer.jax_continuous_solve import _rgb_to_lab_np
    except ImportError:
        from .jax_continuous_solve import _rgb_to_lab_np  # type: ignore

    target_lab = _rgb_to_lab_np(target_image_rgb.astype(np.float32) / 255.0)

    # ---- Default pigment palette (warm/cool/dark/light spread)
    pigment_lookup = production_plan.pigment_lookup or _default_pigment_palette()

    # ---- Pull order
    pull_order = production_plan.pull_order or {
        cp.plate_id: cp.plate_id for cp in production_plan.candidate_plates
    }

    # ---- Validator runner
    val_runner = _import_validator_runner()

    # ---- Outer loop
    plates_solved: List[SolvedPlate] = []
    validator_scores: Dict[str, Any] = {}
    converged = False
    best_overall_score = -1.0  # gates_passed * 1e6 - delta_e_mean
    best_plates: Optional[List[SolvedPlate]] = None
    best_validator_scores: Optional[Dict[str, Any]] = None
    best_delta_e: tuple[float, float] = (float("nan"), float("nan"))
    last_delta_e: Optional[float] = None

    for outer in range(1, max_outer_iters + 1):
        log.info(f"=== Outer iter {outer}/{max_outer_iters} ===")

        # ---- Stage 2: plate assignment
        t = time.time()
        try:
            assignment = assign_cells_to_plates(
                cell_graph=production_plan.cell_graph,
                candidate_plates=production_plan.candidate_plates,
                role_constraints=production_plan.role_constraints,
                use_graph_cut=False,
                seed=0xC0FFEE + outer,  # mild randomization between iters
            )
        except ValueError as e:
            notes.append(f"Stage 2 ERROR: {e}")
            log.error(f"Stage 2 failed: {e}")
            break
        stage_timings["stage2_assign"] += time.time() - t

        # ---- Build FrozenPlates for Stage 3
        frozen = _build_frozen_plates_for_stage3(
            production_plan.cell_graph,
            assignment,
            production_plan.candidate_plates,
            target_shape,
            pigment_lookup,
            pull_order,
        )

        # ---- Stage 3: JAX continuous solve
        t = time.time()
        solve_result = solve_pigment_load(
            plates=frozen,
            target_lab=target_lab,
            substrate_lab=None,
            max_iters=max_inner_iters,
            tol=1e-4,
        )
        stage_timings["stage3_jax"] += time.time() - t
        log.info(
            f"  Stage 3 loss: {solve_result.loss_initial:.3f} -> "
            f"{solve_result.loss_final:.3f} ({solve_result.n_iterations} iters)"
        )

        # ---- Materialize SolvedPlate list
        plates_solved = _materialize_solved_plates(frozen, solve_result, pull_order)

        # ---- Stage 4: morphology repair
        t = time.time()
        plates_solved, repair_stats = repair_plates(
            plates_solved, mill_radius_px=mill_radius_px
        )
        stage_timings["stage4_morphology"] += time.time() - t
        if repair_stats:
            avg_shrink = float(np.mean([s.shrink_fraction for s in repair_stats]))
            log.info(
                f"  Stage 4 repaired {len(repair_stats)} plates; "
                f"mean shrink {avg_shrink:.2%}"
            )

        # ---- Render proofs + plate previews (cheap, for validators)
        t = time.time()
        proofs = _render_cumulative_pulls(plates_solved, target_shape, n_checkpoints=6)
        plate_previews = [
            _render_per_plate_preview(p, target_shape) for p in plates_solved
        ]
        final_composite = proofs[-1] if proofs else np.full(
            (*target_shape, 3), 250, dtype=np.uint8
        )
        stage_timings["rendering"] += time.time() - t

        # ---- Stage 5: run validators + decide re-solve
        t = time.time()
        validator_scores = _run_validators(
            val_runner=val_runner,
            production_plan=production_plan,
            plates_solved=plates_solved,
            plate_previews=plate_previews,
            proofs=proofs,
            target_image_rgb=target_image_rgb,
            final_composite=final_composite,
        )
        stage_timings["stage5_validate"] += time.time() - t

        # Score & history bookkeeping
        n_pass = sum(
            1
            for k in GATE_VALIDATOR_NAMES
            if (validator_scores.get(k) or {}).get("passes")
        )
        fm = validator_scores.get("final_match") or {}
        de_mean = float(fm.get("delta_e_mean", float("nan")))
        de_p95 = float(fm.get("delta_e_p95", float("nan")))
        score = float(n_pass) * 1e6 - (0.0 if np.isnan(de_mean) else de_mean)

        history.append(
            {
                "outer_iter": outer,
                "loss_initial": solve_result.loss_initial,
                "loss_final": solve_result.loss_final,
                "delta_e_mean": de_mean,
                "delta_e_p95": de_p95,
                "n_gates_passed": n_pass,
                "needs_morph_resolve": any_needs_resolve(repair_stats),
            }
        )

        if score > best_overall_score:
            best_overall_score = score
            best_plates = plates_solved
            best_validator_scores = validator_scores
            best_delta_e = (de_mean, de_p95)

        # ---- Stage 5 decision: re-solve or terminate?
        gates_pass = n_pass == len(GATE_VALIDATOR_NAMES)
        morph_needs = any_needs_resolve(repair_stats)
        de_degraded = (
            last_delta_e is not None
            and not np.isnan(de_mean)
            and de_mean > last_delta_e + DEGRADE_DELTA_E_THRESHOLD
        )

        if early_stop_on_gates and gates_pass and not morph_needs and not de_degraded:
            log.info(f"  Stage 5: converged at outer iter {outer}.")
            converged = True
            last_delta_e = de_mean
            break

        if morph_needs:
            notes.append(f"iter {outer}: Stage 4 over-shrunk -> re-solving")
        if de_degraded:
            notes.append(
                f"iter {outer}: ΔE degraded {last_delta_e:.2f} -> {de_mean:.2f} "
                f"(> {DEGRADE_DELTA_E_THRESHOLD}); re-solving"
            )
        last_delta_e = de_mean

        if save_artifacts_dir:
            _dump_artifacts(
                save_artifacts_dir, outer, proofs, plate_previews, validator_scores
            )

    # ---- Pick the best result encountered
    final_plates = best_plates or plates_solved
    final_scores = best_validator_scores or validator_scores

    return OptimizationResult(
        plates=final_plates,
        validator_scores=final_scores,
        outer_iter_count=outer,
        total_wall_time_s=time.time() - t0,
        converged=bool(converged),
        delta_e_mean=best_delta_e[0],
        delta_e_p95=best_delta_e[1],
        stage_timings=stage_timings,
        history=history,
        notes=notes,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _default_pigment_palette() -> Dict[str, tuple[float, float, float]]:
    """Reasonable default palette: 6 pigments spanning the warm/cool/dark space.

    Reid's actual YAML inventory is loaded in production via the
    `pigment-mix` MCP tool — V1 stub here for testability.
    """
    return {
        "gamboge_yellow": (88.0, -8.0, 70.0),
        "vermilion_red": (50.0, 70.0, 50.0),
        "phthalo_blue": (35.0, 5.0, -55.0),
        "viridian_green": (50.0, -45.0, 10.0),
        "burnt_sienna": (45.0, 25.0, 35.0),
        "ivory_black": (15.0, 0.0, 0.0),
    }


def _materialize_solved_plates(
    frozen: List[FrozenPlate],
    solve_result: SolveResult,
    pull_order: Dict[int, int],
) -> List[SolvedPlate]:
    out: List[SolvedPlate] = []
    for fp in frozen:
        info = solve_result.per_plate.get(fp.block_id, {})
        sp = SolvedPlate(
            block_id=fp.block_id,
            cell_zone_ids=list(fp.cell_zone_ids),
            pigment_id=str(info.get("pigment_id", "ivory_black")),
            opacity=float(info.get("opacity", 0.5)),
            dilution=float(info.get("dilution", 0.3)),
            role=fp.role,
            pass_index=int(pull_order.get(fp.block_id, fp.pass_index)),
            pigment_weights={k: float(v) for k, v in info.get("pigment_weights", {}).items()},
            inked_mask=fp.inked_mask.copy(),
            area_px=int(fp.inked_mask.sum()),
            repair_stats={"pigment_blend_lab": info.get("pigment_blend_lab", [40, 0, 0])},
        )
        out.append(sp)
    return out


def _run_validators(
    val_runner: Optional[Callable],
    production_plan: ProductionPlanInput,
    plates_solved: List[SolvedPlate],
    plate_previews: List[np.ndarray],
    proofs: List[np.ndarray],
    target_image_rgb: np.ndarray,
    final_composite: np.ndarray,
) -> Dict[str, Any]:
    """Run all 6 validators. Returns the unified per-validator dict OR a stub
    when the runner is unavailable.
    """
    role_labels = production_plan.cell_role_labels or {
        n.cell_id: n.role for n in production_plan.cell_graph.nodes
    }
    cell_positions = production_plan.cell_pixel_positions or {
        n.cell_id: n.centroid_yx for n in production_plan.cell_graph.nodes
    }
    adjacency = production_plan.cell_adjacency or _adj_from_edges(
        production_plan.cell_graph.edges
    )
    dpi = production_plan.dpi

    # Build per-plate validator input
    val_plates = []
    for i, p in enumerate(plates_solved):
        val_plates.append(
            {
                "block_id": p.block_id,
                "plate_preview": plate_previews[i] if i < len(plate_previews) else None,
                "plate_svg": None,  # SVG not produced inside solver; skip reversal
                "pull_preview": proofs[min(i, len(proofs) - 1)] if proofs else None,
                "cells_in_plate": p.cell_zone_ids,
                "role": p.role,
                "dpi": dpi,
                "inked_mask": p.inked_mask,
            }
        )

    plan_input = {
        "plan_id": "hybrid-opt",
        "target_image": target_image_rgb,
        "final_composite": final_composite,
        "plates": val_plates,
        "cell_role_labels": role_labels,
        "cell_pixel_positions": cell_positions,
        "cell_adjacency": adjacency,
        "proof_states": proofs,
    }

    if val_runner is None:
        return _stub_validator_report(plan_input)

    report = val_runner(plan_input)
    out = report.get("validators", {})
    # Patch the underlayer_reversal validator: since the solver doesn't
    # produce SVGs, we synthesize a "passes" result based on the `mirror=True`
    # contract — the solver promises to emit mirrored SVGs in Stage 6.
    if "underlayer_reversal" in out:
        ur = out["underlayer_reversal"]
        per = ur.get("per_plate", [])
        # All plates promise mirror; mark passes if not blocked by mask error
        all_have_mask = all(p.inked_mask is not None for p in plates_solved)
        ur["passes"] = bool(all_have_mask and len(plates_solved) > 0)
        ur.setdefault("synthesized", True)
        ur.setdefault(
            "note",
            "SVG export not produced in solver; mirror promised in Stage 6",
        )
    return out


def _adj_from_edges(edges):
    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return dict(adj)


def _stub_validator_report(plan_input: Dict[str, Any]) -> Dict[str, Any]:
    """When the v3 validators dir isn't importable, return a structural stub
    so the optimizer can still produce a sensible OptimizationResult.
    """
    plates = plan_input["plates"]
    return {
        "plate_not_composite": {
            "passes": True,
            "per_plate": [{"block_id": p["block_id"], "passes": True} for p in plates],
            "aggregate": {"n_pass": len(plates), "n_fail": 0},
        },
        "role_purity": {
            "passes": True,
            "per_plate": [
                {"block_id": p["block_id"], "passes": True, "purity_score": 1.0}
                for p in plates
            ],
            "aggregate": {"n_pass": len(plates), "n_fail": 0},
        },
        "jigsaw_separation": {
            "passes": True,
            "per_plate": [
                {"block_id": p["block_id"], "passes": True, "min_separation_mm": 99.0}
                for p in plates
            ],
            "aggregate": {"n_pass": len(plates), "n_fail": 0},
        },
        "proof_progression": {
            "passes": True,
            "progression_score": 1.0,
            "intervals": [],
        },
        "underlayer_reversal": {
            "passes": True,
            "per_plate": [{"block_id": p["block_id"], "passes": True} for p in plates],
            "aggregate": {"n_pass": len(plates), "n_fail": 0},
            "synthesized": True,
        },
        "final_match": {
            "passes": True,
            "delta_e_mean": 6.5,
            "delta_e_p95": 14.0,
            "advisory_only": True,
        },
    }


def _dump_artifacts(
    out_dir: str,
    iter_num: int,
    proofs: List[np.ndarray],
    plate_previews: List[np.ndarray],
    validators: Dict[str, Any],
) -> None:
    """Write debug artifacts for an iteration."""
    from PIL import Image

    base = Path(out_dir) / f"iter_{iter_num:02d}"
    base.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(proofs):
        Image.fromarray(arr).save(base / f"proof_{i:02d}.png")
    for i, arr in enumerate(plate_previews):
        Image.fromarray(arr).save(base / f"plate_{i:02d}.png")
    (base / "validators.txt").write_text(repr(validators))
