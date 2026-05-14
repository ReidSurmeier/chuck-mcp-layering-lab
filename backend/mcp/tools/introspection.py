"""D9B — Tier 3 introspection tools (read-only + pigment guidance)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import forward_render_jax


def get_pigments() -> ToolResult[dict[str, Any]]:
    """Return the Chuck layering-lab pigment and adaptive wash library."""
    pigments = []
    names = forward_render_jax.PIGMENT_NAMES
    for idx, name in enumerate(names):
        r, g, b = forward_render_jax.PIGMENT_RGB_255[idx].tolist()
        pigments.append({
            "pigment_id": name,
            "name": name.replace("_", " ").title(),
            "rgb": [int(r), int(g), int(b)],
            "hex": f"#{r:02x}{g:02x}{b:02x}",
        })
    return ToolResult(ok=True, data={"pigments": pigments, "count": len(pigments),
                                     "catalog": "chuck_layering_lab_flexible"})


def get_emma_priors() -> ToolResult[dict[str, Any]]:
    """Return Emma-derived priors (hue families, accent rule, keyblock rule)."""
    return ToolResult(ok=True, data={
        "hue_families": ["cream", "cool", "flesh", "warm", "shadow", "detail", "accent"],
        "default_M_prior": 6,
        "M_prior_range": [4, 12],
        "accent_block_rule": "is_accent_block if pigment_count > 4 OR family_count > 2",
        "underprint_area_ratio_range": [1.5, 4.0],
        "keyblock_rule": "darkest detail impression gets highest order_step",
        "pigments_per_block_target": {"mean": 3.0, "cap": 5, "floor": 1},
        "production_reference_scale": {
            "woodblocks": 27,
            "colors": 113,
            "pulls": 132,
            "note": (
                "Use this as an Emma-scale complexity reference. Current S5 "
                "m_prior is a compressed study budget, not a production pull count."
            ),
        },
    })


def suggest_pigment_mix(
    target_hex: str,
    *,
    max_pigments: int = 3,
    candidate_limit: int = 5,
) -> ToolResult[dict[str, Any]]:
    """Suggest premix ratios for a target color using available Chuck pigments."""
    try:
        target = _parse_hex_color(target_hex)
    except ValueError as exc:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="INVALID_TARGET_COLOR",
                message=str(exc),
                hint="use a 6-digit hex color such as #c65a40",
                recoverable=True,
            )
        ])

    max_pigments = int(max(1, min(max_pigments, 3)))
    candidate_limit = int(max(1, min(candidate_limit, 12)))
    recipes = _mix_candidates(target, max_pigments=max_pigments, limit=candidate_limit)
    return ToolResult(ok=True, data={
        "target_hex": _rgb_to_hex(target),
        "target_rgb": [round(float(v), 4) for v in target.tolist()],
        "catalog": "chuck_layering_lab_flexible",
        "recipes": recipes,
        "guidance": (
            "Ratios are premix starting points by volume/weight. Make swatches on "
            "the same paper, adjust with water/paste for opacity, and prefer a "
            "separate plate when the color shift is regional or needs a crisp edge."
        ),
        "render_note": (
            "This is RGB premix guidance, not an overprint-glazing prediction. "
            "Use calibration swatches for final color decisions."
        ),
    })


def get_defaults() -> ToolResult[dict[str, Any]]:
    """Return locked technical defaults from research-v23-mcp-defaults.md."""
    return ToolResult(ok=True, data={
        "solve_profile_walltime_s": {"fast": 60, "default": 180, "thorough": 600},
        "dE2000_target": {"mean": 1.5, "p95": 3.0},
        "ambiguous_band": {"pixel_dE_gt": 3.0, "mask_mean_gt": 1.5},
        "block_count_target": 6,
        "block_count_cap": 10,
        "block_count_note": (
            "legacy study-stack target; Emma-scale production planning can require "
            "dozens of blocks and 100+ pulls"
        ),
        "min_island_px_at_300dpi": 60,
        "kento_dilation_px_at_300dpi": 3,
        "schema_version": "v23.0",
        "calibration_default": "chuck_layering_lab_24",
        "render_tier_default": "t1_mixbox",
    })


def _parse_hex_color(value: str) -> np.ndarray:
    raw = value.strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) != 6:
        raise ValueError(f"target_hex must have 6 hex digits, got {value!r}")
    try:
        rgb = [int(raw[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError as exc:
        raise ValueError(f"target_hex contains non-hex characters: {value!r}") from exc
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _rgb_to_hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.round(rgb * 255.0), 0, 255).astype(int).tolist()
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _simplex_weights(k: int, step: float = 0.05) -> list[np.ndarray]:
    if k == 1:
        return [np.asarray([1.0], dtype=np.float32)]
    units = int(round(1.0 / step))
    out: list[np.ndarray] = []
    if k == 2:
        for i in range(units + 1):
            out.append(np.asarray([i / units, 1.0 - i / units], dtype=np.float32))
        return out
    for i in range(units + 1):
        for j in range(units - i + 1):
            l = units - i - j
            out.append(np.asarray([i / units, j / units, l / units], dtype=np.float32))
    return out


def _mix_candidates(
    target_rgb: np.ndarray,
    *,
    max_pigments: int,
    limit: int,
) -> list[dict[str, Any]]:
    import itertools

    pigments = forward_render_jax.PIGMENT_TABLE.astype(np.float32)
    names = forward_render_jax.PIGMENT_NAMES
    scored: list[tuple[float, tuple[int, ...], np.ndarray, np.ndarray]] = []
    for k in range(1, max_pigments + 1):
        weights = _simplex_weights(k)
        for combo in itertools.combinations(range(len(names)), k):
            colors = pigments[list(combo)]
            best_err = float("inf")
            best_mix = colors[0]
            best_w = weights[0]
            for w in weights:
                mixed = np.sum(colors * w[:, None], axis=0)
                err = float(_delta_e76(tuple(target_rgb.tolist()), tuple(mixed.tolist())))
                # Slight bias toward simpler bench recipes when color match ties.
                err += 0.05 * (k - 1)
                if err < best_err:
                    best_err = err
                    best_mix = mixed
                    best_w = w
            scored.append((best_err, combo, best_w, best_mix))

    scored.sort(key=lambda row: row[0])
    recipes: list[dict[str, Any]] = []
    for err, combo, weights, mixed in scored[:limit]:
        parts = []
        for idx, weight in sorted(zip(combo, weights, strict=True), key=lambda x: -x[1]):
            if float(weight) <= 0.0:
                continue
            parts.append({
                "pigment_id": int(idx),
                "pigment_name": names[int(idx)],
                "ratio_pct": round(float(weight) * 100.0, 1),
            })
        recipes.append({
            "mixed_hex": _rgb_to_hex(mixed),
            "mixed_rgb": [round(float(v), 4) for v in mixed.tolist()],
            "delta_e76": round(float(err), 3),
            "parts": parts,
        })
    return recipes


def solver_telemetry(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Read solver run summary from a persisted plan."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "pyramid_levels_completed": 0,
                "iters_per_level": [],
                "loss_per_level": [],
                "rule_loss_breakdown": {},
                "exit_reason": "unknown_plan",
                "wall_time_s": 0.0,
                "divergence_flags": [],
                "optimized_shape": [],
                "downsample_scale": 1.0,
            },
            errors=[exc.error],
        )
    profile_iters = {"fast": 60, "default": 180, "thorough": 400}
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "pyramid_levels_completed": 1,  # single-level for day-1
            "iters_per_level": [profile_iters.get(plan.solve_profile, 180)],
            "loss_per_level": [],  # full per-iter trace lands at D14.b
            "rule_loss_breakdown": {},
            "exit_reason": plan.solver_status,
            "wall_time_s": plan.solver_wall_s,
            "divergence_flags": [],
            "solve_profile": plan.solve_profile,
            "optimized_shape": plan.solver_optimized_shape,
            "downsample_scale": plan.solver_downsample_scale,
            "impression_count": len(plan.impressions),
            "block_count": plan.block_count,
        },
    )


_PIGMENT_NAMES = forward_render_jax.PIGMENT_NAMES


def _delta_e76(rgb_a: tuple[float, float, float], rgb_b: tuple[float, float, float]) -> float:
    """Quick ΔE76 in CIE Lab. Cheap + monotonic with perceptual difference."""
    def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
        r, g, b = (max(0.0, min(1.0, c)) for c in rgb)
        def _linearise(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        r_l, g_l, b_l = _linearise(r), _linearise(g), _linearise(b)
        # sRGB -> XYZ (D65)
        x = 0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l
        y = 0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l
        z = 0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l
        # XYZ -> Lab (D65 ref white)
        xn, yn, zn = 0.95047, 1.0, 1.08883
        def _f(t: float) -> float:
            return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)
        fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_lab = 200.0 * (fy - fz)
        return L, a, b_lab
    la, aa, ba = _srgb_to_lab(rgb_a)
    lb, ab, bb = _srgb_to_lab(rgb_b)
    return float(((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2) ** 0.5)


def dE_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    """Per-pixel ΔE — render single pixel from persisted alpha_stack vs target."""
    if x < 0 or y < 0:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_COORDS",
                           message=f"x and y must be >= 0, got ({x}, {y})",
                           recoverable=True),
        ])

    from pathlib import Path

    import numpy as _np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if y >= plan.height or x >= plan.width:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="OUT_OF_BOUNDS",
                           message=f"({x}, {y}) outside plan bounds {plan.width}x{plan.height}",
                           recoverable=True),
        ])

    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "x": x, "y": y, "dE": None,
                  "target_rgb": None, "rendered_rgb": None},
            errors=[WoodblockError(
                tier="degraded", code="NO_SOLVER_OUTPUT",
                message="plan has no persisted alpha_stack — solver did not run",
                hint="re-run propose_stack with solver enabled",
                recoverable=True,
            )],
        )

    target_path = Path(plan.alpha_stack_path).parent / "target.npy"
    if not target_path.is_file():
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "x": x, "y": y, "dE": None,
                  "target_rgb": None, "rendered_rgb": None},
            errors=[WoodblockError(
                tier="degraded", code="NO_TARGET_CACHE",
                message="plan has no persisted target image — older plan?",
                recoverable=True,
            )],
        )

    alpha_stack = _np.load(plan.alpha_stack_path)  # (M, H, W)
    target = _np.load(target_path)                  # (H, W, 3)

    alpha_pixel = alpha_stack[:, y, x]  # (M,)
    pigment_idx = _np.asarray(plan.pigment_idx, dtype=_np.int32)

    import jax.numpy as jnp
    alpha_hwm = jnp.asarray(alpha_pixel[None, None, :], dtype=jnp.float32)  # (1, 1, M)
    rendered = forward_render_jax.forward_render(
        alpha_hwm, jnp.asarray(pigment_idx, dtype=jnp.int32),
    )
    rendered_rgb = tuple(float(c) for c in _np.asarray(rendered[0, 0]))
    target_rgb = tuple(float(c) for c in target[y, x])
    dE = _delta_e76(target_rgb, rendered_rgb)

    return ToolResult(ok=True, data={
        "plan_id": plan_id, "x": x, "y": y,
        "dE": round(dE, 3),
        "target_rgb": [round(c, 4) for c in target_rgb],
        "rendered_rgb": [round(c, 4) for c in rendered_rgb],
        "metric": "deltaE76",
    })


def pigment_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    """Per-pixel impression stack — load alpha_stack at pixel and emit per-impression rows."""
    if x < 0 or y < 0:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_COORDS",
                           message=f"x and y must be >= 0, got ({x}, {y})",
                           recoverable=True),
        ])

    from pathlib import Path

    import numpy as _np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if y >= plan.height or x >= plan.width:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="OUT_OF_BOUNDS",
                           message=f"({x}, {y}) outside plan bounds {plan.width}x{plan.height}",
                           recoverable=True),
        ])

    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "x": x, "y": y, "impressions": []},
            errors=[WoodblockError(
                tier="degraded", code="NO_SOLVER_OUTPUT",
                message="plan has no persisted alpha_stack — solver did not run",
                recoverable=True,
            )],
        )

    alpha_stack = _np.load(plan.alpha_stack_path)  # (M, H, W)
    alpha_pixel = alpha_stack[:, y, x]  # (M,)
    impressions_meta = plan.impressions

    rows = []
    for i, alpha in enumerate(alpha_pixel):
        a = float(alpha)
        if a < 0.01:
            continue  # paper-clear — skip noise
        pid = int(plan.pigment_idx[i]) if i < len(plan.pigment_idx) else None
        meta = impressions_meta[i] if i < len(impressions_meta) else {}
        rows.append({
            "order_step": int(meta.get("order_step", i)),
            "impression_id": meta.get("id", f"imp_{i:02d}"),
            "pigment_id": pid,
            "pigment_name": (
                _PIGMENT_NAMES[pid] if pid is not None and 0 <= pid < len(_PIGMENT_NAMES)
                else f"pigment_{pid}"
            ),
            "alpha": round(a, 4),
        })
    rows.sort(key=lambda r: r["order_step"])

    return ToolResult(ok=True, data={
        "plan_id": plan_id, "x": x, "y": y,
        "impressions": rows,
        "alpha_threshold": 0.01,
    })


def _load_cell_graph(plan: Any) -> tuple[dict[str, Any], np.ndarray] | tuple[None, None]:
    if not getattr(plan, "cell_graph_path", None) or not getattr(plan, "cell_labels_path", None):
        return None, None
    graph_path = Path(plan.cell_graph_path)
    labels_path = Path(plan.cell_labels_path)
    if not graph_path.is_file() or not labels_path.is_file():
        return None, None
    return json.loads(graph_path.read_text()), np.load(labels_path)


def cell_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    """Return cell-graph region metadata at a target pixel."""
    if x < 0 or y < 0:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_COORDS",
                           message=f"x and y must be >= 0, got ({x}, {y})",
                           recoverable=True),
        ])

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    if y >= plan.height or x >= plan.width:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="OUT_OF_BOUNDS",
                           message=f"({x}, {y}) outside plan bounds {plan.width}x{plan.height}",
                           recoverable=True),
        ])

    graph, labels = _load_cell_graph(plan)
    if graph is None or labels is None:
        return ToolResult(ok=True, data={"plan_id": plan_id, "x": x, "y": y, "cell": None},
                          errors=[WoodblockError(
                              tier="degraded", code="NO_CELL_GRAPH",
                              message="plan has no persisted cell graph",
                              recoverable=True,
                          )])

    cell_id = int(labels[y, x])
    cells = {int(c["cell_id"]): c for c in graph.get("cells", [])}
    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "x": x,
        "y": y,
        "cell_id": cell_id,
        "cell": cells.get(cell_id),
        "graph_summary": graph.get("diagnostics", {}),
    })


def _render_numpy(alpha_stack: np.ndarray, pigment_idx: np.ndarray) -> np.ndarray:
    h, w = alpha_stack.shape[1:]
    composite = np.broadcast_to(forward_render_jax.PAPER_RGB, (h, w, 3)).copy()
    pigments = forward_render_jax.PIGMENT_TABLE[pigment_idx]
    for alpha, pigment in zip(alpha_stack, pigments, strict=True):
        a = np.clip(alpha[..., None], 0.0, 1.0)
        composite = composite * (1.0 - a) + pigment[None, None, :] * a
    return composite.astype(np.float32)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.round(rgb * 255.0), 0, 255).astype(int).tolist()
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def inspect_cell(plan_id: str, cell_id: int) -> ToolResult[dict[str, Any]]:
    """Inspect target/rendered color and active impressions for one cell."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    graph, labels = _load_cell_graph(plan)
    if graph is None or labels is None:
        return ToolResult(ok=True, data={"plan_id": plan_id, "cell_id": cell_id, "cell": None},
                          errors=[WoodblockError(
                              tier="degraded", code="NO_CELL_GRAPH",
                              message="plan has no persisted cell graph",
                              recoverable=True,
                          )])
    cells = {int(c["cell_id"]): c for c in graph.get("cells", [])}
    if int(cell_id) not in cells:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="UNKNOWN_CELL_ID",
                           message=f"cell_id {cell_id!r} not in plan {plan_id}",
                           recoverable=True),
        ])

    mask = labels == int(cell_id)
    rows: list[dict[str, Any]] = []
    rendered_mean = None
    target_mean = None
    d_e = None
    if plan.alpha_stack_path and Path(plan.alpha_stack_path).is_file():
        alpha_stack = np.load(plan.alpha_stack_path)
        pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)
        for i, alpha in enumerate(alpha_stack):
            mean_alpha = float(alpha[mask].mean()) if mask.any() else 0.0
            if mean_alpha < 0.005:
                continue
            pid = int(pigment_idx[i]) if i < len(pigment_idx) else -1
            rows.append({
                "order_step": i + 1,
                "impression_id": (
                    plan.impressions[i].get("id", f"imp_{i + 1:03d}")
                    if i < len(plan.impressions) else f"imp_{i + 1:03d}"
                ),
                "pigment_id": pid,
                "pigment_name": (
                    _PIGMENT_NAMES[pid] if 0 <= pid < len(_PIGMENT_NAMES) else f"pigment_{pid}"
                ),
                "mean_alpha": round(mean_alpha, 4),
                "coverage_in_cell_pct": round(float((alpha[mask] > 0.05).mean() * 100.0), 3),
            })

        target_path = Path(plan.alpha_stack_path).parent / "target.npy"
        if target_path.is_file():
            rendered = _render_numpy(alpha_stack, pigment_idx)
            target = np.load(target_path)
            rendered_mean = rendered[mask].mean(axis=0)
            target_mean = target[mask].mean(axis=0)
            d_e = _delta_e76(tuple(target_mean.tolist()), tuple(rendered_mean.tolist()))

    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "cell_id": int(cell_id),
        "cell": cells[int(cell_id)],
        "active_impressions": rows,
        "target_mean_hex": _rgb_to_hex(target_mean) if target_mean is not None else None,
        "rendered_mean_hex": _rgb_to_hex(rendered_mean) if rendered_mean is not None else None,
        "cell_delta_e76": round(float(d_e), 3) if d_e is not None else None,
    })


def score_printability(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Score carve/inking printability before SVG export."""
    from scipy import ndimage as _ndi

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="NO_SOLVER_OUTPUT",
                           message="plan has no alpha_stack to score",
                           recoverable=True),
        ])

    alpha_stack = np.load(plan.alpha_stack_path)
    graph, labels = _load_cell_graph(plan)
    active_threshold = 0.08
    tiny_px = 60
    per_impression: list[dict[str, Any]] = []
    tiny_total = 0
    component_total = 0
    partial_cell_total = 0
    for i, alpha in enumerate(alpha_stack):
        active = alpha >= active_threshold
        cc, ncomp = _ndi.label(active)
        sizes = np.bincount(cc.ravel())[1:]
        tiny = int((sizes < tiny_px).sum()) if sizes.size else 0
        partial_cells = 0
        if labels is not None:
            for cid in np.unique(labels).tolist():
                cell = labels == int(cid)
                frac = float(active[cell].mean())
                if 0.05 < frac < 0.85:
                    partial_cells += 1
        pid = int(plan.pigment_idx[i]) if i < len(plan.pigment_idx) else -1
        per_impression.append({
            "impression_id": (
                plan.impressions[i].get("id", f"imp_{i + 1:03d}")
                if i < len(plan.impressions) else f"imp_{i + 1:03d}"
            ),
            "pigment_name": (
                _PIGMENT_NAMES[pid] if 0 <= pid < len(_PIGMENT_NAMES) else f"pigment_{pid}"
            ),
            "coverage_pct": round(float(active.mean() * 100.0), 3),
            "component_count": int(ncomp),
            "tiny_component_count": tiny,
            "partial_cell_count": int(partial_cells),
        })
        tiny_total += tiny
        component_total += int(ncomp)
        partial_cell_total += int(partial_cells)

    overlap = 0.0
    pairs = 0
    for i in range(alpha_stack.shape[0]):
        for j in range(i + 1, alpha_stack.shape[0]):
            overlap += float(np.mean(alpha_stack[i] * alpha_stack[j]))
            pairs += 1
    overlap = overlap / float(max(pairs, 1))
    low_alpha_pct = float(((alpha_stack > 0.01) & (alpha_stack < active_threshold)).mean() * 100.0)
    # Use pressure terms instead of raw counts so the score remains useful on
    # dense Emma-scale plans. Raw component counts are still returned above.
    m = max(1, int(alpha_stack.shape[0]))
    cell_count = max(1, len(graph.get("cells", [])) if graph else 1)
    component_pressure = 18.0 * float(np.log1p(component_total / float(m * 64)))
    tiny_pressure = 25.0 * (tiny_total / float(max(component_total, 1)))
    partial_pressure = 35.0 * (partial_cell_total / float(max(cell_count * m, 1)))
    overlap_pressure = min(25.0, 800.0 * overlap)
    low_alpha_pressure = min(20.0, 0.8 * low_alpha_pct)
    penalty = (
        component_pressure
        + tiny_pressure
        + partial_pressure
        + overlap_pressure
        + low_alpha_pressure
    )
    score = max(0.0, min(100.0, 100.0 - penalty))
    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "score_0_100": round(score, 2),
        "active_threshold": active_threshold,
        "tiny_component_px": tiny_px,
        "component_count_total": component_total,
        "tiny_component_count_total": tiny_total,
        "partial_cell_count_total": partial_cell_total,
        "pairwise_overlap_mean": round(overlap, 6),
        "low_alpha_coverage_pct": round(low_alpha_pct, 3),
        "cell_graph_present": labels is not None,
        "pressure_terms": {
            "component": round(component_pressure, 3),
            "tiny": round(tiny_pressure, 3),
            "partial_cell": round(partial_pressure, 3),
            "overlap": round(overlap_pressure, 3),
            "low_alpha": round(low_alpha_pressure, 3),
        },
        "per_impression": per_impression,
    })


__all__ = ["get_pigments", "get_emma_priors", "suggest_pigment_mix", "get_defaults",
           "solver_telemetry", "dE_at", "pigment_at", "cell_at", "inspect_cell",
           "score_printability"]
