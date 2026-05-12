"""D9A — Tier 0 core flow tool decorators (10 tools).

Mock-real hybrid per addendum-v5: every tool wired with stable signature
+ valid Pydantic + ``ToolResult[T]``. Real logic where backing module
exists; structured ``IMPL_PENDING`` ``WoodblockError`` everywhere else.

When the FastMCP server lands in D19, each tool gets the ``@server.tool``
decorator on top of these plain Python entries.

Per addendum-v4 WB-LANG-02: every t1_mixbox recipe carries the
"as if pre-mixed" qualifier so artists never confuse Mixbox prediction
with overprint physics.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23 import orchestrator as _orch
from backend.services.v23.core import templates as _templates
from backend.services.v23.stages import s1_ingest

_VALID_SOLVE_PROFILES = ("fast", "default", "thorough")
_VALID_FOCUS_MODES = ("composite", "heatmap", "per_impression", "confidence", "quad", "recipe", "pixel")


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded",
        code=code,
        message=f"{code} — backing implementation lands in a later substep",
        hint=hint,
        recoverable=True,
    )


# ---------------------------------------------------------------------------
# T0.1 — ingest_reference_image (REAL: wraps s1_ingest.ingest_reference_image)
# ---------------------------------------------------------------------------


def ingest_reference_image(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    return ToolResult(
        ok=True,
        data={
            "image_sha256": handle.image_sha256,
            "width": handle.width,
            "height": handle.height,
            "session_id": handle.session_id,
        },
    )


# ---------------------------------------------------------------------------
# T0.2 — analyze_image (REAL measurables; no subject_label per fix 5)
# ---------------------------------------------------------------------------


def analyze_image(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    family_areas = _classify_into_families(handle.array)
    hints = _templates.pick_template_hints(family_areas=family_areas)

    data = {
        "image_sha256": handle.image_sha256,
        "width": handle.width,
        "height": handle.height,
        "mpx": round((handle.width * handle.height) / 1_000_000, 4),
        "family_areas": family_areas,
        "dominant_family": hints["dominant_family"],
        "max_family_area_pct": hints["max_family_area_pct"],
        "flesh_area_pct": hints["flesh_area_pct"],
        "family_count_above_10pct": hints["family_count_above_10pct"],
        "est_solver_s": _estimate_solver_s(handle.width * handle.height),
    }
    return ToolResult(ok=True, data=data)


def _classify_into_families(rgb: Any) -> dict[str, float]:
    """Cheap measurable hue-family clustering — coarse OKLab bucketing."""
    import numpy as np

    arr = (rgb.astype("float32") / 255.0).reshape(-1, 3)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    warm = (r > g) & (r > b)
    cool = (b > r) & (b > g)
    green = (g > r) & (g > b)

    counts = {
        "cream": float(((lum > 0.85) & warm).mean()),
        "cool": float(((lum > 0.40) & (lum <= 0.85) & cool).mean()),
        "flesh": float(((lum > 0.55) & (lum <= 0.80) & warm & ~(r > 0.85)).mean()),
        "warm": float(((lum > 0.30) & (lum <= 0.70) & warm & (r > 0.55)).mean()),
        "shadow": float(((lum > 0.20) & (lum <= 0.50) & (green | cool)).mean()),
        "detail": float((lum <= 0.20).mean()),
        "accent": float(((lum > 0.30) & ~warm & ~cool & ~green).mean()),
    }
    total = sum(counts.values())
    if total > 0:
        counts = {k: v / total for k, v in counts.items()}
    return counts


def _estimate_solver_s(npx: int) -> float:
    """Rough solver wall-time estimate from pixel count."""
    return round(0.5 + (npx / 1_000_000) * 15.0, 1)


# ---------------------------------------------------------------------------
# T0.3 — build_hue_family_map (REAL: family-area dict from same classifier)
# ---------------------------------------------------------------------------


def build_hue_family_map(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    family_areas = _classify_into_families(handle.array)
    return ToolResult(
        ok=True,
        data={
            "image_sha256": handle.image_sha256,
            "family_areas": family_areas,
            "accent_pigments": [],  # populated in D9B Tier 3 introspection
        },
    )


# ---------------------------------------------------------------------------
# T0.4 — propose_stack (DEGRADED: real solver lands at D10+)
# ---------------------------------------------------------------------------


def propose_stack(
    path: str,
    *,
    solve_profile: Literal["fast", "default", "thorough"] = "default",
    m_prior: int | None = None,
    strategy_template: str | None = None,
) -> ToolResult[dict[str, Any]]:
    """Run S1→S2→S3 + template suggest. S5 solver still IMPL_PENDING.

    Real ``plan_id`` + persisted ``plan.json`` under the active session
    even though impressions are empty until the real solver wires at D10.
    """
    try:
        plan = _orch.run_pipeline_partial(
            path,
            solve_profile=solve_profile,
            m_prior=m_prior,
            strategy_template=strategy_template,
        )
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan.plan_id,
            "session_id": plan.session_id,
            "image_sha256": plan.image_sha256,
            "solve_profile": plan.solve_profile,
            "strategy_template": plan.suggested_template,
            "template_confidence": plan.template_confidence,
            "m_prior": m_prior or 6,
            "impression_count": len(plan.impressions),
            "dominant_family": plan.dominant_family,
            "family_areas": plan.family_areas,
            "sam_region_count": len(plan.sam_regions),
            "hue_family_map_path": plan.hue_family_map_path,
            "reconstruction_dE_mean": plan.reconstruction_dE_mean,
            "reconstruction_dE_p95": plan.reconstruction_dE_p95,
            "solver_wall_s": plan.solver_wall_s,
            "solver_status": plan.solver_status,
        },
        errors=[
            _impl_pending(
                "IMPL_PENDING_SOLVER",
                "S1→S2→S3 + template suggestion are real; S4-S10 (Tan warm-start + JAX inverse solver + three-state mask + DSATUR + SVG + ZIP) land incrementally in D10+ ticks",
            )
        ],
    )


# ---------------------------------------------------------------------------
# T0.5 — inspect_plan (MOCK: real artifact resolution lands at D10+)
# ---------------------------------------------------------------------------


def inspect_plan(
    plan_id: str,
    focus: Literal["composite", "heatmap", "per_impression", "confidence", "quad", "recipe", "pixel"] = "composite",
) -> ToolResult[dict[str, Any]]:
    if focus not in _VALID_FOCUS_MODES:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_FOCUS_MODE",
                    message=f"focus must be one of {_VALID_FOCUS_MODES}, got {focus!r}",
                    recoverable=True,
                )
            ],
        )
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "focus": focus, "artifact_path": None},
        errors=[
            _impl_pending(
                "IMPL_PENDING_INSPECT",
                "inspect_plan artifact resolution lands at D10; placeholder returned",
            )
        ],
    )


# ---------------------------------------------------------------------------
# T0.6 — forward_render (alias simulate_candidate_stack — both routed to same impl)
# ---------------------------------------------------------------------------


def forward_render(plan_id: str) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "composite_path": None, "dE_map_path": None},
        errors=[
            _impl_pending(
                "IMPL_PENDING_FORWARD",
                "forward_render reads Plan tensor + invokes forward_render_jax; lands at D10",
            )
        ],
    )


def simulate_candidate_stack(plan_id: str) -> ToolResult[dict[str, Any]]:
    return forward_render(plan_id)


# ---------------------------------------------------------------------------
# T0.7 — score_stack_delta_e (cheap ΔE lookup, mock returns 0.0)
# ---------------------------------------------------------------------------


def score_stack_delta_e(plan_id: str, region: dict | None = None) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "dE_mean": 0.0, "dE_p95": 0.0, "region": region},
        errors=[
            _impl_pending(
                "IMPL_PENDING_DE_LOOKUP",
                "ΔE lookup reads cached scores from Plan; lands at D10",
            )
        ],
    )


# ---------------------------------------------------------------------------
# T0.8 — score_candidate_stack (5-component combined score per fix 4)
# ---------------------------------------------------------------------------


def score_candidate_stack(plan_id: str) -> ToolResult[dict[str, Any]]:
    weights = {
        "visual_match": 0.40,
        "carveability": 0.20,
        "simplicity": 0.15,
        "underprint_utility": 0.15,
        "template_fit": 0.10,
    }
    # Mock scores; D10 populates from real Plan
    components = {
        "visual_match": 0.5,
        "carveability": 0.5,
        "simplicity": 0.5,
        "underprint_utility": 0.5,
        "template_fit": 0.5,
    }
    overall = sum(weights[k] * components[k] for k in weights)
    data = {
        "plan_id": plan_id,
        "overall": overall,
        **components,
        "component_weights": weights,
        "notes": (
            "Mock scores — D10 wires real component calculations. "
            "Reported as if pre-mixed in a well (Mixbox t1); actual overprint "
            "may shift colors ΔE 4–8 deeper into the stack."
        ),
    }
    return ToolResult(ok=True, data=data)


# ---------------------------------------------------------------------------
# T0.9 — export_print_plan (ZIP + recipe.md; mock zips empty content)
# ---------------------------------------------------------------------------


def export_print_plan(plan_id: str, out_dir: str | None = None) -> ToolResult[dict[str, Any]]:
    out = Path(out_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{plan_id}.zip"
    recipe_path = out / f"{plan_id}_recipe.md"

    recipe_md = generate_print_recipe_report(plan_id).data["markdown"]
    recipe_path.write_text(recipe_md)
    # Empty ZIP placeholder
    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("recipe.md", recipe_md)

    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "zip_path": str(zip_path), "recipe_path": str(recipe_path)},
        errors=[
            _impl_pending(
                "IMPL_PENDING_EXPORT",
                "real ZIP composition (impressions/, blocks/, tensors/) lands at D10/D13",
            )
        ],
    )


# ---------------------------------------------------------------------------
# T0.10 — generate_print_recipe_report (plain-language print recipe)
# ---------------------------------------------------------------------------


def generate_print_recipe_report(plan_id: str, format: str = "markdown") -> ToolResult[dict[str, Any]]:
    md = _MOCK_RECIPE.format(plan_id=plan_id)
    return ToolResult(ok=True, data={"plan_id": plan_id, "format": format, "markdown": md})


_MOCK_RECIPE = """# Print recipe — {plan_id}

> **Note on color simulation (t1_mixbox)**: this recipe was rendered as if
> pigments were pre-mixed in a well before application. Actual mokuhanga
> overprint glazing may shift colors by ΔE 4–8 vs. the simulated composite,
> especially on stacks deeper than 3 impressions. Upload a swatch overprint
> matrix to unlock t2_empirical for accurate prediction with your own pigments.

## Impressions (light → dark)

Impression 01: pale cream support, broad face/base areas
Impression 02: cool blue support under shadows and hair edges
Impression 03: pink flesh field, face midtones
Impression 04: warm red/orange accents on cheeks and lips
Impression 05: teal shadow shapes
Impression 06: dark key/detail — hair, eyes, mouth, nose lines

## Notes

- v23-MCP day-1 ship: solver returns mock plan_ids. Real plan generation
  arrives at D10 (close_emma corpus gate).
- All confidence labels in mock mode are reported as ``ambiguous`` per
  addendum-v3 fix 4 until real ΔE scores fire.
"""


__all__ = [
    "ingest_reference_image",
    "analyze_image",
    "build_hue_family_map",
    "propose_stack",
    "inspect_plan",
    "forward_render",
    "simulate_candidate_stack",
    "score_stack_delta_e",
    "score_candidate_stack",
    "export_print_plan",
    "generate_print_recipe_report",
]
