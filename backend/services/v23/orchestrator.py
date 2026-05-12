"""D10.b — v23-MCP pipeline orchestrator.

Chains S1 (ingest) → S2 (SAM) → S3 (hue family) → suggest_template
into a single ``run_pipeline_partial(image_path, solve_profile)`` call.
S4-S10 still ship as ``IMPL_PENDING_*`` placeholders inside the
resulting :class:`PartialPlan` — they wire in D10 real solver + D11 HITL
+ D13 carve real per the build chain.

The Plan JSON persists under ``~/.woodblock/v23/sessions/<sid>/plans/<plan_id>/plan.json``
so subsequent tool calls (``inspect_plan``, ``forward_render``,
``export_print_plan``, etc.) can load + read without re-running the
pipeline.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.mcp.errors import WoodblockError
from backend.services.v23 import session as _sess
from backend.services.v23.core import templates as _templates
from backend.services.v23.stages import (
    s1_ingest,
    s2_sam,
    s3_hue_family,
    s4_warmstart,
    s5_solver,
)

_VALID_SOLVE_PROFILES = ("fast", "default", "thorough")
_SCHEMA_VERSION = "v23.0"


@dataclass(frozen=True)
class PartialPlan:
    """In-memory pipeline result. Persisted as JSON under the session."""

    plan_id: str
    session_id: str
    image_sha256: str
    width: int
    height: int
    solve_profile: str
    schema_version: str
    family_areas: dict[str, float]
    dominant_family: str
    hue_family_map_path: str | None
    sam_regions: list[dict[str, Any]]
    suggested_template: str | None
    template_confidence: float
    template_reason: str
    solver_status: str = "IMPL_PENDING"
    impressions: list[dict[str, Any]] = field(default_factory=list)
    reconstruction_dE_mean: float | None = None
    reconstruction_dE_p95: float | None = None
    solver_wall_s: float = 0.0
    created_at: str = ""


class OrchestratorError(Exception):
    """Raised on input-validation refusals or upstream stage failures."""

    def __init__(self, error: WoodblockError) -> None:
        super().__init__(error.message)
        self.error = error


def _plan_dir(session_id: str, plan_id: str) -> Path:
    base = _sess.paths.session_dir(session_id) / "plans" / plan_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _persist_plan(plan: PartialPlan) -> Path:
    p = _plan_dir(plan.session_id, plan.plan_id) / "plan.json"
    payload = asdict(plan)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def run_pipeline_partial(
    image_path: str,
    *,
    solve_profile: Literal["fast", "default", "thorough"] = "default",
    m_prior: int | None = None,
    strategy_template: str | None = None,
) -> PartialPlan:
    """Run S1→S2→S3 + template suggestion. Persist + return PartialPlan."""
    if solve_profile not in _VALID_SOLVE_PROFILES:
        raise OrchestratorError(WoodblockError(
            tier="refusal", code="INVALID_SOLVE_PROFILE",
            message=f"solve_profile must be one of {_VALID_SOLVE_PROFILES}, got {solve_profile!r}",
            hint=f"use one of: {', '.join(_VALID_SOLVE_PROFILES)}",
            recoverable=True,
        ))

    # S1 — ingest
    try:
        handle = s1_ingest.ingest_reference_image(image_path)
    except s1_ingest.IngestError as exc:
        raise OrchestratorError(exc.error) from exc

    # S2 — SAM region prior (degrades gracefully if sidecar unreachable).
    # ``WOODBLOCK_DISABLE_SAM=1`` env skips S2 entirely (test rings + offline dev).
    sam_regions: list[dict[str, Any]] = []
    if os.environ.get("WOODBLOCK_DISABLE_SAM") != "1":
        try:
            canonical_path = _plan_dir(handle.session_id, "_tmp") / f"{handle.image_sha256}.png"
            canonical_path.write_bytes(handle.canonical_bytes)
            s2 = s2_sam.run_s2_sam(canonical_path, image_sha256=handle.image_sha256)
            sam_regions = [
                {"region_id": r.region_id, "bbox": list(r.bbox), "area_px": r.area_px,
                 "mask_path": str(r.mask_path), "mean_oklab": list(r.mean_oklab)}
                for r in s2.regions
            ]
        except s2_sam.SamGatewayError:
            # SAM unavailable — continue without regions (degraded mode)
            pass
        except Exception:
            # Any other transport failure (connection refused, DNS, etc.)
            # also drops to degraded mode rather than killing the pipeline.
            pass

    # S3 — hue family classification + per-family map PNG
    s3 = s3_hue_family.run_s3_hue_family(handle.array, image_sha256=handle.image_sha256)

    # Template suggestion (measurable hints only — Opus picks)
    suggestion = _templates.suggest_template(family_areas=s3.family_areas)

    # S4 + S5 — Tan warm-start + JAX L-BFGS inverse solver.
    # ``WOODBLOCK_DISABLE_SOLVER=1`` env bypasses S4+S5 (test ring default).
    impressions: list[dict[str, Any]] = []
    solver_status = "IMPL_PENDING"
    reconstruction_dE_mean: float | None = None
    reconstruction_dE_p95: float | None = None
    solver_wall_s = 0.0
    if os.environ.get("WOODBLOCK_DISABLE_SOLVER") != "1":
        try:
            warm = s4_warmstart.tan_to_pigment_warmstart(
                handle.array, target_palette_size=8
            )
            target = handle.array.astype("float32") / 255.0
            result = s5_solver.run_s5_solver(
                target_rgb=target,
                pigment_idx=__import__("numpy").asarray(warm.pigment_idx, dtype="int32"),
                alpha_init=warm.alpha_stack,
                solve_profile=solve_profile,
            )
            impressions = result.impressions
            solver_status = "OK"
            solver_wall_s = result.wall_s
            # Rough RGB-L2 → ΔE proxy: final_loss is sum-squared over (H*W*3).
            # Per-pixel MSE -> sqrt -> heuristic ΔE in RGB scale [0..255].
            n = handle.width * handle.height * 3
            mse = result.final_loss / max(n, 1)
            rms = float(mse ** 0.5)
            reconstruction_dE_mean = round(rms * 255.0 * 0.1, 3)  # crude scale
            reconstruction_dE_p95 = round(rms * 255.0 * 0.2, 3)
        except Exception:
            solver_status = "FAILED"

    plan_id = f"plan_{int(time.time() * 1000)}_{handle.image_sha256[:8]}"
    plan = PartialPlan(
        plan_id=plan_id,
        session_id=handle.session_id,
        image_sha256=handle.image_sha256,
        width=handle.width,
        height=handle.height,
        solve_profile=solve_profile,
        schema_version=_SCHEMA_VERSION,
        family_areas=s3.family_areas,
        dominant_family=s3.dominant_family,
        hue_family_map_path=str(s3.label_map_path) if s3.label_map_path else None,
        sam_regions=sam_regions,
        suggested_template=strategy_template or suggestion.template_id,
        template_confidence=suggestion.confidence,
        template_reason=suggestion.reason,
        solver_status=solver_status,
        impressions=impressions,
        reconstruction_dE_mean=reconstruction_dE_mean,
        reconstruction_dE_p95=reconstruction_dE_p95,
        solver_wall_s=solver_wall_s,
        created_at=_now_iso(),
    )
    _persist_plan(plan)
    return plan


def load_plan(plan_id: str) -> PartialPlan:
    """Load a persisted PartialPlan by plan_id from the active session."""
    sid = _sess.current_session()
    if sid is None:
        raise OrchestratorError(WoodblockError(
            tier="refusal", code="NO_ACTIVE_SESSION",
            message="no active session — call set_session() or ingest first",
            recoverable=True,
        ))
    plan_file = _plan_dir(sid, plan_id) / "plan.json"
    if not plan_file.is_file():
        # Try other sessions (plan_id is globally unique within ULID prefix)
        sessions_root = _sess.paths.WB_DATA_DIR / "sessions"
        for sdir in sessions_root.iterdir() if sessions_root.is_dir() else []:
            candidate = sdir / "plans" / plan_id / "plan.json"
            if candidate.is_file():
                plan_file = candidate
                break
        else:
            raise OrchestratorError(WoodblockError(
                tier="refusal", code="PLAN_NOT_FOUND",
                message=f"plan {plan_id!r} not found in any session",
                hint="check current_session() + list_sessions() to find the right session",
                recoverable=True,
            ))
    payload = json.loads(plan_file.read_text())
    return PartialPlan(**payload)


__all__ = ["PartialPlan", "OrchestratorError", "run_pipeline_partial", "load_plan"]
