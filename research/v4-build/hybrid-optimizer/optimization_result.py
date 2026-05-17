"""OptimizationResult and supporting dataclasses for the hybrid alternating optimizer.

Per docs/v2-design-locked-2026-05-16.md Phase 4:

    "Hybrid alternating optimization (NOT pure α-maps):
     1. cell graph / region proposal,
     2. plate assignment with graph cut or ILP-style exclusivity,
     3. JAX continuous solve for opacity/dilution/color per pull,
     4. morphology repair and component scoring,
     5. re-solve after repair, not just accept degraded dE."

This module defines the result envelope. Downstream consumers:
    - backend/mcp/chuck/plan_emma_print tool: returns this to the web UI
    - research/v3-construction/validators-reconstruction: scores the .plates
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SolvedPlate:
    """One physical block after Stage 3/4. Mirrors `backend...Plate` but
    is solver-internal: cell IDs are frozen by Stage 2, continuous fields
    are filled by Stage 3, the inked mask is repaired by Stage 4.

    Field order matches docs/v2-design-locked-2026-05-16.md domain objects.
    """

    block_id: int
    cell_zone_ids: List[int]
    pigment_id: str
    opacity: float
    dilution: float
    role: str  # underlayer_light | local_chroma | regional_mass | key_detail
    pass_index: int  # which pull (1..132)
    pigment_weights: Dict[str, float] = field(default_factory=dict)
    inked_mask: Optional[np.ndarray] = None  # H x W binary, may be None pre-render
    area_px: int = 0
    repair_stats: Dict[str, Any] = field(default_factory=dict)
    mirror: bool = True

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "block_id": int(self.block_id),
            "cell_zone_ids": [int(x) for x in self.cell_zone_ids],
            "pigment_id": str(self.pigment_id),
            "opacity": float(self.opacity),
            "dilution": float(self.dilution),
            "role": str(self.role),
            "pass_index": int(self.pass_index),
            "pigment_weights": {k: float(v) for k, v in self.pigment_weights.items()},
            "area_px": int(self.area_px),
            "repair_stats": _jsonify(self.repair_stats),
            "mirror": bool(self.mirror),
        }
        if include_mask and self.inked_mask is not None:
            d["inked_mask_shape"] = list(self.inked_mask.shape)
            d["inked_mask_dtype"] = str(self.inked_mask.dtype)
        return d


@dataclass
class OptimizationResult:
    """Output of `alternating_loop.optimize`.

    Attributes:
        plates: solved plates after the final iteration. Length matches the
            production plan's plate count.
        validator_scores: dict keyed by validator name; values are the
            per-validator detail dicts from
            research/v3-construction/validators-reconstruction.
        outer_iter_count: how many outer loops actually ran (1..max).
        total_wall_time_s: end-to-end wall time of optimize().
        converged: True iff all 5 gating validators pass AND no outer loop
            reduced overall score.
        delta_e_mean: ΔE_2000 mean across visible regions (advisory).
        delta_e_p95: ΔE_2000 p95 (advisory).
        stage_timings: per-stage cumulative wall time (s) across outer loops.
        history: trail of `{iter, loss, delta_e_mean, n_gates_passed}` for
            debugging / convergence plots.
        notes: human-readable notes, e.g. "Stage 5 re-solved on iter 2".
    """

    plates: List[SolvedPlate]
    validator_scores: Dict[str, Any]
    outer_iter_count: int
    total_wall_time_s: float
    converged: bool
    delta_e_mean: float = float("nan")
    delta_e_p95: float = float("nan")
    stage_timings: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def n_gates_passed(self) -> int:
        """How many of the 5 gating validators passed (final_match is advisory)."""
        gates = (
            "plate_not_composite",
            "role_purity",
            "jigsaw_separation",
            "proof_progression",
            "underlayer_reversal",
        )
        return sum(
            1 for g in gates if (self.validator_scores.get(g) or {}).get("passes")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plates": [p.to_dict() for p in self.plates],
            "validator_scores": _jsonify(self.validator_scores),
            "outer_iter_count": int(self.outer_iter_count),
            "total_wall_time_s": float(self.total_wall_time_s),
            "converged": bool(self.converged),
            "delta_e_mean": _safe_float(self.delta_e_mean),
            "delta_e_p95": _safe_float(self.delta_e_p95),
            "n_gates_passed": int(self.n_gates_passed()),
            "stage_timings": {k: float(v) for k, v in self.stage_timings.items()},
            "history": _jsonify(self.history),
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def save(self, path: str) -> Path:
        out = Path(path)
        out.write_text(self.to_json())
        return out


def _jsonify(o: Any) -> Any:
    """Recursively convert numpy types to plain Python for json."""
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _safe_float(float(o))
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _safe_float(v: float) -> float:
    """JSON does not allow NaN/Inf — coerce to None-equivalent (0.0) for round-trip."""
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(f) or np.isinf(f):
        return 0.0
    return f


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _safe_float(float(o))
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
