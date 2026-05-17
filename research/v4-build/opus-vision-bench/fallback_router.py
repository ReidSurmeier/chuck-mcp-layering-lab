"""
fallback_router.py — production routing decision: Opus vs MediaPipe.

Two-tier decision:

  1. Global gate (per audit response §1): if Opus's mean Jaccard across the
     10-overlay benchmark is >= GLOBAL_JACCARD_THRESHOLD (default 0.95),
     allow Opus to write cell IDs. Otherwise, route ALL cell-ID work to
     MediaPipe.

  2. Per-region gate: even if the global gate passes, any individual
     region whose mean Jaccard < PER_REGION_JACCARD_FLOOR (default 0.85)
     is routed to MediaPipe (the LLM can still suggest semantic
     interpretation for that region, but the geometry comes from
     MediaPipe).

Public API:

    decision = route_to_opus_or_mediapipe(bench_results)
    # decision.global_route in {"opus", "mediapipe"}
    # decision.per_region_route[name] in {"opus", "mediapipe"}
    # decision.reason -> human-readable string for the audit log
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from jaccard_evaluator import BenchResult


Route = Literal["opus", "mediapipe"]

GLOBAL_JACCARD_THRESHOLD = 0.95
PER_REGION_JACCARD_FLOOR = 0.85


@dataclass
class RoutingDecision:
    global_route: Route
    per_region_route: dict[str, Route] = field(default_factory=dict)
    overall_mean_jaccard: float = 0.0
    reason: str = ""
    global_threshold: float = GLOBAL_JACCARD_THRESHOLD
    per_region_floor: float = PER_REGION_JACCARD_FLOOR

    def is_go(self) -> bool:
        return self.global_route == "opus"

    def to_dict(self) -> dict:
        return {
            "global_route": self.global_route,
            "per_region_route": dict(self.per_region_route),
            "overall_mean_jaccard": self.overall_mean_jaccard,
            "global_threshold": self.global_threshold,
            "per_region_floor": self.per_region_floor,
            "is_go": self.is_go(),
            "reason": self.reason,
        }


def route_to_opus_or_mediapipe(
    bench_results: BenchResult,
    *,
    global_threshold: float = GLOBAL_JACCARD_THRESHOLD,
    per_region_floor: float = PER_REGION_JACCARD_FLOOR,
) -> RoutingDecision:
    """Produce a routing decision from benchmark results."""
    overall = bench_results.overall_mean_jaccard()

    per_region_route: dict[str, Route] = {}
    per_region_summary = bench_results.per_region_summary()
    for region, stats in per_region_summary.items():
        if stats["mean"] >= per_region_floor:
            per_region_route[region] = "opus"
        else:
            per_region_route[region] = "mediapipe"

    if overall >= global_threshold:
        global_route: Route = "opus"
        reason = (
            f"OPUS GO — overall mean Jaccard {overall:.3f} >= "
            f"global threshold {global_threshold}. "
            f"{sum(1 for v in per_region_route.values() if v == 'mediapipe')} "
            f"regions are still pinned to MediaPipe by the per-region floor "
            f"({per_region_floor})."
        )
    else:
        global_route = "mediapipe"
        reason = (
            f"MEDIAPIPE — overall mean Jaccard {overall:.3f} < "
            f"global threshold {global_threshold}. Opus is not yet trusted "
            f"to write cell IDs. Ship MediaPipe; revisit after a model bump "
            f"or after Opus prompt iteration."
        )

    return RoutingDecision(
        global_route=global_route,
        per_region_route=per_region_route,
        overall_mean_jaccard=overall,
        reason=reason,
        global_threshold=global_threshold,
        per_region_floor=per_region_floor,
    )
