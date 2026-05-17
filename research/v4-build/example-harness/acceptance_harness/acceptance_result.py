"""Dataclasses for the acceptance harness output.

The harness intentionally returns STRUCTURAL metrics only. The audit acceptance
rule is verbatim:

    "if a human says 'this looks like slop' against the example sheet, the run
    fails regardless of dE."

So `human_eyeball_required` is always True. Downstream tooling (the MCP server,
the chuck.reidsurmeier.wtf web UI) is responsible for surfacing the sheet to a
human and capturing the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlateMetric:
    """Per-plate structural score.

    Mirrors the design-doc validator family in
    docs/v2-design-locked-2026-05-16.md "Acceptance Test (locked)" — but at
    the *harness* level we only compute cheap structural proxies. The real
    validator suite lives in backend/services/v23/validators.

    Attributes:
        plate_index: 0-based block index as it appears on the contact sheet.
        coverage_fraction: fraction of the plate canvas that is inked
            (non-background pixels). Real jigsaw plates land 0.05-0.40;
            v13-style residual full-face plates land > 0.60.
        plate_not_composite_score: cheap proxy = 1.0 - similarity_to_final.
            Per the design doc, real plates score 0.946-1.000 and v13-style
            residuals score 0.133. HIGH score = good (jigsaw plate).
    """

    plate_index: int
    coverage_fraction: float
    plate_not_composite_score: float


@dataclass
class AcceptanceSheetResult:
    """Return type of render_acceptance_sheet().

    Attributes:
        sheet_path: absolute path to the rendered 4-row contact sheet PNG.
        reference_examples_used: list of source filenames from
            /srv/woodblock-share/Examples that contributed cells.
        proof_checkpoints_rendered: list of pull indices actually pulled from
            the plan dir (may be < 8 if the plan has < 8 cumulative pulls).
        plate_count_rendered: how many plate previews ended up in row 3
            (capped at 8).
        alpha_count_rendered: how many alpha snapshots ended up in row 4
            (capped at 8).
        proof_progression_score: structural proxy in [0, 1]. Computed as the
            mean per-step pixelwise dissimilarity across the current proof row
            (row 2). Reference scale: a healthy progression scores > 0.05.
        plate_metrics: list[PlateMetric] for every plate rendered in row 3.
        human_eyeball_required: always True — the audit rule mandates a human
            verdict before the run is considered passing.
        warnings: human-readable warnings (missing files, fallbacks taken).
    """

    sheet_path: Path
    reference_examples_used: list[str] = field(default_factory=list)
    proof_checkpoints_rendered: list[int] = field(default_factory=list)
    plate_count_rendered: int = 0
    alpha_count_rendered: int = 0
    proof_progression_score: float = 0.0
    plate_metrics: list[PlateMetric] = field(default_factory=list)
    human_eyeball_required: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for JSON/MCP transport."""
        d = asdict(self)
        d["sheet_path"] = str(self.sheet_path)
        return d
