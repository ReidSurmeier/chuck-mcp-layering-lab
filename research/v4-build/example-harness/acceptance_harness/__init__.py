"""chuck-mcp v4 — Example Harness (Phase 1).

Acceptance-sheet generator for human eyeball validation of chuck-mcp plan output
against reference mokuhanga/Chuck Close progressive proofs in
/srv/woodblock-share/Examples.

See docs/v2-design-locked-2026-05-16.md §"Build sequence" Phase 1.
"""

from .acceptance_result import AcceptanceSheetResult, PlateMetric
from .acceptance_harness import render_acceptance_sheet
from .example_loader import (
    load_reference_proofs,
    load_woodblock_print_process,
    REFERENCE_EXAMPLES_DIR,
)

__all__ = [
    "AcceptanceSheetResult",
    "PlateMetric",
    "render_acceptance_sheet",
    "load_reference_proofs",
    "load_woodblock_print_process",
    "REFERENCE_EXAMPLES_DIR",
]

__version__ = "0.1.0"
