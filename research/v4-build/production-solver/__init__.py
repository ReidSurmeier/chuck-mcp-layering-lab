"""chuck-mcp v4-build production-solver.

First stage of the alternating optimizer: the cell-graph PROPOSAL stage.
Produces a ProductionPlan with adaptive plate count and multi-pull-per-block
schedule BEFORE any JAX continuous solve runs.

Per `docs/audit-response-and-reconstruction-plan-2026-05-17.md` Phase 2:
    "Solve production structure directly. Adaptive plate count (24-30 prior
     for Emma-scale). Multi-pull-per-block as first-class variables.
     Block/pull identity solved WITH target reconstruction."

Modules:
    plate_count_estimator   — adaptive plate count from image stats.
    multi_pull_assigner     — assigns 1-5 pulls per block (~132 total Emma).
    production_plan_builder — composes the full ProductionPlan.
    production_plan         — ProductionPlan + PullSpec dataclasses.

Note: the folder name uses a hyphen (production-solver) per the v4-build
audit spec, so this package is not directly importable via "import
production_solver". Use ``test_production_solver._load_package`` (or add
this directory to ``sys.path`` and import the submodules by name) to
consume the API from external code.
"""
import os
import sys

# Make sibling modules importable when this package is loaded as a folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from production_plan import (  # noqa: E402
    PullSpec,
    PlateSpec,
    ProductionPlan,
    Role,
    ROLES,
)
from plate_count_estimator import estimate_plate_count, CellGraphStats  # noqa: E402
from multi_pull_assigner import assign_pulls_per_block  # noqa: E402
from production_plan_builder import build_production_plan  # noqa: E402
import plate_count_estimator  # noqa: E402,F401
import multi_pull_assigner  # noqa: E402,F401
import production_plan  # noqa: E402,F401
import production_plan_builder  # noqa: E402,F401

__all__ = [
    "PullSpec",
    "PlateSpec",
    "ProductionPlan",
    "Role",
    "ROLES",
    "estimate_plate_count",
    "CellGraphStats",
    "assign_pulls_per_block",
    "build_production_plan",
]
