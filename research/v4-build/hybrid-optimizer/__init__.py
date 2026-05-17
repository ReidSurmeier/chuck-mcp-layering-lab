"""chuck-mcp v4-build hybrid-optimizer.

Phase 4 of the audit-corrected build sequence (per
`docs/v2-design-locked-2026-05-16.md` and
`docs/audit-response-and-reconstruction-plan-2026-05-17.md`):

    "Hybrid alternating optimization (NOT pure α-maps):
        1. cell graph / region proposal,
        2. plate assignment with graph cut or ILP-style exclusivity,
        3. JAX continuous solve for opacity/dilution/color per pull,
        4. morphology repair and component scoring,
        5. re-solve after repair, not just accept degraded dE.
     JAX optimizes continuous pigment/load variables. It does NOT
     invent printable topology from unconstrained alpha."

Public surface:

    optimize(target, production_plan, max_outer_iters=5, max_inner_iters=200)
        -> OptimizationResult
    OptimizationResult
    SolvedPlate

    # Lower-level entry points (for testing / piecewise use):
    assign_cells_to_plates(...)
    solve_pigment_load(...)
    repair_plates(...)
"""

# Absolute imports via the installed alias package (see conftest.py).
# Fall back to relative if the alias is not yet installed (e.g. running
# `python -m hybrid_optimizer.<x>` directly).
try:
    from hybrid_optimizer.alternating_loop import ProductionPlanInput, optimize
    from hybrid_optimizer.graph_cut_assigner import (
        CandidatePlate,
        CellGraph,
        CellNode,
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
        repair_plate_mask,
        repair_plates,
    )
    from hybrid_optimizer.optimization_result import OptimizationResult, SolvedPlate
except ImportError:
    from .alternating_loop import ProductionPlanInput, optimize  # type: ignore
    from .graph_cut_assigner import (  # type: ignore
        CandidatePlate,
        CellGraph,
        CellNode,
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
        repair_plate_mask,
        repair_plates,
    )
    from .optimization_result import OptimizationResult, SolvedPlate  # type: ignore

__all__ = [
    "optimize",
    "OptimizationResult",
    "SolvedPlate",
    "ProductionPlanInput",
    "CellGraph",
    "CellNode",
    "CandidatePlate",
    "RoleConstraints",
    "assign_cells_to_plates",
    "assignment_to_cells_by_plate",
    "FrozenPlate",
    "SolveResult",
    "solve_pigment_load",
    "PlateRepairStats",
    "DEFAULT_MILL_RADIUS_PX",
    "repair_plate_mask",
    "repair_plates",
    "any_needs_resolve",
]
