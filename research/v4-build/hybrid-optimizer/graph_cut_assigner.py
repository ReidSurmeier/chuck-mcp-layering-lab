"""Stage 2: plate assignment via graph-cut (with DSATUR fallback).

Consumes a cell-graph proposal (Stage 1) and a list of candidate plates
(produced by `research/v4-build/production-solver/production_plan_builder.py`
or a synthetic ProductionPlan in tests) and returns:

    {cell_id -> plate_id}

with these load-bearing constraints (per audit Phase 3 + design-doc row 2-3):

    1. EVERY cell is assigned to exactly one plate (no orphans).
    2. Each plate's assigned cells share a role family (purity >= 0.7).
    3. The total inked area per plate is bounded by role-frequency permission
       (early underlayer plates may be large; key_detail plates must be small).
    4. Two highly-similar cells (low ΔE in target image) prefer the same plate
       (graph-cut min-cut weight).

V1 implementation strategy:

    - First, partition the cell graph by ROLE FAMILY. Each role family becomes
      a sub-problem.
    - For each role family, run greedy DSATUR-style assignment: highest-degree
      cell first, assigned to the plate with the lowest "tension" (cost of
      cutting its neighbors), subject to role/area constraints.
    - If `networkx` provides a maxflow/min-cut for our sub-problem AND the
      caller passed `use_graph_cut=True`, run min-cut to refine borders.
      Default V1: pure DSATUR — deterministic, fast, no extra dep.

This is intentionally NOT a globally optimal ILP. The audit (Phase 4) says
"plate assignment with graph cut or ILP-style exclusivity, fallback greedy
DSATUR for V1 is acceptable" — see task spec.

Determinism: assignment is deterministic given (cell_graph, candidate_plates,
role_constraints, seed). Seed defaults to 0xC0FFEE for reproducible tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# Role families from docs/v2-design-locked-2026-05-16.md domain objects.
ALLOWED_ROLES = (
    "underlayer_light",
    "local_chroma",
    "regional_mass",
    "key_detail",
)


@dataclass
class CellNode:
    """One node in the cell graph.

    Attributes:
        cell_id: stable integer id (matches SNIC cell label).
        role: one of ALLOWED_ROLES. Set by the role classifier upstream.
        lab_color: mean Lab color of the cell (L 0..100, a/b -128..127).
        area_px: pixel area of the cell.
        centroid_yx: (y, x) pixel centroid for distance computations.
    """

    cell_id: int
    role: str
    lab_color: Tuple[float, float, float]
    area_px: int = 1
    centroid_yx: Tuple[float, float] = (0.0, 0.0)


@dataclass
class CellGraph:
    """Stage 1 proposal output.

    Attributes:
        nodes: list of CellNode, one per cell.
        edges: undirected list of (cell_a, cell_b) cell-id pairs. Adjacency.
        edge_weights: dict[(a, b) -> float] weight of the edge for graph-cut.
            Lower weight = cheaper to CUT (cells less similar, OK to split).
            Higher weight = expensive to cut (cells very similar, keep together).
    """

    nodes: List[CellNode]
    edges: List[Tuple[int, int]] = field(default_factory=list)
    edge_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def neighbors(self, cell_id: int) -> List[int]:
        nbrs: List[int] = []
        for a, b in self.edges:
            if a == cell_id:
                nbrs.append(b)
            elif b == cell_id:
                nbrs.append(a)
        return nbrs

    def by_id(self) -> Dict[int, CellNode]:
        return {n.cell_id: n for n in self.nodes}


@dataclass
class CandidatePlate:
    """A plate slot the assigner can fill.

    Attributes:
        plate_id: stable integer id.
        role: which role family this plate serves. Cells assigned MUST share role.
        max_area_px: upper bound on total inked area (sum of assigned cell areas).
            None = no cap.
        pigment_id: pigment that will be used. Cells assigned should be color-
            compatible (the graph-cut similarity will prefer this naturally).
    """

    plate_id: int
    role: str
    max_area_px: Optional[int] = None
    pigment_id: str = ""


@dataclass
class RoleConstraints:
    """Per-role constraints. Defaults are conservative for Emma-scale.

    Attributes:
        max_plates_per_role: cap on plates per role family (informational —
            assigner respects whatever is in `candidate_plates`).
        max_cells_per_plate: cap on cell count for a single plate. None = inf.
        min_cells_per_plate: floor (drop the plate if below — V2).
    """

    max_plates_per_role: Dict[str, int] = field(
        default_factory=lambda: {
            "underlayer_light": 6,
            "local_chroma": 8,
            "regional_mass": 8,
            "key_detail": 6,
        }
    )
    max_cells_per_plate: Optional[int] = None
    min_cells_per_plate: int = 0


def _assignment_tension(
    cell: CellNode,
    plate: CandidatePlate,
    current_assignment: Dict[int, int],
    graph: CellGraph,
    plate_area: Dict[int, int],
    plate_count: Dict[int, int],
    n_role_plates: int,
    load_balance_weight: float = 0.5,
) -> float:
    """Cost of assigning `cell` to `plate` — graph-cut + load balance.

    Lower = better. Sum of:
      1. graph-cut: edge weights to neighbors on DIFFERENT plates
         (penalty for splitting similar neighbors).
      2. graph-bonus: small NEGATIVE term for neighbors already on the
         SAME plate (reward for joining a started zone).
      3. load-balance: penalty proportional to current plate count divided
         by avg-per-plate target. Without this, DSATUR collapses all cells
         of a role onto the first plate of that role.
    """
    if cell.role != plate.role:
        return float("inf")  # hard role constraint
    cost = 0.0
    same_plate_bonus = 0.0
    for nb in graph.neighbors(cell.cell_id):
        nb_plate = current_assignment.get(nb)
        if nb_plate is None:
            continue
        w = graph.edge_weights.get(
            (min(cell.cell_id, nb), max(cell.cell_id, nb)), 1.0
        )
        if nb_plate != plate.plate_id:
            cost += w
        else:
            same_plate_bonus += w
    cost -= 0.5 * same_plate_bonus  # join-similar bonus

    # Load balance: penalty rises linearly with current cell count
    if n_role_plates > 0:
        # target = ideal cells/plate if balanced
        cost += load_balance_weight * plate_count[plate.plate_id]
    return cost


def _dsatur_order(graph: CellGraph) -> List[int]:
    """Order cells by descending degree (DSATUR-style starting heuristic).

    Ties broken by descending area (bigger cells first — anchors).
    """
    by_id = graph.by_id()
    degree: Dict[int, int] = defaultdict(int)
    for a, b in graph.edges:
        degree[a] += 1
        degree[b] += 1
    return sorted(
        [n.cell_id for n in graph.nodes],
        key=lambda c: (-degree[c], -by_id[c].area_px),
    )


def assign_cells_to_plates(
    cell_graph: CellGraph,
    candidate_plates: List[CandidatePlate],
    role_constraints: Optional[RoleConstraints] = None,
    use_graph_cut: bool = False,
    seed: int = 0xC0FFEE,
) -> Dict[int, int]:
    """Assign every cell to exactly one plate.

    Args:
        cell_graph: Stage 1 proposal.
        candidate_plates: plates the solver may fill. There must be >= 1 plate
            per distinct role appearing in `cell_graph.nodes`.
        role_constraints: per-role caps. Defaults to RoleConstraints().
        use_graph_cut: if True (V2), refine with networkx min-cut. V1: ignored.
        seed: deterministic tie-breaker.

    Returns:
        dict[cell_id -> plate_id]. Length == len(cell_graph.nodes).

    Raises:
        ValueError: if any role in cell_graph has zero candidate plates.
    """
    constraints = role_constraints or RoleConstraints()
    rng = np.random.default_rng(seed)

    # ---- 0. Validate: every role present in cells has at least one plate
    cell_roles = {n.role for n in cell_graph.nodes}
    plate_roles = {p.role for p in candidate_plates}
    missing = cell_roles - plate_roles
    if missing:
        raise ValueError(
            f"No candidate plate for roles: {missing}. "
            f"Need at least one CandidatePlate per role in cell_graph."
        )

    # ---- 1. Bucket candidate plates by role
    plates_by_role: Dict[str, List[CandidatePlate]] = defaultdict(list)
    for cp in candidate_plates:
        plates_by_role[cp.role].append(cp)

    # ---- 2. Walk cells in DSATUR order
    order = _dsatur_order(cell_graph)
    by_id = cell_graph.by_id()

    assignment: Dict[int, int] = {}
    # Running area & count per plate
    plate_area: Dict[int, int] = defaultdict(int)
    plate_count: Dict[int, int] = defaultdict(int)

    for cell_id in order:
        cell = by_id[cell_id]
        candidates = plates_by_role[cell.role]

        # Filter by area cap
        def _allowed(p: CandidatePlate) -> bool:
            if p.max_area_px is not None and plate_area[p.plate_id] + cell.area_px > p.max_area_px:
                return False
            if (
                constraints.max_cells_per_plate is not None
                and plate_count[p.plate_id] + 1 > constraints.max_cells_per_plate
            ):
                return False
            return True

        allowed = [p for p in candidates if _allowed(p)]
        if not allowed:
            # Fall back to least-filled plate of the right role (soft over-cap)
            allowed = sorted(candidates, key=lambda p: plate_area[p.plate_id])

        # Pick best by graph-cut tension; deterministic tie-break by plate_id
        n_role_plates = len(candidates)
        scored = [
            (
                _assignment_tension(
                    cell,
                    p,
                    assignment,
                    cell_graph,
                    plate_area,
                    plate_count,
                    n_role_plates,
                ),
                p.plate_id,
                p,
            )
            for p in allowed
        ]
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0][2]
        assignment[cell_id] = best.plate_id
        plate_area[best.plate_id] += cell.area_px
        plate_count[best.plate_id] += 1

    # ---- 3. Sanity: every cell has an assignment
    assert len(assignment) == len(cell_graph.nodes), (
        f"orphans! assigned {len(assignment)} of {len(cell_graph.nodes)}"
    )

    # ---- 4. Optional graph-cut refinement (V2)
    if use_graph_cut:
        try:
            assignment = _refine_with_min_cut(assignment, cell_graph, candidate_plates)
        except Exception:
            # min-cut refinement is best-effort; greedy answer stands
            pass

    return assignment


def _refine_with_min_cut(
    initial: Dict[int, int],
    graph: CellGraph,
    candidates: List[CandidatePlate],
) -> Dict[int, int]:
    """V2 stub — currently a no-op pass-through.

    A full min-cut refinement would build, per pair of adjacent plates of
    the same role, a 2-coloring problem and run networkx maximum_flow.
    Left intentionally simple for V1; greedy DSATUR already satisfies the
    "all cells assigned, role-respecting" gate.
    """
    return initial


def assignment_to_cells_by_plate(
    assignment: Dict[int, int],
) -> Dict[int, List[int]]:
    """Inverse view: {plate_id -> [cell_id, ...]}."""
    out: Dict[int, List[int]] = defaultdict(list)
    for cell_id, plate_id in assignment.items():
        out[plate_id].append(cell_id)
    # Sort cell ids for deterministic ordering
    for p in out:
        out[p].sort()
    return dict(out)
