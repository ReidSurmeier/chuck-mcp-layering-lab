"""DSATUR graph coloring with OKLab color-aware tiebreaking.

Standard DSATUR (Brélaz 1979, "New methods to color the vertices of a
graph", CACM 22(4)) greedily colors vertices in order of saturation
degree -- the number of distinct colors already used by neighbors --
breaking ties by graph degree. The choice of *which* color to assign,
once a vertex is selected, is "first valid index". For block-printing
plate assignment we want a perceptual choice instead: among the valid
plates, prefer the one whose existing members are closest in OKLab
(Ottosson 2020) to the new vertex's color. This produces plates that
are not only adjacency-valid but also visually coherent, which matters
because a single block in a reduction print is inked with a single
pigment.

networkx ships ``greedy_color`` with strategy="DSATUR" but exposes no
hook for color choice. This module is a clean reimplementation that
adds a single hook (``_choose_block``) for the color-aware tiebreak.

The ``color_grouping_weight`` parameter (``w``) interpolates:

* ``w = 0.0`` -- pure DSATUR. Always pick the smallest valid block.
* ``w = 1.0`` -- pure color grouping. Open a new block aggressively
  whenever the nearest valid plate's centroid is more than a tight
  OKLab threshold away.
* ``w = 0.6`` -- recommended default. Color-coherence dominates but
  the algorithm still reuses blocks when colors are similar.

Threshold formula: ``tau = 0.15 * (1 - w) + 0.04 * w``. At ``w = 0.5``
this is ``~0.095`` (perceptually noticeable difference); at ``w = 1.0``
it drops to ``0.04`` (just-noticeable). The gate ``w >= 0.5`` ensures
we never *open* extra blocks in DSATUR-leaning mode, even though we
still bias the tiebreak.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

__all__ = [
    "BlockAssignment",
    "dsatur_color_aware",
]


OKLab = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BlockAssignment:
    """Result of a color-aware DSATUR coloring.

    Attributes:
        color_to_block: Mapping from the graph's node id (a color key,
            typically a hex string or pigment id) to the assigned block
            index in ``[0, block_count)``.
        block_count: Number of distinct blocks used.
        block_centroids_oklab: Mapping from block index to the mean
            OKLab coordinate of all colors assigned to that block.
        intra_block_variance: Average squared OKLab distance from each
            assigned color to its block's centroid. Lower is better:
            ``0.0`` means every block contains a single color. Useful
            as a quality metric when comparing weights.
    """

    color_to_block: dict[str, int]
    block_count: int
    block_centroids_oklab: dict[int, tuple[float, float, float]]
    intra_block_variance: float


def _delta_e_oklab(a: OKLab, b: OKLab) -> float:
    """Euclidean distance in OKLab. Ottosson 2020 shows this is a
    reasonable perceptual ``deltaE`` for moderate color differences.
    """
    dl = a[0] - b[0]
    da = a[1] - b[1]
    db = a[2] - b[2]
    return float(np.sqrt(dl * dl + da * da + db * db))


def _threshold_for_weight(w: float) -> float:
    """Linear interpolation between the loose DSATUR threshold (0.15)
    and the tight color-grouping threshold (0.04). See module docstring.
    """
    return 0.15 * (1.0 - w) + 0.04 * w


def _pick_next_vertex(
    uncolored: set[str],
    saturation: dict[str, int],
    graph: nx.Graph[str],
) -> str:
    """DSATUR selection rule: max saturation, tiebreak by max graph
    degree, final tiebreak by node id for determinism.
    """
    best: str | None = None
    best_key: tuple[int, int, str] | None = None
    for v in uncolored:
        key = (saturation[v], graph.degree(v), v)
        if best_key is None or key > best_key:
            best_key = key
            best = v
    assert best is not None, "called with empty uncolored set"
    return best


def _choose_block(
    vertex: str,
    valid_blocks: list[int],
    vertex_color: OKLab,
    centroids: dict[int, OKLab],
    weight: float,
) -> tuple[int, bool]:
    """Pick a block index for ``vertex`` from the adjacency-valid set.

    Returns ``(block_index, opened_new)``. If ``valid_blocks`` is
    empty, or the color-aware gate triggers, a new block index is
    returned with ``opened_new=True``.
    """
    # Standard DSATUR with w == 0: always smallest valid index.
    if weight <= 0.0:
        if not valid_blocks:
            return (len(centroids), True)
        return (min(valid_blocks), False)

    # If no valid existing block, we must open a new one regardless.
    if not valid_blocks:
        return (len(centroids), True)

    # Score each valid block by OKLab distance to vertex's color.
    distances = [(_delta_e_oklab(vertex_color, centroids[b]), b) for b in valid_blocks]
    distances.sort()  # ascending distance, tiebreak ascending block index
    best_dist, best_block = distances[0]

    # Color-grouping gate: only active when weight >= 0.5.
    if weight >= 0.5:
        tau = _threshold_for_weight(weight)
        if best_dist > tau:
            return (len(centroids), True)

    return (best_block, False)


def _validate_inputs(
    graph: nx.Graph[str],
    color_oklab: dict[str, tuple[float, float, float]],
    weight: float,
) -> None:
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"color_grouping_weight must be in [0, 1], got {weight}")
    missing = [v for v in graph.nodes if v not in color_oklab]
    if missing:
        raise ValueError(
            f"color_oklab missing entries for {len(missing)} node(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


def dsatur_color_aware(
    graph: nx.Graph[str],
    color_oklab: dict[str, tuple[float, float, float]],
    color_grouping_weight: float = 0.6,
) -> BlockAssignment:
    """Custom DSATUR with an OKLab color-aware tiebreak hook.

    Args:
        graph: An undirected ``networkx.Graph`` whose nodes are color
            keys (e.g. hex strings, pigment ids) and whose edges mean
            "these two colors must not share a block". Self-loops are
            ignored.
        color_oklab: A mapping from every node in ``graph`` to its
            OKLab coordinate ``(L, a, b)``.
        color_grouping_weight: ``w`` in ``[0, 1]``. See module docstring.

    Returns:
        A ``BlockAssignment``. The coloring is guaranteed to be proper:
        no edge connects two nodes assigned to the same block.

    Raises:
        ValueError: If ``color_grouping_weight`` is outside ``[0, 1]``
            or ``color_oklab`` is missing any node in ``graph``.
    """
    _validate_inputs(graph, color_oklab, color_grouping_weight)

    nodes = list(graph.nodes)
    if not nodes:
        return BlockAssignment(
            color_to_block={},
            block_count=0,
            block_centroids_oklab={},
            intra_block_variance=0.0,
        )

    assignment: dict[str, int] = {}
    # block_index -> list of OKLab tuples (members), kept for centroid update
    members: dict[int, list[OKLab]] = {}
    centroids: dict[int, OKLab] = {}
    # neighbor_blocks[v] = set of block indices used by colored neighbors of v
    neighbor_blocks: dict[str, set[int]] = {v: set() for v in nodes}
    saturation: dict[str, int] = {v: 0 for v in nodes}
    uncolored: set[str] = set(nodes)

    while uncolored:
        v = _pick_next_vertex(uncolored, saturation, graph)
        forbidden = neighbor_blocks[v]
        valid_blocks = [b for b in centroids if b not in forbidden]
        raw = color_oklab[v]
        vertex_color: OKLab = (float(raw[0]), float(raw[1]), float(raw[2]))
        block, opened = _choose_block(
            v, valid_blocks, vertex_color, centroids, color_grouping_weight
        )

        # Commit assignment.
        assignment[v] = block
        if opened:
            members[block] = [vertex_color]
            centroids[block] = vertex_color
        else:
            members[block].append(vertex_color)
            arr = np.asarray(members[block], dtype=np.float64)
            mean = arr.mean(axis=0)
            centroids[block] = (float(mean[0]), float(mean[1]), float(mean[2]))

        # Update neighbor saturation & forbidden sets.
        for u in graph.neighbors(v):
            if u == v:
                continue  # ignore self-loops
            if block not in neighbor_blocks[u]:
                neighbor_blocks[u].add(block)
                if u in uncolored:
                    saturation[u] = len(neighbor_blocks[u])

        uncolored.remove(v)

    # Compute intra-block variance (avg squared OKLab distance to centroid).
    total_sq = 0.0
    total_count = 0
    for block, member_list in members.items():
        c = centroids[block]
        for m in member_list:
            d = _delta_e_oklab(m, c)
            total_sq += d * d
            total_count += 1
    variance = total_sq / total_count if total_count > 0 else 0.0

    return BlockAssignment(
        color_to_block=assignment,
        block_count=len(centroids),
        block_centroids_oklab=dict(centroids),
        intra_block_variance=variance,
    )
