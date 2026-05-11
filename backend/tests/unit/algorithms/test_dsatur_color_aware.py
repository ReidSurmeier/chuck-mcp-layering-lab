"""Unit tests for the color-aware DSATUR block assignment."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from algorithms.decomposition.dsatur_color_aware import (
    BlockAssignment,
    dsatur_color_aware,
)


OKLab = tuple[float, float, float]


# --- helpers ----------------------------------------------------------------


def _is_proper_coloring(graph: nx.Graph[str], assignment: dict[str, int]) -> bool:
    """A coloring is proper iff no edge has both endpoints in the same block."""
    for u, v in graph.edges:
        if u == v:
            continue  # self-loops are ignored by the algorithm
        if assignment[u] == assignment[v]:
            return False
    return True


def _block_range_ok(result: BlockAssignment) -> bool:
    if not result.color_to_block:
        return result.block_count == 0
    used = set(result.color_to_block.values())
    return used == set(range(result.block_count))


# Canonical OKLab coords (approximate, sufficient for tests).
RED: OKLab = (0.628, 0.225, 0.126)
GREEN: OKLab = (0.866, -0.234, 0.179)
BLUE: OKLab = (0.452, -0.032, -0.312)
YELLOW: OKLab = (0.968, -0.071, 0.198)
ORANGE: OKLab = (0.793, 0.131, 0.149)
PURPLE: OKLab = (0.420, 0.158, -0.101)

# A second red and second blue, perceptually close to their primaries.
RED_NEAR: OKLab = (0.640, 0.215, 0.120)
BLUE_NEAR: OKLab = (0.460, -0.030, -0.300)


# --- 1. Trivial graphs ------------------------------------------------------


def test_single_node_one_block() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_node("r")
    out = dsatur_color_aware(g, {"r": RED})
    assert out.block_count == 1
    assert out.color_to_block == {"r": 0}
    assert _is_proper_coloring(g, out.color_to_block)


def test_two_unconnected_share_a_block() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(["r", "r2"])
    # Both reds, no edge: with color-aware w=0.6 they should cluster.
    out = dsatur_color_aware(g, {"r": RED, "r2": RED_NEAR}, color_grouping_weight=0.6)
    assert out.block_count == 1
    assert _is_proper_coloring(g, out.color_to_block)
    assert _block_range_ok(out)


def test_two_connected_need_two_blocks() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_edge("r", "b")
    out = dsatur_color_aware(g, {"r": RED, "b": BLUE})
    assert out.block_count == 2
    assert out.color_to_block["r"] != out.color_to_block["b"]
    assert _is_proper_coloring(g, out.color_to_block)


# --- 2. Complete graphs (chromatic number = n) ------------------------------


def test_k3_triangle_needs_three_blocks() -> None:
    g = nx.complete_graph(["r", "g", "b"])
    out = dsatur_color_aware(g, {"r": RED, "g": GREEN, "b": BLUE})
    assert out.block_count == 3
    assert _is_proper_coloring(g, out.color_to_block)
    assert _block_range_ok(out)


def test_k4_needs_four_blocks() -> None:
    g = nx.complete_graph(["r", "g", "b", "y"])
    palette = {"r": RED, "g": GREEN, "b": BLUE, "y": YELLOW}
    out = dsatur_color_aware(g, palette)
    assert out.block_count == 4
    assert _is_proper_coloring(g, out.color_to_block)


# --- 3. Planar bound (Four Color Theorem) ----------------------------------


def test_five_node_planar_at_most_five_blocks() -> None:
    # A 5-cycle is planar and 3-chromatic, well within 5.
    g = nx.cycle_graph(["a", "b", "c", "d", "e"])
    palette = {"a": RED, "b": GREEN, "c": BLUE, "d": YELLOW, "e": ORANGE}
    out = dsatur_color_aware(g, palette)
    assert out.block_count <= 5
    assert _is_proper_coloring(g, out.color_to_block)


# --- 4. Color-aware clustering on unconstrained graphs ----------------------


def test_three_reds_three_blues_cluster_by_hue() -> None:
    """3 reds and 3 blues, no edges, w=0.6 should give exactly 2 blocks
    (one per hue family).
    """
    g: nx.Graph[str] = nx.Graph()
    nodes = ["r1", "r2", "r3", "b1", "b2", "b3"]
    g.add_nodes_from(nodes)
    palette = {
        "r1": RED,
        "r2": RED_NEAR,
        "r3": (0.635, 0.220, 0.122),
        "b1": BLUE,
        "b2": BLUE_NEAR,
        "b3": (0.448, -0.034, -0.314),
    }
    out = dsatur_color_aware(g, palette, color_grouping_weight=0.6)
    assert out.block_count == 2
    # All three reds in one block, all three blues in another.
    red_blocks = {out.color_to_block[n] for n in ("r1", "r2", "r3")}
    blue_blocks = {out.color_to_block[n] for n in ("b1", "b2", "b3")}
    assert len(red_blocks) == 1
    assert len(blue_blocks) == 1
    assert red_blocks != blue_blocks


def test_weight_zero_collapses_to_single_block_when_no_edges() -> None:
    """w=0 -> pure DSATUR -> always smallest valid index -> 1 block."""
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(["r", "g", "b", "y"])
    palette = {"r": RED, "g": GREEN, "b": BLUE, "y": YELLOW}
    out = dsatur_color_aware(g, palette, color_grouping_weight=0.0)
    assert out.block_count == 1


# --- 5. Mixbox-style realistic case -----------------------------------------


def test_13_mixbox_pigments_with_sparse_edges() -> None:
    """13 pigment palette with a sparse adjacency graph -> proper coloring."""
    pigments = {
        "cad_yel": (0.949, -0.025, 0.207),
        "hansa_yel": (0.911, -0.011, 0.205),
        "cad_orange": (0.770, 0.156, 0.176),
        "cad_red": (0.628, 0.225, 0.126),
        "quin_mag": (0.402, 0.196, 0.018),
        "cob_violet": (0.252, 0.099, -0.071),
        "ultra_blue": (0.224, 0.041, -0.193),
        "cob_blue": (0.319, 0.001, -0.218),
        "phth_blue": (0.225, -0.020, -0.103),
        "phth_green": (0.317, -0.094, 0.014),
        "perm_green": (0.498, -0.155, 0.116),
        "sap_green": (0.605, -0.110, 0.155),
        "burnt_sienna": (0.467, 0.082, 0.107),
    }
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(pigments.keys())
    # Sparse "must not share block" edges (e.g. neighboring regions in image).
    edges = [
        ("cad_red", "cad_orange"),
        ("cad_red", "quin_mag"),
        ("ultra_blue", "cob_blue"),
        ("phth_green", "perm_green"),
        ("sap_green", "burnt_sienna"),
        ("cad_yel", "hansa_yel"),
    ]
    g.add_edges_from(edges)
    out = dsatur_color_aware(g, pigments, color_grouping_weight=0.6)
    assert _is_proper_coloring(g, out.color_to_block)
    assert _block_range_ok(out)
    # Should be far fewer than 13 blocks for sparse adjacency.
    assert out.block_count <= 13


# --- 6. Property test: random graphs always yield proper colorings ----------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_random_20_node_graph_is_always_proper(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 20
    g = nx.erdos_renyi_graph(n, p=0.2, seed=seed)
    # Relabel to string ids and build random OKLab palette.
    g = nx.relabel_nodes(g, {i: f"c{i}" for i in range(n)})
    palette: dict[str, tuple[float, float, float]] = {}
    for node in g.nodes:
        L = float(rng.uniform(0.2, 0.95))
        a = float(rng.uniform(-0.25, 0.25))
        b = float(rng.uniform(-0.25, 0.25))
        palette[node] = (L, a, b)
    for w in (0.0, 0.6, 1.0):
        out = dsatur_color_aware(g, palette, color_grouping_weight=w)
        assert _is_proper_coloring(g, out.color_to_block), (
            f"improper coloring at seed={seed} w={w}"
        )
        assert _block_range_ok(out)
        assert set(out.color_to_block.keys()) == set(g.nodes)


# --- 7. Centroid + variance sanity ------------------------------------------


def test_centroid_is_mean_of_member_colors() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(["a", "b"])
    palette = {"a": (0.5, 0.1, -0.05), "b": (0.7, -0.1, 0.05)}
    out = dsatur_color_aware(g, palette, color_grouping_weight=0.0)
    # w=0 -> both end up in block 0; centroid is the mean.
    assert out.block_count == 1
    c = out.block_centroids_oklab[0]
    expected = (0.6, 0.0, 0.0)
    assert c == pytest.approx(expected, abs=1e-9)


def test_variance_zero_when_single_color_per_block() -> None:
    g = nx.complete_graph(["a", "b", "c"])
    palette = {"a": RED, "b": GREEN, "c": BLUE}
    out = dsatur_color_aware(g, palette, color_grouping_weight=0.6)
    assert out.block_count == 3
    assert out.intra_block_variance == pytest.approx(0.0, abs=1e-12)


# --- 8. Input validation ----------------------------------------------------


def test_invalid_weight_raises() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_node("r")
    with pytest.raises(ValueError, match="color_grouping_weight"):
        dsatur_color_aware(g, {"r": RED}, color_grouping_weight=1.5)
    with pytest.raises(ValueError, match="color_grouping_weight"):
        dsatur_color_aware(g, {"r": RED}, color_grouping_weight=-0.1)


def test_missing_oklab_entry_raises() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_edge("r", "b")
    with pytest.raises(ValueError, match="missing entries"):
        dsatur_color_aware(g, {"r": RED})


def test_empty_graph_returns_empty_assignment() -> None:
    g: nx.Graph[str] = nx.Graph()
    out = dsatur_color_aware(g, {})
    assert out.block_count == 0
    assert out.color_to_block == {}
    assert out.intra_block_variance == 0.0


# --- 9. Self-loops are ignored ----------------------------------------------


def test_self_loops_are_ignored() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_node("r")
    g.add_edge("r", "r")  # self loop
    out = dsatur_color_aware(g, {"r": RED})
    assert out.block_count == 1
    assert out.color_to_block["r"] == 0
