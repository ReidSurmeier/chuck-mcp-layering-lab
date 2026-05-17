"""SNIC-real superpixel cell proposer for v4 production_plan_builder.

Replaces the fixed-grid placeholder in `chuck_mcp_v2.plan_emma._grid_cell_graph`
with a real, image-driven SNIC/SLIC superpixel proposal.

Public API: :func:`snic_proposer.propose_cells`.
"""
