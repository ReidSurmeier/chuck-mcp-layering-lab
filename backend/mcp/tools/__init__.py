"""v23 MCP tool decorators across 7 tiers per addendum-v5.

Tier 0 — Core flow (10 tools)        → ``core.py``
Tier 1 — HITL refinement (8 tools)   → ``hitl.py``
Tier 2 — Calibration (5 tools)       → ``calibration.py``
Tier 3 — Introspection (6 tools)     → ``introspection.py``
Tier 4 — Session (4 tools)           → ``session.py``
Tier 5 — Carve handoff (3 tools)     → ``carve.py``
Tier 6 — Overlay (4 tools)           → ``overlay.py``

Total day-1 surface: 40 tools. Every tool returns ``ToolResult[T]`` with
structured ``WoodblockError`` list. Tools whose backing implementation
ships incrementally return ``degraded`` tier with code ``IMPL_PENDING``
+ a hint pointing at the substep that adds real logic.
"""
