"""Pytest config: expose the v5-overnight mokuhanga-pigments package plus the
canonical v3 mokuhanga-rule-classifier and chuck_mcp_v2 packages on sys.path.

This mirrors the conftest style used by other v5-overnight siblings.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
V3_RULE_DIR = REPO_ROOT / "research" / "v3-construction" / "mokuhanga-rule-classifier"
PROD_SOLVER_DIR = REPO_ROOT / "research" / "v4-build" / "production-solver"
HYBRID_DIR = REPO_ROOT / "research" / "v4-build" / "hybrid-optimizer"

for p in (HERE, V3_RULE_DIR, PROD_SOLVER_DIR, HYBRID_DIR, REPO_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
