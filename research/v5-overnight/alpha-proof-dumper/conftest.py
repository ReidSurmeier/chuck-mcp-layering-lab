"""pytest bootstrap for the alpha-proof-dumper TDD package.

Sets sys.path so we can import:
  - alpha_proof_dumper.* (this module)
  - chuck_mcp_v2.* (the pipeline we patch)
  - hybrid_optimizer.* (the OptimizationResult shape we consume)
  - production_solver.* (used by plan_emma)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force CPU JAX — GPU init dominates wall time for small synth plans.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # research/v5-overnight/alpha-proof-dumper -> repo root

# The package dir name contains a hyphen so it isn't a valid Python identifier.
# Make it importable under the alias `alpha_proof_dumper`.
import importlib.util


def _install_alpha_proof_dumper_alias() -> None:
    if "alpha_proof_dumper" in sys.modules:
        return
    pkg_init = _HERE / "__init__.py"
    if not pkg_init.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "alpha_proof_dumper",
        pkg_init,
        submodule_search_locations=[str(_HERE)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["alpha_proof_dumper"] = module
    spec.loader.exec_module(module)


_install_alpha_proof_dumper_alias()

# Make chuck_mcp_v2 + research packages importable
for sub in (
    "",  # for chuck_mcp_v2 (top of repo)
    "research/v4-build/hybrid-optimizer",
    "research/v4-build/production-solver",
    "research/v3-construction/cell-zone-renderer",
):
    p = _REPO_ROOT / sub if sub else _REPO_ROOT
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
