"""pytest conftest — register the hyphenated dir as a python package.

The directory `research/v4-build/hybrid-optimizer/` has a hyphen in its
name, which is not a valid Python identifier. We expose the package under
the clean alias `hybrid_optimizer` so test code can `import hybrid_optimizer`.

This ALSO normalizes the situation where the package's own `__init__.py`
contains relative imports — they need a real package context to resolve.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Default tests to CPU JAX — GPU init dominates wall-time for small synth plans.
# To run on an NVIDIA GPU, set HYBRID_OPTIMIZER_JAX_PLATFORM=gpu or cuda.
_requested_platform = os.environ.get("HYBRID_OPTIMIZER_JAX_PLATFORM", "cpu")
_jax_platform = "cuda" if _requested_platform == "gpu" else _requested_platform
os.environ.setdefault("JAX_PLATFORMS", _jax_platform)
# Hide CUDA libs in test mode unless explicitly requested.
if _jax_platform == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_alias_package() -> None:
    if "hybrid_optimizer" in sys.modules:
        return
    # Pre-register child modules so that the package __init__'s relative
    # imports succeed in any order.
    pkg_spec = importlib.util.spec_from_file_location(
        "hybrid_optimizer",
        _HERE / "__init__.py",
        submodule_search_locations=[str(_HERE)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["hybrid_optimizer"] = pkg
    # We do NOT exec_module(pkg) yet — we need its submodules loaded first.
    for child in (
        "optimization_result",
        "graph_cut_assigner",
        "morphology_repair",
        "jax_continuous_solve",
        "alternating_loop",
    ):
        spec = importlib.util.spec_from_file_location(
            f"hybrid_optimizer.{child}", _HERE / f"{child}.py"
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"hybrid_optimizer.{child}"] = m
        spec.loader.exec_module(m)  # may use relative imports back into pkg
    # Now exec the __init__.py — its `from .x import Y` will find all the
    # pre-registered submodules in sys.modules.
    pkg_spec.loader.exec_module(pkg)


_install_alias_package()
