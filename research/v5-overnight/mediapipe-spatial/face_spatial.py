"""chuck-mcp v5 — Face Spatial Wrapper

Thin re-export of the v3 `face_region_mapper` so the v5-overnight build can
import a stable name (`face_spatial`) and not depend on the v3 folder layout.

The v3 implementation carries the load-bearing Chuck Close σ=21 Gaussian-blur
cascade — DO NOT reimplement here. The wrapper exists so plan_emma can call:

    from face_spatial import extract_face_regions, FaceRegion

without leaking the v3 path hierarchy. All semantics (FaceRegion dataclass,
19-region vocabulary, gauss cascade, hair/background fallbacks) come straight
from the v3 module.

Imports are routed through importlib so we can load the v3 file with a
"-" in its parent directory (mediapipe-face-spatial) — Python's import
system cannot resolve hyphenated package names natively.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_V3_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "v3-construction"
    / "mediapipe-face-spatial"
)

# Ensure the v3 dir is on sys.path so that face_region_mapper's local imports
# (`from region_vocabulary import ...`) resolve to the v3 version, not the v5
# fallback or any stale copy.
_v3_str = str(_V3_DIR)
if _v3_str not in sys.path:
    sys.path.insert(0, _v3_str)


def _load_v3_module(module_name: str, filename: str):
    """Load a v3 sibling module by absolute path. Caches via sys.modules under
    a `v3_<module_name>` alias so the v5 module table stays clean.
    """
    alias = f"v3_{module_name}"
    if alias in sys.modules:
        return sys.modules[alias]
    path = _V3_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"v3 mediapipe module not found at {path}. "
            f"Cannot wrap face_region_mapper from v5-overnight."
        )
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v3 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Eagerly import — surface any FileNotFoundError / model-missing errors at
# import time, not at first call.
_v3_vocab = _load_v3_module("region_vocabulary", "region_vocabulary.py")
_v3_mapper = _load_v3_module("face_region_mapper", "face_region_mapper.py")
_v3_merger = _load_v3_module("merge_regions_with_cells", "merge_regions_with_cells.py")


# ----------------------------------------------------------------------------
# Public re-exports
# ----------------------------------------------------------------------------
FaceRegion = _v3_mapper.FaceRegion
extract_face_regions = _v3_mapper.extract_face_regions

REGION_VOCABULARY = _v3_vocab.REGION_VOCABULARY
list_supported_regions = _v3_vocab.list_supported_regions
resolve_region_name = _v3_vocab.resolve_region_name

# The v3 module's `merge_face_regions_with_snic_cells` is the canonical
# implementation. Re-expose alongside the geometry helper for callers that
# pre-compute SNIC cell geometry once and reuse across many region queries.
merge_face_regions_with_snic_cells = _v3_merger.merge_face_regions_with_snic_cells
compute_snic_cell_geometry = _v3_merger.compute_snic_cell_geometry
resolve_cell_to_primary_region = _v3_merger.resolve_cell_to_primary_region


__all__ = [
    "FaceRegion",
    "extract_face_regions",
    "REGION_VOCABULARY",
    "list_supported_regions",
    "resolve_region_name",
    "merge_face_regions_with_snic_cells",
    "compute_snic_cell_geometry",
    "resolve_cell_to_primary_region",
]
