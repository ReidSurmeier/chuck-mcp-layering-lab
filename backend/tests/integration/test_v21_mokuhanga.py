"""Integration tests for the V21 mokuhanga separation orchestrator.

These tests will be ``skipped`` automatically until every sibling
module the orchestrator imports has landed. Once all imports resolve
the body runs end-to-end on a real corpus image and a tiny synthetic
sanity image.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

# Defer the orchestrator import: sibling agents may still be building
# the algorithm modules. If any import fails we skip the whole module
# rather than crash collection.
pytestmark: pytest.MarkDecorator = pytest.mark.skipif(False, reason="")

separate_mokuhanga: Any
V21Params: Any
V21Result: Any
try:
    from services.separation_v21_mokuhanga import (  # type: ignore[import-not-found]
        V21Params as _V21Params,
    )
    from services.separation_v21_mokuhanga import (
        V21Result as _V21Result,
    )
    from services.separation_v21_mokuhanga import (
        separate_mokuhanga as _separate,
    )

    separate_mokuhanga = _separate
    V21Params = _V21Params
    V21Result = _V21Result
    _IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 — surface any import failure cleanly
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _find_corpus_root() -> Path | None:
    import os
    env = os.environ.get("WOODBLOCK_REPO_ROOT")
    if env:
        p = Path(env)
        return p if (p / "corpus").is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "corpus").is_dir():
            return parent
    return None

_REPO_ROOT = _find_corpus_root()
_CORPUS_IMAGE = (
    _REPO_ROOT / "corpus" / "hiroshige_edo_116" / "original.jpeg"
    if _REPO_ROOT else Path("/_missing_fixture")
)


pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"v21 sibling module not ready: {_IMPORT_ERROR}",
)


def _synthetic_image(w: int = 100, h: int = 100) -> bytes:
    """4-color blocky synthetic image — exercises pipeline end-to-end."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = (220, 40, 40)  # red
    arr[: h // 2, w // 2 :] = (40, 200, 60)  # green
    arr[h // 2 :, : w // 2] = (40, 60, 220)  # blue
    arr[h // 2 :, w // 2 :] = (240, 220, 50)  # yellow
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _assert_valid_zip(zip_bytes: bytes) -> zipfile.ZipFile:
    assert isinstance(zip_bytes, (bytes, bytearray))
    assert len(zip_bytes) > 0
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    assert zf.testzip() is None
    return zf


def _names_with_prefix(zf: zipfile.ZipFile, prefix: str) -> list[str]:
    return [n for n in zf.namelist() if n.startswith(prefix)]


def test_synthetic_pipeline_runs_end_to_end() -> None:
    image_bytes = _synthetic_image()
    result = separate_mokuhanga(image_bytes)

    assert isinstance(result, V21Result)
    assert result.block_count >= 1
    assert result.pigment_count >= 1
    assert result.pigment_count <= 13

    zf = _assert_valid_zip(result.zip_bytes)
    names = zf.namelist()
    assert "composite.png" in names
    assert "manifest.json" in names
    assert "print_order.csv" in names
    assert _names_with_prefix(zf, "png/")
    assert _names_with_prefix(zf, "svg/")
    assert _names_with_prefix(zf, "blocks/")
    assert _names_with_prefix(zf, "impressions/")

    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    editor = manifest.get("editor", {})
    assert editor.get("mode") == "v21_mokuhanga"


@pytest.mark.skipif(
    not _CORPUS_IMAGE.exists(),
    reason=f"corpus image missing: {_CORPUS_IMAGE}",
)
def test_corpus_hiroshige_pipeline() -> None:
    image_bytes = _CORPUS_IMAGE.read_bytes()
    params = V21Params()
    result = separate_mokuhanga(image_bytes, params=params)

    assert isinstance(result, V21Result)
    assert result.block_count >= 1
    assert 1 <= result.pigment_count <= 13
    # Loose threshold for first MVP iteration.
    assert result.reconstruction_dE_mean < 30.0

    zf = _assert_valid_zip(result.zip_bytes)
    names = zf.namelist()
    assert "composite.png" in names
    assert "manifest.json" in names
    assert "print_order.csv" in names
    assert _names_with_prefix(zf, "png/")
    assert _names_with_prefix(zf, "svg/")
    assert _names_with_prefix(zf, "blocks/")
    assert _names_with_prefix(zf, "impressions/")

    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    assert manifest["editor"]["mode"] == "v21_mokuhanga"
