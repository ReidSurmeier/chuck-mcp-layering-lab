"""TDD Cycle 1 — MediaPipe runs on real Emma from the v5 wrapper.

Failing first: the wrapper module `face_spatial` does not yet exist at the
v5-overnight level. Once `face_spatial.extract_face_regions` is implemented,
this test must pass against the canonical Chuck Close Emma image.

Acceptance:
- >= 19 regions populated (v3 vocabulary)
- every region has a binary mask whose shape matches the source image (H, W)
- at least one region has source field containing "mediapipe" (proves the
  Chuck Close gauss21 blur cascade fired — fallback path is NOT acceptable
  for Emma which is the canonical input)
"""
from __future__ import annotations
import cv2
import pytest
from pathlib import Path


EMMA_PATH = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")


@pytest.fixture(scope="module")
def emma_path() -> Path:
    if not EMMA_PATH.exists():
        pytest.skip(f"Emma input image not found at {EMMA_PATH}")
    return EMMA_PATH


@pytest.fixture(scope="module")
def emma_shape(emma_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(emma_path))
    assert img is not None, f"cv2 could not read {emma_path}"
    return img.shape[:2]


def test_mediapipe_extracts_19_regions_on_emma(emma_path: Path, emma_shape: tuple[int, int]) -> None:
    """The v5 wrapper must expose `extract_face_regions` and the v3 vocabulary
    (19 named regions) must all be populated for the canonical Emma input.
    """
    import face_spatial  # the module we are about to write

    regions = face_spatial.extract_face_regions(str(emma_path))

    assert isinstance(regions, dict), "extract_face_regions must return a dict"
    assert len(regions) >= 19, (
        f"expected >=19 regions (v3 vocabulary), got {len(regions)}: "
        f"{sorted(regions)}"
    )

    H, W = emma_shape
    for name, region in regions.items():
        mask = region.mask
        assert mask is not None, f"region {name!r} has no mask"
        assert mask.shape == (H, W), (
            f"region {name!r} mask shape={mask.shape} != image ({H},{W})"
        )

    # Chuck Close gauss21 fallback must have fired — not the heuristic ellipse.
    sources = {r.source for r in regions.values()}
    mp_sources = [s for s in sources if "mediapipe" in s.lower() or "selfie" in s.lower()]
    assert mp_sources, (
        f"no mediapipe / selfie source recorded — fallback heuristic path was "
        f"taken on Emma, which is NOT acceptable (Emma must hit the gauss21 "
        f"cascade). Sources observed: {sources}"
    )


def test_mediapipe_records_gauss_strategy_on_emma(emma_path: Path) -> None:
    """The Chuck Close σ=21 blur cascade is the load-bearing finding from v3.
    At least one region must record the gauss strategy in its `source` string
    so downstream debugging can confirm the cascade fired.
    """
    import face_spatial

    regions = face_spatial.extract_face_regions(str(emma_path))
    sources = " ".join(r.source for r in regions.values())
    assert "gauss" in sources.lower(), (
        f"no gauss strategy recorded on Emma — cascade did not fire. "
        f"Sources: {sorted({r.source for r in regions.values()})}"
    )
