"""TDD Cycle 2 — Each SNIC cell gets assigned to exactly one face region.

The function under test (`merge_cells_with_regions.assign_cells_to_regions`)
takes:
    - the dict of FaceRegion objects from `extract_face_regions`
    - a (H, W) integer label map (SLIC stand-in for SNIC during this build)

And returns:
    - dict[cell_id -> region_name]   (primary region for each cell)

Acceptance:
    - every cell_id present in the label map appears exactly once in the
      returned dict (no duplicates, no orphans)
    - every region_name is in the v3 canonical vocabulary
    - cells on the face_oval ring resolve to face-area regions, NOT to
      background (sanity check on the assignment direction)
"""
from __future__ import annotations
import cv2
import numpy as np
import pytest
from pathlib import Path


EMMA_PATH = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")


@pytest.fixture(scope="module")
def emma_path() -> Path:
    if not EMMA_PATH.exists():
        pytest.skip(f"Emma input image not found at {EMMA_PATH}")
    return EMMA_PATH


@pytest.fixture(scope="module")
def emma_slic_labels(emma_path: Path) -> np.ndarray:
    """Build a SLIC superpixel label map at low resolution (fast, deterministic).
    SLIC stands in for SNIC until the real proposer ships from `snic-real`.
    """
    from skimage.segmentation import slic
    img = cv2.imread(str(emma_path))
    # Downscale for test speed — full-res SLIC takes several seconds and is
    # not needed to exercise the assignment logic.
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    labels_small = slic(
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
        n_segments=300,
        compactness=10.0,
        start_label=0,
    ).astype(np.int32)
    # Upscale labels back to full-res via nearest-neighbor so the mask coords
    # line up with the face_region masks (which are at full image resolution).
    labels = cv2.resize(labels_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return labels.astype(np.int32)


@pytest.fixture(scope="module")
def emma_face_regions(emma_path: Path):
    import face_spatial
    return face_spatial.extract_face_regions(str(emma_path))


def test_each_snic_cell_assigned_to_one_face_region(
    emma_face_regions, emma_slic_labels
) -> None:
    """Every cell in the SLIC label map gets exactly one region label.

    Note: cells that fall entirely outside ALL face regions are allowed to
    receive the fallback label 'background' (matches v3 semantics — the
    'background' region in the vocabulary already covers everything outside
    the face/hair).
    """
    import merge_cells_with_regions as mcr
    import face_spatial

    assignment = mcr.assign_cells_to_regions(emma_face_regions, emma_slic_labels)

    assert isinstance(assignment, dict), "assign_cells_to_regions must return a dict"

    unique_cells = set(int(c) for c in np.unique(emma_slic_labels))
    assigned_cells = set(assignment.keys())

    # Exactly one entry per cell
    assert assigned_cells == unique_cells, (
        f"cell coverage mismatch: "
        f"missing={sorted(unique_cells - assigned_cells)[:5]} "
        f"extra={sorted(assigned_cells - unique_cells)[:5]}"
    )

    # Every region label is in the canonical v3 vocabulary
    vocab = set(face_spatial.list_supported_regions())
    bad = {cid: r for cid, r in assignment.items() if r not in vocab}
    assert not bad, f"cells assigned to non-vocab regions: {dict(list(bad.items())[:5])}"


def test_assignment_includes_face_regions_not_only_background(
    emma_face_regions, emma_slic_labels
) -> None:
    """Sanity: Emma is a face — assignment must include at least one face
    region (cheek/forehead/lip/eye). If everything resolves to background, the
    region masks and the SNIC labels are in different coordinate systems.
    """
    import merge_cells_with_regions as mcr
    assignment = mcr.assign_cells_to_regions(emma_face_regions, emma_slic_labels)
    face_like = {
        "left_cheek", "right_cheek", "forehead", "chin",
        "left_jaw", "right_jaw", "left_temple", "right_temple",
        "upper_lip", "lower_lip", "lips", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "nose", "face",
    }
    used = set(assignment.values())
    face_used = used & face_like
    assert face_used, (
        f"no cells assigned to face regions — coord mismatch likely. "
        f"Regions used: {sorted(used)}"
    )


def test_assignment_returns_inverse_region_to_cells(
    emma_face_regions, emma_slic_labels
) -> None:
    """The merge helper should also offer an inverse view: region -> [cell_ids]
    so plate-constraint code can look up 'all cells claimed by left_cheek'
    in O(1) without rescanning the cell dict.
    """
    import merge_cells_with_regions as mcr
    inverse = mcr.assign_regions_to_cells(emma_face_regions, emma_slic_labels)
    assert isinstance(inverse, dict)
    all_cells_in_inverse = set()
    for region, cell_ids in inverse.items():
        assert isinstance(cell_ids, list), (
            f"region {region} value must be list[int], got {type(cell_ids)}"
        )
        all_cells_in_inverse.update(cell_ids)
    unique_cells = set(int(c) for c in np.unique(emma_slic_labels))
    # inverse should cover every cell (every cell has SOME primary region)
    assert all_cells_in_inverse == unique_cells, (
        f"inverse view missing cells: "
        f"missing={sorted(unique_cells - all_cells_in_inverse)[:5]}"
    )
