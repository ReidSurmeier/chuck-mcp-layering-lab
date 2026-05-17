"""
generate_ground_truth.py — produce 10 annotated SNIC-overlay portraits.

For each input portrait we:
  1. compute a SLIC superpixel ("SNIC stand-in") label image with ~600 cells
  2. run MediaPipe FaceMesh + selfie segmenter (Chuck Close sigma=21 blur
     cascade fallback) to get a region->mask dict
  3. join cells to regions via centroid-in-polygon strategy
  4. render a labeled overlay PNG (cells outlined, cell IDs printed)
  5. save ground_truth_regions.json with {region_name: [cell_ids]}

Inputs: real corpus portraits + synthesized stylized faces (when corpus
        has fewer than 10).

Outputs (under <out_dir>/<image_id>/):
    - source_image.png           original RGB
    - snic_labels.npy            int32 (H,W) cell-ID label image
    - input_overlay.png          labeled overlay sent to Opus
    - ground_truth_regions.json  {region_name: [cell_ids]}

Public API:
    generate_overlay(image_path, image_id, out_dir, n_segments=600) -> dict
    generate_dataset(out_dir, force=False) -> list[dict] (10 entries)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.segmentation import slic

# Import MediaPipe pipeline from v3-construction.
_HERE = Path(__file__).resolve().parent
_V3_FACE = _HERE.parent.parent / "v3-construction" / "mediapipe-face-spatial"
sys.path.insert(0, str(_V3_FACE))

from face_region_mapper import extract_face_regions  # noqa: E402
from merge_regions_with_cells import (  # noqa: E402
    compute_snic_cell_geometry,
    merge_face_regions_with_snic_cells,
)
from region_vocabulary import list_supported_regions  # noqa: E402


CORPUS_ROOT = Path("/home/reidsurmeier/src/chuck-mcp-layering-lab/corpus")

# Real corpus portraits that contain a face (verified by hand: emma is the
# canonical Chuck Close; the three "reid_*" portraits are Reid's own
# photo references; toy_print is a synthetic face).
CORPUS_PORTRAITS: list[tuple[str, str]] = [
    ("close_emma_2002", str(CORPUS_ROOT / "close_emma_2002" / "original.jpg")),
    ("reid_mike_portrait", str(CORPUS_ROOT / "reid_mike_portrait" / "original.jpg")),
    ("reid_untitled_01", str(CORPUS_ROOT / "reid_untitled_01" / "original.png")),
    ("reid_untitled_02", str(CORPUS_ROOT / "reid_untitled_02" / "original.png")),
    ("toy_print_face_masks", str(CORPUS_ROOT / "toy_print_face_masks" / "original.png")),
]

# We need 10 total. Synthesize the rest as stylized close-class faces so the
# benchmark covers a range of "noise levels" Opus might face in production.
SYNTH_PORTRAIT_COUNT = 10 - len(CORPUS_PORTRAITS)


# ---------------------------------------------------------------------------
# Synthetic stylized portrait generator
# ---------------------------------------------------------------------------


def _synth_portrait(seed: int, h: int = 800, w: int = 640) -> np.ndarray:
    """Generate a centered head-on portrait with face/hair/background colors.

    Style is deliberately Chuck-Close-ish — soft regions, dot-mosaic
    texture, identifiable face oval, hair crown, eye/lip blobs — so
    MediaPipe FaceLandmarker can lock on with the sigma=21 blur cascade.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # background (warm gradient)
    bg = rng.integers(60, 200, size=3)
    img[:] = bg

    cx, cy = w // 2, int(h * 0.55)
    face_w, face_h = int(w * 0.50), int(h * 0.60)

    # hair behind face
    hair_color = rng.integers(20, 80, size=3).tolist()
    cv2.ellipse(img, (cx, cy - face_h // 6),
                (int(face_w * 0.75), int(face_h * 0.6)),
                0, 180, 360, hair_color, -1)

    # face oval
    face_color = rng.integers(160, 230, size=3).tolist()
    cv2.ellipse(img, (cx, cy), (face_w // 2, face_h // 2),
                0, 0, 360, face_color, -1)

    # cheeks (subtle pink)
    cheek_color = [min(255, c + 20) for c in face_color]
    cheek_color[2] = min(255, cheek_color[2] + 30)  # red bump
    cv2.circle(img, (cx - face_w // 4, cy + face_h // 12), face_w // 9,
               cheek_color, -1)
    cv2.circle(img, (cx + face_w // 4, cy + face_h // 12), face_w // 9,
               cheek_color, -1)

    # eyes
    eye_y = cy - face_h // 6
    eye_dx = face_w // 5
    eye_color = [40, 40, 40]
    cv2.ellipse(img, (cx - eye_dx, eye_y), (face_w // 14, face_h // 28),
                0, 0, 360, eye_color, -1)
    cv2.ellipse(img, (cx + eye_dx, eye_y), (face_w // 14, face_h // 28),
                0, 0, 360, eye_color, -1)
    # irises (small dark)
    cv2.circle(img, (cx - eye_dx, eye_y), max(2, face_w // 35), (15, 25, 60), -1)
    cv2.circle(img, (cx + eye_dx, eye_y), max(2, face_w // 35), (15, 25, 60), -1)

    # eyebrows
    brow_color = [30, 25, 20]
    cv2.ellipse(img, (cx - eye_dx, eye_y - face_h // 14),
                (face_w // 10, face_h // 50), 0, 0, 180, brow_color, -1)
    cv2.ellipse(img, (cx + eye_dx, eye_y - face_h // 14),
                (face_w // 10, face_h // 50), 0, 0, 180, brow_color, -1)

    # nose (subtle shadow blob)
    nose_color = [int(c * 0.88) for c in face_color]
    cv2.ellipse(img, (cx, cy), (face_w // 18, face_h // 12),
                0, 0, 360, nose_color, -1)

    # lips
    lip_color = [60, 60, 180]
    cv2.ellipse(img, (cx, cy + face_h // 5), (face_w // 7, face_h // 22),
                0, 0, 360, lip_color, -1)
    # upper lip (slightly darker)
    cv2.ellipse(img, (cx, cy + face_h // 5 - face_h // 50),
                (face_w // 8, face_h // 40),
                0, 180, 360, [int(c * 0.85) for c in lip_color], -1)

    # Chuck-Close-style dot mosaic noise: random small colored circles
    # across the whole canvas (sparse, low contrast) so the image isn't
    # a perfect cartoon — closer to printable-grade source material.
    n_dots = (h * w) // 200
    for _ in range(n_dots):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        c = rng.integers(0, 256, size=3).tolist()
        # Blend toward existing pixel — keeps mosaic but doesn't destroy oval.
        existing = img[y, x].astype(int)
        blended = ((existing + np.array(c)) // 2).tolist()
        cv2.circle(img, (x, y), r, blended, -1)

    # Gentle blur so MediaPipe's sigma=21 cascade can lock landmarks.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------


def _render_overlay(
    image_bgr: np.ndarray,
    snic_labels: np.ndarray,
    *,
    font_scale: float = 0.32,
    line_thickness: int = 1,
) -> np.ndarray:
    """Draw cell boundaries + cell IDs on top of the source image.

    Output is BGR uint8 of same shape. Used as Opus's vision input.
    """
    H, W = snic_labels.shape
    overlay = image_bgr.copy()

    # Faded background so labels pop.
    overlay = cv2.addWeighted(overlay, 0.55, np.full_like(overlay, 255), 0.45, 0)

    # cell boundaries via gradient on label map
    sx = np.zeros_like(snic_labels, dtype=np.int32)
    sy = np.zeros_like(snic_labels, dtype=np.int32)
    sx[:, 1:] = (snic_labels[:, 1:] != snic_labels[:, :-1]).astype(np.int32)
    sy[1:, :] = (snic_labels[1:, :] != snic_labels[:-1, :]).astype(np.int32)
    edges = (sx + sy) > 0
    overlay[edges] = [0, 0, 0]

    # cell-ID labels (only on cells big enough to fit text)
    geometry = compute_snic_cell_geometry(snic_labels)
    for cid, g in geometry.items():
        if g.pixel_count < 90:
            continue
        cx, cy = int(g.centroid[0]), int(g.centroid[1])
        text = str(cid)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, line_thickness)
        x = max(0, min(W - tw, cx - tw // 2))
        y = max(th, min(H - 1, cy + th // 2))
        # white halo + black text — readable against any cell color
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), line_thickness + 2,
                    cv2.LINE_AA)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), line_thickness, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthEntry:
    image_id: str
    source_path: Path
    overlay_path: Path
    snic_path: Path
    regions_path: Path
    n_cells: int
    n_regions: int


def generate_overlay(
    image_path: str | Path,
    image_id: str,
    out_dir: str | Path,
    *,
    n_segments: int = 600,
    compactness: float = 10.0,
) -> GroundTruthEntry:
    """Produce one ground-truth bundle for a single portrait."""
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(src)
    out_root = Path(out_dir) / image_id
    out_root.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(src))
    if image_bgr is None:
        raise RuntimeError(f"could not read image: {src}")

    # Cache a stable copy under out_dir so the benchmark is reproducible
    # without the corpus.
    source_copy = out_root / "source_image.png"
    cv2.imwrite(str(source_copy), image_bgr)

    # 1. SLIC superpixels (SNIC stand-in)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    labels = slic(img_rgb, n_segments=n_segments,
                  compactness=compactness, start_label=0).astype(np.int32)
    np.save(out_root / "snic_labels.npy", labels)

    # 2. MediaPipe face regions
    regions = extract_face_regions(str(source_copy))

    # 3. Merge cells <-> regions (overlap strategy = more precise for boundary
    #    cells; centroid is faster but produces sharper boundary errors that
    #    blow up Jaccard scores during the benchmark calibration).
    geometry = compute_snic_cell_geometry(labels)
    cell_assign = merge_face_regions_with_snic_cells(
        regions, labels, strategy="overlap",
        overlap_threshold=0.5, precomputed_geometry=geometry,
    )

    # 4. Filter to canonical regions; serialize.
    canonical = set(list_supported_regions())
    gt = {name: sorted(int(c) for c in cells)
          for name, cells in cell_assign.items() if name in canonical}

    # 5. Overlay PNG for Opus.
    overlay = _render_overlay(image_bgr, labels)
    overlay_path = out_root / "input_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    regions_path = out_root / "ground_truth_regions.json"
    regions_path.write_text(json.dumps({
        "image_id": image_id,
        "image_shape": list(image_bgr.shape),
        "n_cells": int(labels.max() + 1),
        "region_vocabulary_size": len(canonical),
        "regions": gt,
    }, indent=2))

    return GroundTruthEntry(
        image_id=image_id,
        source_path=source_copy,
        overlay_path=overlay_path,
        snic_path=out_root / "snic_labels.npy",
        regions_path=regions_path,
        n_cells=int(labels.max() + 1),
        n_regions=len(gt),
    )


def generate_dataset(out_dir: str | Path, *, force: bool = False
                     ) -> list[GroundTruthEntry]:
    """Build the 10-overlay benchmark dataset."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[GroundTruthEntry] = []

    # Corpus portraits first
    used_ids: set[str] = set()
    for image_id, src in CORPUS_PORTRAITS:
        used_ids.add(image_id)
        target_regions = out_root / image_id / "ground_truth_regions.json"
        if target_regions.exists() and not force:
            entries.append(_entry_from_existing(image_id, out_root))
            continue
        try:
            entries.append(generate_overlay(src, image_id, out_root))
        except Exception as exc:  # pragma: no cover — corpus-image failure
            print(f"[gt] failed on corpus {image_id}: {exc}", file=sys.stderr)

    # Synthesized portraits fill the rest.
    synth_index = 0
    while len(entries) < 10 and synth_index < 50:  # cap retries
        image_id = f"synth_face_{synth_index:02d}"
        synth_index += 1
        if image_id in used_ids:
            continue
        used_ids.add(image_id)

        target_regions = out_root / image_id / "ground_truth_regions.json"
        if target_regions.exists() and not force:
            entries.append(_entry_from_existing(image_id, out_root))
            continue

        synth = _synth_portrait(seed=1000 + synth_index)
        synth_path = out_root / f"_tmp_{image_id}.png"
        cv2.imwrite(str(synth_path), synth)
        try:
            entries.append(generate_overlay(synth_path, image_id, out_root))
        except Exception as exc:  # pragma: no cover
            print(f"[gt] failed on synth {image_id}: {exc}", file=sys.stderr)
        finally:
            if synth_path.exists():
                synth_path.unlink()

    if len(entries) < 10:
        raise RuntimeError(
            f"ground-truth generation produced only {len(entries)}/10 overlays"
        )
    return entries[:10]


def _entry_from_existing(image_id: str, out_root: Path) -> GroundTruthEntry:
    root = out_root / image_id
    regions_payload = json.loads((root / "ground_truth_regions.json").read_text())
    return GroundTruthEntry(
        image_id=image_id,
        source_path=root / "source_image.png",
        overlay_path=root / "input_overlay.png",
        snic_path=root / "snic_labels.npy",
        regions_path=root / "ground_truth_regions.json",
        n_cells=int(regions_payload["n_cells"]),
        n_regions=len(regions_payload["regions"]),
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_HERE / "ground_truth"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    print(f"# generating ground truth into {out}")
    entries = generate_dataset(out, force=args.force)
    for e in entries:
        print(f"  {e.image_id:30s}  cells={e.n_cells:4d}  regions={e.n_regions}")
    print(f"# total: {len(entries)} overlays")
