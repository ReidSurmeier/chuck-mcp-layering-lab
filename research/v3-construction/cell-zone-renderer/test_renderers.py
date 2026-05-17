"""Synthetic 27-plate Emma harness — exercises all four renderers.

Run with:

    /home/reidsurmeier/src/chuck-mcp-layering-lab/.venv-renderer/bin/python \
        /home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/cell-zone-renderer/test_renderers.py

Outputs (under ./out/):
    block_01.svg, block_01.preview.png             — single-plate SVG + PNG
    block_NN.preview.png (NN=01..27)               — preview for each plate
    pull_001.png ... pull_<N>.png                  — pull-by-pull cumulative
    proof_state_sheet.png                          — 8-up cumulative
    plate_and_pull_sheet.png                       — process diagram
    all_blocks_contact_sheet.png                   — the v13 replacement

A "synthetic Emma plan" means: we lay a grid of SNIC-shaped cell
zones (clipped Voronoi-ish patches) over a 1024×1024 canvas, assign
each cell to one of 27 plates by hue family, and let the renderers
do the rest. No real Emma image needed — the rendering logic must
work on any valid plan.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union, voronoi_diagram

from cz_types import CellZone, Plate, ProofState, Pull
from plate_renderer import (
    PlateRenderConfig,
    render_plate_preview,
    render_plate_svg,
)
from pull_renderer import blank_proof_state, render_pull
from proof_state_assembler import (
    DEFAULT_CHECKPOINTS,
    assemble_plate_and_pull_sheet,
    assemble_proof_sheet,
)
from contact_sheet_renderer import save_contact_sheet


# --- synthetic plan -----------------------------------------------------------


PLATE_PALETTE: list[tuple[str, tuple[float, float, float], str, float]] = [
    # name, RGB 0..1, role, opacity
    ("yellow_ochre",      (0.92, 0.78, 0.30), "underlayer_light", 0.55),
    ("naples_yellow",     (0.96, 0.86, 0.55), "underlayer_light", 0.50),
    ("pale_pink",         (0.95, 0.78, 0.78), "underlayer_light", 0.55),
    ("cerulean_pale",     (0.65, 0.80, 0.92), "underlayer_light", 0.55),
    ("apricot",           (0.97, 0.72, 0.52), "underlayer_light", 0.60),
    ("pale_green",        (0.78, 0.90, 0.72), "underlayer_light", 0.55),
    ("vermilion",         (0.87, 0.32, 0.20), "local_chroma",     0.85),
    ("magenta",           (0.86, 0.30, 0.55), "local_chroma",     0.85),
    ("cobalt_blue",       (0.20, 0.36, 0.78), "local_chroma",     0.85),
    ("alizarin",          (0.65, 0.18, 0.32), "local_chroma",     0.80),
    ("phthalo_green",     (0.10, 0.45, 0.40), "local_chroma",     0.85),
    ("ultramarine",       (0.12, 0.20, 0.62), "local_chroma",     0.85),
    ("cadmium_orange",    (0.95, 0.50, 0.18), "local_chroma",     0.85),
    ("violet",            (0.45, 0.25, 0.68), "local_chroma",     0.85),
    ("burnt_sienna",      (0.62, 0.34, 0.22), "regional_mass",    0.80),
    ("raw_umber",         (0.45, 0.30, 0.20), "regional_mass",    0.80),
    ("payne_grey",        (0.30, 0.34, 0.40), "regional_mass",    0.80),
    ("warm_grey",         (0.55, 0.50, 0.45) ,"regional_mass",    0.80),
    ("muted_olive",       (0.55, 0.55, 0.30), "regional_mass",    0.80),
    ("rose_madder",       (0.78, 0.30, 0.35), "regional_mass",    0.80),
    ("ultramarine_deep",  (0.10, 0.16, 0.55), "regional_mass",    0.85),
    ("vandyke_brown",     (0.35, 0.22, 0.16), "key_detail",       0.95),
    ("ivory_black",       (0.10, 0.09, 0.10), "key_detail",       0.95),
    ("bone_black",        (0.14, 0.13, 0.13), "key_detail",       0.95),
    ("chinese_blue",      (0.06, 0.18, 0.42), "key_detail",       0.92),
    ("indigo",            (0.10, 0.14, 0.30), "key_detail",       0.92),
    ("lamp_black",        (0.05, 0.05, 0.06), "key_detail",       0.95),
]
assert len(PLATE_PALETTE) == 27


def _voronoi_cells(width: int, height: int, n_cells: int, seed: int) -> list[Polygon]:
    rng = np.random.default_rng(seed)
    pts = rng.random((n_cells, 2), dtype=np.float32)
    pts[:, 0] *= width
    pts[:, 1] *= height
    from shapely.geometry import MultiPoint, box
    mp = MultiPoint([(float(p[0]), float(p[1])) for p in pts])
    envelope = box(0, 0, width, height)
    diag = voronoi_diagram(mp, envelope=envelope)
    polys = []
    for g in diag.geoms:
        clipped = g.intersection(envelope)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            polys.append(clipped)
        elif hasattr(clipped, "geoms"):
            for sub in clipped.geoms:
                if isinstance(sub, Polygon) and not sub.is_empty:
                    polys.append(sub)
    return polys


def _face_region_mask(width: int, height: int) -> Polygon:
    """A loose 'where the portrait sits' mask — center oval ish."""
    cx, cy = width / 2.0, height * 0.50
    rx, ry = width * 0.36, height * 0.45
    pts = []
    for i in range(64):
        t = 2 * math.pi * i / 64
        pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return Polygon(pts)


def make_synthetic_emma_plates(
    width: int = 1024,
    height: int = 1024,
    n_cells: int = 320,
    n_plates: int = 27,
    seed: int = 17,
) -> tuple[list[Plate], tuple[int, int]]:
    """Build 27 :class:`Plate` objects from a synthetic Voronoi cell graph.

    Cells are assigned to plates with a bias toward:
      - underlayers: large connected patches spanning the face
      - local_chroma: scattered accent cells
      - regional_mass: contiguous patches at face edges
      - key_detail: high-frequency clusters around the center
    """
    rng = np.random.default_rng(seed)
    polys = _voronoi_cells(width, height, n_cells, seed)
    face = _face_region_mask(width, height)

    # Score cells by "centeredness" and "size"
    cell_data = []
    for i, p in enumerate(polys):
        c = p.centroid
        d_norm = math.hypot(c.x - width / 2.0, c.y - height / 2.0) / (width * 0.7)
        a_norm = float(p.area) / (width * height / float(len(polys)))
        in_face = 1.0 if face.contains(c) else 0.0
        cell_data.append((i, p, d_norm, a_norm, in_face))

    # Bucket cells into plates 1..27 by hue/role family
    plates: list[Plate] = []
    cells_by_plate: dict[int, list[CellZone]] = {pid: [] for pid in range(1, n_plates + 1)}

    for ci, poly, d_norm, a_norm, in_face in cell_data:
        # Choose a role first based on geometry
        r = rng.random()
        if d_norm < 0.35 and in_face and r < 0.40:
            role_pool = [pid for pid in range(1, n_plates + 1)
                         if PLATE_PALETTE[pid - 1][2] == "underlayer_light"]
        elif d_norm < 0.25 and in_face and r < 0.70:
            role_pool = [pid for pid in range(1, n_plates + 1)
                         if PLATE_PALETTE[pid - 1][2] == "key_detail"]
        elif r < 0.55:
            role_pool = [pid for pid in range(1, n_plates + 1)
                         if PLATE_PALETTE[pid - 1][2] == "local_chroma"]
        else:
            role_pool = [pid for pid in range(1, n_plates + 1)
                         if PLATE_PALETTE[pid - 1][2] == "regional_mass"]

        # Bias selection so each plate gets a sane number of cells (≥3
        # per plate roughly). Pick the plate in role_pool that has
        # the fewest cells so far.
        plate_id = min(role_pool, key=lambda pid: len(cells_by_plate[pid]))

        # Target color: roughly tint of pigment toward the cell's
        # local average target. Synthetic = pigment itself with noise.
        pc = PLATE_PALETTE[plate_id - 1][1]
        tgt = (
            float(np.clip(pc[0] + rng.normal(0, 0.05), 0, 1)),
            float(np.clip(pc[1] + rng.normal(0, 0.05), 0, 1)),
            float(np.clip(pc[2] + rng.normal(0, 0.05), 0, 1)),
        )
        cz = CellZone(
            zone_id=ci,
            polygon=poly,
            target_rgb=tgt,
            plate_id=plate_id,
        )
        cells_by_plate[plate_id].append(cz)

    for pid in range(1, n_plates + 1):
        name, rgb, role, opacity = PLATE_PALETTE[pid - 1]
        czs = cells_by_plate[pid]
        # Make sure every plate has at least one cell zone — if empty,
        # synthesize one tiny zone at a deterministic position so the
        # SVG isn't completely empty.
        if not czs:
            rng_p = np.random.default_rng(seed + pid)
            cx = rng_p.uniform(width * 0.1, width * 0.9)
            cy = rng_p.uniform(height * 0.1, height * 0.9)
            r = 28
            poly = Polygon([
                (cx - r, cy - r), (cx + r, cy - r),
                (cx + r, cy + r), (cx - r, cy + r),
            ])
            czs = [CellZone(zone_id=10_000 + pid, polygon=poly,
                            target_rgb=rgb, plate_id=pid)]
        plates.append(Plate(
            block_id=pid,
            cell_zones=czs,
            pigment_color=rgb,
            pigment_name=name,
            opacity=opacity,
            role=role,  # type: ignore[arg-type]
            pass_index=1,
            mirror=True,
        ))

    return plates, (width, height)


# --- run the harness ----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "out")
    parser.add_argument("--size", type=int, default=1024,
                        help="Source image and proof state size (square)")
    parser.add_argument("--preview-size", type=int, default=512,
                        help="Plate preview PNG size for individual plates")
    parser.add_argument("--n-pulls", type=int, default=132,
                        help="Total pulls to render for the proof series")
    parser.add_argument("--n-cells", type=int, default=320)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer pulls, smaller previews")
    args = parser.parse_args()

    if args.quick:
        args.n_pulls = 32
        args.preview_size = 320
        args.size = 768

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "plates").mkdir(exist_ok=True)
    (out / "pulls").mkdir(exist_ok=True)

    width = height = args.size
    print(f"[1/6] generating synthetic 27-plate plan at {width}x{height}...")
    t0 = time.time()
    plates, src_size = make_synthetic_emma_plates(
        width=width, height=height, n_cells=args.n_cells
    )
    print(f"      done in {time.time() - t0:.2f}s — {len(plates)} plates, "
          f"{sum(len(p.cell_zones) for p in plates)} total cells")

    cfg = PlateRenderConfig(
        preview_size_px=(args.preview_size, args.preview_size),
        mill_radius_px=4.0,
    )

    print(f"[2/6] rendering plate SVG + preview PNGs (27)...")
    t0 = time.time()
    for plate in plates:
        svg_path = out / "plates" / f"block_{plate.block_id:02d}.svg"
        png_path = out / "plates" / f"block_{plate.block_id:02d}.preview.png"
        render_plate_svg(plate, svg_path, width=width, height=height,
                         mirror=True, cfg=cfg)
        render_plate_preview(plate, png_path,
                             src_size=src_size, mirror=True, cfg=cfg)
    print(f"      done in {time.time() - t0:.2f}s")

    print(f"[3/6] rendering pulls 1..{args.n_pulls} (cumulative)...")
    t0 = time.time()
    proof = blank_proof_state((width, height))
    pulls_imgs: list[np.ndarray] = []
    rng = np.random.default_rng(1)
    # Order plates with a sensible role progression:
    role_order = ["underlayer_light", "local_chroma",
                  "regional_mass", "key_detail"]
    ordered_plates = sorted(
        plates,
        key=lambda p: (role_order.index(p.role), p.block_id),
    )
    # Cycle through plates until we hit n_pulls, varying opacity a bit
    for i in range(args.n_pulls):
        plate = ordered_plates[i % len(ordered_plates)]
        op_mult = float(np.clip(0.7 + rng.normal(0, 0.12), 0.3, 1.0))
        proof = render_pull(
            plate, proof,
            opacity=op_mult,
            ink_density=1.0,
            softness_px=1.6,
            src_size=src_size,
        )
        pulls_imgs.append(proof.copy())
        if (i + 1) % 16 == 0:
            print(f"      pull {i + 1}/{args.n_pulls}  (rolling t={time.time() - t0:.1f}s)")
    # Save a few full-res cumulative pulls
    save_idxs = sorted(set(DEFAULT_CHECKPOINTS + [args.n_pulls]))
    for idx in save_idxs:
        if idx <= len(pulls_imgs):
            Image.fromarray(
                (np.clip(pulls_imgs[idx - 1], 0, 1) * 255).astype(np.uint8), "RGB"
            ).save(out / "pulls" / f"pull_{idx:03d}.png")
    print(f"      done in {time.time() - t0:.2f}s")

    print(f"[4/6] assembling proof-state sheet...")
    t0 = time.time()
    cp = [c for c in DEFAULT_CHECKPOINTS if c <= args.n_pulls] + \
         [args.n_pulls] * (8 - len([c for c in DEFAULT_CHECKPOINTS if c <= args.n_pulls]))
    cp = cp[:8]
    sheet = assemble_proof_sheet(
        pulls_imgs,
        checkpoint_indices=cp,
        cols=4, rows=2,
        cell_size=(420, 580),
    )
    sheet.save(out / "proof_state_sheet.png")
    print(f"      done in {time.time() - t0:.2f}s -> {out / 'proof_state_sheet.png'}")

    print(f"[5/6] assembling plate+pull pair sheet (first 8 plates)...")
    t0 = time.time()
    plate_imgs = []
    pair_pulls = []
    proof2 = blank_proof_state((width, height))
    for plate in ordered_plates[:8]:
        with Image.open(out / "plates" / f"block_{plate.block_id:02d}.preview.png") as im:
            plate_imgs.append(im.copy())
        proof2 = render_pull(plate, proof2, opacity=1.0,
                             softness_px=1.6, src_size=src_size)
        pair_pulls.append(proof2.copy())
    pair_sheet = assemble_plate_and_pull_sheet(
        plate_imgs, pair_pulls, cols=4, cell_size=(280, 380)
    )
    pair_sheet.save(out / "plate_and_pull_sheet.png")
    print(f"      done in {time.time() - t0:.2f}s")

    print(f"[6/6] rendering all-blocks contact sheet...")
    t0 = time.time()
    cs_path = save_contact_sheet(
        plates,
        out / "all_blocks_contact_sheet.png",
        cols=7, cell_size=(280, 360),
        src_size=src_size, cfg=cfg,
    )
    print(f"      done in {time.time() - t0:.2f}s -> {cs_path}")

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
