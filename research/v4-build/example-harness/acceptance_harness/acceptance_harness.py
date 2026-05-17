"""Acceptance harness: render the 4-row contact sheet.

Per docs/audit-response-and-reconstruction-plan-2026-05-17.md §Phase 1, verbatim:

    "Make a side-by-side contact sheet generator: reference proof row, current
    proof row, current block row, alpha row. Acceptance rule: if a human says
    'this looks like slop' against the example sheet, the run fails regardless
    of dE."

Plan-directory layout assumed (matches the v3-audit-thorough-main output we
froze as Phase 0 failing baseline):

    plan_output_dir/
    ├── cumulative_pull_NN.png      # row 2 source (cumulative proof states)
    ├── final_composite.png         # used for plate_not_composite proxy
    ├── target.png                  # informational
    ├── plates/                     # row 3 source (preferred)
    │   ├── block_NN.preview.png    # mirrored block previews
    │   ├── block_NN.svg            # mirrored carving file (rasterized if no preview)
    │   └── ...
    ├── alpha_masks/                # row 4 source (preferred)
    │   ├── alpha_NN.png
    │   └── ...
    └── alpha_masks_contact_sheet.png  # row-4 fallback if no per-plate dumps

The harness is forgiving: missing slots become labeled placeholder tiles so a
human still gets a sheet they can eyeball. Warnings are returned in the
AcceptanceSheetResult.warnings list.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .acceptance_result import AcceptanceSheetResult, PlateMetric
from .example_loader import (
    DEFAULT_TILE_H,
    DEFAULT_TILE_W,
    REFERENCE_EXAMPLES_DIR,
    _make_placeholder_tile,
    list_available_examples,
    load_reference_proofs,
)

# Sheet geometry — fixed so visual review across runs is comparable.
TILES_PER_ROW = 8
TILE_W = 256
TILE_H = 256
ROW_LABEL_H = 28
ROW_GAP = 12
SHEET_PAD = 24
HEADER_H = 56

# Row order is load-bearing — matches the audit-doc spec.
ROW_LABELS = (
    "Row 1 — reference proofs (Chuck Close self-portrait 2015, Pace Editions progressive proof series)",
    "Row 2 — current run cumulative proofs (8 evenly-spaced checkpoints)",
    "Row 3 — current run plates (first 8, mirrored, jigsaw cells only)",
    "Row 4 — current run alpha-map snapshots (normalized for visibility, debug-only)",
)


# ---------------------------------------------------------------------------
# Internal helpers (pure, easy to unit-test if we ever want to)
# ---------------------------------------------------------------------------


_PULL_PATTERNS = (
    re.compile(r"^cumulative_pull_(\d+)\.png$", re.IGNORECASE),
    re.compile(r"^pull_(\d+)\.png$", re.IGNORECASE),
    re.compile(r"^proof_(\d+)\.png$", re.IGNORECASE),
)
_PLATE_PREVIEW_PATTERNS = (
    re.compile(r"^block_(\d+)\.preview\.png$", re.IGNORECASE),
    re.compile(r"^plate_(\d+)\.preview\.png$", re.IGNORECASE),
    re.compile(r"^block_(\d+)\.png$", re.IGNORECASE),
    re.compile(r"^plate_(\d+)\.png$", re.IGNORECASE),
)
_ALPHA_PATTERNS = (
    re.compile(r"^alpha_(\d+)\.png$", re.IGNORECASE),
    re.compile(r"^alpha_mask_(\d+)\.png$", re.IGNORECASE),
    re.compile(r"^plane_(\d+)\.png$", re.IGNORECASE),
)


def _scan_indexed_files(
    directory: Path,
    patterns: tuple[re.Pattern[str], ...],
) -> list[tuple[int, Path]]:
    """Return [(index, path), ...] sorted by index for files matching any pattern."""
    if not directory.exists():
        return []
    found: dict[int, Path] = {}
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        for pat in patterns:
            m = pat.match(entry.name)
            if m:
                idx = int(m.group(1))
                # First match wins (preview > raw).
                found.setdefault(idx, entry)
                break
    return sorted(found.items(), key=lambda kv: kv[0])


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Pick `count` indices from a 0..total-1 range, evenly spaced inclusive of endpoints.

    For total=12, count=8 → [0, 1, 3, 4, 6, 7, 9, 11] style spacing.
    For total < count, returns every available index (no padding).
    """
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    # Use linspace then dedupe while preserving order; np.linspace is well-tested.
    raw = np.linspace(0, total - 1, num=count).round().astype(int).tolist()
    seen: list[int] = []
    for v in raw:
        if v not in seen:
            seen.append(v)
    return seen


def _load_and_fit(path: Path, target_w: int, target_h: int) -> Image.Image:
    """Load `path`, downsample-preserve-aspect to fit inside (target_w, target_h),
    then center on a black canvas of the exact target size.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (16, 16, 16))
    offx = (target_w - img.width) // 2
    offy = (target_h - img.height) // 2
    canvas.paste(img, (offx, offy))
    return canvas


def _normalize_alpha_tile(path: Path, target_w: int, target_h: int) -> Image.Image:
    """Load an alpha-map dump and STRETCH its dynamic range so a debug viewer can see it.

    Real alpha maps are often near-zero floats stored as 8-bit PNGs — they read
    as solid black without this. We linearly remap [min, max] -> [0, 255].
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        # All-same pixel — synthesize a flat mid-gray so the user can tell it's "empty".
        normed = np.full_like(arr, 128, dtype=np.uint8)
    else:
        normed = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    norm_img = Image.fromarray(normed, mode="L").convert("RGB")
    norm_img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (16, 16, 16))
    canvas.paste(
        norm_img,
        ((target_w - norm_img.width) // 2, (target_h - norm_img.height) // 2),
    )
    return canvas


def _compose_row(tiles: list[Image.Image], label: str) -> Image.Image:
    """Compose a labeled row of fixed-size tiles."""
    row_w = TILES_PER_ROW * TILE_W + (TILES_PER_ROW - 1) * 4  # tight inter-tile gap
    row_h = TILE_H + ROW_LABEL_H
    row = Image.new("RGB", (row_w, row_h), (20, 20, 20))

    draw = ImageDraw.Draw(row)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    if font is not None:
        draw.text((6, 6), label, fill=(220, 220, 220), font=font)

    # Pad tile list if short, trim if long.
    padded: list[Image.Image] = list(tiles[:TILES_PER_ROW])
    while len(padded) < TILES_PER_ROW:
        padded.append(
            _make_placeholder_tile(TILE_W, TILE_H, f"slot {len(padded) + 1}\n(empty)")
        )

    x = 0
    for tile in padded:
        if tile.size != (TILE_W, TILE_H):
            tile = tile.resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
        row.paste(tile, (x, ROW_LABEL_H))
        x += TILE_W + 4
    return row


def _compose_sheet(rows: list[Image.Image], header_text: str) -> Image.Image:
    """Stack rows vertically with padding + a header bar."""
    if not rows:
        raise ValueError("at least one row required")
    inner_w = max(r.width for r in rows)
    inner_h = sum(r.height for r in rows) + ROW_GAP * (len(rows) - 1)
    sheet_w = inner_w + 2 * SHEET_PAD
    sheet_h = inner_h + 2 * SHEET_PAD + HEADER_H
    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))

    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    if font is not None:
        draw.text((SHEET_PAD, SHEET_PAD), header_text, fill=(255, 200, 80), font=font)

    y = SHEET_PAD + HEADER_H
    for row in rows:
        sheet.paste(row, (SHEET_PAD, y))
        y += row.height + ROW_GAP
    return sheet


def _proof_progression_score(proof_tiles: list[Image.Image]) -> float:
    """Mean per-step normalized pixel-difference across consecutive proof tiles.

    Cheap structural proxy for the real `proof_progression_score` validator. A
    healthy progressive proof series moves significantly between checkpoints
    (> 0.05); a stalled solver that re-renders the same proof scores near 0.
    """
    if len(proof_tiles) < 2:
        return 0.0
    diffs: list[float] = []
    for a, b in zip(proof_tiles[:-1], proof_tiles[1:], strict=False):
        if a.size != b.size:
            b = b.resize(a.size, Image.Resampling.LANCZOS)
        arr_a = np.asarray(a, dtype=np.float32) / 255.0
        arr_b = np.asarray(b, dtype=np.float32) / 255.0
        diffs.append(float(np.abs(arr_a - arr_b).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def _plate_not_composite_proxy(
    plate_tile: Image.Image, final_tile: Image.Image | None
) -> float:
    """Cheap proxy for plate_not_composite_score (HIGH = good jigsaw plate).

    Returns 1.0 - cosine_similarity(plate_flat, final_flat) clipped to [0, 1].
    Real validator uses a richer feature space; this is the harness-level
    smell-test only.
    """
    if final_tile is None:
        return 1.0  # No baseline → unknown, optimistic.
    if plate_tile.size != final_tile.size:
        final_tile = final_tile.resize(plate_tile.size, Image.Resampling.LANCZOS)
    a = np.asarray(plate_tile, dtype=np.float32).reshape(-1)
    b = np.asarray(final_tile, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    cos = float(np.dot(a, b) / denom)
    return float(np.clip(1.0 - cos, 0.0, 1.0))


def _coverage_fraction(plate_tile: Image.Image, bg_threshold: int = 18) -> float:
    """Fraction of plate pixels that are not near-black background."""
    arr = np.asarray(plate_tile.convert("RGB"))
    # "Inked" = any channel above the bg_threshold (defaults to harness bg ~16).
    inked = (arr.max(axis=-1) > bg_threshold)
    return float(inked.mean())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_acceptance_sheet(
    plan_output_dir: Path,
    reference_examples_dir: Path = REFERENCE_EXAMPLES_DIR,
    output_path: Path | None = None,
) -> AcceptanceSheetResult:
    """Render the 4-row acceptance contact sheet for a chuck-mcp plan directory.

    Args:
        plan_output_dir: directory containing cumulative_pull_NN.png +
            (optionally) plates/ and alpha_masks/ subdirs. The
            v3-audit-thorough-main baseline layout is the canonical example.
        reference_examples_dir: defaults to /srv/woodblock-share/Examples;
            override for tests.
        output_path: where to write the rendered PNG. Defaults to
            `<plan_output_dir>/acceptance_sheet.png`.

    Returns:
        AcceptanceSheetResult — see acceptance_result.py.

    Raises:
        FileNotFoundError: if plan_output_dir does not exist.
    """
    plan_output_dir = Path(plan_output_dir)
    if not plan_output_dir.exists():
        raise FileNotFoundError(f"plan_output_dir not found: {plan_output_dir}")
    reference_examples_dir = Path(reference_examples_dir)
    if output_path is None:
        output_path = plan_output_dir / "acceptance_sheet.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    warnings: list[str] = []

    # ---- Row 1: reference proofs ---------------------------------------------------
    ref_tiles_raw = load_reference_proofs(reference_examples_dir, target_count=TILES_PER_ROW)
    ref_tiles = [
        _fit_pil_image(t, TILE_W, TILE_H) for t in ref_tiles_raw
    ]
    reference_examples_used = list_available_examples(reference_examples_dir)
    if not reference_examples_used:
        warnings.append(
            f"reference_examples_dir empty or missing: {reference_examples_dir}"
        )

    # ---- Row 2: current cumulative proofs ------------------------------------------
    pull_files = _scan_indexed_files(plan_output_dir, _PULL_PATTERNS)
    if not pull_files:
        warnings.append(
            "no cumulative_pull_NN.png / pull_NN.png files found in plan dir"
        )
    pick = _evenly_spaced_indices(len(pull_files), TILES_PER_ROW)
    proof_records = [pull_files[i] for i in pick]
    proof_tiles = [
        _load_and_fit(path, TILE_W, TILE_H) for (_, path) in proof_records
    ]
    proof_checkpoints_rendered = [idx for (idx, _) in proof_records]

    # ---- Row 3: current plate previews ---------------------------------------------
    plates_dir = plan_output_dir / "plates"
    plate_files = _scan_indexed_files(plates_dir, _PLATE_PREVIEW_PATTERNS)
    if not plate_files:
        # Try plan root as fallback (some pipelines drop block_NN.png at root).
        plate_files = _scan_indexed_files(plan_output_dir, _PLATE_PREVIEW_PATTERNS)
    if not plate_files:
        warnings.append(
            "no block_NN.preview.png / plate_NN.png files found "
            "(checked plates/ and plan root)"
        )
    plate_records = plate_files[:TILES_PER_ROW]
    plate_tiles = [_load_and_fit(p, TILE_W, TILE_H) for (_, p) in plate_records]

    # Final composite (for plate_not_composite proxy).
    final_path = plan_output_dir / "final_composite.png"
    final_tile_for_metric: Image.Image | None = None
    if final_path.exists():
        try:
            final_tile_for_metric = _load_and_fit(final_path, TILE_W, TILE_H)
        except Exception as e:  # pragma: no cover — defensive
            warnings.append(f"failed to load final_composite.png: {e!r}")
    else:
        warnings.append("final_composite.png missing — plate_not_composite proxy unreliable")

    plate_metrics: list[PlateMetric] = []
    for i, tile in enumerate(plate_tiles):
        plate_metrics.append(
            PlateMetric(
                plate_index=i,
                coverage_fraction=_coverage_fraction(tile),
                plate_not_composite_score=_plate_not_composite_proxy(
                    tile, final_tile_for_metric
                ),
            )
        )

    # ---- Row 4: alpha-map snapshots (normalized) -----------------------------------
    alpha_dir = plan_output_dir / "alpha_masks"
    alpha_files = _scan_indexed_files(alpha_dir, _ALPHA_PATTERNS)
    if not alpha_files:
        alpha_files = _scan_indexed_files(plan_output_dir, _ALPHA_PATTERNS)
    if not alpha_files:
        warnings.append(
            "no alpha_NN.png / plane_NN.png files found — row 4 will be placeholders. "
            "Tip: dump per-plane alpha snapshots from the solver for richer debugging."
        )
    alpha_records = alpha_files[:TILES_PER_ROW]
    alpha_tiles = [_normalize_alpha_tile(p, TILE_W, TILE_H) for (_, p) in alpha_records]

    # ---- Compose -------------------------------------------------------------------
    rows = [
        _compose_row(ref_tiles, ROW_LABELS[0]),
        _compose_row(proof_tiles, ROW_LABELS[1]),
        _compose_row(plate_tiles, ROW_LABELS[2]),
        _compose_row(alpha_tiles, ROW_LABELS[3]),
    ]
    header = (
        f"chuck-mcp v4 acceptance sheet — plan: {plan_output_dir.name} — "
        f"HUMAN EYEBALL REQUIRED (audit Phase 1)"
    )
    sheet = _compose_sheet(rows, header)
    sheet.save(output_path, format="PNG", optimize=False)

    elapsed = time.perf_counter() - started

    return AcceptanceSheetResult(
        sheet_path=output_path,
        reference_examples_used=reference_examples_used,
        proof_checkpoints_rendered=proof_checkpoints_rendered,
        plate_count_rendered=len(plate_tiles),
        alpha_count_rendered=len(alpha_tiles),
        proof_progression_score=_proof_progression_score(proof_tiles),
        plate_metrics=plate_metrics,
        human_eyeball_required=True,
        warnings=warnings + [f"render_seconds={elapsed:.3f}"],
    )


def _fit_pil_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """In-memory analog of _load_and_fit for tiles already in memory (row 1)."""
    img = img.convert("RGB").copy()
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (16, 16, 16))
    canvas.paste(
        img,
        ((target_w - img.width) // 2, (target_h - img.height) // 2),
    )
    return canvas
