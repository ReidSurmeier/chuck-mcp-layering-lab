"""Per-pull alpha / cumulative proof / mirrored plate dumper for chuck-mcp v4.

The acceptance_harness (research/v4-build/example-harness/) renders a 4-row
contact sheet that requires on-disk PNG artifacts. The plan_emma pipeline
historically only writes `production_plan.json` + `hybrid_result.json`, so
rows 2/3/4 of the sheet show "NOT FOUND" placeholders.

`dump_run_artifacts` walks the OptimizationResult's `plates: list[SolvedPlate]`
in pass-order and emits, for each pull:

  - alphas/pull_NNN_alpha.png          — raw alpha snapshot (normalized to
                                          0..255 for human eyeball debugging)
  - pulls/pull_NNN.png                 — cumulative composite up to that pull
  - plates/block_NN.png +
    plates/block_NN.preview.png        — mirrored plate preview (wood ground
                                          + inked pigment, like the v3
                                          plate_renderer but pure NumPy)

It also emits 7-9 checkpoint proofs (proofs/proof_NN_after_pull_MMM.png) and
the acceptance-harness-preferred names at the plan root
(cumulative_pull_NN.png, alpha_masks/alpha_NN.png).

Cycle 1: alphas/pull_NNN_alpha.png
Cycle 2: plates/block_NN.preview.png  (mirrored)
Cycle 3: pulls/pull_NNN.png + cumulative_pull_NN.png aliases
Cycle 4: proofs/proof_NN_after_pull_MMM.png
Cycle 5: acceptance harness consumes the new outputs successfully.

The output tree mirrors the layout in
/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/cell-zone-renderer/
without requiring shapely / svgwrite — solver outputs are raster masks so
we operate in numpy/PIL throughout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


log = logging.getLogger("alpha_proof_dumper")
if not log.handlers:
    import sys

    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[alpha-dump %(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PAPER_RGB = np.array([0.96, 0.94, 0.88], dtype=np.float32)  # washi white
WOOD_BASE = np.array([0.78, 0.62, 0.42], dtype=np.float32)  # plywood maple
WOOD_DARK = np.array([0.52, 0.36, 0.22], dtype=np.float32)  # darker grain
DEFAULT_CHECKPOINT_COUNT = 7  # locked design uses 7-9 evenly spaced checkpoints


# ---------------------------------------------------------------------------
# Pigment lookups
# ---------------------------------------------------------------------------


def _pigment_rgb_from_plate(plate: Any) -> np.ndarray:
    """Extract a sane sRGB triple in 0..1 for a SolvedPlate.

    Priority order:
      1. `plate.repair_stats["pigment_blend_lab"]` (list[3]) → Lab to sRGB.
      2. `plate.pigment_color` (already 0..1 RGB).
      3. fallback: dark ivory black-ish based on `plate.opacity`.
    """
    lab = None
    rs = getattr(plate, "repair_stats", {}) or {}
    if isinstance(rs.get("pigment_blend_lab"), (list, tuple)) and len(rs["pigment_blend_lab"]) == 3:
        lab = np.asarray(rs["pigment_blend_lab"], dtype=np.float32)

    if lab is None:
        pc = getattr(plate, "pigment_color", None)
        if pc is not None:
            return np.clip(np.asarray(pc, dtype=np.float32), 0.0, 1.0)

    if lab is None:
        # Pigment lookup fallback — match the hybrid optimizer's defaults.
        lab = _PIGMENT_LAB_FALLBACK.get(
            getattr(plate, "pigment_id", "") or "",
            np.array([15.0, 0.0, 0.0], dtype=np.float32),
        )

    try:
        from skimage.color import lab2rgb  # type: ignore

        rgb = np.clip(lab2rgb(lab.reshape(1, 1, 3)).reshape(3), 0.0, 1.0)
        return rgb.astype(np.float32)
    except Exception:
        # Conservative numerical Lab→sRGB fallback (no skimage).
        return _lab_to_rgb_numpy(lab)


_PIGMENT_LAB_FALLBACK = {
    "gamboge_yellow": np.array([88.0, -8.0, 70.0], dtype=np.float32),
    "vermilion_red": np.array([50.0, 70.0, 50.0], dtype=np.float32),
    "phthalo_blue": np.array([35.0, 5.0, -55.0], dtype=np.float32),
    "viridian_green": np.array([50.0, -45.0, 10.0], dtype=np.float32),
    "burnt_sienna": np.array([45.0, 25.0, 35.0], dtype=np.float32),
    "ivory_black": np.array([15.0, 0.0, 0.0], dtype=np.float32),
}


def _lab_to_rgb_numpy(lab: np.ndarray) -> np.ndarray:
    """Pure-numpy CIE Lab → linear sRGB → sRGB, single-pixel approximation.

    Used only as a fallback when skimage is missing; tests use skimage path.
    """
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    Y = (L + 16.0) / 116.0
    X = a / 500.0 + Y
    Z = Y - b / 200.0
    # Reference white D65
    Xw, Yw, Zw = 0.95047, 1.00000, 1.08883

    def _f_inv(t: float) -> float:
        return t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)

    Xc, Yc, Zc = Xw * _f_inv(X), Yw * _f_inv(Y), Zw * _f_inv(Z)
    # Linear sRGB
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float32,
    )
    rgb_lin = M @ np.array([Xc, Yc, Zc], dtype=np.float32)

    def _gamma(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1 / 2.4)) - 0.055

    return np.clip(np.array([_gamma(c) for c in rgb_lin], dtype=np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mask normalization & alpha PNG dump
# ---------------------------------------------------------------------------


def _normalize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize/clip a plate's inked_mask to (H, W) float32 in 0..1."""
    if mask is None:
        return np.zeros(target_shape, dtype=np.float32)
    if mask.dtype != np.float32:
        m = mask.astype(np.float32)
    else:
        m = mask.copy()
    # Many solver masks are 0/1 uint8; some are already 0..1 float; some are
    # 0..255 uint8. Normalize all into 0..1.
    if m.max() > 1.5:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)
    if m.shape != target_shape:
        pil = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize(
            (target_shape[1], target_shape[0]), Image.Resampling.NEAREST
        )
        m = np.asarray(pil, dtype=np.float32) / 255.0
    return m


def _alpha_png(mask: np.ndarray, softness_px: float = 0.0) -> Image.Image:
    """Render a 0..1 mask as an 8-bit grayscale PNG, optionally softened.

    Stretches dynamic range so the file is human-eyeballable even if the
    mask is sparse — this is the same trick the acceptance_harness uses.
    """
    m = mask.astype(np.float32)
    if softness_px > 0:
        pil = Image.fromarray((m * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=softness_px)
        )
        m = np.asarray(pil, dtype=np.float32) / 255.0
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-6:
        # Empty alpha — emit a flat mid-gray placeholder so a reviewer can
        # tell the snapshot exists but the layer is empty.
        flat = np.full_like(m, 0.5, dtype=np.float32)
        return Image.fromarray((flat * 255).astype(np.uint8), mode="L")
    stretched = (m - lo) / (hi - lo)
    return Image.fromarray(np.clip(stretched * 255.0, 0, 255).astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# Pull-by-pull cumulative compositor
# ---------------------------------------------------------------------------


def _compose_pull(
    prev_state: np.ndarray,
    mask: np.ndarray,
    pigment_rgb: np.ndarray,
    opacity: float,
    dilution: float,
) -> np.ndarray:
    """Apply one pull on top of the previous proof state.

    Porter-Duff "source over" with effective opacity = plate.opacity * (1 -
    0.6 * dilution). This matches the hybrid optimizer's cheap compositor
    in `_render_cumulative_pulls`.
    """
    eff = float(opacity) * (1.0 - 0.6 * float(dilution))
    eff = max(0.0, min(1.0, eff))
    alpha = (mask * eff)[..., None]
    out = prev_state * (1.0 - alpha) + pigment_rgb[None, None, :] * alpha
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _blank_paper(shape: tuple[int, int], paper: np.ndarray = PAPER_RGB) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    out[..., 0] = paper[0]
    out[..., 1] = paper[1]
    out[..., 2] = paper[2]
    return out


# ---------------------------------------------------------------------------
# Plate preview (mirrored)
# ---------------------------------------------------------------------------


def _wood_ground(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Procedural plywood ground (pure numpy, no scipy)."""
    H, W = shape
    rng = np.random.default_rng(seed)
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    # Long horizontal stripes (wood grain).
    stripe_freq = 14 + (seed % 6)
    warp = 4.0 * np.sin(2 * np.pi * x / max(W, 1) * 1.4 + (seed % 7))
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (y + warp) * stripe_freq / max(H, 1))
    stripes = 0.6 + 0.4 * stripes  # 0.6..1.0
    # Mid-frequency blotches.
    mid = rng.random((max(4, H // 6), max(4, W // 6)), dtype=np.float32)
    mid_img = Image.fromarray((mid * 255).astype(np.uint8), mode="L").resize(
        (W, H), Image.Resampling.BILINEAR
    ).filter(ImageFilter.GaussianBlur(radius=1.5))
    mid_arr = np.asarray(mid_img, dtype=np.float32) / 255.0
    # Fine speckle.
    fine = (rng.random((H, W), dtype=np.float32) - 0.5) * 0.06
    blend = (1.0 - stripes) * 0.65 + (mid_arr - 0.5) * 0.30 + 0.25
    blend = np.clip(blend, 0.0, 1.0)[..., None]
    rgb = WOOD_BASE[None, None, :] * (1.0 - blend) + WOOD_DARK[None, None, :] * blend
    rgb = rgb + fine[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _draw_kento(arr: np.ndarray, mirror: bool) -> np.ndarray:
    """Stamp a simple L + bar registration cue on the bottom of the plate."""
    H, W = arr.shape[:2]
    short = min(H, W)
    kagi = max(3, int(0.045 * short))
    bar_w = max(3, int(0.05 * short))
    bar_h = max(2, int(kagi * 0.18))
    margin = max(2, int(W * 0.03))
    if mirror:
        bx = margin
        by = H - margin
        # L pointing right
        arr[by - bar_h : by, bx : bx + kagi] = 0.25
        arr[by - kagi : by, bx : bx + bar_h] = 0.25
        hx = int(W * 0.55)
    else:
        bx = W - margin
        by = H - margin
        arr[by - bar_h : by, bx - kagi : bx] = 0.25
        arr[by - kagi : by, bx - bar_h : bx] = 0.25
        hx = int(W * 0.45)
    hy = H - margin
    arr[hy - bar_h : hy, max(0, hx - bar_w // 2) : min(W, hx + bar_w // 2)] = 0.25
    return arr


def _render_plate_preview(
    plate: Any,
    target_shape: tuple[int, int],
    mirror: bool = True,
) -> np.ndarray:
    """Wood-ground + mirrored inked pigment region. Returns float32 H×W×3 in 0..1."""
    block_id = int(getattr(plate, "block_id", 0))
    seed = (block_id * 137 + 11) % 9973
    ground = _wood_ground(target_shape, seed=seed)
    mask = _normalize_mask(getattr(plate, "inked_mask", None), target_shape)
    if mirror:
        mask = mask[:, ::-1].copy()
    pigment_rgb = _pigment_rgb_from_plate(plate)
    # Pigment is wet ink, so use a soft alpha multiplier capped at 0.86
    alpha = (mask * min(0.95, float(getattr(plate, "opacity", 0.6) or 0.6) * 0.95))[..., None]
    out = ground * (1.0 - alpha) + pigment_rgb[None, None, :] * alpha
    out = _draw_kento(out, mirror=mirror)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint picker
# ---------------------------------------------------------------------------


def _pick_checkpoints(total: int, count: int = DEFAULT_CHECKPOINT_COUNT) -> list[int]:
    """Pick `count` 1-based pull indices evenly spaced across 1..total.

    Always includes 1 and total. Dedupes while preserving order.
    """
    if total <= 0:
        return []
    if total <= count:
        return list(range(1, total + 1))
    raw = np.linspace(1, total, num=count).round().astype(int).tolist()
    seen: list[int] = []
    for v in raw:
        v = int(v)
        if v not in seen and 1 <= v <= total:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB"
    ).save(path, format="PNG", optimize=False)


def _save_alpha(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=False)


@dataclass
class DumpResult:
    alphas: list[Path]
    plates: list[Path]
    pulls: list[Path]
    proofs: list[Path]
    cumulative_aliases: list[Path]
    alpha_aliases: list[Path]


def dump_run_artifacts(
    target_rgb: np.ndarray,
    plates: Sequence[Any],
    out_dir: Path | str,
    *,
    checkpoint_count: int = DEFAULT_CHECKPOINT_COUNT,
    mirror_plates: bool = True,
    target_shape: tuple[int, int] | None = None,
) -> dict[str, list[Path]]:
    """Emit per-pull alphas, mirrored plate previews, cumulative pull PNGs,
    and a 7-checkpoint proof series.

    Args:
        target_rgb: H×W×3 uint8 array (the source image). Used only to size
            the output canvas if target_shape is None.
        plates: list of SolvedPlate (or any object with .inked_mask, .opacity,
            .dilution, .pass_index, .block_id, .repair_stats["pigment_blend_lab"]).
        out_dir: directory where artifacts are written. Created if missing.
        checkpoint_count: how many evenly spaced proof checkpoints to dump
            (default 7 per the locked design).
        mirror_plates: True → horizontal flip on plate previews.
        target_shape: override (H, W). Inferred from target_rgb otherwise.

    Returns:
        dict with keys "alphas", "plates", "pulls", "proofs",
        "cumulative_aliases", "alpha_aliases" — each a list of Path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_shape is None:
        if target_rgb is None:
            raise ValueError("either target_rgb or target_shape must be given")
        target_shape = target_rgb.shape[:2]
    H, W = target_shape

    # Sort plates by pass_index then block_id — pulls happen in that order.
    ordered = sorted(
        plates,
        key=lambda p: (
            int(getattr(p, "pass_index", 0) or 0),
            int(getattr(p, "block_id", 0) or 0),
        ),
    )
    n = len(ordered)

    alpha_paths: list[Path] = []
    plate_paths: list[Path] = []
    pull_paths: list[Path] = []
    proof_paths: list[Path] = []
    cumulative_alias_paths: list[Path] = []
    alpha_alias_paths: list[Path] = []

    # Pre-pick checkpoint indices (1-based) so we know which pulls to snapshot.
    checkpoints = _pick_checkpoints(n, count=checkpoint_count) if n > 0 else []
    checkpoint_lookup = {c: i + 1 for i, c in enumerate(checkpoints)}  # 1-based slot

    state = _blank_paper(target_shape)

    for pull_idx, plate in enumerate(ordered, start=1):
        # --- cycle 1: alpha snapshot ---
        mask = _normalize_mask(getattr(plate, "inked_mask", None), target_shape)
        alpha_img = _alpha_png(mask, softness_px=0.0)
        ap = out_dir / "alphas" / f"pull_{pull_idx:03d}_alpha.png"
        _save_alpha(alpha_img, ap)
        alpha_paths.append(ap)

        # --- cycle 2: mirrored plate preview (one per plate, but indexed by
        # block_id to match the acceptance_harness regex block_NN.png).
        block_id = int(getattr(plate, "block_id", pull_idx) or pull_idx)
        preview = _render_plate_preview(plate, target_shape, mirror=mirror_plates)
        pp = out_dir / "plates" / f"block_{block_id:02d}.png"
        _save_rgb(preview, pp)
        plate_paths.append(pp)
        # Harness prefers `block_NN.preview.png` — write a duplicate so both
        # the legacy block_NN.png and the preview-suffix name match.
        pp_preview = out_dir / "plates" / f"block_{block_id:02d}.preview.png"
        _save_rgb(preview, pp_preview)

        # --- cycle 3: cumulative state after this pull ---
        pigment_rgb = _pigment_rgb_from_plate(plate)
        state = _compose_pull(
            state,
            mask=mask,
            pigment_rgb=pigment_rgb,
            opacity=float(getattr(plate, "opacity", 0.5) or 0.5),
            dilution=float(getattr(plate, "dilution", 0.3) or 0.3),
        )
        pul = out_dir / "pulls" / f"pull_{pull_idx:03d}.png"
        _save_rgb(state, pul)
        pull_paths.append(pul)

        # --- cycle 4: 7-checkpoint proof series ---
        if pull_idx in checkpoint_lookup:
            slot = checkpoint_lookup[pull_idx]
            pr = (
                out_dir
                / "proofs"
                / f"proof_{slot:02d}_after_pull_{pull_idx:03d}.png"
            )
            _save_rgb(state, pr)
            proof_paths.append(pr)
            # Harness row 2 prefers `cumulative_pull_NN.png` at the plan root.
            alias_idx = slot
            alias = out_dir / f"cumulative_pull_{alias_idx:02d}.png"
            _save_rgb(state, alias)
            cumulative_alias_paths.append(alias)
            # Harness row 4 prefers `alpha_masks/alpha_NN.png` — write
            # the slot-numbered alphas there too.
            aa = out_dir / "alpha_masks" / f"alpha_{alias_idx:02d}.png"
            _save_alpha(_alpha_png(mask, softness_px=0.0), aa)
            alpha_alias_paths.append(aa)

    # Final-composite is the last cumulative state — write it both at the root
    # (for the harness plate_not_composite proxy) and as the canonical "after
    # pull N" file.
    if n > 0:
        _save_rgb(state, out_dir / "final_composite.png")

    log.info(
        "dumped %d alphas, %d plates, %d pulls, %d proofs into %s",
        len(alpha_paths),
        len(plate_paths),
        len(pull_paths),
        len(proof_paths),
        out_dir,
    )

    return {
        "alphas": alpha_paths,
        "plates": plate_paths,
        "pulls": pull_paths,
        "proofs": proof_paths,
        "cumulative_aliases": cumulative_alias_paths,
        "alpha_aliases": alpha_alias_paths,
    }


__all__ = ["dump_run_artifacts", "DumpResult"]
