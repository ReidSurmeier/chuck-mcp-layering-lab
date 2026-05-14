#!/usr/bin/env python3
"""Generate methodology-style cumulative proof states for Emma review."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.v23.core import color, forward_render_jax  # noqa: E402

DEFAULT_INPUT = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")
DEFAULT_OUT_ROOT = Path("/srv/woodblock-share/chuck-methodology-proofs")


def _as_u8(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _load_rgb(path: Path, max_side: int | None = None) -> NDArray[np.float32]:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save(path: Path, rgb: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_u8(rgb), "RGB").save(path)


def _paper_lab(target: NDArray[np.float32]) -> NDArray[np.float32]:
    lab = color.srgb_to_lab(target)
    h, w = lab.shape[:2]
    m = max(8, min(h, w) // 24)
    border = np.concatenate([
        lab[:m].reshape(-1, 3),
        lab[-m:].reshape(-1, 3),
        lab[:, :m].reshape(-1, 3),
        lab[:, -m:].reshape(-1, 3),
    ])
    return np.median(border, axis=0).astype(np.float32)


def _smooth(mask: NDArray[np.float32], radius: float) -> NDArray[np.float32]:
    arr = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def _quantized_field(
    target: NDArray[np.float32],
    mask: NDArray[np.float32],
    *,
    k: int,
    seed: int,
) -> NDArray[np.float32]:
    active = mask > 0.04
    out = np.broadcast_to(forward_render_jax.PAPER_RGB.astype(np.float32), target.shape).copy()
    if active.sum() < 16:
        return out
    lab = color.srgb_to_lab(target)
    pixels = lab[active].astype(np.float32)
    if pixels.shape[0] > 22000:
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(pixels.shape[0], size=22000, replace=False)]
    unique = np.unique(np.round(pixels, 2), axis=0)
    k = max(1, min(k, len(unique)))
    if k == 1:
        centroid_lab = unique[:1]
    else:
        try:
            centroid_lab, _ = kmeans2(pixels, k, minit="++", seed=seed)
        except Exception:
            centroid_lab = unique[:k]
    all_lab = lab[active].astype(np.float32)
    d2 = np.sum((all_lab[:, None, :] - centroid_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    centroid_rgb = []
    rgb_pixels = target[active]
    for i in range(centroid_lab.shape[0]):
        slot = rgb_pixels[labels == i]
        centroid_rgb.append(slot.mean(axis=0) if slot.size else rgb_pixels.mean(axis=0))
    out[active] = np.asarray(centroid_rgb, dtype=np.float32)[labels]
    return out


def _apply_block(
    current: NDArray[np.float32],
    target: NDArray[np.float32],
    mask: NDArray[np.float32],
    *,
    strength: float,
    colors: int,
    seed: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    field = _quantized_field(target, mask, k=colors, seed=seed)
    a = np.clip(mask[..., None] * strength, 0.0, 1.0)
    after = current * (1.0 - a) + field * a
    block = np.broadcast_to(forward_render_jax.PAPER_RGB.astype(np.float32), target.shape).copy()
    block = block * (1.0 - a) + field * a
    return after.astype(np.float32), block.astype(np.float32)


def _mask_bank(target: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
    lab = color.srgb_to_lab(target)
    paper = _paper_lab(target)
    delta = color.delta_e76(lab, paper)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    active = np.clip((delta - 2.0) / 18.0, 0.0, 1.0)
    warm = active * np.clip((b + 2.0) / 28.0, 0.0, 1.0) * np.clip((L - 30.0) / 55.0, 0.0, 1.0)
    red = active * np.clip((a - 6.0) / 24.0, 0.0, 1.0) * np.clip(chroma / 44.0, 0.0, 1.0)
    cool = active * np.clip((-b + 8.0) / 32.0, 0.0, 1.0)
    green = active * np.clip((-a + 6.0) / 26.0, 0.0, 1.0) * np.clip((b + 6.0) / 26.0, 0.0, 1.0)
    dark = active * np.clip((58.0 - L) / 34.0, 0.0, 1.0)
    face_shadow = dark * np.clip((L - 28.0) / 32.0, 0.0, 1.0) * np.clip((b + 12.0) / 42.0, 0.0, 1.0)
    cool_light = np.maximum(cool, green) * np.clip((L - 34.0) / 42.0, 0.0, 1.0)
    background_dark = dark * np.clip((42.0 - L) / 30.0, 0.0, 1.0)
    key = active * np.clip((42.0 - L) / 24.0, 0.0, 1.0)
    chroma_m = active * np.clip((chroma - 12.0) / 36.0, 0.0, 1.0)
    detail = active * np.clip(delta / 42.0, 0.0, 1.0)
    light = active * np.clip((L - 52.0) / 38.0, 0.0, 1.0)
    return {
        "pale_scaffold": _smooth(light, 1.8),
        "pale_warm": _smooth(light * warm, 2.6),
        "pale_red": _smooth(light * red, 2.2),
        "pale_cool": _smooth(light * np.maximum(cool, green), 2.4),
        "pale_line": _smooth(light * detail, 1.2),
        "warm_mass": _smooth(warm, 3.0),
        "red_orange": _smooth(red, 2.0),
        "cool_green": _smooth(cool_light, 2.4),
        "shadow_mass": _smooth(face_shadow, 2.6),
        "background_dark": _smooth(background_dark, 2.2),
        "key_detail": _smooth(key, 1.1),
        "chroma_detail": _smooth(chroma_m, 1.2),
        "final_residual": _smooth(detail, 0.45),
    }


def _estimate_block_count(target: NDArray[np.float32], masks: dict[str, NDArray[np.float32]]) -> int:
    lab = color.srgb_to_lab(target)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    active = float((masks["final_residual"] > 0.06).mean())
    dark = float((masks["background_dark"] > 0.08).mean())
    chroma_load = float((chroma > 18.0).mean())
    gx = np.abs(L[:, 1:] - L[:, :-1])
    gy = np.abs(L[1:, :] - L[:-1, :])
    edge = float(((gx.mean() + gy.mean()) / 18.0).clip(0.0, 1.0))
    estimate = 12.0 + active * 7.0 + dark * 8.0 + chroma_load * 11.0 + edge * 9.0
    return max(14, min(38, int(round(estimate))))


def _block_plan(block_count: int) -> list[tuple[str, str, float, int]]:
    core = [
        ("pale warm support", "pale_warm", 0.26, 6),
        ("pale pink support", "pale_red", 0.24, 5),
        ("pale cool support", "pale_cool", 0.22, 5),
        ("pale detail scaffold", "pale_line", 0.18, 8),
        ("warm face mass", "warm_mass", 0.30, 18),
        ("orange/red build", "red_orange", 0.22, 16),
        ("subtle cool build", "cool_green", 0.16, 16),
        ("brown face shadow", "shadow_mass", 0.24, 18),
        ("warm reinforcement", "warm_mass", 0.24, 20),
        ("red local accents", "red_orange", 0.28, 18),
        ("chroma cells", "chroma_detail", 0.22, 20),
        ("shadow reinforcement", "shadow_mass", 0.32, 20),
        ("small color corrections", "chroma_detail", 0.28, 22),
        ("key begins", "key_detail", 0.24, 14),
        ("warm glaze", "warm_mass", 0.18, 24),
        ("cool/green glaze", "cool_green", 0.24, 22),
        ("red/chroma density", "red_orange", 0.24, 22),
        ("face shadow density", "shadow_mass", 0.34, 22),
        ("background dark field", "background_dark", 0.40, 20),
        ("background color density", "background_dark", 0.48, 24),
        ("key/detail density", "key_detail", 0.42, 16),
        ("chroma correction", "chroma_detail", 0.36, 24),
        ("deep key", "key_detail", 0.48, 18),
        ("final color correction", "final_residual", 0.46, 42),
        ("final residual build", "final_residual", 0.62, 56),
        ("final residual density", "final_residual", 0.78, 72),
        ("final proof match", "final_residual", 0.90, 96),
    ]
    if block_count <= len(core):
        keep = list(range(block_count - 3)) + [len(core) - 3, len(core) - 2, len(core) - 1]
        return [core[i] for i in keep]
    extra = [
        ("extra warm split", "warm_mass", 0.16, 28),
        ("extra chroma split", "chroma_detail", 0.24, 28),
        ("extra cool split", "cool_green", 0.22, 26),
        ("extra shadow split", "shadow_mass", 0.30, 26),
    ]
    out = core[:-3]
    while len(out) < block_count - 3:
        out.append(extra[(len(out) - len(core)) % len(extra)])
    out.extend(core[-3:])
    return out[:block_count]


def _proof_endpoints(block_count: int) -> list[int]:
    batch = 3 if block_count <= 18 else 5 if block_count >= 34 else 4
    endpoints = list(range(batch, block_count + 1, batch))
    if endpoints[-1] != block_count:
        endpoints.append(block_count)
    return endpoints


def build_methodology_proofs(target: NDArray[np.float32]) -> dict[str, Any]:
    paper = np.broadcast_to(forward_render_jax.PAPER_RGB.astype(np.float32), target.shape).copy()
    current = paper.copy()
    masks = _mask_bank(target)
    block_plan = _block_plan(_estimate_block_count(target, masks))
    endpoints = set(_proof_endpoints(len(block_plan)))
    proofs: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    batch_start = 0
    for block_no, (name, mask_name, strength, colors) in enumerate(block_plan, start=1):
        current, block = _apply_block(
            current, target, masks[mask_name],
            strength=strength, colors=colors, seed=1000 + block_no,
        )
        entry = {
            "block_no": block_no, "name": name, "mask": mask_name,
            "strength": strength, "colors": colors, "rgb": block,
            "after_rgb": current.copy(),
        }
        blocks.append(entry)
        if block_no in endpoints:
            batch = blocks[batch_start:block_no]
            batch_preview = paper.copy()
            for item in batch:
                batch_preview = np.minimum(1.0, batch_preview * 0.50 + item["rgb"] * 0.50)
            proofs.append({
                "proof_no": len(proofs) + 1,
                "name": f"Proof {len(proofs) + 1:02d} after B{block_no:02d}",
                "rgb": current.copy(),
                "batch_preview": batch_preview,
                "blocks": batch,
            })
            batch_start = block_no
    return {"proofs": proofs, "blocks": blocks, "final": current.astype(np.float32)}


def _proof_sheet(proofs: list[dict[str, Any]], out_path: Path) -> None:
    thumb_w = 360
    margin = 20
    label_h = 34
    img_h, img_w = proofs[0]["rgb"].shape[:2]
    thumb_h = int(round(thumb_w * img_h / float(img_w)))
    cols = 4
    rows = int(np.ceil(len(proofs) / cols))
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin,
                              rows * (thumb_h + label_h) + (rows + 1) * margin),
                      (245, 241, 230))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for i, proof in enumerate(proofs):
        x = margin + (i % cols) * (thumb_w + margin)
        y = margin + (i // cols) * (thumb_h + label_h + margin)
        draw.text((x, y), str(proof["name"]), fill=(20, 20, 20), font=font)
        img = Image.fromarray(_as_u8(proof["rgb"]), "RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y + label_h))
    sheet.save(out_path)


def _batch_sheet(proofs: list[dict[str, Any]], out_path: Path) -> None:
    thumb_w = 360
    margin = 20
    label_h = 34
    img_h, img_w = proofs[0]["batch_preview"].shape[:2]
    thumb_h = int(round(thumb_w * img_h / float(img_w)))
    cols = 4
    rows = int(np.ceil(len(proofs) / cols))
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin,
                              rows * (thumb_h + label_h) + (rows + 1) * margin),
                      (245, 241, 230))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for i, proof in enumerate(proofs):
        names = ", ".join(block["name"] for block in proof["blocks"])
        x = margin + (i % cols) * (thumb_w + margin)
        y = margin + (i // cols) * (thumb_h + label_h + margin)
        draw.text((x, y), f"Batches for {proof['name']}: {names}"[:48], fill=(20, 20, 20), font=font)
        img = Image.fromarray(_as_u8(proof["batch_preview"]), "RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y + label_h))
    sheet.save(out_path)


def _pull_sheet(blocks: list[dict[str, Any]], out_path: Path) -> None:
    thumb_w = 220
    margin = 16
    label_h = 28
    img_h, img_w = blocks[0]["after_rgb"].shape[:2]
    thumb_h = int(round(thumb_w * img_h / float(img_w)))
    cols = 7 if len(blocks) >= 24 else 6
    rows = int(np.ceil(len(blocks) / cols))
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * margin,
                              rows * (thumb_h + label_h) + (rows + 1) * margin),
                      (245, 241, 230))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for i, block in enumerate(blocks):
        x = margin + (i % cols) * (thumb_w + margin)
        y = margin + (i // cols) * (thumb_h + label_h + margin)
        draw.text((x, y), f"B{block['block_no']:02d} {block['name']}"[:28], fill=(20, 20, 20), font=font)
        img = Image.fromarray(_as_u8(block["after_rgb"]), "RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y + label_h))
    sheet.save(out_path)


def _side_by_side(target: NDArray[np.float32], final: NDArray[np.float32], out_path: Path) -> None:
    left = Image.fromarray(_as_u8(target), "RGB")
    right = Image.fromarray(_as_u8(final), "RGB")
    gap = max(24, left.width // 26)
    sheet = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), (245, 241, 230))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + gap, 0))
    sheet.save(out_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_name = args.run_name or f"methodology-proofs-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _load_rgb(args.input, args.max_side)
    result = build_methodology_proofs(target)
    _save(out_dir / "target.png", target)
    for block in result["blocks"]:
        _save(out_dir / f"pull_{block['block_no']:02d}_after.png", block["after_rgb"])
        _save(out_dir / f"block_{block['block_no']:02d}.png", block["rgb"])
    for proof in result["proofs"]:
        _save(out_dir / f"proof_{proof['proof_no']:02d}.png", proof["rgb"])
        _save(out_dir / f"proof_{proof['proof_no']:02d}_batch_additions.png", proof["batch_preview"])
    _pull_sheet(result["blocks"], out_dir / "methodology_full_pull_preview.png")
    _proof_sheet(result["proofs"], out_dir / "methodology_proof_sheet.png")
    (out_dir / "methodology_proof_preview.png").write_bytes((out_dir / "methodology_proof_sheet.png").read_bytes())
    _batch_sheet(result["proofs"], out_dir / "methodology_batch_additions_sheet.png")
    _save(out_dir / "final_methodology_composite.png", result["final"])
    _side_by_side(target, result["final"], out_dir / "target_vs_methodology_final.png")
    d_e = color.delta_e_summary(result["final"], target)
    summary = {
        "run_name": run_name,
        "input": str(args.input),
        "out_dir": str(out_dir),
        "proof_count": len(result["proofs"]),
        "block_count": len(result["blocks"]),
        "proof_end_blocks": [int(proof["blocks"][-1]["block_no"]) for proof in result["proofs"]],
        "dE_mean": round(float(d_e["dE_mean"]), 3),
        "dE_p95": round(float(d_e["dE_p95"]), 3),
        "note": "methodology proof-state prototype; cumulative proof states, not final CNC plate geometry",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    latest = args.out_root / "latest-emma"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out_dir, target_is_directory=True)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-side", type=int, default=1400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
