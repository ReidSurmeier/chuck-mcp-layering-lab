#!/usr/bin/env python3
"""Run a v23 pipeline validation pass and emit review artifacts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from backend.mcp.tools import hitl, introspection, planning  # noqa: E402
from backend.services.v23 import orchestrator  # noqa: E402
from backend.services.v23.core import color, forward_render_jax  # noqa: E402
from scripts.carousel_slides import build_carousel_slides  # noqa: E402
DEFAULT_INPUT = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")
DEFAULT_OUTPUT_ROOT = Path("/srv/woodblock-share/output-images")
DEFAULT_CLEAN_ROOT = Path("/srv/woodblock-share/chuck-clean-outputs")
DEFAULT_CAROUSEL_ROOT = Path("/srv/woodblock-share/chuck-carousel-slides")

REFERENCE_OUTPUTS: tuple[tuple[str, Path], ...] = (
    ("old_m10_clean_9_pull",
     Path("/srv/woodblock-share/chuck-clean-outputs/chuck-mcp-thorough-m10-main-20260512-150141/00_final_composite.png")),
    ("old_rolebased_joint_9_pull",
     Path("/srv/woodblock-share/chuck-clean-outputs/chuck-rolebased-joint-main-20260512-160644/00_final_composite.png")),
    ("lab_tints_jigsaw_v2_8_pull",
     Path("/srv/woodblock-share/chuck-clean-outputs/chuck-layering-lab-tints-jigsaw-v2-main-20260514-142316/final_composite.png")),
    ("cellgraph_no_repair_8_pull",
     Path("/srv/woodblock-share/chuck-clean-outputs/chuck-cellgraph-fast-main-20260514-154234/final_composite.png")),
    ("old_residual_25_plate",
     Path("/srv/woodblock-share/output-images/emma-expanded-palette-sharp-20260512-141922/diagnostics/residual_stack_variant_0022_final.png")),
    ("old_residual_32_plate",
     Path("/srv/woodblock-share/output-images/emma-expanded-palette-sharp-20260512-141922/diagnostics/residual_stack_variant_0004_final.png")),
)


def _branch_name() -> str:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "local"
    return out or "detached"


def _default_run_name() -> str:
    branch = _branch_name().replace("/", "-")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"chuck-v23-{branch}-{stamp}"


def _as_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_u8(rgb), "RGB").save(path)


def _render_incremental(
    alpha_stack: np.ndarray,
    pigment_idx: np.ndarray,
) -> list[np.ndarray]:
    pigments = forward_render_jax.PIGMENT_TABLE[pigment_idx]
    h, w = alpha_stack.shape[1:]
    composite = np.broadcast_to(forward_render_jax.PAPER_RGB, (h, w, 3)).astype(np.float32).copy()
    frames: list[np.ndarray] = []
    for alpha, pigment in zip(alpha_stack, pigments, strict=True):
        a = np.clip(alpha[..., None], 0.0, 1.0)
        composite = composite * (1.0 - a) + pigment[None, None, :] * a
        frames.append(composite.astype(np.float32).copy())
    return frames


def _render_final(alpha_stack: np.ndarray, pigment_idx: np.ndarray) -> np.ndarray:
    alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
    return np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(pigment_idx, dtype=jnp.int32),
        )
    ).astype(np.float32)


def _sheet(
    items: list[tuple[str, np.ndarray]],
    path: Path,
    *,
    cols: int = 3,
    thumb_w: int = 390,
    label_h: int = 34,
) -> None:
    if not items:
        return
    h, w = items[0][1].shape[:2]
    thumb_h = max(1, int(round(thumb_w * h / float(w))))
    rows = int(np.ceil(len(items) / cols))
    margin = 18
    sheet_size = (cols * thumb_w + (cols + 1) * margin,
                  rows * (thumb_h + label_h) + (rows + 1) * margin)
    sheet = Image.new("RGB", sheet_size, (245, 241, 230))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (label, rgb) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        draw.text((x, y), label[:64], fill=(20, 20, 20), font=font)
        img = Image.fromarray(_as_u8(rgb), "RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _mask_sheet(
    alpha_stack: np.ndarray,
    pigment_names: list[str],
    out_path: Path,
) -> None:
    items: list[tuple[str, np.ndarray]] = []
    for i, alpha in enumerate(alpha_stack):
        gray = np.clip(alpha, 0.0, 1.0)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        items.append((f"Alpha {i + 1:02d}: {pigment_names[i]}", rgb))
    _sheet(items, out_path, cols=3, thumb_w=390, label_h=34)


def _target_vs_composite(target: np.ndarray, composite: np.ndarray, out_path: Path) -> None:
    target_img = Image.fromarray(_as_u8(target), "RGB")
    comp_img = Image.fromarray(_as_u8(composite), "RGB")
    gap = max(24, target_img.width // 28)
    sheet = Image.new("RGB", (target_img.width + comp_img.width + gap,
                              max(target_img.height, comp_img.height)), (245, 241, 230))
    sheet.paste(target_img, (0, 0))
    sheet.paste(comp_img, (target_img.width + gap, 0))
    sheet.save(out_path)


def _cell_graph_preview(target: np.ndarray, labels_path: str | None, out_path: Path) -> None:
    if not labels_path or not Path(labels_path).is_file():
        return
    labels = np.load(labels_path)
    img = _as_u8(target)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    img[boundary] = np.array([255, 220, 0], dtype=np.uint8)
    Image.fromarray(img, "RGB").save(out_path)


def _artwork_rect_mask(target: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            target[:64].reshape(-1, 3),
            target[-64:].reshape(-1, 3),
            target[:, :64].reshape(-1, 3),
            target[:, -64:].reshape(-1, 3),
        ],
        axis=0,
    )
    paper_lab = np.median(color.srgb_to_lab(border), axis=0)
    target_lab = color.srgb_to_lab(target)
    paper_delta = color.delta_e76(target_lab, paper_lab)
    off_paper = paper_delta > 4.0
    ys, xs = np.where(off_paper)
    mask = np.zeros(off_paper.shape, dtype=bool)
    if ys.size == 0:
        mask[:] = True
        return mask
    mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1] = True
    return mask


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _comparison(
    target_path: Path,
    target: np.ndarray,
    current_name: str,
    current_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    target_img = Image.open(target_path).convert("RGB")
    size = target_img.size
    mask = _artwork_rect_mask(target)
    rows: list[dict[str, Any]] = []
    refs = list(REFERENCE_OUTPUTS) + [(current_name, current_path)]
    for name, path in refs:
        if not path.is_file():
            rows.append({"name": name, "path": str(path), "missing": True})
            continue
        img = _load_rgb(path, size)
        d_e = color.rgb_delta_e76(img, target)
        lab = color.srgb_to_lab(img)
        chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "mean_dE_mask": round(float(np.mean(d_e[mask])), 3),
                "p95_dE_mask": round(float(np.percentile(d_e[mask], 95)), 3),
                "mean_chroma_mask": round(float(np.mean(chroma[mask])), 3),
            }
        )

    payload = {
        "target": str(target_path),
        "artwork_rect_mask_pct": round(float(mask.mean() * 100.0), 3),
        "rows": rows,
    }
    (out_dir / "comparison_against_old.json").write_text(json.dumps(payload, indent=2) + "\n")
    _comparison_sheet(target_path, rows, out_dir / "comparison_against_old_contact_sheet.png")
    return payload


def _comparison_sheet(target_path: Path, rows: list[dict[str, Any]], out_path: Path) -> None:
    items: list[tuple[str, Path, str]] = [("target_input", target_path, "")]
    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_file():
            continue
        metric = (
            f"dE {row['mean_dE_mask']}/{row['p95_dE_mask']} C {row['mean_chroma_mask']}"
            if not row.get("missing")
            else "missing"
        )
        items.append((str(row["name"]), path, metric))

    target_img = Image.open(target_path).convert("RGB")
    thumb_w = 360
    thumb_h = int(round(thumb_w * target_img.height / float(target_img.width)))
    cols = 3
    margin = 18
    label_h = 54
    sheet_rows = int(np.ceil(len(items) / cols))
    sheet_size = (cols * thumb_w + (cols + 1) * margin,
                  sheet_rows * (thumb_h + label_h) + (sheet_rows + 1) * margin)
    sheet = Image.new("RGB", sheet_size, (245, 241, 230))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (name, path, metric) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        draw.text((x, y), name[:46], fill=(20, 20, 20), font=font)
        if metric:
            draw.text((x, y + 18), metric, fill=(20, 20, 20), font=font)
        sheet.paste(img, (x, y + label_h))
    sheet.save(out_path)


def _copy_clean_to_output(clean_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in clean_dir.iterdir():
        if src.is_file():
            dst = output_dir / src.name
            dst.write_bytes(src.read_bytes())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_sam:
        os.environ["WOODBLOCK_DISABLE_SAM"] = "1"
    if args.role_warmup:
        os.environ["WOODBLOCK_ROLE_WARMUP"] = "1"
    if args.max_pixels:
        os.environ["WOODBLOCK_SOLVER_MAX_PIXELS"] = str(args.max_pixels)

    run_name = args.run_name or _default_run_name()
    clean_dir = args.clean_root / run_name
    output_dir = args.output_root / run_name
    clean_dir.mkdir(parents=True, exist_ok=True)

    plan = orchestrator.run_pipeline_partial(
        str(args.input),
        solve_profile=args.solve_profile,
        m_prior=args.m_prior,
    )
    if not plan.alpha_stack_path:
        raise RuntimeError(f"solver did not emit alpha_stack for plan {plan.plan_id}")

    alpha_stack = np.load(plan.alpha_stack_path)
    pigment_idx = np.load(Path(plan.alpha_stack_path).parent / "pigment_idx.npy")
    target = np.load(Path(plan.alpha_stack_path).parent / "target.npy")
    pigment_names = [forward_render_jax.PIGMENT_NAMES[int(pid)] for pid in pigment_idx.tolist()]
    final = _render_final(alpha_stack, pigment_idx)
    frames = _render_incremental(alpha_stack, pigment_idx)

    _save_rgb(clean_dir / "target.png", target)
    _save_rgb(clean_dir / "final_composite.png", final)
    _target_vs_composite(target, final, clean_dir / "target_vs_composite.png")
    for i, frame in enumerate(frames, start=1):
        _save_rgb(clean_dir / f"cumulative_pull_{i:02d}.png", frame)
    _sheet(
        [(f"Pull{i + 1:02d}: {pigment_names[i]}", frame) for i, frame in enumerate(frames)],
        clean_dir / "cumulative_pulls_contact_sheet.png",
        cols=3,
    )
    _mask_sheet(alpha_stack, pigment_names, clean_dir / "alpha_masks_contact_sheet.png")
    _cell_graph_preview(target, plan.cell_labels_path, clean_dir / "cell_graph_preview.png")

    d_e = color.delta_e_summary(final, target)
    printability = introspection.score_printability(plan.plan_id)
    reorg = hitl.propose_plate_reorganization(plan.plan_id)
    batch_result = planning.plan_production_batches(plan.plan_id)
    batch_data = batch_result.data if batch_result.ok else None
    carousel_dir = build_carousel_slides(
        run_name=run_name, carousel_root=args.carousel_root, target_rgb=target,
        final_rgb=final, alpha_stack=alpha_stack, pigment_idx=pigment_idx,
        cumulative_frames=frames, batch_plan=batch_data,
    )
    comparison = _comparison(args.input, target, run_name, clean_dir / "final_composite.png", clean_dir)

    summary = {
        "run_name": run_name,
        "input_path": str(args.input),
        "clean_dir": str(clean_dir),
        "full_dir": str(output_dir),
        "carousel_dir": str(carousel_dir),
        "plan_id": plan.plan_id,
        "solve_profile": plan.solve_profile,
        "m_prior": plan.m_prior,
        "solver_status": plan.solver_status,
        "solver_wall_s": plan.solver_wall_s,
        "solver_optimized_shape": plan.solver_optimized_shape,
        "solver_downsample_scale": plan.solver_downsample_scale,
        "impression_count": len(plan.impressions),
        "block_count": plan.block_count,
        "pigment_order": pigment_names,
        "recomputed_dE_mean": round(float(d_e["dE_mean"]), 3),
        "recomputed_dE_p95": round(float(d_e["dE_p95"]), 3),
        "cell_graph_summary": plan.cell_graph_summary,
        "jigsaw_summary": plan.jigsaw_summary,
        "printability_repair_summary": plan.printability_repair_summary,
        "printability": printability.data if printability.ok else {"errors": [e.code for e in printability.errors]},
        "plate_reorganization": reorg.data if reorg.ok else {"errors": [e.code for e in reorg.errors]},
        "production_batches": batch_data if batch_data else {"errors": [e.code for e in batch_result.errors]},
        "comparison": comparison,
    }
    (clean_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _copy_clean_to_output(clean_dir, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--solve-profile", choices=("fast", "default", "thorough"), default="fast")
    parser.add_argument("--m-prior", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--role-warmup", action="store_true")
    parser.add_argument("--disable-sam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--carousel-root", type=Path, default=DEFAULT_CAROUSEL_ROOT)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(
        {
            "run_name": summary["run_name"],
            "plan_id": summary["plan_id"],
            "clean_dir": summary["clean_dir"],
            "full_dir": summary["full_dir"],
            "carousel_dir": summary["carousel_dir"],
            "dE_mean": summary["recomputed_dE_mean"],
            "dE_p95": summary["recomputed_dE_p95"],
            "printability_score": summary["printability"].get("score_0_100"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
