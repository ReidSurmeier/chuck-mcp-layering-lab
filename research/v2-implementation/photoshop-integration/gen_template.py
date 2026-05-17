"""Production template generator for chuck-mcp v2.

Generates a Photoshop-ready PSD with:
- Background: source image at 30% opacity (locked reference)
- Kento overlay: corner L-marks + center crosshair (locked reference)
- 4-9 empty paintable slots, named by semantic role

Usage (CLI):
    python gen_template.py --source input.png --slots slot_01_yellow,slot_02_pink ... --out template.psd

Usage (programmatic):
    from gen_template import generate_template
    psd_path = generate_template(
        source_image=PIL.Image.open('input.png'),
        slot_names=['slot_01_yellow_cheek', 'slot_02_pink_lip', ...],
        paper_size='A4',
        dpi=300,
        out_path='template.psd',
    )

Tested: psd-tools==1.17.0, Pillow==12.2.0, numpy==2.4.5 on Linux 6.6 (WSL2).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage


# --- Paper sizes (portrait, mm) ---
PAPER_SIZES_MM = {
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "A6": (105, 148),
    # Mokuhanga common: ban-shu (small), chu-han (medium), oban (large).
    "oban":   (250, 360),   # roughly  ~10x14 inches
    "chuban": (190, 250),
    "koban":  (120, 170),
}


@dataclass
class TemplateSpec:
    """Locked geometry for a chuck-mcp v2 template."""

    width_px: int
    height_px: int
    dpi: int
    kento_inset_mm: float = 10.0
    kento_arm_mm: float = 15.0
    kento_stroke_mm: float = 0.5
    crosshair_arm_mm: float = 5.0
    source_opacity_pct: float = 30.0  # 30% = opacity byte 77

    @property
    def kento_inset_px(self) -> int:
        return round(self.kento_inset_mm * self.dpi / 25.4)

    @property
    def kento_arm_px(self) -> int:
        return round(self.kento_arm_mm * self.dpi / 25.4)

    @property
    def kento_stroke_px(self) -> int:
        return max(1, round(self.kento_stroke_mm * self.dpi / 25.4))

    @property
    def crosshair_arm_px(self) -> int:
        return round(self.crosshair_arm_mm * self.dpi / 25.4)

    @property
    def source_opacity_byte(self) -> int:
        return round(self.source_opacity_pct / 100.0 * 255)


def mm_to_px(mm: float, dpi: int) -> int:
    """Convert millimetres to pixels at given DPI."""
    return round(mm * dpi / 25.4)


def make_template_spec(paper_size: str = "A4", dpi: int = 300) -> TemplateSpec:
    """Build a TemplateSpec from a paper-size key."""
    if paper_size not in PAPER_SIZES_MM:
        raise ValueError(f"Unknown paper_size {paper_size!r}; valid: {sorted(PAPER_SIZES_MM)}")
    w_mm, h_mm = PAPER_SIZES_MM[paper_size]
    return TemplateSpec(
        width_px=mm_to_px(w_mm, dpi),
        height_px=mm_to_px(h_mm, dpi),
        dpi=dpi,
    )


def render_source_layer(source: Image.Image, spec: TemplateSpec) -> Image.Image:
    """Fit/letterbox source image to template canvas. Returns RGB image at canvas size."""
    canvas = Image.new("RGB", (spec.width_px, spec.height_px), (255, 255, 255))
    # Fit source into canvas while preserving aspect.
    src = source.convert("RGB").copy()
    src.thumbnail((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
    x = (spec.width_px - src.width) // 2
    y = (spec.height_px - src.height) // 2
    canvas.paste(src, (x, y))
    return canvas


def render_kento_layer(spec: TemplateSpec) -> Image.Image:
    """Generate the kento overlay: corner Ls + center crosshair, black on white.

    Layer is full canvas size, RGB. Use layer opacity to make the white transparent
    is NOT supported in psd-tools directly (it uses pixel data, not blend mode).
    Instead: white pixels in this layer will composite as white over background,
    but at opacity 255 they fully occlude. Compromise: use 'screen-like' approach —
    paint black on white at opacity 255. White overlay washes background visually,
    but corners + crosshair are crisply visible. User paints UNDER this layer.

    Better approach (recommended): set layer blend mode to 'darken' or 'multiply'
    so white = transparent visually. psd-tools writes blend_mode via
    layer.blend_mode = BlendMode.MULTIPLY. See psd_tools.constants.BlendMode.

    For v1 template, paint corners black on transparent — we use an L-mask trick.
    """
    img = Image.new("RGB", (spec.width_px, spec.height_px), (255, 255, 255))
    d = ImageDraw.Draw(img)
    W, H = spec.width_px, spec.height_px
    pad = spec.kento_inset_px
    arm = spec.kento_arm_px
    line = spec.kento_stroke_px

    def L(x: int, y: int, flip_x: bool, flip_y: bool) -> None:
        dx = -1 if flip_x else 1
        dy = -1 if flip_y else 1
        d.line([(x, y), (x + dx * arm, y)], fill=(0, 0, 0), width=line)
        d.line([(x, y), (x, y + dy * arm)], fill=(0, 0, 0), width=line)

    # NW, NE, SW, SE — L opens toward the canvas center
    L(pad,       pad,       flip_x=False, flip_y=False)
    L(W - pad,   pad,       flip_x=True,  flip_y=False)
    L(pad,       H - pad,   flip_x=False, flip_y=True)
    L(W - pad,   H - pad,   flip_x=True,  flip_y=True)

    # Center crosshair (optional but standard for mokuhanga)
    cx, cy = W // 2, H // 2
    ch = spec.crosshair_arm_px
    d.line([(cx - ch, cy), (cx + ch, cy)], fill=(0, 0, 0), width=line)
    d.line([(cx, cy - ch), (cx, cy + ch)], fill=(0, 0, 0), width=line)
    return img


def generate_template(
    source_image: Image.Image,
    slot_names: Iterable[str],
    out_path: str | Path,
    paper_size: str = "A4",
    dpi: int = 300,
    spec: TemplateSpec | None = None,
) -> Path:
    """Generate a chuck-mcp v2 PSD template ready for Photoshop painting.

    Args:
        source_image: PIL image to use as 30% background reference.
        slot_names: Iterable of layer names. Convention: "slot_NN_role_subject"
            e.g. ["slot_01_light_yellow_cheek", "slot_02_pink_lip"].
            Must be 4-9 names.
        out_path: Where to save the .psd
        paper_size: One of PAPER_SIZES_MM keys.
        dpi: Print DPI. 300 standard; use 600 for fine-detail relief carving.
        spec: Override paper_size+dpi entirely.

    Returns:
        Path of the saved .psd.
    """
    slots = list(slot_names)
    if not (4 <= len(slots) <= 9):
        raise ValueError(f"slot_names must have 4-9 entries, got {len(slots)}")
    if spec is None:
        spec = make_template_spec(paper_size=paper_size, dpi=dpi)

    psd = PSDImage.new(mode="RGB", size=(spec.width_px, spec.height_px), depth=8)

    # 1. Source reference layer at 30% opacity
    source_layer = render_source_layer(source_image, spec)
    psd.create_pixel_layer(
        source_layer,
        name="[REF] source_30pct_DO_NOT_PAINT",
        top=0,
        left=0,
        opacity=spec.source_opacity_byte,
    )

    # 2. Kento mark layer at 100% (always visible)
    kento_layer = render_kento_layer(spec)
    psd.create_pixel_layer(
        kento_layer,
        name="[REF] kento_marks_DO_NOT_PAINT",
        top=0,
        left=0,
        opacity=255,
    )

    # 3. Empty paintable slots — 1x1 placeholder, full opacity
    placeholder = Image.new("RGB", (1, 1), (0, 0, 0))
    for slot_name in slots:
        # Convention check: must start with "slot_NN_" where NN is 2-digit index
        if not slot_name.startswith("slot_"):
            raise ValueError(f"slot name must start with 'slot_', got {slot_name!r}")
        psd.create_pixel_layer(
            placeholder,
            name=slot_name,
            top=0,
            left=0,
            opacity=255,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(out_path))
    return out_path


def make_manifest_dict(
    spec: TemplateSpec,
    slot_names: list[str],
    template_path: str,
) -> dict:
    """Build the JSON manifest accompanying the PSD.

    This manifest is stored alongside the template so the ingestion script knows
    exactly what to expect.
    """
    pad = spec.kento_inset_px
    arm = spec.kento_arm_px
    W, H = spec.width_px, spec.height_px
    return {
        "schema_version": "chuck-mcp-v2-template/1.0",
        "template_path": template_path,
        "canvas": {
            "width_px": W,
            "height_px": H,
            "dpi": spec.dpi,
            "color_mode": "RGB",
            "depth": 8,
        },
        "kento": {
            "inset_mm": spec.kento_inset_mm,
            "arm_mm": spec.kento_arm_mm,
            "stroke_mm": spec.kento_stroke_mm,
            "corners_px": {
                "nw": [pad, pad],
                "ne": [W - pad, pad],
                "sw": [pad, H - pad],
                "se": [W - pad, H - pad],
            },
            "center_crosshair_px": [W // 2, H // 2],
            # Bounding boxes for each kento mark (used in mask-overlap validation)
            "no_paint_zones": _kento_bboxes(spec),
        },
        "slots": [
            {"index": i + 1, "name": name}
            for i, name in enumerate(slot_names)
        ],
        "validation": {
            "binary_threshold_pct": 99.0,
            "binary_tolerance": 2,
            "min_coverage_pct": 0.1,
            "max_coverage_pct": 80.0,
        },
    }


def _kento_bboxes(spec: TemplateSpec) -> list[dict]:
    """Bounding boxes for each kento mark region (no-paint zones)."""
    W, H = spec.width_px, spec.height_px
    pad = spec.kento_inset_px
    arm = spec.kento_arm_px
    stroke = spec.kento_stroke_px
    margin = stroke * 2  # generous margin around marks
    cx, cy = W // 2, H // 2
    ch = spec.crosshair_arm_px
    return [
        # Each: x0, y0, x1, y1 (inclusive)
        {"label": "nw_kento", "bbox": [pad - margin, pad - margin, pad + arm + margin, pad + arm + margin]},
        {"label": "ne_kento", "bbox": [W - pad - arm - margin, pad - margin, W - pad + margin, pad + arm + margin]},
        {"label": "sw_kento", "bbox": [pad - margin, H - pad - arm - margin, pad + arm + margin, H - pad + margin]},
        {"label": "se_kento", "bbox": [W - pad - arm - margin, H - pad - arm - margin, W - pad + margin, H - pad + margin]},
        {"label": "center_crosshair", "bbox": [cx - ch - margin, cy - ch - margin, cx + ch + margin, cy + ch + margin]},
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate chuck-mcp v2 PSD template")
    p.add_argument("--source", required=True, help="Path to source image (PNG/JPEG)")
    p.add_argument(
        "--slots",
        required=True,
        help="Comma-separated slot names, e.g. slot_01_yellow,slot_02_pink",
    )
    p.add_argument("--out", required=True, help="Output PSD path")
    p.add_argument("--paper", default="A4", choices=sorted(PAPER_SIZES_MM))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--manifest", default=None, help="Optional manifest JSON output path")
    args = p.parse_args()

    src = Image.open(args.source)
    slot_names = [s.strip() for s in args.slots.split(",") if s.strip()]

    spec = make_template_spec(paper_size=args.paper, dpi=args.dpi)
    out = generate_template(
        source_image=src,
        slot_names=slot_names,
        out_path=args.out,
        spec=spec,
    )
    print(f"Wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    if args.manifest:
        import json
        manifest = make_manifest_dict(spec=spec, slot_names=slot_names, template_path=str(out))
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
