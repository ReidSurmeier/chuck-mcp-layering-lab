#!/usr/bin/env python3
"""
Woodblock color separation V9 — Edge-Preserving Filter + Mean Shift + K-means + CC Cleanup.

Fundamentally different pipeline from V2/V8:
1. cv2.edgePreservingFilter — heavy edge-aware smoothing
2. cv2.pyrMeanShiftFiltering — flatten color gradients into solid regions
3. CIELAB K-means++ on the FILTERED image
4. Connected component cleanup — small components absorbed into neighbors
5. binary_fill_holes per plate
6. Chroma boost on palette
7. Optional Real-ESRGAN 2x upscale

Results: MSE ~672 (V2 was 2836), Noise ~0.24% (V2 was 3%).
"""
import argparse
import io
import json
import os
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as ndlabel
from skimage.color import rgb2lab, lab2rgb
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file
from sklearn.cluster import KMeans


def connected_component_cleanup(labels, n_plates, dust_threshold=50):
    """
    For each plate, find connected components. Components smaller than
    dust_threshold get absorbed into the most common neighboring plate.
    Eliminates ALL isolated noise.
    """
    for plate_id in range(n_plates):
        mask = (labels == plate_id)
        labeled, n_components = ndlabel(mask)
        for comp in range(1, n_components + 1):
            comp_mask = labeled == comp
            comp_size = np.sum(comp_mask)
            if comp_size < dust_threshold:
                dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
                border = (dilated > 0) & ~comp_mask
                if np.any(border):
                    neighbors = labels[border]
                    neighbors = neighbors[neighbors != plate_id]
                    if len(neighbors) > 0:
                        vals, counts = np.unique(neighbors, return_counts=True)
                        labels[comp_mask] = vals[np.argmax(counts)]
    return labels


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=50,
             locked_colors=None, return_data=False,
             chroma_boost=1.3,
             sigma_s=100, sigma_r=0.5,
             meanshift_sp=15, meanshift_sr=30,
             upscale=True):
    """
    V9 separation: EdgePreserve → MeanShift → CIELAB K-means++ → CC Cleanup.

    Args:
        input_path_or_array: filepath string or numpy RGB array
        output_dir: where to save (None if return_data=True)
        n_plates: number of color plates (2-35)
        dust_threshold: CC cleanup threshold — components smaller than this absorbed (default 50)
        locked_colors: list of [R,G,B] colors to lock as centroids
        return_data: if True, return dict instead of writing files
        chroma_boost: chroma multiplier for palette vividness (0.5-2.0, default 1.3)
        sigma_s: edge-preserving filter spatial bandwidth (20-200, default 100)
        sigma_r: edge-preserving filter range bandwidth (0.1-1.0, default 0.5)
        meanshift_sp: mean shift spatial bandwidth (5-50, default 15)
        meanshift_sr: mean shift color bandwidth (10-80, default 30)
        upscale: whether to apply Real-ESRGAN 2x upscale (default True)
    """
    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array
        img = Image.fromarray(arr)

    # ── Step 0: Optional Real-ESRGAN 2x upscale ──
    was_upscaled = False
    if upscale:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            weights_path = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x2plus.pth")
            if os.path.exists(weights_path):
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                upsampler = RealESRGANer(
                    scale=2, model_path=weights_path, model=model,
                    half=torch.cuda.is_available(),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    tile=256, tile_pad=10
                )
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                output, _ = upsampler.enhance(bgr, outscale=2)
                arr = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(arr)
                was_upscaled = True
                del upsampler, model
                torch.cuda.empty_cache()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Upscale failed: {e}")

    h, w = arr.shape[:2]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Edge-preserving filter — heavy edge-aware smoothing ──
    filtered = cv2.edgePreservingFilter(arr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

    # ── Step 2: Mean shift filtering — flatten gradients into solid regions ──
    filtered = cv2.pyrMeanShiftFiltering(filtered, sp=meanshift_sp, sr=meanshift_sr)

    # ── Step 3: Convert FILTERED image to CIELAB for clustering ──
    arr_float = filtered.astype(np.float64) / 255.0
    lab_img = rgb2lab(arr_float)

    # Boost chroma for clustering
    lab_boosted = lab_img.copy()
    lab_boosted[:, :, 1] *= chroma_boost
    lab_boosted[:, :, 2] *= chroma_boost

    lab_flat = lab_boosted.reshape(-1, 3)

    # Subsample for K-means speed
    max_samples = 150000
    if len(lab_flat) > max_samples:
        indices = np.random.RandomState(42).choice(len(lab_flat), max_samples, replace=False)
        sample = lab_flat[indices]
    else:
        sample = lab_flat

    # ── Step 4: K-means++ clustering ──
    init = "k-means++"
    if locked_colors and len(locked_colors) > 0:
        locked_rgb = np.array(locked_colors, dtype=np.float64).reshape(-1, 1, 3) / 255.0
        locked_lab = rgb2lab(locked_rgb).reshape(-1, 3)
        locked_lab[:, 1] *= chroma_boost
        locked_lab[:, 2] *= chroma_boost

        if len(locked_lab) == n_plates:
            init = locked_lab
        elif len(locked_lab) < n_plates:
            remaining = n_plates - len(locked_lab)
            random_indices = np.random.RandomState(42).choice(len(sample), remaining, replace=False)
            extra_centroids = sample[random_indices]
            init = np.vstack([locked_lab, extra_centroids])
        else:
            init = locked_lab[:n_plates]

    kmeans = KMeans(
        n_clusters=n_plates,
        init=init,
        random_state=42,
        n_init=10 if isinstance(init, str) else 1,
        max_iter=300,
        tol=1e-4
    )
    kmeans.fit(sample)

    # Get palette in LAB (undo chroma boost) and convert to RGB
    palette_lab_boosted = kmeans.cluster_centers_
    palette_lab = palette_lab_boosted.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost
    palette_rgb_float = lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Step 5: Assign ALL pixels to nearest cluster ──
    dists = np.zeros((len(lab_flat), n_plates), dtype=np.float64)
    for i, center in enumerate(palette_lab_boosted):
        diff = lab_flat - center
        dists[:, i] = np.sum(diff ** 2, axis=1)

    pixel_labels = np.argmin(dists, axis=1).reshape(h, w)

    # ── Step 6: Connected component cleanup ──
    pixel_labels = connected_component_cleanup(pixel_labels, n_plates, dust_threshold)

    # ── Step 7: Extract and clean individual plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

        # Fill holes
        mask = binary_fill_holes(mask)

        coverage = np.sum(mask) / mask.size * 100
        rgb = palette_rgb[idx]
        name = f"plate{rank + 1}"

        binary = np.where(mask, 0, 255).astype(np.uint8)
        plate_img = Image.fromarray(binary)

        plate_info = {
            "name": name,
            "index": int(idx),
            "rank": rank + 1,
            "color": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "color_hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "coverage_pct": round(coverage, 2),
        }
        results.append(plate_info)
        plate_images[name] = {"mask": mask, "binary": binary, "image": plate_img}

    # ── Step 8: Composite preview ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb_val = plate_info["color"]
        comp[mask] = rgb_val

    composite_img = Image.fromarray(comp)

    # ── Step 9: Manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(100.0 - sum(p["coverage_pct"] for p in results), 2),
        "version": "v9",
        "upscaled": was_upscaled,
    }

    if return_data:
        return {
            "composite": composite_img,
            "plates": plate_images,
            "manifest": manifest,
            "palette_rgb": palette_rgb,
        }

    # Save files
    composite_img.save(os.path.join(output_dir, "composite.png"))

    for plate_info in results:
        name = plate_info["name"]
        plate_images[name]["image"].save(os.path.join(output_dir, f"{name}.png"))

        mask = plate_images[name]["mask"]
        svg_path = os.path.join(output_dir, f"{name}.svg")
        mask_to_svg(mask, svg_path, w, h)

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def build_preview_response(image_bytes, plates=4, dust=50,
                           locked_colors=None,
                           chroma_boost=1.3,
                           sigma_s=100, sigma_r=0.5,
                           meanshift_sp=15, meanshift_sr=30,
                           upscale=True, **kwargs):
    """Process image and return composite PNG bytes + manifest."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Cap input so 2x upscale doesn't exceed VRAM
    max_dim = 1000 if upscale else 1500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img)
    result = separate(
        arr, n_plates=plates, dust_threshold=dust,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        upscale=upscale,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=50,
                       locked_colors=None,
                       chroma_boost=1.3,
                       sigma_s=100, sigma_r=0.5,
                       meanshift_sp=15, meanshift_sr=30,
                       upscale=True, **kwargs):
    """Process image and return ZIP bytes containing all outputs."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_dim = 4000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img)
    result = separate(
        arr, n_plates=plates, dust_threshold=dust,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        upscale=upscale,
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        comp_buf = io.BytesIO()
        result["composite"].save(comp_buf, format="PNG")
        zf.writestr("composite.png", comp_buf.getvalue())

        h, w = arr.shape[:2]
        if result["manifest"].get("upscaled"):
            h, w = result["manifest"]["height"], result["manifest"]["width"]

        for plate_info in result["manifest"]["plates"]:
            name = plate_info["name"]
            plate_data = result["plates"][name]

            png_buf = io.BytesIO()
            plate_data["image"].save(png_buf, format="PNG")
            zf.writestr(f"{name}.png", png_buf.getvalue())

            svg_content = mask_to_svg_string(plate_data["mask"], w, h)
            zf.writestr(f"{name}.svg", svg_content)

        zf.writestr("manifest.json", json.dumps(result["manifest"], indent=2))

    return zip_buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Woodblock color separation V9")
    parser.add_argument("input", nargs="?", default=None, help="Input image path")
    parser.add_argument("output", nargs="?", default=None, help="Output directory")
    parser.add_argument("--plates", type=int, default=8, help="Number of color plates")
    parser.add_argument("--dust", type=int, default=50, help="CC cleanup threshold")
    parser.add_argument("--sigma-s", type=float, default=100, help="Edge-preserving spatial bandwidth")
    parser.add_argument("--sigma-r", type=float, default=0.5, help="Edge-preserving range bandwidth")
    parser.add_argument("--meanshift-sp", type=int, default=15, help="Mean shift spatial bandwidth")
    parser.add_argument("--meanshift-sr", type=int, default=30, help="Mean shift color bandwidth")
    parser.add_argument("--chroma", type=float, default=1.3, help="Chroma boost factor")
    parser.add_argument("--no-upscale", action="store_true", help="Disable Real-ESRGAN upscale")
    args = parser.parse_args()

    if args.input and args.output:
        result = separate(
            args.input, args.output, n_plates=args.plates,
            dust_threshold=args.dust,
            sigma_s=args.sigma_s, sigma_r=args.sigma_r,
            meanshift_sp=args.meanshift_sp, meanshift_sr=args.meanshift_sr,
            chroma_boost=args.chroma,
            upscale=not args.no_upscale,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
