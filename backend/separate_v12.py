#!/usr/bin/env python3
"""
Woodblock color separation V12 — Performance-optimized V11.

New in V12:
1. Vectorized distance calc via scipy cdist (biggest speedup)
2. MiniBatchKMeans instead of KMeans (3-5x faster fitting)
3. Faster connected component cleanup (vectorized size filtering)
4. Reduced max_samples (80k vs 150k)
5. Smaller Gaussian blur kernel (5,5) for smoothing
6. Removed binary_fill_holes (CC cleanup already handles it)
7. LRU-bounded upscale cache (max 5 entries)
8. Optional OKLab color space
"""
import argparse
import hashlib
import io
import json
import os
import zipfile
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import label as ndlabel
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file
from sklearn.cluster import MiniBatchKMeans
from shared_cache import LRUCache


_upscale_cache = LRUCache(maxsize=5)


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes[:4096]).hexdigest()


def upscale_2x(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run Real-ESRGAN 2x on an RGB numpy array. Returns (result, success)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weights_path = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x2plus.pth")
        if not os.path.exists(weights_path):
            return arr, False

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2, model_path=weights_path, model=model,
            half=torch.cuda.is_available(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            tile=256, tile_pad=10
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(bgr, outscale=2)
        result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        del upsampler, model
        torch.cuda.empty_cache()
        return result, True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Upscale failed: {e}")
        return arr, False


def upscale_and_cache(image_bytes: bytes) -> tuple[str, bool, bool]:
    """Upscale image and store in cache. Returns (hash, cached, success)."""
    img_hash = _image_hash(image_bytes)
    if img_hash in _upscale_cache:
        return img_hash, True, True
    arr = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    # Cap input so 2x upscale doesn't exceed VRAM
    h, w = arr.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        arr = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))
    upscaled, success = upscale_2x(arr)
    if success:
        _upscale_cache[img_hash] = upscaled
    return img_hash, False, success


# ── OKLab color space ──
def rgb_to_oklab(rgb_float):
    """Convert linear sRGB (0-1 float) to OKLab."""
    # sRGB to linear
    linear = np.where(rgb_float <= 0.04045, rgb_float / 12.92,
                      ((rgb_float + 0.055) / 1.055) ** 2.4)
    lms_l = 0.4122214708 * linear[..., 0] + 0.5363325363 * linear[..., 1] + 0.0514459929 * linear[..., 2]
    m = 0.2119034982 * linear[..., 0] + 0.6806995451 * linear[..., 1] + 0.1073969566 * linear[..., 2]
    s = 0.0883024619 * linear[..., 0] + 0.2817188376 * linear[..., 1] + 0.6299787005 * linear[..., 2]
    l_ = np.cbrt(lms_l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, A, B], axis=-1)


def connected_component_cleanup(labels, n_plates, dust_threshold=150):
    """
    For each plate, find connected components. Components smaller than
    dust_threshold get absorbed into the most common neighboring plate.
    Optimized: batch all small components per plate using morphological mode filter.
    """
    h, w = labels.shape

    for plate_id in range(n_plates):
        mask = labels == plate_id
        labeled, n_comp = ndlabel(mask)
        if n_comp <= 1:
            continue

        # Vectorized: get component sizes all at once
        comp_sizes = np.bincount(labeled.ravel())
        # Mark all small components at once (skip background = index 0)
        small_mask = np.zeros_like(mask)
        for comp_id in range(1, len(comp_sizes)):
            if comp_sizes[comp_id] < dust_threshold:
                small_mask |= (labeled == comp_id)

        if not np.any(small_mask):
            continue

        # For all small pixels at once: find nearest non-small neighbor
        # Use a dilated version of the labels where small pixels are replaced
        # by the majority of their 5x5 neighborhood
        kernel = np.ones((5, 5), np.uint8)
        # Temporarily set small pixels to -1
        temp_labels = labels.copy()
        temp_labels[small_mask] = -1

        # Dilate valid labels into the gaps
        for _ in range(3):  # 3 passes to fill gaps
            dilated = cv2.dilate(
                (temp_labels + 1).astype(np.uint16),  # shift so 0 = was -1
                kernel
            ).astype(np.int32) - 1
            still_empty = temp_labels == -1
            temp_labels[still_empty] = dilated[still_empty]

        # Apply: only update pixels that were in small components
        labels[small_mask] = temp_labels[small_mask]
        # Any remaining -1 pixels (isolated) keep original label
        labels[labels == -1] = plate_id

    return labels


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=150,
             locked_colors=None, return_data=False,
             chroma_boost=1.3,
             sigma_s=100, sigma_r=0.5,
             meanshift_sp=15, meanshift_sr=30,
             upscale=True, img_hash=None,
             color_space="cielab",
             progress_callback=None):
    """
    V12 separation: EdgePreserve → MeanShift → CIELAB/OKLab K-means++ → CC Cleanup → Smooth → Merge suggestions.

    Args:
        input_path_or_array: filepath string or numpy RGB array
        output_dir: where to save (None if return_data=True)
        n_plates: number of color plates (2-35)
        dust_threshold: CC cleanup threshold
        locked_colors: list of [R,G,B] colors to lock as centroids
        return_data: if True, return dict instead of writing files
        chroma_boost: chroma multiplier for palette vividness
        sigma_s: edge-preserving filter spatial bandwidth
        sigma_r: edge-preserving filter range bandwidth
        meanshift_sp: mean shift spatial bandwidth
        meanshift_sr: mean shift color bandwidth
        upscale: whether to apply Real-ESRGAN 2x upscale
        img_hash: if provided, check upscale cache instead of re-upscaling
        color_space: "cielab" (default) or "oklab"
        progress_callback: optional callable(stage, pct) for progress reporting
    """
    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array
        img = Image.fromarray(arr)

    report("Upscaling image", 5)
    # ── Step 0: Optional Real-ESRGAN 2x upscale (with cache support) ──
    was_upscaled = False
    if upscale:
        if img_hash and img_hash in _upscale_cache:
            arr = _upscale_cache[img_hash]
            img = Image.fromarray(arr)
            was_upscaled = True
        else:
            upscaled_arr, success = upscale_2x(arr)
            if success:
                arr = upscaled_arr
                img = Image.fromarray(arr)
                was_upscaled = True

    h, w = arr.shape[:2]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    report("Separating colors", 10)
    # ── Step 1: Edge-preserving filter ──
    filtered = cv2.edgePreservingFilter(arr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

    # ── Step 2: Mean shift filtering ──
    filtered = cv2.pyrMeanShiftFiltering(filtered, sp=meanshift_sp, sr=meanshift_sr)

    # ── Step 3: Convert FILTERED image to color space for clustering ──
    arr_float = filtered.astype(np.float64) / 255.0

    if color_space == "oklab":
        lab_img = rgb_to_oklab(arr_float)
    else:
        lab_img = rgb2lab(arr_float)

    lab_boosted = lab_img.copy()
    lab_boosted[:, :, 1] *= chroma_boost
    lab_boosted[:, :, 2] *= chroma_boost

    lab_flat = lab_boosted.reshape(-1, 3)

    max_samples = 80000
    if len(lab_flat) > max_samples:
        indices = np.random.RandomState(42).choice(len(lab_flat), max_samples, replace=False)
        sample = lab_flat[indices]
    else:
        sample = lab_flat

    report("Clustering", 40)
    # ── Step 4: MiniBatchKMeans clustering ──
    init = "k-means++"
    if locked_colors and len(locked_colors) > 0:
        if color_space == "oklab":
            locked_float = np.array(locked_colors, dtype=np.float64).reshape(-1, 3) / 255.0
            locked_lab = rgb_to_oklab(locked_float)
        else:
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

    kmeans = MiniBatchKMeans(
        n_clusters=n_plates,
        init=init,
        random_state=42,
        n_init=3 if isinstance(init, str) else 1,
        batch_size=min(10000, len(sample)),
        max_iter=100,
    )
    kmeans.fit(sample)

    # Get palette in LAB (undo chroma boost) and convert to RGB
    palette_lab_boosted = kmeans.cluster_centers_
    palette_lab = palette_lab_boosted.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost

    if color_space == "oklab":
        # For OKLab, use CIELAB palette conversion for RGB output
        # Re-derive palette RGB from the original filtered image clusters
        palette_rgb_float = lab2rgb(rgb2lab(
            (filtered.astype(np.float64) / 255.0).reshape(-1, 1, 3)[:n_plates]
        ).reshape(-1, 3).reshape(-1, 1, 3)).reshape(-1, 3)
        # Fallback: assign via nearest cluster centers in original space
        # Actually, just convert boosted centers back properly
        arr_float_for_palette = filtered.astype(np.float64) / 255.0
        # Find the pixel closest to each cluster center in oklab space
        oklab_flat = rgb_to_oklab(arr_float_for_palette).reshape(-1, 3)
        oklab_flat_boosted = oklab_flat.copy()
        oklab_flat_boosted[:, 1] *= chroma_boost
        oklab_flat_boosted[:, 2] *= chroma_boost
        palette_rgb = np.zeros((n_plates, 3), dtype=np.uint8)
        for i, center in enumerate(palette_lab_boosted):
            dists_i = np.sum((oklab_flat_boosted - center) ** 2, axis=1)
            nearest_idx = np.argmin(dists_i)
            palette_rgb[i] = filtered.reshape(-1, 3)[nearest_idx]
    else:
        palette_rgb_float = lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Step 5: Assign ALL pixels to nearest cluster (vectorized cdist) ──
    dists = cdist(lab_flat, palette_lab_boosted, metric='sqeuclidean')
    pixel_labels = np.argmin(dists, axis=1).reshape(h, w)

    report("Cleaning up", 70)
    # ── Step 6: Connected component cleanup ──
    pixel_labels = connected_component_cleanup(pixel_labels, n_plates, dust_threshold)

    # ── Step 6.5: Label map boundary smoothing (zero-gap guaranteed) ──
    plate_scores = np.zeros((n_plates, h, w), dtype=np.float32)
    for i in range(n_plates):
        mask_float = (pixel_labels == i).astype(np.float32)
        plate_scores[i] = cv2.GaussianBlur(mask_float, (5, 5), sigmaX=1.5)

    score_sums = plate_scores.sum(axis=0)
    has_score = score_sums > 0.01
    smoothed_labels = pixel_labels.copy()
    smoothed_labels[has_score] = np.argmax(plate_scores[:, has_score], axis=0)
    pixel_labels = smoothed_labels

    # ── Step 7: Extract and clean individual plates ──
    if color_space == "oklab":
        # For brightness ordering with OKLab, L channel is index 0
        brightness_order = np.argsort([c[0] for c in palette_lab])
    else:
        brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx
        # Skip binary_fill_holes — CC cleanup already handles gaps

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

    report("Building output", 90)
    # ── Step 8: Composite preview ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb_val = plate_info["color"]
        comp[mask] = rgb_val

    composite_img = Image.fromarray(comp)

    # ── Step 9: Merge suggestions ──
    merge_suggestions = []
    for i in range(len(palette_lab)):
        for j in range(i + 1, len(palette_lab)):
            delta_e = deltaE_ciede2000(
                palette_lab[i].reshape(1, 1, 3),
                palette_lab[j].reshape(1, 1, 3)
            ).item()
            if delta_e < 15:
                merge_suggestions.append({
                    'plate_a': int(i),
                    'plate_b': int(j),
                    'delta_e': round(float(delta_e), 1)
                })

    # ── Step 10: Manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(100.0 - sum(p["coverage_pct"] for p in results), 2),
        "version": "v12",
        "upscaled": was_upscaled,
        "merge_suggestions": merge_suggestions,
    }

    if return_data:
        return {
            "composite": composite_img,
            "plates": plate_images,
            "manifest": manifest,
            "palette_rgb": palette_rgb,
            "pixel_labels": pixel_labels,
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


def apply_merge(pixel_labels, palette_rgb, merge_pairs, n_plates):
    """
    Apply merge operations: for each (plate_a, plate_b), replace all
    plate_a pixels with plate_b's label.
    Returns updated pixel_labels.
    """
    labels = pixel_labels.copy()
    for plate_a, plate_b in merge_pairs:
        labels[labels == plate_a] = plate_b
    return labels


def build_preview_response(image_bytes, plates=4, dust=50,
                           locked_colors=None,
                           chroma_boost=1.3,
                           sigma_s=100, sigma_r=0.5,
                           meanshift_sp=15, meanshift_sr=30,
                           upscale=True, img_hash=None,
                           color_space="cielab",
                           progress_callback=None, **kwargs):
    """Process image and return composite PNG bytes + manifest."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
        upscale=upscale, img_hash=img_hash,
        color_space=color_space,
        progress_callback=progress_callback,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_merge_response(image_bytes, merge_pairs, plates=4, dust=50,
                         locked_colors=None,
                         chroma_boost=1.3,
                         sigma_s=100, sigma_r=0.5,
                         meanshift_sp=15, meanshift_sr=30,
                         upscale=True, img_hash=None,
                         color_space="cielab", **kwargs):
    """Run separation, apply merges, return new composite + manifest."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
        upscale=upscale, img_hash=img_hash,
        color_space=color_space,
    )

    pixel_labels = result["pixel_labels"]
    palette_rgb = result["palette_rgb"]
    h, w = pixel_labels.shape

    # Apply merges
    pixel_labels = apply_merge(pixel_labels, palette_rgb, merge_pairs, plates)

    # Determine which plates remain active
    active_labels = np.unique(pixel_labels)
    palette_lab_all = rgb2lab(palette_rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    brightness_order = np.argsort([palette_lab_all[idx][0] for idx in active_labels])

    # Rebuild results with merged plates
    new_results = []
    plate_images = {}
    for rank, order_idx in enumerate(brightness_order):
        idx = active_labels[order_idx]
        mask = pixel_labels == idx
        coverage = np.sum(mask) / mask.size * 100
        rgb = palette_rgb[idx]
        name = f"plate{rank + 1}"
        binary = np.where(mask, 0, 255).astype(np.uint8)

        plate_info = {
            "name": name,
            "index": int(idx),
            "rank": rank + 1,
            "color": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "color_hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "coverage_pct": round(coverage, 2),
        }
        new_results.append(plate_info)
        plate_images[name] = {"mask": mask, "binary": binary}

    # New composite
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(new_results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        comp[mask] = plate_info["color"]

    composite_img = Image.fromarray(comp)

    # New merge suggestions for remaining plates
    merge_suggestions = []
    for i_idx in range(len(active_labels)):
        for j_idx in range(i_idx + 1, len(active_labels)):
            li, lj = active_labels[i_idx], active_labels[j_idx]
            delta_e = deltaE_ciede2000(
                palette_lab_all[li].reshape(1, 1, 3),
                palette_lab_all[lj].reshape(1, 1, 3)
            ).item()
            if delta_e < 15:
                merge_suggestions.append({
                    'plate_a': int(li),
                    'plate_b': int(lj),
                    'delta_e': round(float(delta_e), 1)
                })

    manifest = {
        "width": w,
        "height": h,
        "num_plates": len(active_labels),
        "plates": new_results,
        "paper_pct": round(100.0 - sum(p["coverage_pct"] for p in new_results), 2),
        "version": "v12",
        "upscaled": result["manifest"].get("upscaled", False),
        "merge_suggestions": merge_suggestions,
    }

    buf = io.BytesIO()
    composite_img.save(buf, format="PNG")
    return buf.getvalue(), manifest


def build_zip_response(image_bytes, plates=4, dust=50,
                       locked_colors=None,
                       chroma_boost=1.3,
                       sigma_s=100, sigma_r=0.5,
                       meanshift_sp=15, meanshift_sr=30,
                       upscale=True, img_hash=None,
                       color_space="cielab",
                       progress_callback=None, **kwargs):
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
        upscale=upscale, img_hash=img_hash,
        color_space=color_space,
        progress_callback=progress_callback,
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
            zf.writestr(f"png/{name}.png", png_buf.getvalue())

            svg_content = mask_to_svg_string(plate_data["mask"], w, h)
            zf.writestr(f"svg/{name}.svg", svg_content)

        zf.writestr("manifest.json", json.dumps(result["manifest"], indent=2))

    return zip_buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Woodblock color separation V12 (fast)")
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
    parser.add_argument("--color-space", choices=["cielab", "oklab"], default="cielab", help="Color space")
    args = parser.parse_args()

    if args.input and args.output:
        result = separate(
            args.input, args.output, n_plates=args.plates,
            dust_threshold=args.dust,
            sigma_s=args.sigma_s, sigma_r=args.sigma_r,
            meanshift_sp=args.meanshift_sp, meanshift_sr=args.meanshift_sr,
            chroma_boost=args.chroma,
            upscale=not args.no_upscale,
            color_space=args.color_space,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
