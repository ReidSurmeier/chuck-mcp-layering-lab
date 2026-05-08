#!/usr/bin/env python3
"""
Woodblock color separation V14 — Two-pass gradient-aware fusion.

Combines v12's smooth separation (edgePreservingFilter + pyrMeanShiftFiltering)
with v13's detail preservation (raw pixel assignment) using a gradient-based
fusion that picks smooth labels in flat regions and detail labels near edges.

Key idea:
1. Pass 1 (smooth): Filter image, cluster in CIELAB, get smooth_labels
2. Pass 2 (detail): Assign RAW pixels to the SAME centroids from Pass 1
3. Fusion: Use Sobel gradient magnitude to blend — high gradient = detail, low = smooth
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
from scipy.ndimage import label as ndlabel, median_filter
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


def connected_component_cleanup(labels, n_plates, dust_threshold=150):
    """
    For each plate, find connected components. Components smaller than
    dust_threshold get absorbed into the most common neighboring plate.
    Optimized: batch all small components per plate using morphological dilation.
    """
    h, w = labels.shape

    for plate_id in range(n_plates):
        mask = labels == plate_id
        labeled, n_comp = ndlabel(mask)
        if n_comp <= 1:
            continue

        comp_sizes = np.bincount(labeled.ravel())
        small_mask = np.zeros_like(mask)
        for comp_id in range(1, len(comp_sizes)):
            if comp_sizes[comp_id] < dust_threshold:
                small_mask |= (labeled == comp_id)

        if not np.any(small_mask):
            continue

        kernel = np.ones((5, 5), np.uint8)
        temp_labels = labels.copy()
        temp_labels[small_mask] = -1

        for _ in range(3):
            dilated = cv2.dilate(
                (temp_labels + 1).astype(np.uint16),
                kernel
            ).astype(np.int32) - 1
            still_empty = temp_labels == -1
            temp_labels[still_empty] = dilated[still_empty]

        labels[small_mask] = temp_labels[small_mask]
        labels[labels == -1] = plate_id

    return labels


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=150,
             use_edges=True, edge_sigma=1.5,
             locked_colors=None, return_data=False,
             chroma_boost=1.3,
             sigma_s=60, sigma_r=0.3,
             meanshift_sp=10, meanshift_sr=20,
             shadow_threshold=8, highlight_threshold=95,
             median_size=3,
             upscale=True, img_hash=None,
             detail_strength=0.5,
             progress_callback=None):
    """
    V14 separation: Two-pass gradient-aware fusion.

    Pass 1 clusters on filtered image (smooth labels).
    Pass 2 assigns raw pixels to same centroids (detail labels).
    Gradient magnitude controls which labels win per pixel.

    Args:
        input_path_or_array: filepath string or numpy RGB array
        output_dir: where to save (None if return_data=True)
        n_plates: number of color plates (2-35)
        dust_threshold: CC cleanup threshold
        use_edges: whether to use Canny edge detection for linework
        edge_sigma: Canny sigma parameter
        locked_colors: list of [R,G,B] colors to lock as centroids
        return_data: if True, return dict instead of writing files
        chroma_boost: chroma multiplier for palette vividness
        sigma_s: edge-preserving filter spatial bandwidth
        sigma_r: edge-preserving filter range bandwidth
        meanshift_sp: mean shift spatial bandwidth
        meanshift_sr: mean shift color bandwidth
        shadow_threshold: L* lower bound for content masking
        highlight_threshold: L* upper bound for content masking
        median_size: median filter kernel size for label map cleanup
        upscale: whether to apply Real-ESRGAN 2x upscale
        img_hash: if provided, check upscale cache instead of re-upscaling
        detail_strength: 0.0 = all smooth, 1.0 = all detail (default 0.5)
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
    # ══════════════════════════════════════════════════════════════
    # PASS 1: Smooth separation (v12 approach — filtered image)
    # ══════════════════════════════════════════════════════════════
    filtered = cv2.edgePreservingFilter(arr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    filtered = cv2.pyrMeanShiftFiltering(filtered, sp=meanshift_sp, sr=meanshift_sr)

    filtered_float = filtered.astype(np.float64) / 255.0
    lab_filtered = rgb2lab(filtered_float)
    lab_filtered_boosted = lab_filtered.copy()
    lab_filtered_boosted[:, :, 1] *= chroma_boost
    lab_filtered_boosted[:, :, 2] *= chroma_boost

    lab_filtered_flat = lab_filtered_boosted.reshape(-1, 3)

    max_samples = 80000
    if len(lab_filtered_flat) > max_samples:
        indices = np.random.RandomState(42).choice(len(lab_filtered_flat), max_samples, replace=False)
        sample = lab_filtered_flat[indices]
    else:
        sample = lab_filtered_flat

    report("Clustering", 40)
    # Clustering on filtered image
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

    kmeans = MiniBatchKMeans(
        n_clusters=n_plates,
        init=init,
        random_state=42,
        n_init=3 if isinstance(init, str) else 1,
        batch_size=min(10000, len(sample)),
        max_iter=100,
    )
    kmeans.fit(sample)

    palette_lab_boosted = kmeans.cluster_centers_
    palette_lab = palette_lab_boosted.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost

    palette_rgb_float = lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # Smooth labels: assign filtered pixels to centroids
    smooth_dists = cdist(lab_filtered_flat, palette_lab_boosted, metric='sqeuclidean')
    smooth_labels = np.argmin(smooth_dists, axis=1).reshape(h, w)

    # ══════════════════════════════════════════════════════════════
    # PASS 2: Detail separation (raw pixels → same centroids)
    # ══════════════════════════════════════════════════════════════
    arr_float = arr.astype(np.float64) / 255.0
    lab_raw = rgb2lab(arr_float)
    lab_raw_boosted = lab_raw.copy()
    lab_raw_boosted[:, :, 1] *= chroma_boost
    lab_raw_boosted[:, :, 2] *= chroma_boost

    lab_raw_flat = lab_raw_boosted.reshape(-1, 3)
    detail_dists = cdist(lab_raw_flat, palette_lab_boosted, metric='sqeuclidean')
    detail_labels = np.argmin(detail_dists, axis=1).reshape(h, w)

    # ══════════════════════════════════════════════════════════════
    # GRADIENT-AWARE FUSION
    # ══════════════════════════════════════════════════════════════
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to 0-1
    gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)

    # Sigmoid soft threshold — detail_strength controls the crossover point
    threshold = 1.0 - detail_strength
    detail_weight = 1.0 / (1.0 + np.exp(-15 * (gradient_mag - threshold)))

    # Fuse labels
    pixel_labels = np.where(detail_weight > 0.5, detail_labels, smooth_labels)

    # ══════════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ══════════════════════════════════════════════════════════════

    # Median filter cleanup on fused label map
    median_sz = max(1, int(median_size))
    if median_sz % 2 == 0:
        median_sz += 1
    label_median = min(median_sz, 3)
    pixel_labels = median_filter(pixel_labels.astype(np.int32), size=label_median)

    report("Cleaning up", 70)
    # Fast CC cleanup (batch dilation)
    pixel_labels = connected_component_cleanup(pixel_labels, n_plates, dust_threshold)

    # Label map boundary smoothing (zero-gap guaranteed)
    plate_scores = np.zeros((n_plates, h, w), dtype=np.float32)
    for i in range(n_plates):
        mask_float = (pixel_labels == i).astype(np.float32)
        plate_scores[i] = cv2.GaussianBlur(mask_float, (5, 5), sigmaX=1.5)

    score_sums = plate_scores.sum(axis=0)
    has_score = score_sums > 0.01
    smoothed_labels = pixel_labels.copy()
    smoothed_labels[has_score] = np.argmax(plate_scores[:, has_score], axis=0)
    pixel_labels = smoothed_labels

    # ── Extract and clean individual plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

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
    # ── Composite preview ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb_val = plate_info["color"]
        comp[mask] = rgb_val

    composite_img = Image.fromarray(comp)

    # ── Merge suggestions ──
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

    # ── Manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(100.0 - sum(p["coverage_pct"] for p in results), 2),
        "version": "v14",
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
                           use_edges=True, edge_sigma=1.5,
                           sigma_s=60, sigma_r=0.3,
                           meanshift_sp=10, meanshift_sr=20,
                           shadow_threshold=8, highlight_threshold=95,
                           median_size=3,
                           upscale=True, img_hash=None,
                           detail_strength=0.5,
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
        use_edges=use_edges, edge_sigma=edge_sigma,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
        detail_strength=detail_strength,
        progress_callback=progress_callback,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_merge_response(image_bytes, merge_pairs, plates=4, dust=50,
                         locked_colors=None,
                         chroma_boost=1.3,
                         use_edges=True, edge_sigma=1.5,
                         sigma_s=60, sigma_r=0.3,
                         meanshift_sp=10, meanshift_sr=20,
                         shadow_threshold=8, highlight_threshold=95,
                         median_size=3,
                         upscale=True, img_hash=None,
                         detail_strength=0.5, **kwargs):
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
        use_edges=use_edges, edge_sigma=edge_sigma,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
        detail_strength=detail_strength,
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
        "version": "v14",
        "upscaled": result["manifest"].get("upscaled", False),
        "merge_suggestions": merge_suggestions,
    }

    buf = io.BytesIO()
    composite_img.save(buf, format="PNG")
    return buf.getvalue(), manifest


def build_zip_response(image_bytes, plates=4, dust=50,
                       locked_colors=None,
                       chroma_boost=1.3,
                       use_edges=True, edge_sigma=1.5,
                       sigma_s=60, sigma_r=0.3,
                       meanshift_sp=10, meanshift_sr=20,
                       shadow_threshold=8, highlight_threshold=95,
                       median_size=3,
                       upscale=True, img_hash=None,
                       detail_strength=0.5,
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
        use_edges=use_edges, edge_sigma=edge_sigma,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        sigma_s=sigma_s, sigma_r=sigma_r,
        meanshift_sp=meanshift_sp, meanshift_sr=meanshift_sr,
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
        detail_strength=detail_strength,
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
    parser = argparse.ArgumentParser(description="Woodblock color separation V14 (hybrid gradient fusion)")
    parser.add_argument("input", nargs="?", default=None, help="Input image path")
    parser.add_argument("output", nargs="?", default=None, help="Output directory")
    parser.add_argument("--plates", type=int, default=8, help="Number of color plates")
    parser.add_argument("--dust", type=int, default=50, help="CC cleanup threshold")
    parser.add_argument("--sigma-s", type=float, default=60, help="Edge-preserving spatial bandwidth")
    parser.add_argument("--sigma-r", type=float, default=0.3, help="Edge-preserving range bandwidth")
    parser.add_argument("--meanshift-sp", type=int, default=10, help="Mean shift spatial bandwidth")
    parser.add_argument("--meanshift-sr", type=int, default=20, help="Mean shift color bandwidth")
    parser.add_argument("--chroma", type=float, default=1.3, help="Chroma boost factor")
    parser.add_argument("--detail", type=float, default=0.5, help="Detail strength 0.0-1.0")
    parser.add_argument("--no-upscale", action="store_true", help="Disable Real-ESRGAN upscale")
    args = parser.parse_args()

    if args.input and args.output:
        result = separate(
            args.input, args.output, n_plates=args.plates,
            dust_threshold=args.dust,
            sigma_s=args.sigma_s, sigma_r=args.sigma_r,
            meanshift_sp=args.meanshift_sp, meanshift_sr=args.meanshift_sr,
            chroma_boost=args.chroma,
            detail_strength=args.detail,
            upscale=not args.no_upscale,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
