#!/usr/bin/env python3
"""
Woodblock color separation V15 — SAM-guided color separation.

Uses SAM2.1 to segment the image into objects FIRST, then runs K-means
within each object region independently. This prevents similar-colored
objects from merging.
"""
import argparse
import hashlib
import io
import json
import os
import tempfile
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
    return hashlib.md5(image_bytes[:4096]).hexdigest()


def upscale_2x(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run Real-ESRGAN 2x on an RGB numpy array. Returns (result, success)."""
    try:
        import torch
        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
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
        import gc
        torch.cuda.empty_cache()
        gc.collect()
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


# ── SAM model cache ──
_sam_model = None


def get_sam_model():
    global _sam_model
    if _sam_model is None:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        from ultralytics import SAM
        _sam_model = SAM("sam2.1_t.pt")
        # Force SAM to CPU — GPU is reserved for ESRGAN upscaler
        _sam_model.model.cpu()
    return _sam_model


def release_sam():
    """No-op — SAM stays on CPU."""
    pass


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
             shadow_threshold=8, highlight_threshold=95,
             median_size=3,
             upscale=True, img_hash=None,
             progress_callback=None):
    """
    V15 separation: SAM segmentation -> per-region K-means -> Canny edges -> CC cleanup.
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

    report("Upscaling image (2x)", 5)
    # ── Step 0: Optional Real-ESRGAN 2x upscale ──
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

    report("Detecting strokes", 15)
    # ── Step 0.5: Line-aware pre-pass — detect thin dark strokes ──
    # Convert to grayscale and find dark strokes (signatures, text, fine linework)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold to find dark marks on lighter background
    # Block size 31 catches both fine pen strokes and printed text
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    # Also catch mid-tone strokes (signatures are often not pure black)
    # Use a global threshold relative to the median background
    # Use median of non-white pixels (the card, not surrounding white)
    card_pixels = gray[gray < 240]
    bg_median = np.median(card_pixels) if len(card_pixels) > 0 else np.median(gray)
    dark_global = (gray < bg_median * 0.85).astype(np.uint8) * 255

    # Also detect via local contrast (catches faint signatures on colored backgrounds)
    gray_float = gray.astype(np.float32)
    local_mean = cv2.blur(gray_float, (51, 51))
    local_contrast = local_mean - gray_float  # positive = darker than surroundings
    local_contrast_mask = (local_contrast > 8).astype(np.uint8) * 255

    # Combine all detections
    stroke_mask = ((adaptive_thresh > 0) | (dark_global > 0) | (local_contrast_mask > 0)).astype(np.uint8)

    # Clean up: remove very large regions (those are color areas, not strokes)
    # Keep only thin structures via morphological opening
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN, thin_kernel)

    # Dilate slightly to capture full stroke width
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_mask = cv2.dilate(stroke_mask, dilate_kernel, iterations=1)

    # Remove large connected components (> 2% of image = not a stroke)
    from scipy.ndimage import label as ndlabel_func
    labeled_strokes, n_stroke_comps = ndlabel_func(stroke_mask)
    if n_stroke_comps > 0:
        comp_sizes = np.bincount(labeled_strokes.ravel())[1:]
        max_stroke_size = h * w * 0.02  # 2% threshold
        for comp_id in range(1, len(comp_sizes) + 1):
            if comp_sizes[comp_id - 1] > max_stroke_size:
                stroke_mask[labeled_strokes == comp_id] = 0

    stroke_mask_bool = stroke_mask.astype(bool)

    report("Segmenting objects (SAM)", 25)
    # ── Step 1: SAM segmentation ──
    import torch
    if torch.cuda.is_available():
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    model = get_sam_model()

    # Save image to temp file for SAM
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        Image.fromarray(arr).save(f.name)
        temp_path = f.name

    # Grid of points for segmentation — balanced density
    step = max(50, min(h, w) // 18)
    points = []
    for y in range(step // 2, h - step // 2, step):
        for x in range(step // 2, w - step // 2, step):
            points.append([x, y])
    points_arr = np.array(points)
    labels_arr = np.ones(len(points_arr))

    try:
        results = model(temp_path, points=points_arr, labels=labels_arr, device="cpu")
        masks = results[0].masks.data.cpu().numpy()
    except (RuntimeError, Exception) as e:
        print(f"SAM failed ({e}), falling back to single-region mode")
        masks = np.ones((1, h, w), dtype=bool)
    finally:
        os.unlink(temp_path)
        release_sam()  # Free GPU for ESRGAN

    report("Merging regions", 40)
    # ── Step 2: Merge overlapping/small SAM masks into coherent regions ──
    region_map = np.zeros((h, w), dtype=np.int32)
    mask_areas = [(i, m.sum()) for i, m in enumerate(masks)]
    mask_areas.sort(key=lambda x: x[1], reverse=True)  # largest first

    for mask_idx, area in mask_areas:
        mask = masks[mask_idx]
        if mask.shape != (h, w):
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h))) > 128
        unclaimed = region_map == 0
        region_map[mask & unclaimed] = mask_idx + 1

    # Fill unclaimed pixels — use color similarity to nearest claimed pixel
    if np.any(region_map == 0):
        # First pass: small kernel dilation (3x3) to fill thin gaps
        kernel_small = np.ones((3, 3), np.uint8)
        filled = region_map.copy()
        for _ in range(20):  # more passes with smaller kernel = finer fill
            dilated = cv2.dilate(filled.astype(np.uint16), kernel_small).astype(np.int32)
            still_empty = filled == 0
            if not np.any(still_empty):
                break
            filled[still_empty] = dilated[still_empty]
        # Second pass: larger kernel for any remaining
        if np.any(filled == 0):
            kernel_large = np.ones((7, 7), np.uint8)
            for _ in range(5):
                dilated = cv2.dilate(filled.astype(np.uint16), kernel_large).astype(np.int32)
                still_empty = filled == 0
                if not np.any(still_empty):
                    break
                filled[still_empty] = dilated[still_empty]
        region_map = filled

    # ── Step 2.5: Split SAM regions by color saturation ──
    # If a SAM region contains both high-sat (colored) and low-sat (neutral) pixels,
    # split it into two sub-regions so K-means doesn't mix them
    hsv_split = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_split = hsv_split[:, :, 1]

    next_region_id = region_map.max() + 1
    for region_id in np.unique(region_map):
        if region_id == 0:
            continue
        region_mask = region_map == region_id
        region_sats = sat_split[region_mask]

        # Check if this region has BOTH high-sat and low-sat pixels
        high_sat_count = np.sum(region_sats > 70)
        low_sat_count = np.sum(region_sats <= 70)
        total = len(region_sats)

        # Only split if both groups are significant (>15% each)
        if high_sat_count > total * 0.15 and low_sat_count > total * 0.15:
            # Split: high-sat pixels get a new region ID
            high_sat_mask = region_mask & (sat_split > 70)
            region_map[high_sat_mask] = next_region_id
            next_region_id += 1

    unique_regions = np.unique(region_map)
    unique_regions = unique_regions[unique_regions > 0]
    n_sam_masks = len(masks)

    report("Clustering colors", 50)
    # ── Step 3: Per-region K-means clustering ──
    arr_float = arr.astype(np.float64) / 255.0
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

    # Handle locked colors
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

    # Now assign pixels using color distance within SAM regions
    pixel_labels = np.zeros((h, w), dtype=np.int32)

    # Pre-compute centroid properties
    palette_lab_for_check = palette_lab_boosted.copy()
    palette_lab_for_check[:, 1] /= chroma_boost
    palette_lab_for_check[:, 2] /= chroma_boost
    centroid_chroma = np.sqrt(palette_lab_for_check[:, 1]**2 + palette_lab_for_check[:, 2]**2)

    hsv_assign = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_assign = hsv_assign[:, :, 1]

    for region_id in unique_regions:
        region_mask = region_map == region_id
        region_pixels = lab_boosted[region_mask]
        region_sats = sat_assign[region_mask]

        if len(region_pixels) == 0:
            continue

        # Check if this region is predominantly colored
        mean_sat = np.mean(region_sats)

        dists = cdist(region_pixels, palette_lab_boosted, metric='sqeuclidean')

        if mean_sat > 60:
            # COLORED region: mask out neutral/dark centroids
            # Only allow assignment to centroids with chroma > 8
            for c_idx in range(n_plates):
                if centroid_chroma[c_idx] < 8:
                    dists[:, c_idx] = np.inf  # block this centroid for colored regions

        region_labels = np.argmin(dists, axis=1)
        pixel_labels[region_mask] = region_labels

    report("Smoothing plates", 65)
    # ── Step 4: Median filter + morphological cleanup ──
    # Median filter first to remove salt-and-pepper noise
    median_size_val = min(max(1, int(median_size)), 5)
    if median_size_val % 2 == 0:
        median_size_val += 1
    pixel_labels = median_filter(pixel_labels.astype(np.int32), size=median_size_val)

    # Morphological closing per plate to fill small holes (vectorized, no pixel loops)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_labels = pixel_labels.copy()
    for plate_id in range(n_plates):
        mask = (pixel_labels == plate_id).astype(np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        # Fill holes: where closed says yes but original says no, AND no other plate has a stronger claim
        fill_candidates = (closed > 0) & (mask == 0)
        closed_labels[fill_candidates] = plate_id
    pixel_labels = closed_labels

    report("Filling strokes", 75)
    # ── Step 4.5: Two-pass stroke fill + color protection ──
    hsv_pp = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue_pp = hsv_pp[:, :, 0]
    sat_pp = hsv_pp[:, :, 1]
    val_pp = hsv_pp[:, :, 2]

    palette_lab_pp = palette_lab_boosted.copy()
    palette_lab_pp[:, 1] /= chroma_boost
    palette_lab_pp[:, 2] /= chroma_boost
    darkest_plate = int(np.argmin(palette_lab_pp[:, 0]))
    reddest_plate = int(np.argmax(palette_lab_pp[:, 1]))
    centroid_chroma_pp = np.sqrt(palette_lab_pp[:, 1]**2 + palette_lab_pp[:, 2]**2)

    # PASS 1: Fill dark strokes — but ONLY if the pixel ITSELF is low-saturation
    # This is the key insight: dark strokes (signatures, text) have low saturation
    # Colored illustration pixels have high saturation even when dark
    # So pixel saturation is the discriminator, not neighborhood saturation
    fill_mask = stroke_mask_bool & (gray < bg_median * 0.85) & (sat_pp < 100)
    pixel_labels[fill_mask] = darkest_plate

    # PASS 2: Fix red-hue pixels stuck on neutral plates
    red_hue = ((hue_pp < 15) | (hue_pp > 165)) & (sat_pp > 80) & (val_pp > 50)
    for plate_id in range(n_plates):
        if centroid_chroma_pp[plate_id] < 10:
            fix_mask = red_hue & (pixel_labels == plate_id)
            pixel_labels[fix_mask] = reddest_plate

    report("Detecting edges", 82)
    # ── Step 5: Canny edge assignment ──
    if use_edges:
        from skimage.color import rgb2gray
        from skimage.feature import canny
        from scipy.ndimage import binary_dilation

        # Get palette_lab for brightness ordering
        palette_lab = palette_lab_boosted.copy()
        palette_lab[:, 1] /= chroma_boost
        palette_lab[:, 2] /= chroma_boost

        gray = rgb2gray(arr)
        edges = canny(gray, sigma=edge_sigma, low_threshold=0.04, high_threshold=0.12)
        edges = binary_dilation(edges, iterations=1)

        ey, ex = np.where(edges)
        for y, x in zip(ey, ex):
            y0, y1 = max(0, y - 3), min(h, y + 4)
            x0, x1 = max(0, x - 3), min(w, x + 4)
            nbr = pixel_labels[y0:y1, x0:x1]
            plates_nearby = np.unique(nbr)
            plates_nearby = plates_nearby[plates_nearby < n_plates]
            if len(plates_nearby) > 0:
                darkest = min(plates_nearby, key=lambda p: palette_lab[p][0])
                pixel_labels[y, x] = darkest

    report("Correcting holes", 88)
    # ── Step 6: CC cleanup ──
    pixel_labels = connected_component_cleanup(pixel_labels, n_plates, dust_threshold)

    report("Cleaning up", 95)
    # ── Step 7: Get palette RGB ──
    palette_lab = palette_lab_boosted.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost

    palette_rgb_float = lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Step 8: Extract and clean individual plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results_list = []
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
        results_list.append(plate_info)
        plate_images[name] = {"mask": mask, "binary": binary, "image": plate_img}

    # ── Step 9: Composite preview ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results_list):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb_val = plate_info["color"]
        comp[mask] = rgb_val

    composite_img = Image.fromarray(comp)

    # ── Step 10: Merge suggestions ──
    merge_suggestions = []
    for i in range(len(palette_lab)):
        for j in range(i + 1, len(palette_lab)):
            delta_e = deltaE_ciede2000(
                palette_lab[i].reshape(1, 1, 3),
                palette_lab[j].reshape(1, 1, 3)
            ).item()
            if 2.0 < delta_e < 15:
                merge_suggestions.append({
                    'plate_a': int(i),
                    'plate_b': int(j),
                    'delta_e': round(float(delta_e), 1)
                })

    # ── Step 11: Manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results_list,
        "paper_pct": round(100.0 - sum(p["coverage_pct"] for p in results_list), 2),
        "version": "v18",
        "upscaled": was_upscaled,
        "merge_suggestions": merge_suggestions,
        "sam_masks": n_sam_masks,
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

    for plate_info in results_list:
        name = plate_info["name"]
        plate_images[name]["image"].save(os.path.join(output_dir, f"{name}.png"))

        mask = plate_images[name]["mask"]
        svg_path = os.path.join(output_dir, f"{name}.svg")
        mask_to_svg(mask, svg_path, w, h)

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def apply_merge(pixel_labels, palette_rgb, merge_pairs, n_plates):
    """Apply merge operations."""
    labels = pixel_labels.copy()
    for plate_a, plate_b in merge_pairs:
        labels[labels == plate_a] = plate_b
    return labels


def build_preview_response(image_bytes, plates=4, dust=50,
                           locked_colors=None,
                           chroma_boost=1.3,
                           use_edges=True, edge_sigma=1.5,
                           shadow_threshold=8, highlight_threshold=95,
                           median_size=3,
                           upscale=True, img_hash=None,
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
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
        progress_callback=progress_callback,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_merge_response(image_bytes, merge_pairs, plates=4, dust=50,
                         locked_colors=None,
                         chroma_boost=1.3,
                         use_edges=True, edge_sigma=1.5,
                         shadow_threshold=8, highlight_threshold=95,
                         median_size=3,
                         upscale=True, img_hash=None, **kwargs):
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
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
    )

    pixel_labels = result["pixel_labels"]
    palette_rgb = result["palette_rgb"]
    h, w = pixel_labels.shape

    pixel_labels = apply_merge(pixel_labels, palette_rgb, merge_pairs, plates)

    active_labels = np.unique(pixel_labels)
    active_labels = active_labels[active_labels < plates]
    palette_lab_all = rgb2lab(palette_rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    brightness_order = np.argsort([palette_lab_all[idx][0] for idx in active_labels])

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

    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(new_results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        comp[mask] = plate_info["color"]

    composite_img = Image.fromarray(comp)

    merge_suggestions = []
    for i_idx in range(len(active_labels)):
        for j_idx in range(i_idx + 1, len(active_labels)):
            li, lj = active_labels[i_idx], active_labels[j_idx]
            delta_e = deltaE_ciede2000(
                palette_lab_all[li].reshape(1, 1, 3),
                palette_lab_all[lj].reshape(1, 1, 3)
            ).item()
            if 2.0 < delta_e < 15:
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
        "version": "v18",
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
                       shadow_threshold=8, highlight_threshold=95,
                       median_size=3,
                       upscale=True, img_hash=None,
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
        shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold,
        median_size=median_size,
        upscale=upscale, img_hash=img_hash,
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
    parser = argparse.ArgumentParser(description="Woodblock color separation V15 (SAM-guided)")
    parser.add_argument("input", nargs="?", default=None, help="Input image path")
    parser.add_argument("output", nargs="?", default=None, help="Output directory")
    parser.add_argument("--plates", type=int, default=8, help="Number of color plates")
    parser.add_argument("--dust", type=int, default=50, help="CC cleanup threshold")
    parser.add_argument("--no-edges", action="store_true", help="Disable Canny edge detection")
    parser.add_argument("--edge-sigma", type=float, default=1.5, help="Canny edge sigma")
    parser.add_argument("--chroma", type=float, default=1.3, help="Chroma boost factor")
    parser.add_argument("--shadow", type=int, default=8, help="Shadow threshold (L*)")
    parser.add_argument("--highlight", type=int, default=95, help="Highlight threshold (L*)")
    parser.add_argument("--median", type=int, default=3, help="Median filter size for label map")
    parser.add_argument("--no-upscale", action="store_true", help="Disable Real-ESRGAN upscale")
    args = parser.parse_args()

    if args.input and args.output:
        result = separate(
            args.input, args.output, n_plates=args.plates,
            dust_threshold=args.dust,
            use_edges=not args.no_edges, edge_sigma=args.edge_sigma,
            chroma_boost=args.chroma,
            shadow_threshold=args.shadow, highlight_threshold=args.highlight,
            median_size=args.median,
            upscale=not args.no_upscale,
        )
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
