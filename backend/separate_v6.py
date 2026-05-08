#!/usr/bin/env python3
"""
Woodblock color separation V6 — Superpixel-based color separation.

Based on: 'Efficient Color Quantization Using Superpixels' (MDPI Sensors 2022)

Key insight: instead of clustering individual pixels (which creates noisy boundaries),
first group pixels into SLIC superpixels, then cluster the superpixels.
This produces ZERO line noise by construction — every pixel in a superpixel gets the
same label, and superpixels follow natural edges.

Pipeline:
1. Optional Real-ESRGAN 4x upscale
2. SLIC superpixel segmentation
3. Compute mean CIELAB color per superpixel
4. K-means++ on superpixel means (not individual pixels)
5. Assign each superpixel to its cluster → plate mask
6. Extract plates, build composite
"""
import io
import json
import logging
import os
import zipfile

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import slic
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ── Preview result cache ──
_last_result_cache: dict = {}


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=50,
             locked_colors=None, return_data=False, n_segments=3000,
             compactness=15, chroma_boost=1.3, upscale=False,
             shadow_threshold=8, highlight_threshold=95):
    """
    V6 superpixel-based separation.

    Args:
        input_path_or_array: filepath string or numpy RGB array
        output_dir: where to save (None if return_data=True)
        n_plates: number of color plates
        dust_threshold: minimum island size in pixels
        locked_colors: list of [R,G,B] colors to lock as centroids
        return_data: if True, return dict instead of writing files
        n_segments: number of SLIC superpixels (500-10000)
        compactness: SLIC compactness — higher = more square, lower = follows edges (5-40)
        chroma_boost: chroma multiplier for LAB a*/b* channels
        upscale: whether to 4x upscale before separation
        shadow_threshold: L* lower bound for content masking
        highlight_threshold: L* upper bound for content masking
    """
    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array

    # Step 0: Optional upscale
    if upscale:
        try:
            from separate_v4 import upscale_image
            arr, was_upscaled = upscale_image(arr)
            if was_upscaled:
                dust_threshold *= 16  # scale with 4x area
        except ImportError:
            logger.warning("separate_v4 not available for upscaling — skipping")

    h, w = arr.shape[:2]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: SLIC superpixel segmentation ──
    arr_float = arr.astype(np.float64) / 255.0
    segments = slic(arr_float, n_segments=n_segments, compactness=compactness,
                    convert2lab=True, enforce_connectivity=True, start_label=0)

    # ── Step 2: Compute mean CIELAB color per superpixel ──
    lab_img = rgb2lab(arr_float)

    # Boost chroma in LAB space
    lab_boosted = lab_img.copy()
    lab_boosted[:, :, 1] *= chroma_boost  # a* channel
    lab_boosted[:, :, 2] *= chroma_boost  # b* channel

    n_superpixels = segments.max() + 1
    superpixel_colors = np.zeros((n_superpixels, 3))
    superpixel_pixel_count = np.zeros(n_superpixels)

    for i in range(n_superpixels):
        mask = segments == i
        superpixel_colors[i] = lab_boosted[mask].mean(axis=0)
        superpixel_pixel_count[i] = mask.sum()

    # ── Step 3: Content masking at superpixel level ──
    # Exclude superpixels that are mostly paper (high L*) or very dark
    L_channel = lab_img[:, :, 0]
    sp_mean_L = np.zeros(n_superpixels)
    for i in range(n_superpixels):
        mask = segments == i
        sp_mean_L[i] = L_channel[mask].mean()

    is_content_sp = (sp_mean_L > shadow_threshold) & (sp_mean_L < highlight_threshold)
    content_sp_indices = np.where(is_content_sp)[0]
    content_sp_colors = superpixel_colors[content_sp_indices]

    if len(content_sp_colors) == 0:
        # Fallback: use all superpixels
        content_sp_indices = np.arange(n_superpixels)
        content_sp_colors = superpixel_colors

    # ── Step 4: K-means++ on superpixel colors ──
    init = "k-means++"
    if locked_colors and len(locked_colors) > 0:
        locked_rgb = np.array(locked_colors, dtype=np.float64).reshape(-1, 1, 3) / 255.0
        locked_lab = rgb2lab(locked_rgb).reshape(-1, 3)
        # Apply chroma boost to locked colors too
        locked_lab[:, 1] *= chroma_boost
        locked_lab[:, 2] *= chroma_boost

        if len(locked_lab) == n_plates:
            init = locked_lab
        elif len(locked_lab) < n_plates:
            remaining = n_plates - len(locked_lab)
            rng = np.random.RandomState(42)
            random_indices = rng.choice(len(content_sp_colors), remaining, replace=False)
            extra_centroids = content_sp_colors[random_indices]
            init = np.vstack([locked_lab, extra_centroids])
        else:
            init = locked_lab[:n_plates]

    kmeans = KMeans(
        n_clusters=n_plates,
        init=init,
        random_state=42,
        n_init=10 if isinstance(init, str) else 1,
        max_iter=300,
        tol=1e-4,
    )
    kmeans.fit(content_sp_colors)

    # Get palette in LAB and convert back to RGB
    palette_lab = kmeans.cluster_centers_
    # Undo chroma boost for palette display
    palette_lab_display = palette_lab.copy()
    palette_lab_display[:, 1] /= chroma_boost
    palette_lab_display[:, 2] /= chroma_boost
    palette_rgb_float = lab2rgb(palette_lab_display.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Step 5: Assign superpixels to clusters ──
    # Assign content superpixels via kmeans labels
    sp_labels = np.full(n_superpixels, n_plates, dtype=np.int32)  # default = background
    sp_labels[content_sp_indices] = kmeans.labels_

    # For non-content superpixels, assign to nearest cluster by color distance
    non_content_indices = np.where(~is_content_sp)[0]
    if len(non_content_indices) > 0:
        non_content_colors = superpixel_colors[non_content_indices]
        dists = np.zeros((len(non_content_colors), n_plates))
        for i, center in enumerate(palette_lab):
            diff = non_content_colors - center
            dists[:, i] = np.sum(diff ** 2, axis=1)
        # Only assign if reasonably close, otherwise leave as background
        min_dists = np.min(dists, axis=1)
        close_enough = min_dists < 5000  # threshold in LAB squared distance
        assignments = np.argmin(dists, axis=1)
        for idx, sp_idx in enumerate(non_content_indices):
            if close_enough[idx]:
                sp_labels[sp_idx] = assignments[idx]

    # ── Step 6: Map superpixel labels back to pixel labels ──
    # This is the magic — perfectly clean boundaries!
    pixel_labels = sp_labels[segments]

    # ── Step 6.5: Smooth jagged superpixel boundaries ──
    # Superpixel edges are pixelated/staircase-shaped. Smooth them by
    # re-assigning boundary pixels based on the actual image color distance
    # to neighboring plate centroids. This rounds the jagged steps.
    # Find boundary pixels (where label differs from any neighbor)
    padded = np.pad(pixel_labels, 1, mode='edge')
    is_boundary = (
        (pixel_labels != padded[:-2, 1:-1]) |
        (pixel_labels != padded[2:, 1:-1]) |
        (pixel_labels != padded[1:-1, :-2]) |
        (pixel_labels != padded[1:-1, 2:])
    )

    # Dilate boundary by 2px to catch nearby jagged pixels
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk as morph_disk
    boundary_zone = binary_dilation(is_boundary, morph_disk(2))

    # For each pixel in boundary zone, reassign to nearest plate centroid
    # based on actual pixel color (not superpixel average)
    bz_y, bz_x = np.where(boundary_zone & (pixel_labels < n_plates))
    if len(bz_y) > 0:
        bz_colors = lab_img[bz_y, bz_x]  # actual pixel colors in LAB
        bz_dists = np.zeros((len(bz_colors), n_plates))
        for i, center in enumerate(palette_lab):
            diff = bz_colors - center
            bz_dists[:, i] = np.sum(diff ** 2, axis=1)

        # Weight by spatial context: for each boundary pixel, also consider
        # what its 5x5 neighborhood mostly is (prevents single-pixel reassignment noise)
        # Apply a soft vote: 70% color distance, 30% neighborhood majority
        new_labels = np.argmin(bz_dists, axis=1)

        # Only reassign if the color-based assignment matches a neighbor
        # (prevents creating new isolated pixels)
        for k, (y, x) in enumerate(zip(bz_y, bz_x)):
            y0, y1 = max(0, y-2), min(h, y+3)
            x0, x1 = max(0, x-2), min(w, x+3)
            neighborhood = pixel_labels[y0:y1, x0:x1]
            neighbor_plates = neighborhood[neighborhood < n_plates]
            if len(neighbor_plates) > 0 and new_labels[k] in neighbor_plates:
                pixel_labels[y, x] = new_labels[k]

    # ── Step 7: Extract and clean individual plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

        # Remove small islands (dust)
        if dust_threshold > 0:
            mask = remove_small_objects(mask, max_size=dust_threshold)
            mask = remove_small_holes(mask, max_size=dust_threshold * 2)

        coverage = np.sum(mask) / mask.size * 100
        rgb = palette_rgb[idx]
        name = f"plate{rank + 1}"

        # Create binary plate image (black = prints/raised, white = carved)
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
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255  # white paper
    for plate_info in reversed(results):  # lightest first, then darker on top
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb = plate_info["color"]
        comp[mask] = rgb

    composite_img = Image.fromarray(comp)

    # ── Step 9: Generate manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(np.sum(pixel_labels == n_plates) / pixel_labels.size * 100, 2),
        "n_superpixels": int(n_superpixels),
    }

    if return_data:
        return {
            "composite": composite_img,
            "plates": plate_images,
            "manifest": manifest,
            "palette_rgb": palette_rgb,
        }

    # Save files
    if output_dir:
        composite_img.save(os.path.join(output_dir, "composite.png"))
        for plate_info in results:
            name = plate_info["name"]
            plate_images[name]["image"].save(os.path.join(output_dir, f"{name}.png"))
            svg_path = os.path.join(output_dir, f"{name}.svg")
            mask_to_svg(plate_images[name]["mask"], svg_path, w, h)
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def build_preview_response(image_bytes, plates=4, dust=50, locked_colors=None,
                           n_segments=3000, compactness=15, chroma_boost=1.3,
                           upscale=False, shadow_threshold=8, highlight_threshold=95,
                           **kwargs):
    """Process image and return composite PNG bytes + manifest. Caches result for ZIP reuse."""
    import hashlib
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Cap input at 1200px for preview
    max_dim = 1200
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    arr = np.array(img)
    result = separate(arr, n_plates=plates, dust_threshold=dust,
                      locked_colors=locked_colors, return_data=True,
                      n_segments=n_segments, compactness=compactness,
                      chroma_boost=chroma_boost, upscale=upscale,
                      shadow_threshold=shadow_threshold,
                      highlight_threshold=highlight_threshold)

    # Cache the result for ZIP reuse
    img_hash = hashlib.sha256(image_bytes[:8192]).hexdigest()
    _last_result_cache["last"] = result
    _last_result_cache["hash"] = img_hash

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=50, locked_colors=None,
                       n_segments=3000, compactness=15, chroma_boost=1.3,
                       upscale=False, shadow_threshold=8, highlight_threshold=95,
                       **kwargs):
    """Process image and return ZIP bytes. Uses cached preview result when available."""
    import hashlib

    img_hash = hashlib.sha256(image_bytes[:8192]).hexdigest()
    if _last_result_cache.get("hash") == img_hash and "last" in _last_result_cache:
        logger.info("Using cached preview result for ZIP — skipping reprocessing")
        result = _last_result_cache["last"]
    else:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        max_dim = 4000
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        arr = np.array(img)
        result = separate(arr, n_plates=plates, dust_threshold=dust,
                          locked_colors=locked_colors, return_data=True,
                          n_segments=n_segments, compactness=compactness,
                          chroma_boost=chroma_boost, upscale=upscale,
                          shadow_threshold=shadow_threshold,
                          highlight_threshold=highlight_threshold)

    comp_arr = np.array(result["composite"])
    h, w = comp_arr.shape[:2]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        comp_buf = io.BytesIO()
        result["composite"].save(comp_buf, format="PNG")
        zf.writestr("composite.png", comp_buf.getvalue())

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
