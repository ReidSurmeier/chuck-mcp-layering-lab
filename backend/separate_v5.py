#!/usr/bin/env python3
"""
Woodblock color separation V5 — noise-free, sharp-edge, flat-shape plates.

Key innovation: targeted line noise removal without edge rounding.
Instead of median filter (which rounds corners), uses a mode filter
that ONLY changes pixels that are isolated or in thin 1px strips.
Edges between large color regions are preserved exactly.

Based on V2 CIELAB clustering but with:
1. No source image blur
2. No label map median filter
3. No morphological close
4. Targeted mode filter for line noise only
5. Connected component analysis for clean geometry
"""
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.color import rgb2lab, lab2rgb
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.cluster import KMeans
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file


def remove_line_noise(labels, bg_label, iterations=3):
    """
    Remove thin line noise from label map without rounding edges.

    Strategy: for each pixel, check if it's "thin" — meaning it differs
    from BOTH neighbors in at least one axis (horizontal or vertical).
    If thin, replace with the most common neighbor label.

    This removes 1px-wide noise lines while preserving all edges between
    large solid regions.
    """
    h, w = labels.shape
    result = labels.copy()

    for _ in range(iterations):
        changed = 0
        padded = np.pad(result, 1, mode='edge')

        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        # A pixel is "thin horizontally" if it differs from both left AND right
        thin_h = (result != left) & (result != right)
        # A pixel is "thin vertically" if it differs from both up AND down
        thin_v = (result != up) & (result != down)
        # A pixel is "isolated" if it differs from ALL 4 neighbors
        isolated = thin_h & thin_v
        # Thin = isolated OR (thin in one direction AND the two matching neighbors agree)
        thin = isolated | (thin_h & (left == right)) | (thin_v & (up == down))

        # Don't touch background
        thin = thin & (result != bg_label)

        # For thin pixels, replace with most common non-self neighbor
        thin_y, thin_x = np.where(thin)
        for y, x in zip(thin_y, thin_x):
            neighbors = []
            if y > 0:
                neighbors.append(result[y-1, x])
            if y < h-1:
                neighbors.append(result[y+1, x])
            if x > 0:
                neighbors.append(result[y, x-1])
            if x < w-1:
                neighbors.append(result[y, x+1])
            # Also diagonal neighbors for better context
            if y > 0 and x > 0:
                neighbors.append(result[y-1, x-1])
            if y > 0 and x < w-1:
                neighbors.append(result[y-1, x+1])
            if y < h-1 and x > 0:
                neighbors.append(result[y+1, x-1])
            if y < h-1 and x < w-1:
                neighbors.append(result[y+1, x+1])

            neighbors = [n for n in neighbors if n != bg_label]
            if neighbors:
                # Most common neighbor
                vals, counts = np.unique(neighbors, return_counts=True)
                result[y, x] = vals[np.argmax(counts)]
                changed += 1

        if changed == 0:
            break

    return result


def remove_line_noise_fast(labels, bg_label, iterations=3):
    """
    Vectorized version of line noise removal — much faster for large images.
    """
    h, w = labels.shape
    result = labels.copy()

    for _ in range(iterations):
        padded = np.pad(result, 1, mode='edge')

        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        # Thin detection
        thin_h = (result != left) & (result != right) & (left == right)
        thin_v = (result != up) & (result != down) & (up == down)
        isolated = (result != left) & (result != right) & (result != up) & (result != down)

        thin = (thin_h | thin_v | isolated) & (result != bg_label)

        if not np.any(thin):
            break

        # For horizontally thin pixels where left==right, take that value
        fix_h = thin_h & ~thin_v & (result != bg_label)
        result[fix_h] = left[fix_h]

        # For vertically thin pixels where up==down, take that value
        fix_v = thin_v & ~thin_h & (result != bg_label)
        result[fix_v] = up[fix_v]

        # For isolated pixels, use a 3x3 mode
        iso_y, iso_x = np.where(isolated & (result != bg_label))
        for y, x in zip(iso_y, iso_x):
            y0, y1 = max(0, y-1), min(h, y+2)
            x0, x1 = max(0, x-1), min(w, x+2)
            patch = result[y0:y1, x0:x1].ravel()
            patch = patch[patch != bg_label]
            if len(patch) > 0:
                vals, counts = np.unique(patch, return_counts=True)
                result[y, x] = vals[np.argmax(counts)]

    return result


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=30,
             use_edges=True, edge_sigma=1.5, locked_colors=None, return_data=False,
             median_size=1, chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95):
    """V5 separation — clean geometry, no line noise, sharp edges."""

    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array
        img = Image.fromarray(arr)

    h, w = arr.shape[:2]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── NO source image blur — preserve all original detail ──
    arr_clean = arr

    # ── CIELAB conversion ──
    arr_float = arr_clean.astype(np.float64) / 255.0
    lab_img = rgb2lab(arr_float)

    # Chroma boost
    lab_img[:, :, 1] *= chroma_boost
    lab_img[:, :, 2] *= chroma_boost

    # ── Content masking ──
    L_channel = lab_img[:, :, 0]
    is_content = (L_channel > shadow_threshold) & (L_channel < highlight_threshold)

    content_pixels_lab = lab_img[is_content]

    # Subsample for speed
    max_samples = 200000
    if len(content_pixels_lab) > max_samples:
        indices = np.random.RandomState(42).choice(len(content_pixels_lab), max_samples, replace=False)
        sample = content_pixels_lab[indices]
    else:
        sample = content_pixels_lab

    # ── K-means++ in CIELAB ──
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
            init = np.vstack([locked_lab, sample[random_indices]])
        else:
            init = locked_lab[:n_plates]

    kmeans = KMeans(
        n_clusters=n_plates,
        init=init,
        random_state=42,
        n_init=10 if isinstance(init, str) else 1,
        max_iter=300,
    )
    kmeans.fit(sample)

    palette_lab = kmeans.cluster_centers_
    palette_lab_unboost = palette_lab.copy()
    palette_lab_unboost[:, 1] /= chroma_boost
    palette_lab_unboost[:, 2] /= chroma_boost
    palette_rgb_float = lab2rgb(palette_lab_unboost.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Assign pixels ──
    lab_flat = lab_img.reshape(-1, 3)
    dists = np.zeros((len(lab_flat), n_plates), dtype=np.float64)
    for i, center in enumerate(palette_lab):
        diff = lab_flat - center
        dists[:, i] = np.sum(diff ** 2, axis=1)

    pixel_labels = np.argmin(dists, axis=1).reshape(h, w)
    bg_label = n_plates
    pixel_labels[~is_content] = bg_label

    # ── TARGETED LINE NOISE REMOVAL ──
    # Phase 1: remove thin 1px strips and isolated pixels (fast vectorized)
    pixel_labels = remove_line_noise_fast(pixel_labels, bg_label, iterations=5)

    # Phase 2: vectorized majority vote for remaining junction noise
    # A pixel is noise if fewer than 3 of its 8 neighbors share its label
    def majority_vote_filter(labels_2d, bg, passes=3):
        for _ in range(passes):
            padded = np.pad(labels_2d, 1, mode='edge')
            # Count how many of 8 neighbors match the center pixel
            match_count = np.zeros_like(labels_2d, dtype=int)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    shifted = padded[1+dy:1+dy+labels_2d.shape[0], 1+dx:1+dx+labels_2d.shape[1]]
                    match_count += (shifted == labels_2d).astype(int)

            # Pixels with < 3 matching neighbors are noise
            is_noise = (match_count < 3) & (labels_2d != bg)

            if not np.any(is_noise):
                break

            # Replace noise pixels with the most common neighbor via scipy mode filter
            from scipy.ndimage import median_filter
            # Use median on label map but ONLY for noise pixels
            smoothed = median_filter(labels_2d.astype(np.int32), size=3)
            labels_2d[is_noise] = smoothed[is_noise]

        return labels_2d

    pixel_labels = majority_vote_filter(pixel_labels, bg_label, passes=3)

    # ── Edge detection (optional) ──
    if use_edges:
        from skimage.color import rgb2gray
        from skimage.feature import canny
        gray = rgb2gray(arr_clean)
        edges = canny(gray, sigma=edge_sigma, low_threshold=0.04, high_threshold=0.12)
        edges = binary_dilation(edges, iterations=1)

        ey, ex = np.where(edges)
        for y, x in zip(ey, ex):
            y0, y1 = max(0, y - 3), min(h, y + 4)
            x0, x1 = max(0, x - 3), min(w, x + 4)
            nbr = pixel_labels[y0:y1, x0:x1]
            plates = np.unique(nbr)
            plates = plates[plates < n_plates]
            if len(plates) > 0:
                darkest = min(plates, key=lambda p: palette_lab[p][0])
                pixel_labels[y, x] = darkest

    # ── Extract plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])
    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

        # Fill internal holes
        mask = binary_fill_holes(mask)

        # Remove only very small dust (keep all detail)
        mask = remove_small_objects(mask, max_size=dust_threshold)
        mask = remove_small_holes(mask, max_size=dust_threshold * 2)

        # NO morphological close — preserve sharp geometry

        coverage = np.sum(mask) / mask.size * 100
        rgb = palette_rgb[idx]
        name = f"plate{rank + 1}"

        binary_img = np.where(mask, 0, 255).astype(np.uint8)
        plate_img = Image.fromarray(binary_img)

        plate_info = {
            "name": name,
            "index": int(idx),
            "rank": rank + 1,
            "color": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "color_hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "coverage_pct": round(coverage, 2),
        }
        results.append(plate_info)
        plate_images[name] = {"mask": mask, "binary": binary_img, "image": plate_img}

    # ── Composite ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb = plate_info["color"]
        comp[mask] = rgb

    composite_img = Image.fromarray(comp)

    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(np.sum(pixel_labels == bg_label) / pixel_labels.size * 100, 2),
        "version": "v5",
    }

    if return_data:
        return {"composite": composite_img, "plates": plate_images, "manifest": manifest, "palette_rgb": palette_rgb}

    if output_dir:
        composite_img.save(os.path.join(output_dir, "composite.png"))
        for plate_info in results:
            name = plate_info["name"]
            plate_images[name]["image"].save(os.path.join(output_dir, f"{name}.png"))
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def build_preview_response(image_bytes, plates=4, dust=30, use_edges=True,
                           edge_sigma=1.5, locked_colors=None,
                           median_size=1, chroma_boost=1.3,
                           shadow_threshold=8, highlight_threshold=95):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = 1500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    arr = np.array(img)
    result = separate(arr, n_plates=plates, dust_threshold=dust,
                      use_edges=use_edges, edge_sigma=edge_sigma,
                      locked_colors=locked_colors, return_data=True,
                      median_size=median_size, chroma_boost=chroma_boost,
                      shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold)

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")
    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=30, use_edges=True,
                       edge_sigma=1.5, locked_colors=None,
                       median_size=1, chroma_boost=1.3,
                       shadow_threshold=8, highlight_threshold=95):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    result = separate(arr, n_plates=plates, dust_threshold=dust,
                      use_edges=use_edges, edge_sigma=edge_sigma,
                      locked_colors=locked_colors, return_data=True,
                      median_size=median_size, chroma_boost=chroma_boost,
                      shadow_threshold=shadow_threshold, highlight_threshold=highlight_threshold)

    h, w = arr.shape[:2]
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        comp_buf = io.BytesIO()
        result["composite"].save(comp_buf, format="PNG")
        zf.writestr("composite.png", comp_buf.getvalue())
        for plate_info in result["manifest"]["plates"]:
            name = plate_info["name"]
            png_buf = io.BytesIO()
            result["plates"][name]["image"].save(png_buf, format="PNG")
            zf.writestr(f"{name}.png", png_buf.getvalue())
            zf.writestr(f"{name}.svg", mask_to_svg_string(result["plates"][name]["mask"], w, h))
        zf.writestr("manifest.json", json.dumps(result["manifest"], indent=2))
    return zip_buf.getvalue()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Woodblock V5 — clean geometry")
    parser.add_argument("input", help="Input image")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--plates", type=int, default=6)
    parser.add_argument("--dust", type=int, default=30)
    parser.add_argument("--no-edges", action="store_true")
    args = parser.parse_args()
    result = separate(args.input, args.output, n_plates=args.plates,
                      dust_threshold=args.dust, use_edges=not args.no_edges)
    print(json.dumps(result, indent=2))
