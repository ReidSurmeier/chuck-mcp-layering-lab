#!/usr/bin/env python3
"""
Woodblock color separation V7 — K-means + Dense CRF.

The definitive algorithm: V2's CIELAB K-means++ pixel clustering for detail
preservation, followed by fully-connected CRF (Dense CRF) for spatially
coherent labels that respect image edges.

This is the standard technique from modern segmentation (DeepLab, etc.):
- K-means finds the right colors (preserves detail)
- CRF fixes the boundaries (removes noise, edges follow image edges)
- No heuristic smoothing needed (no median, no morphological ops)
- The bilateral kernel naturally smooths within regions while keeping edges sharp
"""
import argparse
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2lab, lab2rgb
from skimage.feature import canny
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def apply_dense_crf(image_rgb, labels, n_labels, sxy=3, srgb=13, compat=10, iterations=10):
    """
    Refine label map using Dense CRF.

    Args:
        image_rgb: H x W x 3 uint8 — original RGB image
        labels: H x W int32 (0 to n_labels-1)
        n_labels: number of classes
        sxy: Gaussian spatial bandwidth (higher = smoother boundaries)
        srgb: Bilateral color bandwidth (higher = less color-sensitive)
        compat: Bilateral compatibility (higher = stronger edge preservation)
        iterations: CRF inference iterations
    """
    h, w = labels.shape

    d = dcrf.DenseCRF2D(w, h, n_labels)

    # Unary potentials from K-means labels (high confidence = 0.9)
    U = unary_from_labels(
        labels.astype(np.int32).ravel(), n_labels,
        gt_prob=0.9, zero_unsure=False
    )
    d.setUnaryEnergy(U)

    # Pairwise Gaussian — encourages nearby pixels to have same label
    d.addPairwiseGaussian(sxy=sxy, compat=3)

    # Pairwise Bilateral — encourages pixels with similar color AND position
    # to have same label. This is the key: smooths noise while PRESERVING
    # edges where color changes.
    d.addPairwiseBilateral(
        sxy=sxy, srgb=srgb,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat
    )

    # Inference
    Q = d.inference(iterations)

    # Get MAP labels
    refined_labels = np.argmax(Q, axis=0).reshape(h, w)
    return refined_labels


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=50,
             use_edges=True, edge_sigma=1.5, locked_colors=None, return_data=False,
             chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
             crf_spatial=3, crf_color=13, crf_compat=10, crf_iterations=10):
    """
    V7 separation: CIELAB K-means++ → Dense CRF refinement → minimal cleanup.

    Args:
        input_path_or_array: filepath string or numpy RGB array
        output_dir: where to save (None if return_data=True)
        n_plates: number of color plates
        dust_threshold: minimum island size in pixels
        use_edges: whether to use Canny edge detection
        edge_sigma: Canny sigma parameter
        locked_colors: list of [R,G,B] colors to lock as centroids
        return_data: if True, return dict instead of writing files
        chroma_boost: chroma multiplier for LAB a*/b* channels
        shadow_threshold: L* lower bound for content masking
        highlight_threshold: L* upper bound for content masking
        crf_spatial: Gaussian sxy — spatial smoothness (1-20)
        crf_color: Bilateral srgb — color sensitivity (5-50)
        crf_compat: Bilateral compatibility — edge preservation strength (1-20)
        crf_iterations: CRF inference iterations (1-20)
    """
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

    # ── Step 1: Convert to CIELAB for perceptually uniform clustering ──
    arr_float = arr.astype(np.float64) / 255.0
    lab_img = rgb2lab(arr_float)

    # Boost chroma in LAB space — compensates for K-means centroid averaging
    lab_boosted = lab_img.copy()
    lab_boosted[:, :, 1] *= chroma_boost  # a* channel
    lab_boosted[:, :, 2] *= chroma_boost  # b* channel

    # ── Step 2: Content masking ──
    L_channel = lab_img[:, :, 0]
    is_content = (L_channel > shadow_threshold) & (L_channel < highlight_threshold)

    content_pixels_lab = lab_boosted[is_content]

    # Subsample for K-means speed
    max_samples = 150000
    if len(content_pixels_lab) > max_samples:
        indices = np.random.RandomState(42).choice(len(content_pixels_lab), max_samples, replace=False)
        sample = content_pixels_lab[indices]
    else:
        sample = content_pixels_lab

    # ── Step 3: K-means++ clustering in CIELAB space ──
    init = "k-means++"
    if locked_colors and len(locked_colors) > 0:
        locked_rgb = np.array(locked_colors, dtype=np.float64).reshape(-1, 1, 3) / 255.0
        locked_lab = rgb2lab(locked_rgb).reshape(-1, 3)
        # Apply same chroma boost to locked colors
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

    # Get palette in LAB (undo chroma boost for display) and convert to RGB
    palette_lab_boosted = kmeans.cluster_centers_
    palette_lab = palette_lab_boosted.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost
    palette_rgb_float = lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    palette_rgb = np.clip(palette_rgb_float * 255, 0, 255).astype(np.uint8)

    # ── Step 4: Assign ALL pixels to nearest cluster in boosted CIELAB ──
    lab_flat = lab_boosted.reshape(-1, 3)
    dists = np.zeros((len(lab_flat), n_plates), dtype=np.float64)
    for i, center in enumerate(palette_lab_boosted):
        diff = lab_flat - center
        dists[:, i] = np.sum(diff ** 2, axis=1)

    pixel_labels = np.argmin(dists, axis=1).reshape(h, w)

    # Mark non-content pixels as background
    pixel_labels[~is_content] = n_plates

    # ── Step 5: Dense CRF refinement ──
    # Remap labels: background = 0, plates = 1..n_plates
    # CRF needs contiguous labels starting from 0
    crf_labels = pixel_labels.copy()
    crf_labels[is_content] = pixel_labels[is_content] + 1  # shift plates to 1..n_plates
    crf_labels[~is_content] = 0  # background = 0

    n_crf_labels = n_plates + 1  # background + n_plates

    # Apply CRF with the ORIGINAL RGB image (not boosted/filtered)
    arr_rgb_u8 = arr.astype(np.uint8)
    refined_crf_labels = apply_dense_crf(
        arr_rgb_u8, crf_labels, n_crf_labels,
        sxy=crf_spatial, srgb=crf_color, compat=crf_compat,
        iterations=crf_iterations
    )

    # Remap back: CRF label 0 = background (n_plates), CRF labels 1..n = plates 0..n-1
    pixel_labels = np.where(
        refined_crf_labels == 0,
        n_plates,  # background
        refined_crf_labels - 1  # plate index
    )

    # Restore background mask
    pixel_labels[~is_content] = n_plates

    # ── Step 6: Edge detection and assignment (optional) ──
    if use_edges:
        from skimage.color import rgb2gray
        gray = rgb2gray(arr)
        edges = canny(gray, sigma=edge_sigma, low_threshold=0.04, high_threshold=0.12)
        from scipy.ndimage import binary_dilation
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

    # ── Step 7: Extract and clean individual plates ──
    # NO median filter, NO morphological close — CRF already cleaned boundaries
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

        # Fill holes with binary_fill_holes
        mask = binary_fill_holes(mask)

        # Remove dust with remove_small_objects
        mask = remove_small_objects(mask, min_size=dust_threshold)

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
        "version": "v7",
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


def build_preview_response(image_bytes, plates=4, dust=50, use_edges=True,
                           edge_sigma=1.5, locked_colors=None,
                           chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
                           crf_spatial=3, crf_color=13, crf_compat=10):
    """Process image and return composite PNG bytes + manifest."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Limit size for preview speed
    max_dim = 1200
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
        shadow_threshold=shadow_threshold,
        highlight_threshold=highlight_threshold,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=50, use_edges=True,
                       edge_sigma=1.5, locked_colors=None,
                       chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
                       crf_spatial=3, crf_color=13, crf_compat=10):
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
        shadow_threshold=shadow_threshold,
        highlight_threshold=highlight_threshold,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat,
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        comp_buf = io.BytesIO()
        result["composite"].save(comp_buf, format="PNG")
        zf.writestr("composite.png", comp_buf.getvalue())

        h, w = arr.shape[:2]
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
    parser = argparse.ArgumentParser(description="Woodblock color separation V7 (K-means + Dense CRF)")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--plates", type=int, default=4, help="Number of color plates")
    parser.add_argument("--dust", type=int, default=50, help="Min island size in pixels")
    parser.add_argument("--no-edges", action="store_true", help="Disable edge detection")
    parser.add_argument("--edge-sigma", type=float, default=1.5, help="Canny edge sigma")
    parser.add_argument("--crf-spatial", type=int, default=3, help="CRF spatial bandwidth")
    parser.add_argument("--crf-color", type=int, default=13, help="CRF color bandwidth")
    parser.add_argument("--crf-compat", type=int, default=10, help="CRF compatibility")
    args = parser.parse_args()

    result = separate(args.input, args.output, n_plates=args.plates,
                      dust_threshold=args.dust, use_edges=not args.no_edges,
                      edge_sigma=args.edge_sigma,
                      crf_spatial=args.crf_spatial, crf_color=args.crf_color,
                      crf_compat=args.crf_compat)
    print(json.dumps(result, indent=2))
