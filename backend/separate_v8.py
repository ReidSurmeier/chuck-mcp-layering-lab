#!/usr/bin/env python3
"""
Woodblock color separation V8 — Bilateral + K-means + Tuned CRF + Line Noise Cleanup.

Combines the best of V2 (color fidelity) and V7 (CRF spatial coherence) with
targeted fixes for both:

1. Bilateral filter on source image BEFORE clustering — smooths pixel noise
   while preserving edges, giving K-means cleaner input
2. V2's CIELAB K-means++ for best color fidelity
3. Dense CRF with LOWER spatial smoothing (sxy=2) and HIGHER color sensitivity
   (srgb=5) — respects color edges more tightly than V7's over-smooth params
4. gt_prob=0.95 — high confidence in K-means labels, CRF only fixes boundaries
5. V5's targeted line noise removal as final cleanup
6. binary_fill_holes on each plate
7. NO median filter on label map, NO morphological close
"""
import argparse
import io
import json
import os
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from skimage.color import rgb2lab, lab2rgb
from skimage.feature import canny
from svg_generator import mask_to_svg, mask_to_svg as mask_to_svg_string, mask_to_svg_file
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


# ── Default CRF parameters (tuned via parameter sweep) ──
DEFAULT_CRF_SPATIAL = 5
DEFAULT_CRF_COLOR = 20
DEFAULT_CRF_COMPAT = 10
DEFAULT_GT_PROB = 0.95


def apply_dense_crf(image_rgb, labels, n_labels, sxy=2, srgb=5, compat=5,
                    gt_prob=0.95, iterations=10):
    """
    Refine label map using Dense CRF with tight edge-respecting params.

    Key differences from V7:
    - sxy=2 (not 3): less spatial smoothing, preserves detail
    - srgb=5 (not 13): much more color-sensitive, CRF won't merge dissimilar colors
    - gt_prob=0.95 (not 0.9): higher confidence in K-means, CRF only fixes edges
    """
    h, w = labels.shape

    d = dcrf.DenseCRF2D(w, h, n_labels)

    U = unary_from_labels(
        labels.astype(np.int32).ravel(), n_labels,
        gt_prob=gt_prob, zero_unsure=False
    )
    d.setUnaryEnergy(U)

    # Pairwise Gaussian — spatial smoothness only
    d.addPairwiseGaussian(sxy=sxy, compat=3)

    # Pairwise Bilateral — color + spatial
    d.addPairwiseBilateral(
        sxy=sxy, srgb=srgb,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat
    )

    Q = d.inference(iterations)
    refined_labels = np.argmax(Q, axis=0).reshape(h, w)
    return refined_labels


def remove_line_noise_fast(labels, bg_label, iterations=3):
    """
    Vectorized thin-line noise removal from V5.
    Removes 1px-wide noise lines while preserving edges between solid regions.
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
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            patch = result[y0:y1, x0:x1].ravel()
            patch = patch[patch != bg_label]
            if len(patch) > 0:
                vals, counts = np.unique(patch, return_counts=True)
                result[y, x] = vals[np.argmax(counts)]

    return result


def majority_vote_cleanup(labels, bg_label, passes=2):
    """
    Final cleanup: replace pixels with < 3 matching neighbors.
    Uses median on label map but ONLY for noise pixels.
    """
    from scipy.ndimage import median_filter

    for _ in range(passes):
        padded = np.pad(labels, 1, mode='edge')
        match_count = np.zeros_like(labels, dtype=int)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1 + dy:1 + dy + labels.shape[0],
                                 1 + dx:1 + dx + labels.shape[1]]
                match_count += (shifted == labels).astype(int)

        is_noise = (match_count < 3) & (labels != bg_label)

        if not np.any(is_noise):
            break

        smoothed = median_filter(labels.astype(np.int32), size=3)
        labels[is_noise] = smoothed[is_noise]

    return labels


def separate(input_path_or_array, output_dir=None, n_plates=4, dust_threshold=50,
             use_edges=True, edge_sigma=1.5, locked_colors=None, return_data=False,
             chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
             crf_spatial=None, crf_color=None, crf_compat=None, crf_iterations=10,
             bilateral_d=5, bilateral_sigma_color=50, bilateral_sigma_space=50,
             upscale=True):
    """
    V8 separation: Upscale → Bilateral → CIELAB K-means++ → Dense CRF → Line Noise Cleanup.

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
        crf_spatial: CRF spatial bandwidth (default tuned)
        crf_color: CRF color bandwidth (default tuned)
        crf_compat: CRF compatibility strength (default tuned)
        crf_iterations: CRF inference iterations
        bilateral_d: bilateral filter diameter
        bilateral_sigma_color: bilateral filter color sigma
        bilateral_sigma_space: bilateral filter space sigma
    """
    if crf_spatial is None:
        crf_spatial = DEFAULT_CRF_SPATIAL
    if crf_color is None:
        crf_color = DEFAULT_CRF_COLOR
    if crf_compat is None:
        crf_compat = DEFAULT_CRF_COMPAT

    # Load image
    if isinstance(input_path_or_array, str):
        img = Image.open(input_path_or_array).convert("RGB")
        arr = np.array(img)
    else:
        arr = input_path_or_array
        img = Image.fromarray(arr)

    # ── Step 0: Real-ESRGAN 2x upscale ──
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
                # Free GPU memory
                del upsampler, model
                torch.cuda.empty_cache()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Upscale failed: {e}")

    h, w = arr.shape[:2]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Bilateral filter on source image ──
    # Smooths noise while preserving edges — gives K-means cleaner input
    arr_filtered = cv2.bilateralFilter(
        arr, d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space
    )

    # ── Step 2: Convert to CIELAB for perceptually uniform clustering ──
    arr_float = arr_filtered.astype(np.float64) / 255.0
    lab_img = rgb2lab(arr_float)

    # Boost chroma in LAB space
    lab_boosted = lab_img.copy()
    lab_boosted[:, :, 1] *= chroma_boost
    lab_boosted[:, :, 2] *= chroma_boost

    # ── Step 3: Content masking ──
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

    # ── Step 4: K-means++ clustering in CIELAB space ──
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

    # ── Step 5: Assign ALL pixels to nearest cluster in boosted CIELAB ──
    lab_flat = lab_boosted.reshape(-1, 3)
    dists = np.zeros((len(lab_flat), n_plates), dtype=np.float64)
    for i, center in enumerate(palette_lab_boosted):
        diff = lab_flat - center
        dists[:, i] = np.sum(diff ** 2, axis=1)

    pixel_labels = np.argmin(dists, axis=1).reshape(h, w)
    pixel_labels[~is_content] = n_plates

    # ── Step 6: Dense CRF refinement ──
    # Use the ORIGINAL (unfiltered) RGB for CRF — bilateral was only for clustering
    crf_labels = pixel_labels.copy()
    crf_labels[is_content] = pixel_labels[is_content] + 1  # shift plates to 1..n_plates
    crf_labels[~is_content] = 0  # background = 0

    n_crf_labels = n_plates + 1

    arr_rgb_u8 = arr.astype(np.uint8)
    refined_crf_labels = apply_dense_crf(
        arr_rgb_u8, crf_labels, n_crf_labels,
        sxy=crf_spatial, srgb=crf_color, compat=crf_compat,
        gt_prob=DEFAULT_GT_PROB, iterations=crf_iterations
    )

    # Remap back
    pixel_labels = np.where(
        refined_crf_labels == 0,
        n_plates,  # background
        refined_crf_labels - 1  # plate index
    )

    # Restore background mask
    pixel_labels[~is_content] = n_plates

    # ── Step 7: Targeted line noise removal (from V5) ──
    pixel_labels = remove_line_noise_fast(pixel_labels, n_plates, iterations=5)
    pixel_labels = majority_vote_cleanup(pixel_labels, n_plates, passes=2)

    # ── Step 8: Edge detection and assignment (optional) ──
    if use_edges:
        from skimage.color import rgb2gray
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

    # ── Step 9: Extract and clean individual plates ──
    brightness_order = np.argsort([c[0] for c in palette_lab])

    results = []
    plate_images = {}

    for rank, idx in enumerate(brightness_order):
        mask = pixel_labels == idx

        # Fill holes with binary_fill_holes
        mask = binary_fill_holes(mask)

        # Remove dust
        mask = remove_small_objects(mask, max_size=dust_threshold)

        # Dilate plates to fill gaps (same as V4 — prevents white spaces)
        from skimage.morphology import disk as morph_disk
        mask = binary_dilation(mask, morph_disk(3))
        mask = binary_erosion(mask, morph_disk(1))

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

    # ── Step 10: Composite preview ──
    comp = np.ones((h, w, 3), dtype=np.uint8) * 255
    for plate_info in reversed(results):
        name = plate_info["name"]
        mask = plate_images[name]["mask"]
        rgb = plate_info["color"]
        comp[mask] = rgb

    composite_img = Image.fromarray(comp)

    # ── Step 11: Generate manifest ──
    manifest = {
        "width": w,
        "height": h,
        "num_plates": n_plates,
        "plates": results,
        "paper_pct": round(np.sum(pixel_labels == n_plates) / pixel_labels.size * 100, 2),
        "version": "v8",
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


def build_preview_response(image_bytes, plates=4, dust=50, use_edges=True,
                           edge_sigma=1.5, locked_colors=None,
                           chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
                           crf_spatial=None, crf_color=None, crf_compat=None,
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
        use_edges=use_edges, edge_sigma=edge_sigma,
        locked_colors=locked_colors, return_data=True,
        chroma_boost=chroma_boost,
        shadow_threshold=shadow_threshold,
        highlight_threshold=highlight_threshold,
        crf_spatial=crf_spatial, crf_color=crf_color, crf_compat=crf_compat, upscale=upscale,
    )

    buf = io.BytesIO()
    result["composite"].save(buf, format="PNG")

    return buf.getvalue(), result["manifest"]


def build_zip_response(image_bytes, plates=4, dust=50, use_edges=True,
                       edge_sigma=1.5, locked_colors=None,
                       chroma_boost=1.3, shadow_threshold=8, highlight_threshold=95,
                       crf_spatial=None, crf_color=None, crf_compat=None,
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


def compute_mse(original, composite):
    """Compute MSE between original and composite images."""
    return np.mean((original.astype(np.float64) - composite.astype(np.float64)) ** 2)


def compute_noise_pct(pixel_labels, bg_label):
    """Compute percentage of pixels that are thin-line noise (< 3 matching neighbors)."""
    padded = np.pad(pixel_labels, 1, mode='edge')
    match_count = np.zeros_like(pixel_labels, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy:1 + dy + pixel_labels.shape[0],
                             1 + dx:1 + dx + pixel_labels.shape[1]]
            match_count += (shifted == pixel_labels).astype(int)

    is_noise = (match_count < 2) & (pixel_labels != bg_label)
    return np.sum(is_noise) / pixel_labels.size * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Woodblock color separation V8")
    parser.add_argument("input", nargs="?", default=None, help="Input image path")
    parser.add_argument("output", nargs="?", default=None, help="Output directory")
    parser.add_argument("--plates", type=int, default=8, help="Number of color plates")
    parser.add_argument("--dust", type=int, default=50, help="Min island size in pixels")
    parser.add_argument("--no-edges", action="store_true", help="Disable edge detection")
    parser.add_argument("--edge-sigma", type=float, default=1.5, help="Canny edge sigma")
    parser.add_argument("--sweep", action="store_true",
                        help="Run CRF parameter sweep on the input image")
    args = parser.parse_args()

    if args.sweep or (args.input is None):
        # ── CRF Parameter Sweep ──
        test_path = args.input or "/tmp/woodblock-ref/test-wave.jpg"
        n_plates = args.plates

        print(f"Loading {test_path} with {n_plates} plates...")
        img = Image.open(test_path).convert("RGB")
        # Resize for speed
        max_dim = 600
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        arr = np.array(img)

        sxy_values = [1, 2, 3, 5]
        srgb_values = [5, 10, 15, 20]
        compat_values = [3, 5, 10]

        print(f"\n{'sxy':>4} {'srgb':>5} {'compat':>7} {'MSE':>8} {'noise%':>7}")
        print("-" * 40)

        best_score = float('inf')
        best_params = None
        results_table = []

        for sxy in sxy_values:
            for srgb in srgb_values:
                for compat in compat_values:
                    result = separate(
                        arr.copy(), n_plates=n_plates, dust_threshold=50,
                        use_edges=True, return_data=True,
                        crf_spatial=sxy, crf_color=srgb, crf_compat=compat,
                    )

                    # Compute MSE
                    comp_arr = np.array(result["composite"])
                    mse = compute_mse(arr, comp_arr)

                    # Compute noise from plate masks
                    # Reconstruct pixel_labels from masks
                    pixel_labels = np.full((arr.shape[0], arr.shape[1]), n_plates, dtype=np.int32)
                    for pi in result["manifest"]["plates"]:
                        name = pi["name"]
                        mask = result["plates"][name]["mask"]
                        pixel_labels[mask] = pi["index"]

                    noise = compute_noise_pct(pixel_labels, n_plates)

                    # Combined score: weight MSE heavily but penalize noise
                    combined = mse + noise * 1000

                    results_table.append((sxy, srgb, compat, mse, noise, combined))
                    print(f"{sxy:>4} {srgb:>5} {compat:>7} {mse:>8.1f} {noise:>6.3f}%")

                    if combined < best_score:
                        best_score = combined
                        best_params = (sxy, srgb, compat)

        print(f"\n{'='*40}")
        print(f"BEST: sxy={best_params[0]}, srgb={best_params[1]}, compat={best_params[2]}")
        best_row = [r for r in results_table if (r[0], r[1], r[2]) == best_params][0]
        print(f"  MSE={best_row[3]:.1f}, noise={best_row[4]:.3f}%")
        print("\nRecommended defaults:")
        print(f"  DEFAULT_CRF_SPATIAL = {best_params[0]}")
        print(f"  DEFAULT_CRF_COLOR = {best_params[1]}")
        print(f"  DEFAULT_CRF_COMPAT = {best_params[2]}")

    elif args.input and args.output:
        result = separate(args.input, args.output, n_plates=args.plates,
                          dust_threshold=args.dust, use_edges=not args.no_edges,
                          edge_sigma=args.edge_sigma)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
