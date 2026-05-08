#!/usr/bin/env python3
"""
Overnight optimization loop for woodblock color separation.

Tests parameter combinations, measures MSE + noise, logs results,
and saves the best configuration. Designed to run as a cron job.

Results logged to: /home/reidsurmeier/sites/woodblock-tools/data/optimization/
"""
import csv
import cv2
import itertools
import json
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import label as ndlabel, binary_fill_holes

# Test images
TEST_IMAGES = [
    "/tmp/woodblock-ref/test-wave.jpg",
    "/tmp/woodblock-ref/test-single-print.png",
]

OUTPUT_DIR = "/home/reidsurmeier/sites/woodblock-tools/data/optimization"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
BEST_PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_params.json")


def measure_noise(labels):
    """Count line noise pixels as percentage."""
    padded = np.pad(labels, 1, mode='edge')
    up = padded[:-2, 1:-1]; down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]; right = padded[1:-1, 2:]
    h_lines = (labels != up) & (labels != down) & (up == down)
    v_lines = (labels != left) & (labels != right) & (left == right)
    isolated = (labels != up) & (labels != down) & (labels != left) & (labels != right)
    total = int(np.sum(h_lines) + np.sum(v_lines) + np.sum(isolated))
    pct = total / labels.size * 100
    return total, pct


def measure_mse(original, composite):
    """Per-pixel MSE between original and composite."""
    return float(np.mean((original.astype(float) - composite.astype(float)) ** 2))


def run_separation(img_bgr, n_plates, sigma_s, sigma_r, ms_sp, ms_sr, 
                    chroma_boost, dust_threshold, use_bilateral=True):
    """Run the V9 pipeline with given params. Returns labels, palette, composite."""
    # Step 1: Edge-preserving filter
    if use_bilateral:
        filtered = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        filtered = img_bgr.copy()
    
    # Step 2: Mean shift
    if ms_sp > 0 and ms_sr > 0:
        filtered = cv2.pyrMeanShiftFiltering(filtered, sp=ms_sp, sr=ms_sr)
    
    # Step 3: CIELAB K-means
    rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB).astype(float) / 255
    lab = rgb2lab(rgb)
    lab[:, :, 1] *= chroma_boost
    lab[:, :, 2] *= chroma_boost
    h, w = lab.shape[:2]
    lab_flat = lab.reshape(-1, 3)
    
    max_samples = 150000
    if len(lab_flat) > max_samples:
        sample = lab_flat[np.random.RandomState(42).choice(len(lab_flat), max_samples, replace=False)]
    else:
        sample = lab_flat
    
    km = KMeans(n_clusters=n_plates, init='k-means++', random_state=42, n_init=10, max_iter=300)
    km.fit(sample)
    labels = km.predict(lab_flat).reshape(h, w)
    
    # Step 4: CC cleanup
    for plate_id in range(n_plates):
        mask = (labels == plate_id)
        labeled, n_components = ndlabel(mask)
        for comp in range(1, n_components + 1):
            comp_mask = labeled == comp
            if np.sum(comp_mask) < dust_threshold:
                dilated = cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
                border = (dilated > 0) & ~comp_mask
                if np.any(border):
                    neighbors = labels[border]
                    neighbors = neighbors[neighbors != plate_id]
                    if len(neighbors) > 0:
                        vals, counts = np.unique(neighbors, return_counts=True)
                        labels[comp_mask] = vals[np.argmax(counts)]
    
    # Step 5: Fill holes
    for plate_id in range(n_plates):
        mask = labels == plate_id
        filled = binary_fill_holes(mask)
        labels[filled & ~mask] = plate_id
    
    # Palette
    palette_lab = km.cluster_centers_.copy()
    palette_lab[:, 1] /= chroma_boost
    palette_lab[:, 2] /= chroma_boost
    palette_rgb = np.clip(lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    
    # Composite
    composite = palette_rgb[labels]
    
    return labels, palette_rgb, composite


def run_iteration(iteration_num):
    """Run one optimization iteration with random parameter sampling."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parameter ranges to explore
    param_space = {
        'sigma_s': [40, 60, 80, 100, 120, 150, 200],
        'sigma_r': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'ms_sp': [0, 10, 15, 20, 30, 40],
        'ms_sr': [0, 20, 30, 40, 60],
        'chroma_boost': [1.0, 1.2, 1.3, 1.5],
        'dust_threshold': [30, 50, 80, 100, 150],
        'n_plates': [6, 8, 10, 12],
    }
    
    # Load best params if they exist
    best_score = float('inf')
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_data = json.load(f)
            best_score = best_data.get('combined_score', float('inf'))
    
    # Random sample params from the space
    rng = np.random.RandomState(int(time.time()) % (2**31))
    n_trials = 20  # test 20 combinations per iteration
    
    results = []
    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        # Ensure ms_sp and ms_sr are both 0 or both nonzero
        if params['ms_sp'] == 0:
            params['ms_sr'] = 0
        elif params['ms_sr'] == 0:
            params['ms_sp'] = 0
        
        trial_scores = []
        for img_path in TEST_IMAGES:
            if not os.path.exists(img_path):
                continue
            
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                
                # Resize for speed
                max_dim = 800
                h, w = img_bgr.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
                
                orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                start = time.time()
                labels, palette, composite = run_separation(
                    img_bgr,
                    n_plates=int(params['n_plates']),
                    sigma_s=float(params['sigma_s']),
                    sigma_r=float(params['sigma_r']),
                    ms_sp=int(params['ms_sp']),
                    ms_sr=int(params['ms_sr']),
                    chroma_boost=float(params['chroma_boost']),
                    dust_threshold=int(params['dust_threshold']),
                )
                elapsed = time.time() - start
                
                noise_count, noise_pct = measure_noise(labels)
                mse = measure_mse(orig_rgb, composite)
                
                # Combined score: lower is better
                # Weight: MSE matters more, noise is secondary
                combined = mse * 0.7 + noise_pct * 1000 * 0.3
                
                trial_scores.append({
                    'image': os.path.basename(img_path),
                    'mse': mse,
                    'noise_pct': noise_pct,
                    'noise_count': noise_count,
                    'combined': combined,
                    'elapsed': elapsed,
                })
            except Exception as e:
                print(f"  Trial {trial} failed on {img_path}: {e}")
                continue
        
        if trial_scores:
            avg_combined = np.mean([s['combined'] for s in trial_scores])
            avg_mse = np.mean([s['mse'] for s in trial_scores])
            avg_noise = np.mean([s['noise_pct'] for s in trial_scores])
            
            result = {
                'iteration': iteration_num,
                'trial': trial,
                'timestamp': datetime.now().isoformat(),
                'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in params.items()},
                'avg_mse': float(avg_mse),
                'avg_noise_pct': float(avg_noise),
                'combined_score': float(avg_combined),
                'per_image': trial_scores,
            }
            results.append(result)
            
            if avg_combined < best_score:
                best_score = avg_combined
                best_data = {
                    'combined_score': float(avg_combined),
                    'mse': float(avg_mse),
                    'noise_pct': float(avg_noise),
                    'params': result['params'],
                    'timestamp': datetime.now().isoformat(),
                    'iteration': iteration_num,
                }
                with open(BEST_PARAMS_FILE, 'w') as f:
                    json.dump(best_data, f, indent=2)
                print(f"  ★ NEW BEST: MSE={avg_mse:.0f} noise={avg_noise:.3f}% combined={avg_combined:.0f}")
                
                # Save best composite
                if trial_scores:
                    img_bgr = cv2.imread(TEST_IMAGES[0])
                    max_dim = 800
                    h, w = img_bgr.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
                    _, _, comp = run_separation(img_bgr, **{k: (int(v) if k in ('n_plates','ms_sp','ms_sr','dust_threshold') else float(v)) for k, v in params.items()})
                    Image.fromarray(comp).save(os.path.join(OUTPUT_DIR, f"best_iter{iteration_num}.png"))
    
    # Append to CSV
    csv_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['iteration', 'trial', 'timestamp', 'sigma_s', 'sigma_r', 
                           'ms_sp', 'ms_sr', 'chroma_boost', 'dust_threshold', 'n_plates',
                           'avg_mse', 'avg_noise_pct', 'combined_score'])
        for r in results:
            writer.writerow([
                r['iteration'], r['trial'], r['timestamp'],
                r['params']['sigma_s'], r['params']['sigma_r'],
                r['params']['ms_sp'], r['params']['ms_sr'],
                r['params']['chroma_boost'], r['params']['dust_threshold'],
                r['params']['n_plates'],
                f"{r['avg_mse']:.1f}", f"{r['avg_noise_pct']:.4f}", f"{r['combined_score']:.1f}",
            ])
    
    print(f"Iteration {iteration_num}: tested {len(results)} combos, best combined={best_score:.0f}")
    return best_score


if __name__ == "__main__":
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_iteration(iteration)
