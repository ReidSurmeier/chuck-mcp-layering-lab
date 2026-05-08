#!/usr/bin/env python3
"""Profile v20 pipeline timing breakdown."""
import time
import io
import numpy as np
from PIL import Image

# Create a test image (800x600 with some color regions)
np.random.seed(42)
img = np.zeros((600, 800, 3), dtype=np.uint8)
img[:300, :400] = [200, 50, 50]   # red quadrant
img[:300, 400:] = [50, 50, 200]   # blue quadrant
img[300:, :400] = [50, 180, 50]   # green quadrant
img[300:, 400:] = [40, 40, 40]    # dark quadrant
# Add noise
img = np.clip(img.astype(np.int16) + np.random.randint(-20, 20, img.shape), 0, 255).astype(np.uint8)

buf = io.BytesIO()
Image.fromarray(img).save(buf, format="PNG")
image_bytes = buf.getvalue()

print("=== V20 Pipeline Profiling ===")
print(f"Test image: 800x600 (4 color regions)")

# Time the full pipeline
t0 = time.perf_counter()
from separate_v20 import build_preview_response
t_import = time.perf_counter()
print(f"Import time: {t_import - t0:.2f}s")

t1 = time.perf_counter()
result_bytes, manifest = build_preview_response(
    image_bytes=image_bytes,
    plates=4,
    dust=50,
    upscale=True,
    upscale_scale=2,
)
t2 = time.perf_counter()
print(f"First run (cold): {t2 - t1:.2f}s")
print(f"  upscaled: {manifest.get('upscaled')}")
print(f"  plates: {manifest.get('num_plates')}")

# Second run (cached models)
t3 = time.perf_counter()
result_bytes2, manifest2 = build_preview_response(
    image_bytes=image_bytes,
    plates=4,
    dust=50,
    upscale=True,
    upscale_scale=2,
)
t4 = time.perf_counter()
print(f"Second run (cached): {t4 - t3:.2f}s")

print(f"\nTotal cold: {t2 - t0:.2f}s")
print(f"Total cached: {t4 - t3:.2f}s")
print("=== Done ===")
