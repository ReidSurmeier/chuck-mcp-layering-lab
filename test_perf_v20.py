#!/usr/bin/env python3
"""
Performance benchmarks for v20 color separation pipeline.

Tests ensure the optimized pipeline meets timing requirements:
- Cold start (first request, model loading): <= 45s
- Cached (models already loaded): <= 30s
- Upscale 4x included in cached timing

Run: pytest test_perf_v20.py -v -s
"""
import io
import time

import numpy as np
import pytest
from PIL import Image


def _make_test_image(width=800, height=600):
    """Create a synthetic test image with distinct color regions."""
    np.random.seed(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:300, :400] = [200, 50, 50]
    img[:300, 400:] = [50, 50, 200]
    img[300:, :400] = [50, 180, 50]
    img[300:, 400:] = [40, 40, 40]
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def test_image():
    return _make_test_image()


def test_v20_cold_start_under_45s(test_image):
    """First request including model loading must complete in <= 45s."""
    import importlib
    import separate_v20
    separate_v20._sam_model = None
    separate_v20._esrgan_cache.clear()
    separate_v20._upscale_cache = separate_v20.LRUCache(maxsize=5)
    importlib.reload(separate_v20)

    t0 = time.perf_counter()
    result_bytes, manifest = separate_v20.build_preview_response(
        image_bytes=test_image, plates=4, dust=50,
        upscale=True, upscale_scale=2,
    )
    elapsed = time.perf_counter() - t0

    assert result_bytes is not None
    assert manifest["num_plates"] == 4
    assert elapsed <= 45.0, f"Cold start took {elapsed:.1f}s, limit is 45s"
    print(f"\n  Cold start: {elapsed:.1f}s (limit: 45s)")


def test_v20_cached_under_30s(test_image):
    """Cached request (models loaded) must complete in <= 30s."""
    from separate_v20 import build_preview_response, _sam_model

    if _sam_model is None:
        build_preview_response(
            image_bytes=test_image, plates=4, dust=50,
            upscale=True, upscale_scale=2,
        )

    t0 = time.perf_counter()
    result_bytes, manifest = build_preview_response(
        image_bytes=test_image, plates=4, dust=50,
        upscale=True, upscale_scale=2,
    )
    elapsed = time.perf_counter() - t0

    assert result_bytes is not None
    assert elapsed <= 30.0, f"Cached run took {elapsed:.1f}s, limit is 30s"
    print(f"\n  Cached run: {elapsed:.1f}s (limit: 30s)")


def test_upscale_4x_included_in_timing(test_image):
    """4x upscale must still complete within 30s cached."""
    from separate_v20 import build_preview_response, _sam_model

    if _sam_model is None:
        build_preview_response(
            image_bytes=test_image, plates=4, dust=50,
            upscale=True, upscale_scale=2,
        )

    t0 = time.perf_counter()
    result_bytes, manifest = build_preview_response(
        image_bytes=test_image, plates=4, dust=50,
        upscale=True, upscale_scale=4,
    )
    elapsed = time.perf_counter() - t0

    assert result_bytes is not None
    assert manifest["upscaled"] is True
    assert elapsed <= 30.0, f"4x upscale took {elapsed:.1f}s, limit is 30s"
    print(f"\n  4x upscale cached: {elapsed:.1f}s (limit: 30s)")


def test_output_correctness(test_image):
    """Verify output structure is correct."""
    from separate_v20 import build_preview_response

    result_bytes, manifest = build_preview_response(
        image_bytes=test_image, plates=4, dust=50,
        upscale=True, upscale_scale=2,
    )

    assert manifest["version"] == "v20"
    assert len(manifest["plates"]) == 4
    for plate in manifest["plates"]:
        assert "name" in plate
        assert "color" in plate
        assert len(plate["color"]) == 3

    img = Image.open(io.BytesIO(result_bytes))
    assert img.format == "PNG"
    assert img.size[0] > 0
    print(f"\n  Output: {img.size[0]}x{img.size[1]}, {manifest['num_plates']} plates")
