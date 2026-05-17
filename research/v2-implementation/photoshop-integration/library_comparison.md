# Library Comparison: PSD generation / parsing in 2026

Decision: **psd-tools 1.17.0**.

Below is the full survey. Run date: 2026-05-16.

## TL;DR matrix

| Library | Reads PSD | Writes PSD | Named layers | Layer masks | Pip install | Active in 2026 | Linux | Photoshop opens output? |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **psd-tools 1.17.0** | YES (full) | YES (basic) | YES | YES (reads, write limited) | YES | YES (May 2026) | YES | YES (verified) |
| pytoshop 1.2.1 | YES | YES | YES | YES | YES | NO (last release 2018) | YES | UNCERTAIN |
| PhotoshopAPI 0.9.1 | YES (full) | YES (full, 5-20x faster) | YES | YES | YES | YES (May 2026) | YES | YES (claimed) |
| photoshop-python-api | N/A | N/A (drives Photoshop) | N/A | N/A | YES | YES | NO (needs Photoshop) | YES (it IS Photoshop) |
| Aspose.PSD | YES | YES | YES | YES | YES (proprietary) | YES (commercial) | YES | YES |
| Adobe Photoshop API (cloud) | YES | YES | YES | YES | N/A (REST) | YES | N/A | YES |
| OpenRaster (.ora) via Pillow | N/A | N/A | YES | YES | YES | YES | YES | NO |
| GIMP XCF (gimpformats) | YES | YES | YES | YES | YES | YES | YES | NO |

## Detailed notes

### psd-tools (CHOSEN)

- **GitHub**: https://github.com/psd-tools/psd-tools
- **PyPI**: https://pypi.org/project/psd-tools/
- **Version tested**: 1.17.0 (released 2026-05-11)
- **License**: MIT
- **Stars**: ~1.6k as of May 2026
- **API**: `PSDImage.new()`, `create_pixel_layer(pil_image, name, top, left, opacity)`, `create_group(name)`, `save(path)`
- **Pros**:
  - 16 years of maintenance. Used in CyberAgentAILab/LayerD (ICCV 2025), QwenLM/Qwen-Image-Layered, multiple production tooling.
  - Pure Python. Zero compilation. Cross-platform.
  - Excellent layer iteration / inspection. Reads almost every PSD spec feature.
  - Writes named pixel layers, opacity, groups. Sufficient for our use case.
- **Cons**:
  - Slow on large canvases (we measured 11s for A4@300DPI, 2480×3508, 11 layers, 26MB output).
  - Limited write support for advanced features: type layers, smart objects, adjustment layers, vector masks, layer effects.
  - Composite rendering may differ from Photoshop for non-trivial files (we use simple pixel layers, so this doesn't bite us).
- **Verdict**: PRIMARY CHOICE. Mature, sufficient, lowest risk.

### pytoshop

- **GitHub**: https://github.com/mdboom/pytoshop
- **Last release**: 1.2.1 (2018-11)
- **License**: BSD-3
- **API**: Lower-level. Construct `LayerRecord`, `ChannelImageData`, manually link. More verbose.
- **Pros**: Was once the "writes-PSD" library. Some users still recommend it for write-heavy workflows.
- **Cons**: STALE. Open issue #29 ("Trying to create new psd from scratch") still unresolved 8 years later. Cannot rely on bug fixes.
- **Verdict**: REJECTED. Stale.

### PhotoshopAPI (EmilDohne)

- **GitHub**: https://github.com/EmilDohne/PhotoshopAPI
- **PyPI**: `PhotoshopAPI`
- **Version**: 0.9.1 (released 2026-05-01)
- **License**: MIT
- **Stars**: 339
- **API**: C++20 core with pybind11 bindings. `psapi.LayeredFile_8bit()`, `add_layer(ImageLayer_8bit(name=..., ...))`, `write(path)`.
- **Pros**:
  - Claimed 5-10x faster reads, 20x faster writes vs Photoshop. ~30s for A4@300DPI generation, projected to be sub-second.
  - 20-50% smaller output files due to better compression.
  - First-class layer editing.
  - Supports all bit depths (8/16/32), all color modes Photoshop supports.
- **Cons**:
  - "Still in early development" per README. Pre-1.0.
  - Smaller user base. Less battle-tested.
  - C++ build dependency. If wheels don't exist for our Python + arch combo, pip install fails.
- **Verdict**: SECONDARY CHOICE. Adopt when psd-tools file-size or generation-time becomes a bottleneck. Mid-2026 v1.0 release may be when to switch.

### photoshop-python-api

- **GitHub**: https://github.com/loonghao/photoshop-python-api
- **PyPI**: `photoshop-python-api`
- **License**: MIT
- **API**: Drives Photoshop via COM (Windows) or AppleEvents (macOS). Wraps ExtendScript DOM.
- **Pros**: Full Photoshop fidelity (it IS Photoshop). Type layers, layer effects, smart objects all just work.
- **Cons**:
  - Requires Photoshop installed on the same machine.
  - Linux-incompatible (Photoshop has no Linux build).
  - Slow: every action is an IPC round-trip to Photoshop's scripting engine.
  - Not reproducible — Photoshop version differences cause subtle output differences.
- **Verdict**: REJECTED. Reid's chuck-mcp server runs on Linux. Can't run Photoshop. Even if we could, IPC overhead is prohibitive.

### Aspose.PSD

- **PyPI**: `aspose-psd`
- **License**: Commercial (free trial)
- **Pros**: Industrial-grade, supports every PSD feature, including writing vector masks and layer effects.
- **Cons**: License fee. Per-developer + per-deployment cost. Overkill.
- **Verdict**: REJECTED. Cost not justified.

### Adobe Photoshop API (cloud)

- **URL**: https://developer.adobe.com/photoshop/
- **API**: REST + manifest JSON. Upload PSD, send JSON of layer edits, download result.
- **Pros**: Authoritative Adobe-controlled output. Always matches Photoshop opens.
- **Cons**:
  - Cloud roundtrip. Requires upload + download of large PSDs.
  - Adobe Creative Cloud subscription required.
  - Network dependency.
- **Verdict**: REJECTED. Network latency makes interactive iteration painful. Cost adds up.

### OpenRaster (.ora) via Pillow plugin

- **Format spec**: https://www.openraster.org/baseline/
- **Python lib**: `pillow-ora` or write manually (it's just a ZIP).
- **Pros**: Open standard. Just a ZIP with `stack.xml` + PNGs. Trivial to read/write/diff.
- **Cons**: **Photoshop has not implemented .ora import since 2021 feature request.** Krita / GIMP / MyPaint only.
- **Verdict**: REJECTED. Photoshop is a hard requirement.

### GIMP XCF (gimpformats)

- **PyPI**: `gimpformats`
- **Pros**: Pure-Python XCF parser. Layered. Native to Linux ecosystem.
- **Cons**: **Photoshop does not import XCF.** User would need GIMP, not Photoshop.
- **Verdict**: REJECTED. Reid uses Photoshop.

## Decision tree

```
User has Photoshop on Mac/Windows?  YES → PSD
                                    NO  → reject project (Reid said Photoshop)

Need cross-platform Python write?   YES → psd-tools OR PhotoshopAPI
                                    NO  → photoshop-python-api (drive Photoshop directly)

PhotoshopAPI v1.0 released yet?     YES → consider switching if faster needed
                                    NO  → use psd-tools

Sufficient for current scale?       YES → psd-tools 1.17.0  ← WE ARE HERE
                                    NO  → eval PhotoshopAPI
```

## Future re-evaluation triggers

Switch from psd-tools to PhotoshopAPI when ANY of these:

1. PSD generation time exceeds 30 seconds on A3@600DPI canvases.
2. PSD file size exceeds 200 MB (psd-tools doesn't compress as well).
3. PhotoshopAPI hits 1.0.0 (signals stability).
4. We need vector masks / type layers / layer effects in the template.
5. We need to programmatically toggle blend modes for the kento layer (PhotoshopAPI handles this; psd-tools has partial support).

Sources:
- [psd-tools docs](https://psd-tools.readthedocs.io/en/latest/usage.html)
- [pytoshop issue #29](https://github.com/mdboom/pytoshop/issues/29)
- [PhotoshopAPI README](https://github.com/EmilDohne/PhotoshopAPI)
- [photoshop-python-api](https://photoshop-python-api.readthedocs.io/en/master/)
- [OpenRaster baseline spec](https://www.openraster.org/baseline/)
- [Adobe feature request: ORA import](https://community.adobe.com/t5/photoshop/feature-request-for-photoshop-and-or-character-animator-import-openraster-ora-files/td-p/11760913)
- [gimpformats PyPI](https://pypi.org/project/gimpformats/)
- [Adobe Photoshop File Formats Specification](https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/)
