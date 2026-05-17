# PSD Format Quirks (for chuck-mcp v2)

Things to know about the PSD file format that affect template / mask design.

## 1. Hard size limits

| Limit | Value | Source |
|---|---|---|
| Max canvas width or height (PSD) | 30 000 px | Adobe spec |
| Max canvas width or height (PSB) | 300 000 px | Adobe spec |
| Max file size (PSD) | 2 GB | Adobe spec |
| Max file size (PSB) | 4 EB | Adobe spec (largely theoretical) |
| Max layer count | unlimited in spec, ~8000 practical | Photoshop UI cap historically 8000 |

**Implication for chuck-mcp v2**: A4@1200 DPI = 9921×14036 px = OK for PSD. A3@1200 DPI = 14043×19842 = OK. For anything denser than 1200 DPI, switch to PSB by appending `.psb` to the save path (`psdimage.save('template.psb')`).

We default to 300 DPI = comfortably under all limits.

## 2. Color modes

PSD supports: Bitmap, Grayscale, Indexed, RGB, CMYK, Multichannel, Duotone, Lab.

**chuck-mcp v2 uses RGB only.** Why:

- psd-tools `create_pixel_layer` accepts PIL images; PIL's RGB mode round-trips cleanly.
- Grayscale would be more compact for binary masks, but Photoshop's painting tools default to RGB. Forcing greyscale would break the user's brush color picker.
- The solver only cares about the binarized mask anyway. RGB → grayscale → threshold is two cheap operations.

## 3. Bit depth

PSD supports 1, 8, 16, 32 bits per channel. **We use 8-bit.**

- 1-bit (bitmap) would be ideal for binary masks but Photoshop disables most tools in 1-bit mode.
- 16-bit / 32-bit waste space for our binary-mask use case.

## 4. RGB vs RGBA in psd-tools

**Trap**: `PSDImage.new(mode='RGB')` does NOT support alpha channels for layers. You can pass PIL RGBA images to `create_pixel_layer` and it converts them to RGB internally, losing transparency.

**Implications**:
- The kento layer is RGB with white background. White will visually overlay everything below it. To make white "transparent" we'd need to set the layer's blend mode to `MULTIPLY` or `DARKEN`. psd-tools partial supports: `layer.blend_mode = BlendMode.MULTIPLY` (in 1.17.0).
- For v1 we use opacity-only and accept that the kento layer dims the source-reference underneath. Acceptable because user toggles kento visibility anyway.

**If transparency required**: switch to mode='RGBA' in `PSDImage.new()`. Pre-1.16, this was buggy. 1.17.0 may have fixes — needs testing.

## 5. Layer placement convention

`create_pixel_layer(image, name, top=0, left=0, opacity=...)`:

- `top, left` are the pixel offset of the layer's top-left from the canvas top-left.
- A 1×1 layer at `(top=0, left=0)` occupies pixel `(0, 0)` only.
- The layer can extend beyond canvas bounds — Photoshop allows "off-canvas" content (e.g. for non-destructive trimming). chuck-mcp v2 clips to canvas during ingestion (defensive).

## 6. Layer name encoding

PSD layer names are stored as UTF-16. psd-tools handles encoding automatically. Layer names CAN contain:

- Spaces, punctuation, emoji (try not).
- Up to ~255 chars practical (theoretical limit higher).

**chuck-mcp v2 convention**: ASCII only, snake_case, `slot_NN_role_subject` for paintable slots. `[REF] foo_DO_NOT_PAINT` prefix for locked references.

The bracket-prefix `[REF]` is a Photoshop community convention for "this layer is a reference; do not edit". It also sorts to the top in some layer panels.

## 7. Layer group hierarchy

PSDs can nest groups. psd-tools `create_group(name)` + `group.append(layer)` works fine. Photoshop sees them as folders in the Layers panel.

**chuck-mcp v2 v1 SKIPS groups** for simplicity. v2 plan: put `[REF]` layers in a locked "Reference" group, paintable slots in a "Slots" group.

```python
ref_group = psd.create_group(name='Reference (locked)')
ref_group.append(source_layer)
ref_group.append(kento_layer)
slots_group = psd.create_group(name='Slots (paint here)')
for layer in slot_layers:
    slots_group.append(layer)
```

Note: psd-tools' `create_group` does NOT mark the group as locked. Locking is a Photoshop UI feature stored in layer flags. To lock programmatically, set `layer.lock = LayerLock.LOCK_ALL` after creation (requires psd-tools ≥ 1.16).

## 8. Composite vs PIL conversion

`layer.composite()` renders the layer applying its own mask, clipping layers, and blend mode. Returns a PIL Image at the layer's bounding box, NOT the canvas size.

`layer.topil()` returns raw pixel data without compositing. Same bounding box.

`np.array(layer.composite())` for ingestion. We use `composite` because user may have set blend modes or applied layer effects.

**Empty layers**: `composite()` returns `None` if the layer has zero non-default pixels. We treat None == "user didn't paint anything here".

## 9. Photoshop "save compatibility" mode

When Photoshop saves a PSD, it can optionally save a flattened composite preview alongside the layer data ("Maximize compatibility"). Older readers can render the flattened preview without parsing layers.

**Affects chuck-mcp v2**: If a user disables Maximize Compatibility, the file may be 30% smaller but other readers (Lightroom, mobile previews) won't show the composite. psd-tools doesn't care.

Default in Photoshop is "Ask Before Saving" — user should always click "Yes" for our workflow to be safe.

## 10. PSD compression

PSD uses RLE (run-length encoding) per scanline by default. psd-tools writes RLE-compressed output.

**Our 26 MB A4 file** is mostly the full-color source-reference layer. The kento layer compresses to ~50 KB (mostly white). Empty slots are essentially free.

**Reducing file size**:
- Downsample source-reference to 150 DPI (it's just for visual ref, not solver input). Saves ~75% on background layer.
- Convert source-reference to indexed/palette mode before layering. Saves another 30%.
- Save as PSB then compress externally with zip. PSB has slightly better compression in some cases.

For v1 we accept 26 MB. Cloudflare CDN delivers it in seconds.

## 11. Resource block: print marks, image resolution

PSD has a "image resources" block that stores non-pixel metadata: print settings, color profile, EXIF, print flags (including a `registration_marks` boolean).

**chuck-mcp v2 ignores the resource-block registration_marks**. That flag is for Photoshop's print-output dialog (when sending to a physical printer). Our kento marks are pixel-level, not metadata.

But: we DO want to set the resolution metadata so Photoshop displays the canvas at the correct print size.

```python
# psd-tools 1.17.0:
psd.image_resources['hres'] = (dpi, ...)
psd.image_resources['vres'] = (dpi, ...)
```

(Exact API: see `psd_tools.psd.image_resources.Resolution`. We've omitted from `gen_template.py` v1 for simplicity but should add for production.)

## 12. Coordinate origin

Photoshop / PSD: **top-left origin**, Y down, X right.
numpy / PIL: same (top-left, Y down).
SVG / PDF: top-left, Y down (when not inverted).
OpenGL / CAD: usually bottom-left, Y up.

**Watch out** when handing off to the CNC plate generator if it uses CAD conventions. Always document Y direction in any cross-domain output.

## 13. Filename casing

PSD filenames are case-insensitive on macOS / Windows, case-sensitive on Linux. Use lowercase + underscores. Never `template.PSD` and `template.psd` in the same project.

## 14. Photoshop CC version compatibility

PSDs written by psd-tools 1.17.0 should open in Photoshop CS3 and later. The PSD spec hasn't changed substantially since CS2 (2005). Adobe maintains backward compatibility aggressively.

## 15. ICC profile

PSDs can embed an ICC color profile in the resource block. We don't, for v1. sRGB is assumed.

**Risk**: if user's Photoshop is set to Adobe RGB (1998) or ProPhoto RGB, colors will display shifted. This affects the source-reference brightness but NOT the binary masks (which are 0/255 regardless of color space).

For mokuhanga where Reid is calibrating spectral pigments — eventually we'll want to embed sRGB explicitly. v2 task.

## 16. Layer ID vs name

PSD layers have a numeric ID (unique within the document) AND a name (UTF-16 string, not unique). We address layers by NAME (because users see names in Photoshop). If two layers had the same name they'd collide — gen_template enforces unique names.

## Sources

- [Adobe Photoshop File Format Specification (PDF)](https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/)
- [psd-tools API docs](https://psd-tools.readthedocs.io/en/latest/reference/psd_tools.html)
- [fileformat.info PSD reference](https://www.fileformat.info/format/psd/egff.htm)
- [archiveteam.org PSD entry](http://fileformats.archiveteam.org/wiki/PSD)
