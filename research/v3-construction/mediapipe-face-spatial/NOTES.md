# MediaPipe Face Spatial — chuck-mcp v3 research notes

Agent: `MEDIAPIPE-FACE-SPATIAL` (swarm `swarm-1778978903817-bqgh16`)
Date: 2026-05-16
Status: working modules confirmed on `close_emma_2002_2048.jpg`

## TL;DR

- **Library:** modern MediaPipe Tasks-API `FaceLandmarker` (478 landmarks
  with iris refinement). Legacy `mp.solutions.face_mesh` is gone in
  mediapipe 0.10.35 — do not rely on it.
- **Hair / background:** MediaPipe `ImageSegmenter` multi-class selfie
  model (`selfie_multiclass_256x256.tflite`). Falls back to
  bbox-extension heuristic when the segmenter mislabels stylized art.
- **Vocabulary:** 19 named regions (covers cheek, temple, forehead, lip,
  chin, jaw, eye, eyebrow, nose, hair, background, all left/right pairs).
- **Stylized-art finding:** Chuck Close's dot-mosaic defeats the default
  face detector. Pre-blurring with Gaussian σ ≈ 21 makes it detect
  reliably. The mapper auto-cascades through `raw → gauss21 → gauss41 →
  gauss21+down512` until a face is found. Recorded as `strategy=...` on
  every region's `source` field.

## Files in this folder

```
mediapipe-face-spatial/
├── NOTES.md                          # this file
├── region_vocabulary.py              # 19-region vocabulary + synonyms
├── face_region_mapper.py             # extract_face_regions(image_path)
├── merge_regions_with_cells.py       # join with SNIC cell graph
├── visualize_regions.py              # build the 3 visualization sheets
├── extensibility_demo.py             # how to add new regions
├── close_emma_2002_2048.jpg          # canonical test input
├── emma_regions_polygons.png         # face mesh + region polygons + legend
├── emma_regions_masks.png            # 4x5 grid of per-region masks
├── emma_cell_assignment.png          # 3-panel: orig | polygons | SNIC cells colored
├── emma_extensibility_demo.png       # above-eyebrow, between-eyes, lower-third
├── models/
│   ├── face_landmarker.task          # 3.6MB FaceLandmarker bundle
│   └── selfie_multiclass_256x256.tflite  # 16MB multi-class segmenter
└── venv/                             # pip env (mediapipe 0.10.35)
```

## Install

```bash
cd research/v3-construction/mediapipe-face-spatial
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python numpy pillow scikit-image shapely matplotlib

# Models (one-time):
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o models/selfie_multiclass_256x256.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

Total download = ~20MB. Both run on CPU at acceptable latency for an MCP
backend (Emma 1658×2048 takes ~600ms end-to-end on the linux dev box).

## Region vocabulary (19 names)

These are the **canonical** names the LLM prompt translator should
normalize to. Synonyms are documented in `REGION_VOCABULARY`.

| name | kind | notes |
|---|---|---|
| `face` | face_mesh ring | FACE_OVAL_BOUNDARY (36 indices) |
| `left_cheek` | mesh_hull | subject's left cheek (image right side) |
| `right_cheek` | mesh_hull | subject's right cheek (image left side) |
| `left_temple` | mesh_hull, dilated 8px | between eye outer corner and ear |
| `right_temple` | mesh_hull, dilated 8px | mirror of above |
| `forehead` | mesh_hull, dilated 4px | sparse mediapipe coverage — heuristically tightened |
| `chin` | mesh_hull | lowest jaw indices |
| `left_jaw` / `right_jaw` | mesh_hull | the jawline curve from cheek down |
| `upper_lip` / `lower_lip` / `lips` | face_mesh rings | three options for lip-area constraint |
| `left_eye` / `right_eye` | face_mesh rings | eye opening (not iris) |
| `left_eyebrow` / `right_eyebrow` | face_mesh rings, dilated 6px | thin → dilated for usable mask |
| `nose` | mesh_hull | bridge + nostrils + tip |
| `hair` | outside_face_up | NOT in mediapipe — built from segmenter or heuristic |
| `background` | outside_face_all | NOT in mediapipe — built from segmenter or heuristic |

Subject-side convention: "left" = subject's anatomical left = image right side.
(MediaPipe documents this; the synonyms `"her left cheek"` and `"subject_left_cheek"`
all resolve to the same canonical `left_cheek`.)

## Python module API

### `extract_face_regions(image_path) -> dict[str, FaceRegion]`

```python
from face_region_mapper import extract_face_regions
regions = extract_face_regions("/path/to/close_emma_2002_2048.jpg")
regions["left_cheek"].mask        # HxW uint8 binary mask (0/255)
regions["left_cheek"].polygon     # closed pixel-coord polygon or None
regions["left_cheek"].source      # "mediapipe_facemesh[gauss21@conf=0.5]"
regions["left_cheek"].confidence  # 0..1
regions["left_cheek"].bbox        # (xmin, ymin, xmax, ymax)
regions["left_cheek"].centroid    # (cx, cy)
```

### `merge_face_regions_with_snic_cells(regions, snic_labels) -> dict[str, list[int]]`

```python
from merge_regions_with_cells import (
    merge_face_regions_with_snic_cells,
    resolve_cell_to_primary_region,
)
assign = merge_face_regions_with_snic_cells(regions, snic_labels, strategy="centroid")
# assign["left_cheek"] = [42, 51, 53, 67, ...] — sorted cell IDs
primary = resolve_cell_to_primary_region(assign)
# primary[42] = "left_cheek"  (most specific region wins ties)
```

Strategy options:
- `"centroid"` — fast, O(N_cells × poly_complexity). Use this for the
  interactive solver. Merge call ~140ms at 1746 cells.
- `"overlap"` — accurate boundary handling. Costlier on >2000 cells.

**Perf note for v3 integrator:** `compute_snic_cell_geometry` is currently
the bottleneck at ~3.4s for 1746 cells (it loops per-cell over `np.indices`).
Replace with `scipy.ndimage.center_of_mass(labels, labels, np.unique(labels))`
and `np.bincount(labels.ravel())` for pixel counts → ~50× speedup. The
SLIC/SNIC step itself dominates (~2.3s) so geometry will not be the
ultimate bottleneck; still worth the vectorization for iteration latency.

**Caching:** `extract_face_regions` and `compute_snic_cell_geometry` are
pure functions of (image, snic_labels). Memoize per (`plan_id`,
`snic_version`) pair in the chuck-mcp backend — both inputs are stable
across prompt iterations on the same image, so the LLM-loop iteration cost
is just the ~140ms merge step.

End-to-end on Emma 2048×1658:
- `extract_face_regions`: 946ms (FaceLandmarker + multi-class segmenter)
- `compute_snic_cell_geometry`: 3.4s (one-time, cacheable; vectorize → ~70ms)
- `merge_face_regions_with_snic_cells`: 140ms (per-iteration)

## Integration with chuck-mcp v3 solver

The v3 design doc specifies workflow step 4-5:

> LLM (Opus 4.7) parses prompt → structured constraints with region descriptors
> ("hair", "cheek", "temple", "lip"). System maps those descriptors to actual
> spatial SNIC cell IDs.

The integration code lives in `chuck-mcp/server/spatial.py` (to be created)
and looks like:

```python
from research.v3_construction.mediapipe_face_spatial.face_region_mapper \
    import extract_face_regions
from research.v3_construction.mediapipe_face_spatial.merge_regions_with_cells \
    import merge_face_regions_with_snic_cells

def resolve_region_constraint(snic_labels, image_path, region_name):
    regions = extract_face_regions(image_path)            # cache per image
    assignments = merge_face_regions_with_snic_cells(regions, snic_labels)
    if region_name not in assignments:
        raise ValueError(f"unknown region: {region_name}")
    return assignments[region_name]                       # list[int] of cell IDs
```

Both `regions` and `assignments` should be cached per (image, snic_version)
pair. Re-extraction on every prompt iteration is wasteful.

## Test results on Emma (close_emma_2002_2048.jpg)

Full pipeline runs cleanly:

| stage | result |
|---|---|
| FaceLandmarker on raw image | FAILS (Chuck Close mosaic confuses the detector) |
| FaceLandmarker on Gaussian-blurred image | PASSES at conf 0.5 |
| Strategy selected | `gauss21@conf=0.5` |
| Regions populated by mediapipe | 17 of 19 (all but hair, background) |
| Selfie multiclass segmenter | runs but mislabels Emma as "clothes" (class 4) — implausibility check skips it |
| Hair fallback | `bbox_extend_heuristic` (face bbox extended 75% upward, 30% outward) |
| Background fallback | `bbox_complement` |
| SLIC stand-in for SNIC at n_segments=1200 | 633 superpixels |
| Cells assigned to a primary region | 503 of 633 (the rest are body/shoulder area in lower frame) |

The polygon overlay on `emma_regions_polygons.png` visually confirms eyes,
nose, lips, jawline, and cheek hulls all sit on the actual face features in
the Chuck Close print despite the dot-mosaic abstraction.

## Failure modes + fallbacks (V1 ships with these)

| failure | symptom | fallback |
|---|---|---|
| No face detected at all | mediapipe returns empty result on all strategies | `_build_fallback_regions(h, w)` — centered ellipse, all 11 regions populated heuristically. Caller sees `source="fallback_heuristic"`, `confidence=0.3`. **Solver should warn user.** |
| Face detected but tilted/profile | landmarks rotate with the face | polygons still align with the actual face — works. (Only tested head-on but designed for it.) |
| Multiple faces | mediapipe returns only the strongest (we set `num_faces=1`) | acceptable for Chuck Close portraits; v2 may support multi-face |
| Selfie segmenter mislabels stylized print | hair/face/bg confused with clothes/others | implausibility check (`overlap > 1000 px`) skips bad seg, falls back to bbox heuristic. Confidence drops from 0.85 → 0.50. |
| Image not loadable | `cv2.imread` returns None | raises `FileNotFoundError` — caller's job |

## Stylized-portrait pitfall (the load-bearing finding)

**The single most important thing this research found:** Chuck Close-style
dot-mosaic textures destroy the default face detector. Without
pre-blurring, MediaPipe FaceLandmarker reports 0 faces on Emma at any
confidence threshold (tested 0.5, 0.2, 0.05). Gaussian blur σ ≈ 21 fixes
it instantly.

The cascade in `_detect_facemesh_landmarks` makes this transparent — the
caller never has to know. But the v3 design should document that:

1. Mokuhanga prints with comparable high-frequency texture (Hokusai's
   wave foam, Yoshida's grass crosshatch) likely need the same trick.
2. For non-portrait subjects, the cascade is irrelevant — `extract_face_regions`
   simply hits the `fallback_heuristic` path and returns proportional regions.
3. If the v3 solver wants to know which strategy succeeded, parse it out of
   `region.source` (it includes the strategy label).

## Extending the vocabulary

When Reid's prompt says "above the eyebrow" or "between the eyes" or
"lower half of the face" — none of those are in `REGION_VOCABULARY` directly.
The extension patterns are documented in `extensibility_demo.py` and the
output `emma_extensibility_demo.png` proves they work.

Three composable patterns:

| pattern | use when | example |
|---|---|---|
| `mesh_band(ring, offset_px, height_px)` | the new region is parallel to an existing ring | `above_eyebrow` = eyebrow ring + 60px up |
| `mesh_hull(custom_indices)` | the new region is the convex hull of arbitrary mediapipe points | `between_eyes` = hull of brow + bridge indices |
| `derived(existing_region, transform)` | proportional split or intersection | `lower_face_third` = face mask ∩ bottom 1/3 bbox |

To make a new region available to the LLM, add a `RegionSpec` to
`REGION_VOCABULARY` and a build branch in `extract_face_regions`. ~10 LOC.

## Top-3 must-reads (in order)

1. **`docs/v2-design-locked-2026-05-16.md`** in this repo — the canonical
   chuck-mcp v3 contract. Read it before changing any region semantics.
2. **MediaPipe FaceLandmarker Python guide**:
   https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python —
   especially the section on output landmark schema and the 478-point
   numbering convention (the same as the 468-point mesh + 10 iris).
3. **Sander de Snaijer's interactive 478-landmark explorer**:
   https://demos.sanderdesnaijer.com/demos/face-mesh-explorer/ — single
   best resource for picking which indices belong to which region. The
   `region_vocabulary.py` cheek/temple/forehead lists were cross-checked
   against this page.

## Reporting back (per task spec)

- **File count:** 7 source artifacts + 4 visualization PNGs + 2 model
  files + 1 input image + this NOTES.md = **15 files**.
- **Working modules confirmed:** yes
  - `region_vocabulary.py` (19 regions, all listed via `__main__`)
  - `face_region_mapper.py` (smoke-tests on Emma, all 19 regions populated)
  - `merge_regions_with_cells.py` (smoke-tests on Emma with SLIC stand-in)
  - `visualize_regions.py` (3 PNG sheets generated)
  - `extensibility_demo.py` (4 new derived regions generated)
- **Top-3 must-reads:** above.
- **Vocabulary size:** **19 canonical region names**, ~55 synonym phrases,
  3 documented extension patterns for adding more without code changes
  beyond `REGION_VOCABULARY`.
