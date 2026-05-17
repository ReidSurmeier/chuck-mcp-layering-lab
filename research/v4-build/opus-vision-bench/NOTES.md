# opus-vision-bench — NOTES

Phase BENCH deliverable for chuck-mcp v4-build swarm (`v4-build/opus-vision-bench`).

Audit override #4 (`docs/v2-design-locked-2026-05-16.md` §AUDIT OVERRIDES) requires
Opus 4.7 vision to clear Jaccard/F1 ≥ 0.95 on 10 annotated SNIC overlays
before it is allowed to write cell IDs. MediaPipe is the automatic
fallback below threshold.

This deliverable provides:

- a **real** ground-truth generator (10 overlays from corpus + synthesized
  Chuck-Close-style faces),
- a **real** Opus 4.7 vision extractor (calls `claude -p` via the
  v3-construction transport — billed against Reid's Max plan),
- a Jaccard/F1 evaluator,
- a fallback router that produces the GO/NO-GO decision,
- a runner that executed all 10 overlays end-to-end and produced the
  numbers below.

## File map + line counts

```
/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v4-build/opus-vision-bench/
  __init__.py                        1 LOC
  fallback_router.py               105 LOC   — global + per-region routing decision
  generate_ground_truth.py         384 LOC   — 10 overlays (5 corpus + 5 synth)
  jaccard_evaluator.py             237 LOC   — RegionScore, ImageScore, BenchResult
  opus_cell_id_extractor.py        303 LOC   — claude -p invocation, monkeypatch for Read tool
  run_benchmark.py                 286 LOC   — end-to-end runner; writes JSON + MD
  test_opus_vision_bench.py        248 LOC   — 7 required tests
  bench_output/                              — produced by run_benchmark.py
    bench_results.json                       — raw scores per region per image
    bench_report.md                          — human-readable summary
    ground_truth/<image_id>/                 — per-overlay artifacts
      source_image.png
      input_overlay.png + input_overlay_small.png
      snic_labels.npy
      ground_truth_regions.json
```

Total project LOC: 1564 (1316 in modules, 248 in tests).

## pytest output

```
7 passed in 110.14s

Name                              Stmts  Miss  Cover
fallback_router.py                   33     0   100%
generate_ground_truth.py            165    16    90%
jaccard_evaluator.py                105     8    92%
opus_cell_id_extractor.py           103    43    58%
run_benchmark.py                    132    30    77%
test_opus_vision_bench.py           106     0   100%
__init__.py                           0     0   100%
---- in-package total: 644 stmts / 97 misses = 84.9% ----
```

(The 63% global figure pytest prints is dragged down by the v3-construction
transport + MediaPipe modules that are imported into the sys.path; those
modules already have their own coverage in their own repo and are not in
scope for this benchmark.)

The 7 required tests, all passing:

```
test_ground_truth_generator_produces_10_overlays            PASSED
test_opus_extractor_returns_dict_with_19_regions            PASSED
test_jaccard_evaluator_perfect_match_returns_1              PASSED
test_jaccard_evaluator_disjoint_returns_0                   PASSED
test_fallback_router_routes_opus_above_threshold            PASSED
test_fallback_router_routes_mediapipe_below_threshold       PASSED
test_end_to_end_benchmark_runs_without_error                PASSED
```

## Headline numbers (real, not mocked)

```
images scored      : 9 / 10
images skipped     : 1 (synth_face_00 — claude -p exited non-zero, no stderr)
overall mean Jaccard : 0.027
overall median       : 0.000
overall min          : 0.000
overall mean F1      : 0.043
cost actually spent  : $5.5076 over 9 Opus calls
latency (ms)         : median 52372  min 45862  max 78104
```

## Per-image scores

| image_id | mean Jaccard | min | F1 | cost USD | duration ms |
|---|---|---|---|---|---|
| close_emma_2002      | 0.041 | 0.000 | 0.069 | 0.7113 | 52372 |
| reid_mike_portrait   | 0.000 | 0.000 | 0.000 | 0.5596 | 49433 |
| reid_untitled_01     | 0.106 | 0.000 | 0.166 | 0.7721 | 78104 |
| reid_untitled_02     | 0.000 | 0.000 | 0.000 | 0.5661 | 45862 |
| toy_print_face_masks | 0.000 | 0.000 | 0.000 | 0.5748 | 50509 |
| synth_face_00        | SKIP — `claude -p exited 1; stderr=''` |||||
| synth_face_01        | 0.000 | 0.000 | 0.000 | 0.5615 | 48154 |
| synth_face_02        | 0.053 | 0.000 | 0.084 | 0.5482 | 54809 |
| synth_face_03        | 0.000 | 0.000 | 0.000 | 0.6029 | 52954 |
| synth_face_04        | 0.040 | 0.000 | 0.066 | 0.6111 | 52633 |

The best single image, `reid_untitled_01`, illustrates the failure mode: even
with the highest Jaccard in the dataset, fine-grained anatomy (eyes, lips,
chin, right_cheek) drop to 0.000 while only the largest, least-structured
regions (background 0.461, forehead 0.256, face 0.189) retain any signal.

## Per-region scores (aggregated across 9 images)

| region | mean | median | min | max | n |
|---|---|---|---|---|---|
| background    | **0.124** | 0.000 | 0.000 | 0.461 | 9 |
| face          | **0.082** | 0.000 | 0.000 | 0.300 | 9 |
| hair          | **0.079** | 0.000 | 0.000 | 0.341 | 9 |
| forehead      | **0.052** | 0.000 | 0.000 | 0.256 | 9 |
| left_cheek    | 0.030 | 0.000 | 0.000 | 0.111 | 8 |
| nose          | 0.027 | 0.000 | 0.000 | 0.107 | 9 |
| right_temple  | 0.024 | 0.000 | 0.000 | 0.143 | 6 |
| right_cheek   | 0.007 | 0.000 | 0.000 | 0.056 | 8 |
| chin          | 0.000 | 0.000 | 0.000 | 0.000 | 7 |
| left_eye      | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| left_eyebrow  | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| left_jaw      | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| left_temple   | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| lips          | 0.000 | 0.000 | 0.000 | 0.000 | 8 |
| lower_lip     | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| right_eye     | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| right_eyebrow | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| right_jaw     | 0.000 | 0.000 | 0.000 | 0.000 | 6 |
| upper_lip     | 0.000 | 0.000 | 0.000 | 0.000 | 3 |

### Per-region failure analysis

11 of the 19 canonical regions score **0.000** mean Jaccard across every
image in the dataset:

```
chin, left_eye, left_eyebrow, left_jaw, left_temple, lips,
lower_lip, right_eye, right_eyebrow, right_jaw, upper_lip
```

These are exactly the regions Anthropic's vision docs warn about: small
shapes, dense neighbours, where "spatial reasoning and precise counting
are limited, especially for exact layouts or many small objects."
(`docs/audit-response-and-reconstruction-plan-2026-05-17.md` §1.)

The eight regions with any signal at all (max Jaccard 0.461 for background)
are the largest and least-structured: background, face, hair, forehead,
cheeks, nose, temple. Even those never break 0.5 Jaccard, let alone the
0.95 production threshold.

## Routing decision

### Global gate

```
overall mean Jaccard 0.027 < global threshold 0.95
→ global_route = "mediapipe"
→ is_go = False
```

### Per-region gate

Every single region also falls below the per-region floor of 0.85.

```
{region: "mediapipe" for region in [
    background, chin, face, forehead, hair,
    left_cheek, left_eye, left_eyebrow, left_jaw, left_temple,
    lips, lower_lip, nose,
    right_cheek, right_eye, right_eyebrow, right_jaw, right_temple,
    upper_lip,
]}
```

### Stored in ruflo memory

```
namespace: "v4-build"
key:       "opus_vision_routing_decision"
backend:   sql.js + HNSW  (embedding stored, 384 dim)
```

Anyone in the v4 swarm can recall the verdict with:

```python
mcp__ruflo__embeddings_search(query="opus vision cell id routing decision",
                              namespace="v4-build")
```

## Cost actually spent

| metric | value |
|---|---|
| Opus calls completed | 9 |
| Opus calls skipped | 1 (synth_face_00) |
| total cost (subscription accounting) | **$5.5076** |
| per-call median cost | $0.575 |
| per-call min cost | $0.548 |
| per-call max cost | $0.772 |

(All calls land on `claude-opus-4-7[1m]` per the per-call `modelUsage`
field in the audit log at `~/.chuck-mcp/claude-p-calls.log`.)

## Latency

| metric | value (ms) |
|---|---|
| median API duration | 52372 |
| min | 45862 |
| max | 78104 |
| wall-clock for full bench | 643.9s (~11 min, includes ground-truth gen) |

## Reproduction

```
cd /home/reidsurmeier/src/chuck-mcp-layering-lab/research/v4-build/opus-vision-bench
PY=/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/mediapipe-face-spatial/venv/bin/python
$PY -m pytest test_opus_vision_bench.py -v --cov=. --cov-report=term-missing
$PY run_benchmark.py --out bench_output --timeout-s 300 --retries 1
#                              ^ uses cached ground truth if present
$PY run_benchmark.py --out fresh_output --force-gt
#                              ^ regenerates ground truth
$PY run_benchmark.py --out smoke --dry-run
#                              ^ structural smoke test, no Opus calls
```

## Verdict

**Ship MediaPipe. Opus 4.7 vision is not production-ready for cell-ID
assignment.**

Opus scored 0.027 mean Jaccard against a 0.95 production gate — 35× below
threshold — with 11 of 19 anatomical regions scoring exactly 0.0 across
all nine successful overlays. Even the largest, simplest regions
(background, face, hair, forehead) cap out around 0.46 in the best image
and average <0.13. The audit's "RISKY as assistive classifier, UNSOUND as
sole geometry authority" diagnosis from `audit-response-and-reconstruction-plan-2026-05-17.md`
§1 is confirmed empirically: keep MediaPipe + SNIC as the cell-ID
authority; Opus may still be useful for semantic intent translation, but
it must not be wired to `previous_plan.json` cell assignments.

## Implementation notes / gotchas captured

1. **`claude -p --allowedTools` is variadic.** Placing it before the
   trailing positional prompt causes the prompt to be swallowed as a
   tool name and the CLI prints
   `Error: Input must be provided either through stdin or as a prompt argument`.
   Insert `--allowedTools Read` immediately after `-p` so it is bounded
   by the next named flag.
2. **Schema with 19 required keys eats max-turns budget.** The default
   `--max-turns 3` (chuck-mcp's intent translator default) hits
   `error_max_turns` on every 19-region call. Bumped to 8.
3. **Overlays must be passed as absolute file paths**, not relative —
   `claude -p` does not run with the caller's cwd. Also pre-downsize
   the overlay to ≤1280 px long edge: full-res PNGs blow past practical
   prompt sizes after base64 encoding.
4. **Skipped image `synth_face_00`** failed with `rc=1` and empty
   stderr after 83s — likely a transient. The harness re-runs once and
   then records the skip, per spec.

## What's in `bench_results.json`

The JSON the runner emits has the structure the integration code can
import directly:

```json
{
  "generated_at": "...",
  "n_images_scored": 9,
  "n_images_skipped": 1,
  "skipped": [{"image_id": "synth_face_00", "error": "..."}],
  "cost_usd_total": 5.5076,
  "latency_summary_ms": {...},
  "extraction_records": [...],
  "bench": {
    "overall_mean_jaccard": 0.027,
    "overall_mean_f1": 0.043,
    "per_region_summary": {...},
    "per_image_detail": [...]
  },
  "routing_decision": {
    "global_route": "mediapipe",
    "is_go": false,
    "per_region_route": {...},
    "reason": "..."
  }
}
```
