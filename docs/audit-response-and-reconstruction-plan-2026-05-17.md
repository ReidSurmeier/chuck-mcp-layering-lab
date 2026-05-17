# Chuck MCP Audit Response + Reconstruction Plan - 2026-05-17

This audit is for `chuck-mcp-layering-lab` only. I did not touch
`emma-mokuhanga-mcp`.

## Executive Verdict

The current codebase is running on GPU and the MCP/test surface is real, but the
latest output is not yet the Chuck Close / Yasu Shibata methodology. It is
closer than the very early runs as software, but visually and structurally it
still behaves like a compressed alpha-stack solver with post-hoc production
labels. That is the wrong load-bearing architecture for the reference examples.

The most important finding is this: the production planner proposes 24 plates,
but the solver still optimizes a 12-impression, 3-block study stack first, then
expands it after the fact. That guarantees the plate organization is not truly
inside the objective.

## What I Ran

Environment:

- JAX: `0.10.0`
- Backend: `gpu`
- Device: `CudaDevice(id=0)`
- Note: JAX emits `Could not get kernel mode driver version` warnings in this
  WSL setup, but it still selects the CUDA backend and completes the solve.

Test results:

- v23 stage/direct/unit tests from README subset: `103 passed in 145.53s`.
- MCP transport/scaffold smoke: `10 passed in 3.58s`.
- v3 validators: `15/15 passed`; v13 late blocks 24/25/26 are correctly
  rejected by `plate_not_composite`.
- Underlayer classifier: `8/9 exact`, `1 near`, `0 miss` against the encoded
  Emma annotation after installing `PyYAML`.
- v3 synthetic 27-plate renderer: generated 132-pull proof sheets in
  `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v3-prototype-audit/renderer-full`.

Fresh Emma run:

- Command profile: `thorough`, `m_prior=10`.
- Output folder:
  `/srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main`
- Carousel folder:
  `/srv/woodblock-share/chuck-carousel-slides/2026-05-17_v3-audit-thorough-main`
- Plan id: `plan_1778981663658_66626530`
- Solver wall time: `65.18s`
- Optimized grid: `974 x 789`
- Impressions: `12`
- Blocks: `3`
- Recomputed score from run script: mean dE `12.086`, p95 dE `33.026`
- Printability: `67.86 / 100`

Extra validator pass on the fresh alpha stack:

- `plate_not_composite` rejected 7 of 12 alpha planes.
- Failing planes include broad support/color/key passes: 01, 02, 03, 09, 10,
  11, 12.
- `proof_progression` failed 2 of 11 intervals.
- `final_match` via the v3 validator was still advisory-failing: mean dE
  `7.660`, p95 `20.421`. This differs from the run script because the two
  paths score different masks/downsamples.

## Visual Comparison Against Examples

Reference examples in `/srv/woodblock-share/Examples` show:

- Progressive proofs become recognizable in grouped additions, not in one giant
  final jump.
- Early supports are light and transparent, but they still respect the portrait
  structure and the carved cell vocabulary.
- Individual blocks are grouped by local hue/region decisions. They are not
  full-face residual images, and they are not random isolated specks.
- Overlap is intentional: warm/cool support, red local accents, blue/green
  shadows, and dark key/detail build cumulatively.
- Jigsaw boundaries matter. Each block must be printable as separated brushed
  regions, not just numerically valid pixels.

Current fresh run:

- The cumulative proof sheet is dominated by a green/gold wash by pull 2.
  That is not how the reference first four proofs read.
- The final is too pale through the face, too muddy/blue-green in background
  structure, and has a hard diagonal/banded look.
- The alpha masks still look like solver fields, not carved plates. Several
  masks contain full face/hair/background silhouettes.
- The production pull/block grid is better formatted, but it is a proposal
  only. The file itself reports: "production expansion proposal; does not mutate
  alpha masks."
- The v3 synthetic renderer is useful for renderer mechanics only. It is not a
  proof that the solver understands Emma, because it uses random Voronoi cells
  with synthetic color assignment and no target-image optimization.

v13 comparison:

- v13 final was visually closer to the input, but its late blocks were faded
  full-face composites. The validator correctly catches that.
- Current v3 has better tooling and some better printability checks, but its
  final image and plate construction are not good enough.

## Seven Audit Questions

### 1. Opus 4.7 Vision For Cell-ID Assignment

Verdict: **RISKY as an assistive classifier; UNSOUND as the sole geometry
authority.**

Anthropic's vision documentation confirms Opus 4.7 has higher native image
resolution than earlier models, but also says images above the native long edge
are resized/padded and that coordinate outputs must be rescaled. The same docs
explicitly warn that spatial reasoning and precise counting are limited,
especially for exact layouts or many small objects:
https://platform.claude.com/docs/en/build-with-claude/vision

That directly conflicts with "read a labeled SNIC overlay and return accurate
cell-ID lists for ~1700-2500 cells." The correct architecture is:

- deterministic cell graph and region masks from image processing,
- optional Opus vision for semantic labels and ambiguous art-direction choices,
- hard benchmark before it can write cell IDs.

Minimum benchmark: 10 annotated overlays, 9-12 target regions each, require
region Jaccard/F1 >= 0.95 on cell-ID sets, and automatically fall back to
MediaPipe/SAM/cell-mask geometry below threshold.

MediaPipe should stay available. Google's Face Landmarker docs describe a
model stack that outputs a complete face mesh and 478 3D landmarks:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### 2. Load-Bearing Test Correctness

Verdict: **not yet computationally sufficient.**

A pull cannot be judged only by singleton ablation over the whole final image.
That misses:

- cancellation pairs where pull A only matters when pull B remains,
- hidden underlayers that influence later overprint chroma,
- small but critical regional accents,
- pulls that produce a large global dE but no meaningful local print role.

Concrete gate:

- singleton ablation: fail if no named region has contiguous changed area
  >= 0.20% of artwork area with mean dE2000 >= 2.0 or p95 >= 5.0;
- pair ablation: run top-K pair checks for high-overlap/high-cancellation
  candidates and reject if a pull only passes alone because of another pull's
  artifact;
- regional specificity: report the exact regions/cell IDs where the pull earns
  its place.

Use CIEDE2000 for color-difference math; Sharma/Wu/Dalal's implementation
paper is the right reference for correctness:
https://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf

### 3. Validator Threshold Directions

Verdict: **partially correct, but inconsistent across docs/code/UI specs.**

- `plate_not_composite_score`: behavior is correct in current tests. The code
  implements `badness > 0.6 => reject`; the design-doc form is
  `good_score < 0.6 => reject`. These are equivalent only if everyone uses the
  same naming. The docs still mix "score" and "badness" language.
- `role_purity_score`: direction is correct (`purity >= 0.7` and max 2 role
  families), but it is only meaningful if role labels are independently derived,
  not self-assigned by the same solver that is being validated.
- `jigsaw_separation_score`: direction is correct (`min separation >= 5mm`),
  but the cell-centroid path is too weak. Use mask/component geometry as the
  authority and tune 5mm against brush width/CNC bit constraints.
- `proof_progression_score`: direction is correct, but "visible color shift" is
  too weak. It should compare against expected residual/proof targets, not just
  any dE change.
- `underlayer_reversal_check`: direction is correct for SVG mirrored / pull not
  mirrored, but correlation can be ambiguous. Add kento/registration markers to
  make orientation unambiguous.
- `final_match_score`: correct as advisory. It must not be the main gate,
  because v13 proved a decent dE can still be a bad print plan.

### 4. 3-Batch Staged Solver vs 132-Pull Continuity

Verdict: **current design does not produce Shibata-style continuity.**

The fresh run produced 12 impressions and 3 physical blocks. The production
planner then proposed a 4 + 4 + 16 layout, but explicitly did not mutate the
alpha masks. That means the 24-block methodology is not solved; it is narrated.

The reference method is continuous cumulative construction. The current output
has visible jumps: pale green/yellow dominance, then red, then blue/dark. It is
not 132-pull continuity, and it is not yet a believable 24-block proxy.

### 5. Frontend In-Place Fork Risk

Verdict: **not currently contaminating because `src/app/colorv2/` does not
exist yet; risky when implemented.**

Current tree has `src/app/color-separator/`, `src/app/cnc/`, and shared
`globals.css`/API routes. There is no `src/app/colorv2/` on disk.

In-place is acceptable for a private test route if:

- no global CSS changes are shared without visual regression checks,
- route middleware maps `colorv2.*` separately,
- `/api/chuck/*` is isolated from existing `/api/separate`,
- streaming/progress components are copied then renamed, not edited in place.

For a serious A/B test against another MCP, a separate repo/app remains safer.

### 6. Five-Week Build Estimate

Verdict: **not credible for edition-ready output.**

Five weeks could ship a private research UI and a proof generator. It is not
credible for:

- validated Opus cell-ID replacement,
- production-grade 24-30 block planning,
- 132-pull continuity,
- calibrated pigment/overprint behavior,
- CNC-safe vector export,
- edition-of-10 repeatability.

Adjusted estimate:

- 2-3 weeks: fix solver architecture enough for credible digital proof sheets.
- 4-6 weeks: produce one physical test proof with calibration swatches.
- 8-12+ weeks: edition-capable workflow with validation and carving/export.

### 7. Calibration Deferral vs Edition Of 10

Verdict: **incompatible.**

Uncalibrated V1 can deliver a study/proof, not an edition commitment. Edition
of 10 requires:

- swatch capture under the same camera/light/paper process,
- pigment load and dilution logs,
- per-pull/edition drift checks,
- overprint swatches for important two-layer interactions,
- recipe tolerances.

Without this, the tool can design a plausible print plan but cannot promise
repeatable edition color.

## Reconstruction Plan

### Phase 0 - Freeze Current As Failing Baseline

Keep these outputs as the baseline:

- `/srv/woodblock-share/chuck-clean-outputs/2026-05-17_v3-audit-thorough-main`
- `/srv/woodblock-share/chuck-carousel-slides/2026-05-17_v3-audit-thorough-main`
- `/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v3-prototype-audit/renderer-full`

Do not judge future work by final dE alone. Judge by final match plus plate
construction plus proof progression.

### Phase 1 - Example-Grounded Acceptance Harness

Turn `/srv/woodblock-share/Examples` into a reference harness:

- archive each example with metadata: source, printmaker, proof/block role,
  what it demonstrates;
- define visual criteria: early proof density, local color grouping, dark-key
  timing, background timing, block separability, no full-face residual plates;
- make a side-by-side contact sheet generator:
  reference proof row, current proof row, current block row, alpha row.

Acceptance rule: if a human says "this looks like slop" against the example
sheet, the run fails regardless of dE.

### Phase 2 - Solve Production Structure Directly

Stop solving 12 impressions and expanding afterward. The solver variables must
be production-shaped from the start:

- adaptive plate count, default prior around 24-30 for Emma-scale images;
- batch scaffold: first light support group, second color/depth group, detail
  group;
- multiple pulls per block as first-class variables: opacity, dilution, repeat
  count, and order;
- block/pull identity solved together with target reconstruction.

The 4 + 4 + 16 method is a prior, not a rigid grid. It should flex based on
image complexity and cell graph statistics.

### Phase 3 - Put Plate Organization Into The Objective

Add loss terms that operate before vectorization:

- final image loss;
- checkpoint proof loss against expected cumulative structure;
- `plate_not_composite` penalty for each physical plate;
- cell exclusivity/jigsaw penalty for middle hue shifts;
- global coverage caps by role;
- high-frequency permission by role: early plates may contain carved detail, but
  cannot be full-face residuals;
- load-bearing singleton and pair ablation;
- printability/morphology inside the loop, not as cleanup only.

Important correction: "broad role" does not mean "blurred geometry only."
Yellow can have detailed carved structure. It is first because it is light,
transparent/supportive, and preserves luminance under later colors.

Red should be separate when it is a high-chroma local role that would muddy if
constructed only through overprint mixing.

### Phase 4 - Use Hybrid Optimization, Not Pure Alpha Maps

Use alternating optimization:

1. cell graph / region proposal,
2. plate assignment with graph cut or ILP-style exclusivity,
3. JAX continuous solve for opacity/dilution/color per pull,
4. morphology repair and component scoring,
5. re-solve after repair, not just accept degraded dE.

JAX should optimize continuous pigment/load variables. It should not be asked
to invent printable topology from unconstrained full-resolution alpha maps.

### Phase 5 - Premix Colors Are Flexible

Do not bottleneck on a fixed pigment list. The tool should output:

- target batch color,
- closest available pigment or premix recipe,
- suggested pigment ratios,
- opacity/dilution/load guidance,
- fallback if measured swatch differs.

Pigment catalog is an inventory, not a hard palette.

### Phase 6 - CNC/Printability Before SVG

Vectorization is the last representation, not the cleanup step. A plate should
already be printable before SVG export:

- connected components above minimum area,
- no hairline islands,
- no unbrushable adjacent colors on same block,
- clear jigsaw separations,
- known registration/mirror state.

## Immediate Next Engineering Tasks

1. Replace post-hoc `plan_production_batches` with a solver-facing production
   layout object.
2. Add an example-comparison command that emits a single audit sheet beside the
   reference examples.
3. Add a hard gate: every generated physical block must pass
   `plate_not_composite` before any final dE score is considered.
4. Add pairwise load-bearing ablation for high-overlap pulls.
5. Add `previous_plan.json` cell-assignment benchmark before trusting Opus
   vision for cell IDs.
6. Add a minimal calibration workflow before calling anything edition-ready.

## Packaging Fix Applied

The current declared dependencies were missing modules required by the research
harnesses:

- `PyYAML` for `mokuhanga-rule-classifier/underlayer_proposer.py`
- `shapely` for `cell-zone-renderer/test_renderers.py`

I added `PyYAML>=6.0` to base dependencies and `shapely>=2.0` to the `io`
optional dependency group.

