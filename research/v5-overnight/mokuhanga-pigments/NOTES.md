# NOTES.md — Mokuhanga Pigments Adapter (v5-overnight)

Agent: MOKUHANGA-PIGMENTS
Swarm: swarm-1778989256284-xvs2l5
Date: 2026-05-17
Methodology: TDD London School (red → green → commit per cycle)

This agent wires the v3-construction mokuhanga rule classifier (94.4% match
vs Reid's Emma annotation) into `chuck_mcp_v2.plan_emma` so the production
plan's plates leave the planner with REAL, deterministic pigment assignments
from `chuck_mcp_v2/pigment_library_emma.yaml`, not defaults.

## TDD cycle status

| Cycle | Test                                                          | State |
|-------|---------------------------------------------------------------|-------|
| 1     | pigment library loads 15-25 entries / required pigments       | GREEN |
| 2     | underlayer proposer picks yellow for cheek + temple           | GREEN |
| 3     | plan_emma plates have real pigments not default               | GREEN |
| 4     | mid/dark plates pick pigments by ΔE_76 to plate target Lab    | GREEN |
| 5     | Emma underlayer match score ≥ 85% through the adapter         | GREEN |

`pytest -v` → 8 passed.

## Library entry count

`chuck_mcp_v2/pigment_library_emma.yaml` contains **19 entries** across all
**7 mokuhanga families** the rule classifier emits:

- `light_yellow`: light_yellow, yellow_ochre, raw_sienna
- `pale_pink`: pale_pink, rose_lilac_wash
- `pale_orange`: peach_wash, light_orange, pale_orange
- `pale_red`: vermilion, scarlet, alizarin_crimson
- `pale_blue`: pale_blue, mint_cyan_wash, periwinkle_blue, phthalo_blue
- `pale_green`: pale_green_wash
- `warm_grey`: warm_grey, burnt_umber, ivory_black

Every entry carries `uncalibrated_v1: true` + a `source_note` citing the
manufacturer catalog the Lab values were approximated from (Holbein,
Daniel Smith, Schmincke). V2 will replace with measured swatches per the
spec at `chuck_mcp_v2/pigment_library_emma.yaml:meta.calibration_status`.

## Underlayer match score on Emma

Through the adapter on the synthetic Emma color profile (re-using the v3
test_emma_annotation harness):

```
EXACT=8, NEAR=1, MISS=0   Match rate: 94.4%
```

Identical to the v3-only baseline — wiring did not regress the classifier,
and concrete pigment_name picks (per region) are plausible:

| Region     | Annotated   | Predicted   | Pigment picked       |
|------------|-------------|-------------|----------------------|
| cheek      | light_yellow| light_yellow| raw_sienna           |
| forehead   | light_yellow| pale_orange | peach_wash      (NEAR)|
| lip        | pale_red    | pale_red    | alizarin_crimson     |
| chin       | pale_pink   | pale_pink   | pale_pink            |
| temple     | pale_pink   | pale_pink   | pale_pink            |
| hair       | pale_blue   | pale_blue   | phthalo_blue         |
| eye_white  | pale_blue   | pale_blue   | mint_cyan_wash       |
| background | pale_orange | pale_orange | pale_orange          |
| jaw_neck   | warm_grey   | warm_grey   | warm_grey            |

Same single NEAR on forehead as v3 — defensible artist variant.

## Integration committed

`chuck_mcp_v2/plan_emma.py` now imports the v5 mokuhanga-pigments adapter
and the v3 rule classifier (sys.path injection, matching the existing v5
mediapipe-spatial pattern) and calls `apply_mokuhanga_pigments()` after
`build_production_plan()`. Disable with `--no-mokuhanga-pigments` (env
flag is not registered as a CLI option yet — set via
`args.no_mokuhanga_pigments = True` in callers).

End-to-end smoke test:

```bash
python -m chuck_mcp_v2.plan_emma --synthetic --plan-output /tmp/emma_plan_v5.json \
  --output /tmp/emma_hybrid_v5.json --plate-count 20 --size 96 --cells 64 \
  --max-outer-iters 1 --max-inner-iters 2 --no-early-stop
```

Verifies every plate emits a real `pigment_family` AND every pull emits a
real `pigment_id` (no `PY3_holbein_pale` placeholders).

## Adapter architecture

```
                        ┌──────────────────────────────────────┐
   target_image  ──────►│ chuck_mcp_v2.plan_emma.run()         │
   image (or synth)     │                                      │
                        │  build_production_plan() ────────────┼─► production_plan
                        │  apply_face_region_constraints()     │   (PlateSpec objects;
                        │       (image runs only)              │    underlayer plates
                        │  apply_mokuhanga_pigments() ◄────────┼─── carry region_label
                        └──────────────────────────────────────┘    from v5 spatial)
                                          │
                  ┌───────────────────────┴────────────────────────┐
                  │ mokuhanga_emma.apply_mokuhanga_pigments_to_plan│
                  │                                                │
                  │  1) underlayer_light plates                    │
                  │      → run v3 propose_underlayers()            │
                  │      → for each region: ΔE_76 nearest pigment  │
                  │        within rule classifier's family pick    │
                  │  2) mid/dark plates                            │
                  │      → aggregate region Lab from cells         │
                  │      → ΔE_76 nearest pigment in plate's family │
                  │  3) mutate plate.pigment_family + plate.       │
                  │     pigment_name + plate.pigment_lab + every   │
                  │     pull.pigment_id; append rationale.         │
                  │     DOES NOT set plate.pigment_id (that's a    │
                  │     post-JAX field; setting it flips to_dict() │
                  │     to the wrong schema).                      │
                  └────────────────────────────────────────────────┘
```

## Files in this artifact set

| File                                                                                              | Role                                         |
|---------------------------------------------------------------------------------------------------|----------------------------------------------|
| `chuck_mcp_v2/pigment_library_emma.yaml`                                                          | 19-entry concrete Emma pigment inventory     |
| `research/v5-overnight/mokuhanga-pigments/pigment_library.py`                                     | YAML loader + family index + ΔE helpers      |
| `research/v5-overnight/mokuhanga-pigments/mokuhanga_emma.py`                                      | Adapter: v3 rule classifier → real pigments  |
| `research/v5-overnight/mokuhanga-pigments/conftest.py`                                            | sys.path wiring                              |
| `research/v5-overnight/mokuhanga-pigments/test_pigment_library.py`                                | Cycle 1 tests                                |
| `research/v5-overnight/mokuhanga-pigments/test_mokuhanga_emma.py`                                 | Cycles 2 / 4 / 5 tests                       |
| `research/v5-overnight/mokuhanga-pigments/test_plan_emma_integration.py`                          | Cycle 3 tests                                |
| `chuck_mcp_v2/plan_emma.py` (modified)                                                            | sys.path + apply_mokuhanga_pigments() call   |

## Open work for next agent

1. CLI flag `--no-mokuhanga-pigments` to expose the existing
   `args.no_mokuhanga_pigments` opt-out path on the command line.
2. V2 calibration: replace `lab_values` in `pigment_library_emma.yaml`
   with spectrophotometer readings of physical swatches printed on
   Shiramine washi. Set `uncalibrated_v1: false` per entry.
3. Real-face integration: once the v5 mediapipe-spatial agent populates
   real `region_label` on underlayer plates (cheek/temple/etc instead of
   `underlayer_light_cluster_N`), the underlayer adapter will route via
   the rule classifier exactly as it does in the test harness. Today the
   synthetic-image path produces cluster-labels, which the adapter falls
   back to with a degraded "background" landmark.
4. Consider exposing the diagnostic summary dict returned by
   `apply_mokuhanga_pigments()` in the planner's CLI JSON summary.
