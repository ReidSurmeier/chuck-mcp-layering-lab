# Architecture Review — Deepening opportunities

Date: 2026-05-17

This review uses the repo glossary in `CONTEXT.md` and the current decisions in
`docs/adr/`. It treats V1 acceptance as a visually plausible mokuhanga
**Block**/**Impression**/**proof** plan, with final-match ΔE as telemetry.

## 1. Validator-plan module

**Files**

- `research/v5-overnight/loop-runner/build_validator_plan.py`
- `research/v3-construction/validators-reconstruction/run_all_validators.py`
- `research/v5-overnight/alpha-proof-dumper/dumper.py`

**Problem**

The validator seam is currently a loose JSON dictionary. Callers must know the
difference between **Review preview** and **Validator truth**, how pull-order
alpha files are named, and which validators accept paths versus arrays. That
made it easy to feed `plate_preview` into a geometry gate and report a false
failure.

**Solution**

Create a small typed validator-plan module that owns:

- mapping solved **Impressions** to authoritative **Mask** paths;
- loading JSON-friendly mask paths into arrays;
- exposing one `build_validator_plan(job_dir, artifacts_dir, input_image)` style
  interface.

**Benefits**

Locality improves because naming, polarity, and path rules live in one module.
Tests can target the public validator-plan interface instead of reaching into
three scripts.

## 2. Block/Impression language seam

**Files**

- `chuck_mcp_v2/types.py`
- `research/v4-build/hybrid-optimizer/*`
- `research/v5-overnight/*`
- `backend/mcp/*`

**Problem**

The canonical glossary says **Block**, **Impression**, and **Mask**, but much of
the research code still exposes `Plate` names. This is tolerable internally, but
it leaks into docs, reports, and issues where it confuses physical blocks with
print applications.

**Solution**

Keep legacy `Plate` compatibility internally, but add an explicit glossary
adapter at repo boundaries:

- public manifests and MCP outputs use Block/Impression/Mask terms;
- research modules can keep `Plate` aliases until rewritten;
- docs and acceptance reports translate legacy names at the edge.

**Benefits**

The interface becomes stable for users and agents, while implementation churn is
kept local.

## 3. Carved-region topology module

**Files**

- `research/v5-overnight/snic-real/snic_proposer.py`
- `research/v4-build/hybrid-optimizer/alternating_loop.py`
- `research/v4-build/hybrid-optimizer/morphology_repair.py`

**Problem**

The current mask topology still reads as dot/cell islands. Even after validator
truth and outer-loop fixes, iter 13 remains at `3/5` gates with jigsaw
separation failures on `14/28` blocks. The current interface hands Stage 3 fixed
cell disks; it does not expose a higher-level carved-region proposal that can
represent irregular connected jigsaw shapes like the reference sheets.

**Solution**

Introduce a carved-region proposal module before Stage 3. Its public interface
should output printable **Mask** candidates grouped by role and pull group:

- contiguous regions from SNIC/cell graph merging;
- minimum-distance and mill-radius constraints before optimization;
- role-aware grouping for support, regional mass, local chroma, and key detail.

**Benefits**

Topology becomes an input constraint instead of a repair afterthought. This
aligns with ADR-0005 while improving the shape of the masks the optimizer sees.

## 4. Acceptance-harness split

**Files**

- `research/v4-build/example-harness/`
- `research/v5-overnight/alpha-proof-dumper/dumper.py`
- `/srv/woodblock-share/chuck-mcp-iterations/current-review/*`

**Problem**

The same artifacts are serving human review, validator input, archival
comparison, and carousel display. That mixes contact-sheet readability with
machine validation requirements.

**Solution**

Split outputs into:

- **Review preview** package: contact sheets, carousel slides, proof sheets.
- **Validator truth** package: masks, metadata, proof state paths, final
  composite, input target.
- **Archive** package: dated iteration manifest with pointers to the above.

**Benefits**

A future reviewer can visually compare method sheets without accidentally using
their pixels as validator data.

## 5. Render-tier seam

**Files**

- `research/v4-build/hybrid-optimizer/jax_continuous_solve.py`
- `backend/services/v23/core/forward_render_jax.py`
- `chuck_mcp_v2/pigment_library_emma.yaml`

**Problem**

Color matching is still largely a Mixbox-style **Mixing** approximation. V1 now
treats ΔE as telemetry, but future reconstruction improvement depends on a
clean seam where empirical **Overprint** or K-M render tiers can replace the
current compositor without rewriting topology and validation.

**Solution**

Extract a render-tier adapter with one small interface used by Stage 3 and
review export:

- `render(stack, substrate, tier) -> proof states + final composite`
- `available_calibration() -> tier metadata`

**Benefits**

The solver can improve color physics without touching **Mask** topology or MCP
tool contracts. This also matches ADR-0002's render-tier vocabulary.
