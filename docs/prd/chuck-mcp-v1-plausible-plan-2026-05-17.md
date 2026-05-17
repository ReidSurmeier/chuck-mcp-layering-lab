# PRD — Chuck MCP V1 plausible mokuhanga plan

## Problem Statement

The current Chuck MCP research build can run end-to-end and produce review
sheets, but the output is still not reliable enough as a printmaking plan. Past
iterations mixed human previews with validator truth, replayed the same outer
solve, and produced masks that score better as sparse objects than they read as
carvable Chuck Close-style jigsaw blocks.

The user needs a second MCP experiment, separate from `emma-mokuhanga-mcp`, that
can be tested against another approach. V1 should prioritize a visually
plausible mokuhanga block/proof plan over a hard quantitative reconstruction
claim.

## Solution

Chuck MCP V1 will produce a mask-backed, reviewable, physically plausible
mokuhanga plan:

- an ordered stack of Blocks, Impressions, Masks, Pigments, proof states, and a
  print recipe;
- contact sheets and proof previews that make the incremental methodology easy
  to inspect;
- validators that score authoritative masks and proof states, not review
  imagery;
- topology generation that creates connected carved regions rather than dot-cell
  artifacts;
- quantitative final-match metrics reported as telemetry rather than the sole
  acceptance gate.

## User Stories

1. As Reid, I want Chuck MCP to stay separate from `emma-mokuhanga-mcp`, so that
   I can compare two independent approaches.
2. As Reid, I want one input image to produce a full block/proof plan, so that I
   can test whether the method is printmaking-plausible.
3. As Reid, I want early proofs to build like Chuck Close incremental proof
   sheets, so that I can visually audit the methodology.
4. As Reid, I want Blocks and Impressions grouped into readable batches, so that
   I can understand how support, color mass, chroma, and key/detail develop.
5. As Reid, I want each geometry validator to score masks, so that review
   contact sheets cannot create false failures or passes.
6. As Reid, I want final-match dE to be visible but not the only V1 gate, so
   that the solver does not optimize itself into unprintable pixel masks.
7. As Reid, I want jigsaw regions to be connected and separated enough for
   carving, so that the output could become CNC plates later.
8. As Reid, I want underprints to be designed support structures, so that the
   plan reflects mokuhanga reasoning without claiming recovered artist process.
9. As Reid, I want the artifact folders to be organized by dated iteration, so
   that old and current runs are easy to compare.
10. As Reid, I want a clean README and issue set, so that another agent or human
    can continue without reading the whole chat.
11. As a printmaker, I want pigment batches described as adaptable recipes, so
    that I can mix available pigments instead of being locked to a fixed list.
12. As a developer, I want the outer loop to carry solved state forward, so that
    multiple iterations are real optimization rather than replay.
13. As a developer, I want validator-plan construction in one module, so that
    preview paths, mask paths, and proof paths do not drift.
14. As a developer, I want regression tests at public seams, so that validator
    and solver bugs do not silently return.
15. As a reviewer, I want proof-state sheets and full block sheets saved next to
    the final target, so that visual review is fast.

## Implementation Decisions

- V1 acceptance means visually plausible mokuhanga plan output, not a hard claim
  that the input image has been quantitatively solved.
- Physical-plan validators are acceptance gates. Final-match dE is telemetry
  until empirical overprint calibration is available.
- Validator truth must come from masks and proof states. Review previews are not
  geometry data.
- The outer optimization loop must warm-start from previous repaired masks and
  solved continuous variables.
- The next major algorithmic seam is carved-region topology: generate connected,
  separable Masks before continuous pigment/load solving.
- The pigment list is flexible guidance. Pigment batches may be adaptive recipes
  rather than enforced catalog entries.
- Public terminology should use Block, Impression, Mask, Pigment, Order,
  Underprint, Review preview, and Validator truth.

## Testing Decisions

- Tests should verify behavior through public seams: validator plans, optimizer
  output histories, artifact packages, and MCP tool responses.
- Validator tests must include binary mask paths loaded through JSON plans.
- Optimizer tests must verify that outer iterations carry previous state
  forward.
- Artifact tests should verify that review previews and validator truth are
  saved separately.
- Topology tests should score connectedness, minimum separation, and proof
  readability rather than only final pixel match.

## Out of Scope

- Recovering the artist's actual block sequence from the image.
- Touching `emma-mokuhanga-mcp`.
- Treating dE < 8 as the V1-only hard gate.
- Edition-ready color without swatch calibration.
- Full CNC SVG shipping until carved-region topology passes physical gates.

## Further Notes

The corrected v5 iter 13 run reached 3/5 gates with dE around 18.8. The next
work should improve physical topology and proof methodology before deeper color
physics work.
