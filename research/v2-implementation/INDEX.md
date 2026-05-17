# chuck-mcp v2 Implementation Research Index

Created: 2026-05-16
Swarm: `swarm-1778969836247-ys4o7z` (hierarchical, 6 specialized implementation researchers)
Total artifacts: 54 papers/code + 6 NOTES.md syntheses
Phase: design-locked, implementation-research

## Locked v2 design (from grill-me session 2026-05-16)

| Decision | Value |
|---|---|
| V1 scope | Chuck Close-style portraits, Emma as gold-standard test |
| Block count | 27 physical maple plywood blocks |
| Regions per block | 3-7 jigsaw regions, binary masks (no gradient inside region) |
| Pulls per block | 1-5 (multi-pull, overprint depth) |
| Total pulls | ~132 per print |
| Edition size | 10 |
| Output | Physically printable: CNC blocks + water-based ink + washi |
| Pigment library | Reid's physical inventory, YAML/JSON, calibrated swatches |
| Sketch input | Photoshop PSD + up-front text intent prompt |
| Solver authorship | Reid designs 4-9 underlayers as HARD constraints; solver fills blocks 5-27 + pull order + dark/detail |
| Preview UI | Composite vs target (big) + 27-block grid (primary review) + scrubber + sidebar |
| Verification UI | Side-by-side proof series vs Hokusai/Chuck Close reference (C) + per-pull heatmap (B) + continuity score (A) |
| Acceptance test | Overlap and continuity from pull 1 → last pull; load-bearing test per pull |
| MCP semantics | Stateless single-call; iteration = re-invoke with adjusted inputs |
| V1 handoff | Carving files dump to `~/cnc-carving-jobs/`; Reid prints manually |

## Implementation verdicts

### 1. Binary mask enforcement (binary-mask-jax/)
**Hard-sigmoid STE** as primary. Forward: `z = (sigmoid(logit) > 0.5).astype(float32)`. Backward: sigmoid Jacobian via stable STE form. L-BFGS-B compatible. 3-phase annealing: Heaviside warmup → STE binarization → optional Heaviside refinement.
- Read: `arxiv_1308_3432_bengio_ste.md`, `arxiv_2012_02860_heaviside_topology.md`, `arxiv_1511_00363_binaryconnect.md`
- Escalation if STE pathologies: Heaviside continuation β=4,16,64.

### 2. Load-bearing test (load-bearing-test/)
**Hybrid: gradient×mask inner loop + counterfactual ablation outer + SIM-Shapley offline audit.**
- Inner: `jax.grad(loss)(pulls)` projected onto each mask_k. O(1), ~100-400ms for 132 pulls.
- Outer: counterfactual via `jax.vmap`. O(N), ~1-2s.
- Soft+hard hybrid penalty: L1+hinge inside L-BFGS-B; "drop bottom-K by counterfactual" at outer checkpoints.
- Performance: 2-3s total, under 5s V1 budget.
- Read: `arxiv_2312_11334_optimize_and_reduce.md`, `arxiv_2403_09480_sla_stroke_attribution.md`, `arxiv_2511_13191_birth_of_painting_ordering.md`

### 3. LLM prompt → constraints (llm-prompt-translation/)
**Claude Opus 4.7 + single forced tool-call + strict JSON schema + 3-5 input examples + Python validator + max 3 retries.**
- One tool: `translate_artistic_intent` with `tool_choice: {"type": "tool", "name": "..."}` and `strict: true`.
- Mandatory fields force negation handling: `forbidden_pigment_ids`, `forbidden_hue_families`.
- Reject for V1: OPRO-style iterative loops, fine-tuning, multi-turn, vision-grounded verification.
- Read: `NOTES.md` (worked example), `arxiv_2508_08987_colorgpt.md`, `web_nl4opt_competition.md`
- Cost: $100/mo Max plan covers it.

### 4. Photoshop integration (photoshop-integration/)
**psd-tools 1.17.0** (Python, pure, pip-installable, no Photoshop dependency). Tested end-to-end on Linux: A4@300DPI in 11s.
- Shipped: `gen_template.py` (production template generator), `ingest_masks.py` (9-rule validator), `chuck_template_a4.psd` (working sample).
- Validates: non-binary brushes (soft edge), empty slots, kento-overlap painting, canvas resize, missing kento layer.
- Kento convention: 10mm inset, 15mm arms, 0.5mm stroke. Per-paper-size pixel tables.
- Read: `NOTES.md`, `kento_spec.md`, `mask_validation_rules.md`

### 5. Calibration workflow (calibration-workflow/)
**Cross-polarized DSLR + Calibrite ColorChecker Classic + 95+ CRI LED at 45° + Finlayson 2015 root-polynomial CCM.**
- ~$1400 one-time (Sony A6400 / Fuji X-T30 used + ColorChecker + LED panels + polarizing filters).
- Absolute ΔE_2000: mean 2-3, max 5.
- **Inter-print ΔE_2000: mean 1-1.5, max 2** — meets edition-of-10 sub-ΔE-2 requirement.
- Two separate MCP tools: `bootstrap_pigment` (one-time-per-pigment), `drift_check` (periodic).
- Curtis 1997 two-substrate inverse K-M for (K, S) without spectrophotometer.
- Read: `web_capture_protocol_cross_polarized.md`, `web_curtis_inverse_km_two_swatches.md`, `web_pipeline_python_implementation.md` (460 LOC reference)

### 6. CNC + kento (cnc-mokuhanga-carve/)
**Soto-kento jig pattern (Mike Lyon), NOT per-block kento marks.**
- One registration jig (350×470×12mm) with 280.15×400.15mm block pocket + machined kento (kagi 20mm L + hikitsuke 25mm bar, 1.5mm deep, 25° baren-clearance bevel).
- 27 blocks drop into same jig pocket. Block-to-block drift ±0.18mm (sub-mm requirement met).
- Stock: 280×400×12mm hard maple plywood (matches B4 paper). 27 blocks + jig fit on 2 sheets of 4'×8'×12mm.
- Image area: 227×329mm. Carve depth: 2.5mm.
- Tool set: 1/4" compression (rough) + 1/8" up-spiral (bulk) + 1/16" up-spiral (detail, load-bearing min feature 1.6mm) + 30° V-bit. ~$100-150. Buy 2 spare 1/16".
- Pipeline: chuck-mcp SVG → Vectric VCarve Pro → "ShopBot TC (mm) (*.sbp)" post → .sbp on ShopBot. **OpenSBP, NOT G-code.**
- Read: `NOTES.md`, `web_mike_lyon_cnc_mokuhanga_workflow.md`, `chuck_mcp_v2_tool_config.yaml`

## End-to-end stack (one MCP call)

```
input: (emma.jpg, emma_underlayer.psd, intent_prompt: "warm-tonal, vermilion lip, ...")
  ↓
  parse PSD via psd-tools 1.17.0
  validate masks (9-rule check)
  translate intent prompt via Opus 4.7 tool-call → structured constraints
  ↓
  S3.b: SNIC superpixels (from v1 research)
  S4: warm start with user underlayers as HARD constraints
  S5: staged 3-batch JAX outer loop
      - hard-sigmoid STE for binary masks
      - inner L-BFGS-B per batch (JAXopt)
      - load-bearing inner penalty (gradient×mask)
      - counterfactual checkpoint between batches → drop weak pulls
  S6.b: jigsaw assignment on SNIC polygons
  S6.c: mill-sized morphology gate (1.6mm min feature)
        + ΔE guard + horizontal flip
  S7: DSATUR pack + MaxRects face packer → 27 physical blocks
  ↓
  render per-pull cumulative proofs (132 states)
  render per-pull load-bearing heatmaps
  compute continuity score
  ↓
  serve preview UI: composite/target + 27-block grid + scrubber + sidebar
  serve verification UI: side-by-side proof series vs Hokusai/Chuck Close reference + heatmap + score
  ↓
  on sign-off: write to ~/cnc-carving-jobs/emma-2026-MM-DD/
    - 27 block SVG files (kento-aligned to soto-kento jig, mirror-flipped)
    - 1 jig SVG (one-time, reused across editions)
    - master_proof.png
    - swatch_reference_binder.pdf
    - pull_order_schedule.md
    - ink_quantity_estimate.csv
    - pigment_mix_recipes.md
    - chuck_mcp_v2_tool_config.yaml (CNC settings)
```

## Open grilling questions

Still unanswered as of 2026-05-16 (in grilling order):
- Q15: physical maple stock dimensions + ShopBot bed cadence
  (research now confirms 280×400×12mm + soto-kento jig + one-block-per-session over a week)

Pending grilling intent gaps:
- Iteration loop mechanics (re-sketch + re-prompt + re-solve cadence)
- Calibration setup timing (one-time bootstrap before first Emma run?)
- Failure recovery (mis-carved block, mid-edition mishap)
- Text prompt vocabulary anchors (named pigment families, region names)
- Sign-off gate semantics (auto on continuity score, or always manual?)
