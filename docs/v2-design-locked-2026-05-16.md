# chuck-mcp v2 — Design Locked (AUDIT-CORRECTED 2026-05-17)

Date: 2026-05-16, **audit-corrected 2026-05-17**
Source: grill-me session (28 questions) + two ruflo research swarms (12 implementation agents) + `docs/reconstruction-plan-2026-05-16.md` + **`docs/audit-response-and-reconstruction-plan-2026-05-17.md`** (overrides 5 grilling locks) + Sharma/Wu/Dalal CIEDE2000 paper (gradient-discontinuity gotcha)

This document is the canonical V1 design contract. Build against this.

## AUDIT OVERRIDES (2026-05-17) — read first

The 2026-05-17 audit run produced a fresh Emma plan that FAILED `plate_not_composite` on 7/12 alpha planes, mean ΔE 12.086, p95 33.026. Validator works; solver doesn't. Five grilling locks are overridden:

| # | Prior lock | Override | Source |
|---|---|---|---|
| 1 | Q25: incremental refactor (keep v13 solver, bolt on continuity) | **Solver must be production-shaped from start**: alternating optimization (cell-graph proposal → graph-cut/ILP plate assignment → JAX continuous solve for opacity/dilution/color only → morphology repair → re-solve). JAX is ONE STAGE, not the architecture. Adaptive 24-30 plates, not fixed 12 + post-hoc expansion. | Audit §Phase 2-4 |
| 2 | Q9 + Q16: edition of 10, calibration deferred | **Incompatible.** V1 ships as single artist's proof. Edition-of-10 promoted to V2 (calibration required). | Audit §7 |
| 3 | Q24: V1.0 in 5 weeks at one-block validation | **Not credible.** Adjusted: V1.0 = credible digital proof sheets (~3 weeks). V1.5 = one physical proof (~6 weeks). V2.0 = edition-capable (~12 weeks). | Audit §6 |
| 4 | MediaPipe removed in favor of Opus 4.7 vision | **RISKY without benchmark.** Anthropic vision docs explicitly warn about spatial reasoning + small-object counting. Keep MediaPipe; benchmark Opus on 10 annotated overlays; require Jaccard/F1 ≥ 0.95 before Opus writes cell IDs; auto-fallback to MediaPipe below threshold. | Audit §1 |
| 5 | Q12: load-bearing via singleton ablation only | **Needs pair ablation + regional specificity.** Singleton misses cancellation pairs and small regional accents. Add pair ablation for high-overlap candidates + regional-specific scoring (named region with contiguous changed area ≥ 0.20% AND mean ΔE_2000 ≥ 2.0 OR p95 ≥ 5.0). | Audit §2 |

## CIEDE2000 IMPLEMENTATION CAVEAT (Sharma/Wu/Dalal 2005)

CIEDE2000 has three mathematical discontinuities that **break gradient-based optimization**. From the paper directly:

> "These discontinuities preclude the use of the formula in analysis based on Taylor series approximations and in design techniques using gradient based optimization."

Worst case: discontinuity up to 0.27 ΔE for samples within 5 ΔE*_ab — concentrated around mean hue 143° (deep blue/violet — Emma's hair/shadow zones).

**Required split for chuck-mcp:**

| Use case | Metric |
|---|---|
| JAX solver inner-loop loss (gradient-required) | **ΔE_76 or smooth ΔE_94** — differentiable everywhere |
| Outer-loop counterfactual ablation (discrete comparison) | ΔE_2000 OK |
| Validator scoring (`final_match_score`, after-the-fact) | ΔE_2000 |
| Per-pull contribution heatmap | ΔE_2000 |

**Implementation correctness gates** (chuck-mcp's CIEDE2000 must pass Table I in Sharma 2005 — all 34 pairs to 4 decimal places):
- Use **atan2** (4-quadrant), not atan, for hue
- Keep **signed** ΔC' and ΔH' (not absolute); cross-term R_T depends on signs
- Mean hue h̄' boundary case at |h'_1 − h'_2| > 180° per Eq. (14)
- Hue-diff Δh' sign at exactly 180° apart per Eq. (10)

## Mission

Take an input image, produce a CNC-ready multi-impression mokuhanga plan that an artist can carve and print as an editioned woodblock print in the Chuck Close / Yasu Shibata / Pace Editions methodology. Output proof preview that matches reference progressive-proof series (Hokusai, Chuck Close).

## V1 Scope (locked)

| Decision | Value | Grill Q |
|---|---|---|
| Image class | Chuck Close-style portraits, front-facing, well-lit | Q2 |
| Gold-standard test | Emma 2002 (`close_emma_2002_2048.jpg`) | Q1, Q2 |
| Physical blocks | 27 | Q1 |
| Regions per block face | 3–7 (binary masks, no gradient within region) | Q1 |
| Pulls per block | 1–5 (multi-pull, overprint depth) | Q3 |
| Total pulls per print | ~132 | Q3 |
| Edition size | 10 | Q9 |
| Output type | Physically printable: CNC blocks + water-based ink + washi | Q4 |
| Pigment library | Reid's physical inventory YAML (uncalibrated V1, calibration-ready arch) | Q5 |
| Calibration | Deferred to V2; ships with t1_mixbox + Mixbox approximations | Q16 |

## Workflow (locked)

1. **Upload** image at `chuck.reidsurmeier.wtf`
2. **Auto-upscale** if < 2048px via color-separator's RealESRGAN service over Tailscale
3. **Free-form text prompt** describing artistic intent, pigment preferences, region overrides, forbidden hues. Prompt is OPTIONAL — empty prompt uses built-in Emma defaults.
4. **LLM (Opus 4.7) interpretation** of prompt → structured constraints (forbidden_pigment_ids mandatory). Interpretation panel shown BEFORE solve.
5. **Algorithm proposes baseline underlayers** (4–9 plates from cell graph + face landmarks + hue rules)
6. **Text overrides applied** to algorithmic baseline (text always wins where specified)
7. **Solver fills blocks 5–27** + pull order + dark/detail
8. **Preview** rendered: composite/target side-by-side + 27-block grid (primary review) + scrubber + sidebar
9. **Verification UI** shown: side-by-side proof series vs Hokusai/Chuck Close reference + per-pull load-bearing heatmap + continuity score
10. **User iterates** by re-prompting (anchored to `previous_plan.json`, only affected regions re-optimize). Soft cap at 5 iterations with diminishing-returns nag.
11. **Sign off** triggers `export_carving_files` MCP tool → writes to `~/cnc-carving-jobs/emma-<date>/`
12. **Reid carves manually** on RISD ShopBot, prints edition of 10 with own notebook tracking
13. **Single-block patch** flow available throughout: edit input + re-invoke with `regenerate_blocks: [N]` → emits only `block_NN_vM.svg`

## UI (locked)

**Form factor:** Next.js web app at `colorv2.reidsurmeier.wtf`, **a copy of `color.reidsurmeier.wtf`** — same visual language, same plate loading animation, same component library, same layout grammar. Public access (no auth), via Cloudflare tunnel. The frontend is literally a fork of the color-separator Next.js project rebranded as v2 and rewired to the new backend.

**Backend:** Python MCP service on Linux server as systemd user unit (`chuck-mcp.service`).

**LLM transport:** subprocess to `claude -p` headless (uses Anthropic Max $100/mo subscription, NOT API key). Set `--output-format json` and parse via Python validator with 3 retries on malformed output. Strict tool-use schema enforcement deferred to V2 (would require API migration). The MCP backend wraps `claude -p` invocations behind a `translate_intent_prompt()` Python function so V2 swap to API is one-file. Verified flag set per `research/v3-construction/claude-p-transport/`: `--output-format json --json-schema <schema> --max-turns 3 --no-session-persistence --permission-mode dontAsk --disallowedTools "Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep"`. **Do NOT use `--bare`** — it breaks OAuth subscription (returns "Not logged in" in 67ms).

**Spatial region assignment (REVISED 2026-05-16):** Opus 4.7 vision in the SAME `claude -p` call that translates intent — NOT a separate MediaPipe pipeline. Input: target image (base64) + SNIC cell graph overlay (cell IDs labeled) + Reid's text prompt. Output JSON includes per-region `cell_ids` lists for each named underlayer slot. Vocabulary anchors (19 canonical regions + ~55 synonyms) carried over from `research/v3-construction/mediapipe-face-spatial/region_vocabulary.py` as the LLM's target output schema, but the implementation is replaced by Opus vision. **MediaPipe pipeline parked as V2 escape hatch** — full working code (region_vocabulary, face_region_mapper, merge_regions_with_cells, Chuck Close σ=21 blur cascade fallback) shipped to `research/v3-construction/mediapipe-face-spatial/` in case Opus vision underperforms on production data. V1 ships without MediaPipe dependency. Reasoning: Opus 4.7 vision handles stylized art natively, eliminating the σ=21 blur cascade Chuck Close requires; cell-ID-grained precision is sufficient (we're not doing sub-pixel landmarks); LLM round-trip cost is already paid for intent translation; deterministic iteration achieved via cached cell assignments in `previous_plan.json`.

**Preview surfaces:**
- Top: target image vs current composite, side-by-side, big
- Middle: 27-block grid (primary review — each cell shows synthesized inked block with jigsaw color regions, labeled with block ID + which pulls use it)
- Bottom: scrubber-driven pull progression (1 → 132)
- Sidebar: text prompt + interpretation panel + provenance per underlayer (algorithm vs text-override + which prompt phrase)
- Action bar: "Adjust" / "Re-prompt" / "Single-block patch" / "Sign off → carve"

**Verification surfaces:**
- Side-by-side: your 7-checkpoint proof states vs Hokusai/Chuck Close reference proof grids
- Per-pull load-bearing heatmap (click any pull in scrubber to inspect)
- Continuity score + 6 validator scores (per reconstruction doc)

## Acceptance Test (locked)

Per Q11: **overlap and continuity from pull 1 to the last pull** — buildup of tones from layer to layer, matching Hokusai/Chuck Close reference progression structure.

Per Q12 + reconstruction doc: **every pull earns its place via load-bearing test** — a pull can be fully covered IFF removing it shifts visible final-pixel color by > threshold ΔE somewhere. Wasted pulls are rejected.

Per reconstruction doc Stage 6, the 6 validators that gate sign-off:

| Validator | Function | Definition |
|---|---|---|
| `plate_not_composite_score` | Penalize blocks that look like final image | `1.0 - (cosine_sim(block, final) + composite_likeness) / 2`. HIGH score = good (jigsaw plate); LOW = bad (residual). **Reject if < 0.6**. Verified by cell-zone-renderer agent: real plates score 0.946-1.000, v13-style residuals score 0.133. |
| `role_purity_score` | Each block has clear print role | Each block tagged with role; reject if cell-zones span > 2 role families per block |
| `jigsaw_separation_score` | Zones on a block need brushable boundaries | Min separation between zones on one block ≥ 5mm physical |
| `proof_progression_score` | Proof states add visible families over time | Each proof checkpoint adds ≥ N pixels of new significant color shift |
| `underlayer_reversal_check` | Plates mirrored; pulls not | Validate horizontal flip on `Plate.svg`; validate no flip on `Pull[k].png` |
| `final_match_score` | Color/structure match after all pulls | ΔE_2000 mean across visible regions, advisory not gating |

V1 ship criterion (Q24): engineering complete + one physically carved block + one proof pull on washi. V1.0 tag cuts there.

## Domain objects (per reconstruction doc Stage 1)

```python
@dataclass
class Plate:
    block_id: int              # 1..27
    cell_zone_ids: list[int]   # which SNIC cells are inked on this plate
    pigment_id: str            # from Reid's YAML library
    opacity: float             # 0..1
    role: Literal["underlayer_light", "local_chroma", "regional_mass", "key_detail"]
    pass_index: int            # which of the 1-5 pulls this block contributes to
    mirror: bool = True        # always True for output SVG

@dataclass
class Pull:
    pull_id: int               # 1..132
    plate: Plate
    order_step: int            # absolute print order
    ink_density: float

@dataclass
class ProofState:
    checkpoint_id: int         # 1..7 (after pulls 4, 8, 12, 16, 20, 24, 132)
    pulls_so_far: list[Pull]
    rendered_image: jnp.ndarray
```

α-map representation retained (per Q26) ONLY inside `render_pull()` as the renderer's internal implementation. NOT user-facing. Never an output artifact.

## MCP tool surface (locked)

```python
plan_emma_print(
    image_path: str,
    intent_prompt: str = "",       # free-form, optional
    iteration_anchor_path: str | None = None,  # previous_plan.json
) -> {
    "plan_id": str,
    "preview_url": str,           # https://chuck.reidsurmeier.wtf/preview/<plan_id>
    "continuity_score": float,
    "validator_scores": dict[str, float],
    "interpretation": dict,       # what LLM read from prompt
    "defaults_applied": list[str],
    "iteration_count": int,
    "nag": str | None,            # diminishing-returns nag after iter 5
}

regenerate_blocks(
    plan_id: str,
    blocks: list[int],
    photoshop_mask: str | None = None,  # V2 escape hatch
) -> {plan_id: str, diff: dict, ...}

export_carving_files(
    plan_id: str,
    output_dir: str = "~/cnc-carving-jobs",
) -> {
    "carving_job_path": str,
    "files": list[str],           # SVG + PDF + MD per below
}
```

## Carving job folder structure (V1 output contract)

```
~/cnc-carving-jobs/emma-2026-MM-DD/
├── plates/
│   ├── block_01.svg              # mirrored, kento-jig-aligned, mill-radius-safe
│   ├── block_01.preview.png      # mirrored preview for review
│   ├── block_02.svg
│   ├── ...
│   └── block_27.svg
├── jig/
│   └── soto_kento_jig.svg        # one-time cut, reused across editions
├── proofs/
│   ├── master_proof.png          # the printed reference, pinned to wall during edition
│   ├── target_vs_final.png
│   ├── proof_state_sheet.png     # 8-up cumulative like Examples/Screenshot 2026-05-14
│   ├── all_blocks_plate_contact_sheet.png  # reversed plates only
│   └── plate_and_pull_contact_sheet.png    # each plate above its cumulative result
├── pulls/
│   ├── pull_001.png              # rendered cumulative state after pull 1
│   ├── ...
│   └── pull_132.png
├── recipes/
│   ├── pigment_mix_recipes.md    # per unique color: grams + ml + shelf life
│   ├── ink_quantity_estimate.csv # per pigment for full edition
│   └── pull_order_schedule.md    # markdown table, print + tick off as you go
├── meta/
│   ├── plan_v01.json             # full plan state, git-trackable
│   ├── interpretation.json       # what LLM read from your prompt
│   ├── iterations.csv            # iter_n, continuity_score, delta, user_input_changes
│   ├── validator_scores.json     # 6 validators, pass/fail per metric
│   └── chuck_mcp_v2_tool_config.yaml  # CNC settings (end-mills, depths, post-processor)
└── previous_plan.json            # symlink → plan_vN.json (latest signed-off)
```

## Build sequence (AUDIT-CORRECTED 2026-05-17 per Phase 0-6)

Replaces prior week-by-week. Solver architecture is production-shaped from start (NOT incremental refactor of v13's compressed-stack solver).

| Phase | Scope | Estimate |
|---|---|---|
| **0** | Freeze 2026-05-17 audit run as failing baseline. Do NOT judge future work by final ΔE alone. | done |
| **1** | Example-grounded acceptance harness from `/srv/woodblock-share/Examples`. Side-by-side contact sheet generator (reference row / current proof row / current block row / alpha row). Visual criteria: early proof density, local color grouping, dark-key timing, background timing, block separability, no full-face residual plates. | Week 1 |
| **2** | Solve production structure directly. Adaptive plate count (24-30 prior for Emma-scale). Multi-pull-per-block as first-class variables. Block/pull identity solved WITH target reconstruction. 4+4+16 is a prior, not a rigid grid. | Weeks 2-3 |
| **3** | Plate organization INTO objective. Loss terms: final image, checkpoint proof, `plate_not_composite` per plate, cell exclusivity/jigsaw, role coverage caps, role-frequency permission (yellow can have detailed carved structure if first/transparent), load-bearing singleton+pair ablation, printability in-loop. **ΔE_76 in solver loss, ΔE_2000 in validators** (CIEDE2000 gradient discontinuity per Sharma 2005). | Week 3 |
| **4** | Hybrid alternating optimization (NOT pure α-maps). Cell-graph proposal → plate assignment via graph-cut/ILP exclusivity → JAX continuous solve for opacity/dilution/color per pull → morphology repair + component scoring → re-solve after repair. JAX optimizes pigment/load only. JAX does NOT invent printable topology from unconstrained alpha. | Weeks 4-5 |
| **5** | Premix colors flexible. Planner outputs target batch color + closest available pigment OR premix recipe with ratios + opacity/dilution/load guidance + measured-swatch fallback notes. Pigment catalog is inventory, not hard palette. | Week 5 |
| **6** | CNC/printability BEFORE SVG (not cleanup). Connected components above minimum area + no hairline islands + no unbrushable adjacent colors on same block + clear jigsaw separations + known registration/mirror state. Reject before vectorization, not after. | Week 6 |
| **bench** | Opus vision benchmark: 10 annotated overlays, require Jaccard/F1 ≥ 0.95 for cell-ID assignment before Opus is allowed to write cell IDs. MediaPipe pipeline stays as automatic fallback below threshold. | parallel Week 1-2 |

**V1.0 ship** = Phase 1-3 complete: credible digital proof sheets, validators gating, no v13-style residuals. NOT physical proof. NOT edition. (~Week 3.)
**V1.5 ship** = Phase 4-6 complete: one physical proof pull on washi. Calibration optional. (~Week 6.)
**V2.0 ship** = edition-of-10 capable. Calibration REQUIRED (DSLR rig + ColorChecker + Finlayson 2015 CCM from V2 calibration research). (~Week 12.)

**Immediate engineering tasks (from audit §Reconstruction Plan):**
1. Replace post-hoc `plan_production_batches` with solver-facing production layout object.
2. Add example-comparison command emitting single audit sheet beside reference examples.
3. Hard gate: every generated physical block must pass `plate_not_composite` before any final ΔE is considered.
4. Pairwise load-bearing ablation for high-overlap pulls.
5. Cell-assignment Jaccard benchmark before trusting Opus vision for cell IDs.
6. Minimal calibration workflow before calling anything edition-ready (V2 only).

**Packaging fix applied 2026-05-17:** added `PyYAML>=6.0` to base deps + `shapely>=2.0` to `io` optional deps in pyproject.toml.

## Deferred to V2 (explicit)

- Calibration (DSLR rig, ColorChecker, Finlayson 2015 CCM, `bootstrap_pigment`, `drift_check` MCP tools) — research complete in `research/v2-implementation/calibration-workflow/`
- t2_empirical LUT (Curtis 1997 white+black inverse K-M)
- t3_spectral renderer (Curtis 1997 multilayer K-M + Saunderson 1942 + Berns 2016 36-wavelength K/S)
- Photoshop integration as user-facing primary (psd-tools 1.17.0 working sample in `research/v2-implementation/photoshop-integration/`, parked as V2 escape hatch for users who want pixel-precise underlayer masks)
- Edition session tracking UI
- Auto-CNC dispatch to ShopBot
- Multi-image testing beyond Chuck Close portraits
- Multi-project gallery view
- Auth / private mode

## Research foundation

All decisions are backed by published research in:
- `research/papers/INDEX.md` — first swarm, 71 artifacts, 6 domains
- `research/v2-implementation/INDEX.md` — second swarm, 54 artifacts, 6 domains

Key citations:
- **Reconstruction**: `docs/reconstruction-plan-2026-05-16.md` — load-bearing diagnostic of v13 representation error, the 6-validator framework
- **STE binarization**: Bengio 2013 (`arxiv_1308_3432`), BinaryConnect 2015 (`arxiv_1511_00363`)
- **Continuity / load-bearing**: Optimize & Reduce 2024 (`arxiv_2312_11334`), SLA stroke attribution 2024 (`arxiv_2403_09480`), Birth of a Painting 2025 (`arxiv_2511_13191`)
- **LLM prompt translation**: ColorGPT 2025 (`arxiv_2508_08987`), NL4Opt
- **SNIC superpixels**: Achanta & Süsstrunk 2017 (web)
- **Mokuhanga methodology**: Sultan/Shiff "Chuck Close Prints: Process & Collaboration" 2003, Salter "Japanese Woodblock Printing" 2002
- **CNC mokuhanga**: Mike Lyon wedged-jig method (2017), Mike Lyon "Post-Digital Printmaking" (2012)
- **K-M overprint physics**: Curtis 1997, Saunderson 1942, Zeller 2026 (`arxiv_2603_09139`)
- **Vectorization**: Selinger 2003 Potrace, Vincent 1993 area-opening + opening-by-reconstruction
- **Graph coloring**: Yekezare et al. 2024 (DSATUR optimality on chordal graphs)

## What v13 got wrong (locked diagnosis from reconstruction doc + contact-sheet review)

- **Block outputs are residual α-map dumps, not plates.** The 26-block contact sheet shows ghosted full-face copies at varying opacities. Blocks 24-26 are essentially finished composites. Real plates carry isolated jigsaw regions on wood-grain ground, mirrored.
- **No separation between plate and proof.** v13 produces one image per impression and labels it "block_NN.png" — but the image is the cumulative pull state, not the inked carved block. Two surfaces conflated.
- **No role assignment.** Underlayers, mid-builds, and key-details are all in the same α-map representation. No print role.
- **No reversal.** Block outputs aren't mirrored. They'd print backwards.
- **No jigsaw grouping.** Cells aren't assigned to physical blocks pre-solve; cells are derived from α-map at output time. Spatial coherence within a plate is accidental.
- **No load-bearing test.** Phthalo blue at pull 27 produces 542 components and ivory black at pull 27 produces 690 components because the solver is mopping residual error. No principled penalty for non-contributing pulls.
- **The dE 4.98 score is misleading.** It measures cumulative-render quality (good), not plate-output quality (the actual product). Future runs gated by the 6-validators won't suffer this illusion.

## Open architectural questions (V1 punt list)

These can wait — they're not load-bearing for V1.0:
- Pigment YAML schema details (will emerge from week-3 LLM-translation work)
- Continuity score gating threshold (will be set after first 5 runs)
- Provenance display granularity in web app (V2 cleanup)
- Validator weighting if multiple fail
- Concurrent edition tracking
