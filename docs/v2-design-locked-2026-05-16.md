# chuck-mcp v2 — Design Locked

Date: 2026-05-16
Source: grill-me session (28 questions, all answered) + two ruflo research swarms (12 implementation agents) + `docs/reconstruction-plan-2026-05-16.md`

This document is the canonical V1 design contract. Build against this.

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

**LLM transport:** subprocess to `claude -p` headless (uses Anthropic Max $100/mo subscription, NOT API key). Set `--output-format json` and parse via Python validator with 3 retries on malformed output. Strict tool-use schema enforcement deferred to V2 (would require API migration). The MCP backend wraps `claude -p` invocations behind a `translate_intent_prompt()` Python function so V2 swap to API is one-file.

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
| `plate_not_composite_score` | Penalize blocks that look like final image | `1.0 - (cosine_sim(block, final) + coverage_concentration) / 2`. Reject if > 0.6. |
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

## Build sequence (locked per Q25 + reconstruction doc adjustments)

Incremental refactor of existing chuck-mcp-layering-lab v23 pipeline.

**Week 1 — Representation fix + domain objects.**
- Add `Plate`, `Pull`, `ProofState` dataclasses per reconstruction Stage 1
- SNIC drop-in for SLIC (S3.b cell graph)
- STE binarization for mask outputs (hard-sigmoid + Heaviside warmup; per binary-mask-jax research)
- Wire `plate_not_composite_score` validator from day 1 — blocks any v13-style residual-render output
- Mill-sized morphology gate (area-opening + opening-by-reconstruction sized by physical end-mill)
- Horizontal flip on plate SVG export

**Week 2 — Block partition + role assignment + mirrored plate rendering.**
- Cell-graph role assignment pass (algorithmic baseline)
- Graph partition cells into 27 physical blocks (DSATUR + chordality cert + MaxRects face packer per first-swarm research)
- Render `Plate.svg` from cell-zone assignments (mirrored, kento-jig-aligned)
- Wire the other 5 validators with hard gates
- Render the 5 required review sheets per reconstruction Stage 5

**Week 3 — Continuity + LLM + iteration loop.**
- Continuity objective in solver loss (gradient×mask + counterfactual ablation per load-bearing research)
- Soto-kento jig SVG generation
- LLM prompt translation via Opus 4.7 single forced tool-call + strict schema (per llm-prompt-translation research)
- Algorithm proposes underlayer baseline; text overrides apply on top
- Anchored iteration via `previous_plan.json` warm-start
- Single-block patch (`regenerate_blocks` tool)

**Week 4 — Web app + carving export.**
- chuck.reidsurmeier.wtf Next.js frontend (themed like color.reidsurmeier.wtf)
- Cloudflare tunnel ingress
- Preview UI (composite + block grid + scrubber + sidebar + interpretation panel)
- Verification UI (side-by-side vs Hokusai reference + heatmap + score + 6 validators)
- Sign-off + `export_carving_files` MCP tool
- Auto-upscale integration with color-separator over Tailscale

**Week 5 — One-block validation.**
- Reid carves block_01 (lightest, most forgiving)
- Prints one proof pull on washi using soto-kento jig
- Confirms physical reality matches digital plan
- V1.0 tag cut

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
