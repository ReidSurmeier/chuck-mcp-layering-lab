# chuck-mcp v3 Construction Research Index

Created: 2026-05-16
Swarm: `swarm-1778978903817-bqgh16` (hierarchical, 6 specialized construction researchers)
Total artifacts: 6 NOTES.md syntheses + 17 working Python modules + 73 generated test artifacts + working PSD sample + plus reference docs

## Locked v2 design (post-grilling, post-Q26 α-map decision, post-MediaPipe removal)

See `docs/v2-design-locked-2026-05-16.md` (canonical).

## Six construction-phase verdicts

### 1. Frontend clone (`frontend-clone/`)

**Verdict:** in-place fork. The color-separator Next.js source IS the chuck-mcp-layering-lab repo itself. Add `src/app/colorv2/` parallel to `src/app/color-separator/`, share the Next.js deploy, add a second proxy `/api/chuck/*` → chuck-mcp Python backend.

Plate loading animation = 5 coordinated CSS pieces in `globals.css:1114-1188` + `PlatesGrid.tsx` skeleton render with `animationDelay: i * 60ms` stagger. Battle-hardened SSE streaming in `src/lib/api.ts:fetchPreviewStream:624-745`.

Estimate: 32 hours frontend work, compresses to 5 days if Python stubs land by Day 3.

5 artifacts: INVENTORY.md, CLONE_STRATEGY.md, package_diff.md, new_pages_spec.md, NOTES.md.

### 2. claude -p transport (`claude-p-transport/`)

**Verdict:** working `claude_p.py` smoke-tested against live `claude` v2.1.129. End-to-end Python subprocess module ready to wire into chuck-mcp's MCP tool handler.

Verified flag set:
```bash
claude -p \
  --output-format json \
  --json-schema '<schema>' \
  --max-turns 3 \
  --no-session-persistence \
  --permission-mode dontAsk \
  --disallowedTools "Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep" \
  --append-system-prompt 'JSON only, no prose, no fences.' \
  '<user_prompt>'
```

**Critical gotchas:**
- `--bare` BREAKS OAuth subscription (returns "Not logged in" in 67ms)
- `--json-schema` requires `--max-turns ≥ 2`
- June 15 2026: Anthropic splits subscription and Agent SDK credits — chuck-mcp must reassess transport (likely V2 REST migration trigger)

Cost reality: ~$0.50/call × ~180 calls/mo headroom on $100/mo Max plan. p50 latency ~36s.

11 artifacts: 6 docs + 1 reference impl + 4 live JSON samples.

### 3. MediaPipe face-spatial — PARKED (`mediapipe-face-spatial/`)

**Verdict:** PARKED as V2 escape hatch. Full working pipeline shipped:
- 5 Python modules (region_vocabulary, face_region_mapper, merge_regions_with_cells, visualize_regions, extensibility_demo)
- 19 canonical regions + ~55 synonyms + 3 extension patterns
- Chuck Close-aware fallback cascade (Gaussian σ=21 / σ=41 / down512)
- Selfie segmenter mislabel detection + bbox_extend_heuristic for hair
- 1746 SLIC cells assigned across 19 regions on Emma in 140ms

**Replaced in V1 by:** Opus 4.7 vision in the same `claude -p` call that translates intent. Reasoning: Opus handles stylized art natively (eliminating the σ=21 blur cascade), cell-ID-grained precision is sufficient, LLM round-trip already paid, deterministic iteration via `previous_plan.json` caching.

Vocabulary anchors (19 regions, ~55 synonyms) carried over as Opus's target output schema, but the IMPLEMENTATION moves to Opus. If Opus underperforms in production, this pipeline is ready to unshelve.

15 artifacts retained for V2 / escape hatch use.

### 4. Cell-zone renderer (`cell-zone-renderer/`)

**Verdict:** 4 working renderers + types module + 2 test harnesses. Full production-grade rendering subsystem for plates / pulls / proof states / contact sheets.

| Module | Status |
|---|---|
| `cz_types.py` | dataclasses for CellZone, Plate, Pull, ProofState |
| `plate_renderer.py` | mirrored SVG + PNG preview, wood-grain ground, kento marks |
| `pull_renderer.py` | α-map K-M overprint, masked composite, 132 pulls in 20.1s |
| `proof_state_assembler.py` | 4×2 Chuck Close layout + plate-and-pull sheet |
| `contact_sheet_renderer.py` | 7×4 grid replacing v13 failure mode |

**Critical empirical proof:** 27/27 plates pass `plate_not_composite_score` (0.946-1.000). Adversarial v13-style residual scored 0.133 (correctly rejected). Replacement contact sheet at `out/all_blocks_contact_sheet.png` shows isolated jigsaw regions on wood-grain instead of faded full-face residuals.

**Doc typo caught:** original `plate_not_composite_score` threshold direction inverted — fixed in design doc.

Full 132-pull + 27-plate render at 2048×2048 in 29.2s (under 30s V1 budget).

73 artifacts: 8 source files + 3 generated sheets + 54 per-plate files + 8 per-pull files.

### 5. Validators reconstruction (`validators-reconstruction/`)

**Verdict:** 6 working validator modules + master runner + test harness. 15/15 tests pass, master runner 2.4s (< 3s budget).

| Validator | Implementation |
|---|---|
| `plate_not_composite.py` | V1 gate. Smoking gun: v13 blocks 24/25/26 score 0.999-1.000 badness; blocks 01-23 score 0.40-0.46 |
| `role_purity.py` | V2 gate. Modal-role fraction across cells in plate |
| `jigsaw_separation.py` | V3 gate. Min physical distance between zones |
| `proof_progression.py` | V4 gate. Pixel-area shift per consecutive proof state |
| `underlayer_reversal_check.py` | V5 gate. Mirror flip presence on plate / absence on pull |
| `final_match.py` | V6 advisory. ΔE_2000 mean/median/p95 |

Master runner JSON output compatible with web app verification UI.

12 artifacts: 7 Python modules + NOTES.md + 4 result/report files.

### 6. Mokuhanga rule classifier (`mokuhanga-rule-classifier/`)

**Verdict:** 94.4% match against Reid's annotated Emma reference. 8 of 9 EXACT, 1 NEAR, 0 MISS.

Robustness across 3 non-Emma palettes: pale-skin 91.7%, dark-skin 100%, cool-toned-adversarial 87.5%.

Working modules:
- `underlayer_proposer.py` — entry point `propose_underlayers(target, cell_graph, face_landmarks, pigment_library) -> list[UnderlayerPlate]`
- `apply_text_overrides.py` — 6 override kinds + `diff_against_baseline` for UI interpretation panel
- `rule_table.yaml` — 7 pigment families × 12 face regions × 6 global rules, source-attributed (Salter / Pace / Shibata / Vollmer / Lyon)

Integration contract: outputs designed to pass `role_purity_score` + `plate_not_composite_score` by construction (low opacity 0.15-0.30, region-bounded, single role per plate).

Confidence on Chuck Close portraits: HIGH (rule table designed against Pace progressive-proof forensics).

7 artifacts: 2 working modules + 1 rule table + 2 test harnesses + evaluation output + NOTES.md.

## End-to-end stack (consolidated, post-MediaPipe-removal)

```
input: (image_path, intent_prompt: str = "")
  ↓
  if image < 2048px: route to color-separator's RealESRGAN over Tailscale
  ↓
  S3.b: SNIC superpixels → cell graph (deterministic, polygons native)
  ↓
  single claude -p call (Opus 4.7 vision):
    INPUT: image (base64) + cell_graph_overlay_with_ids + intent_prompt
    OUTPUT JSON: {
      intent: {tonal_direction, forbidden_pigment_ids, ...},
      underlayers: [{role, pigment_family, cell_ids: [...]}],
      defaults_applied: [...]
    }
  ↓
  algorithm proposes underlayer baseline (mokuhanga_rule_classifier.propose_underlayers)
  ↓
  apply_text_overrides(baseline, llm_extracted_overrides)
  ↓
  S5: staged 3-batch JAX outer loop
    - hard-sigmoid STE for binary masks (Bengio 2013, BinaryConnect)
    - per-batch L-BFGS-B inner (JAXopt)
    - continuity loss + load-bearing penalty (gradient×mask + counterfactual)
    - shrinking trust region (Worchel 2023)
    - topo-derivative plate spawning (Mehta 2023)
  ↓
  S6.b: jigsaw assignment on SNIC polygons
  S6.c: mill-radius morphology gate (area-opening + opening-by-reconstruction)
        + ΔE guard + horizontal flip
  ↓
  S7: DSATUR + chordality cert + MaxRects face packer → 27 physical blocks
  ↓
  RENDER:
    - plate_renderer.py → 27 mirrored Plate.svg + previews
    - pull_renderer.py → 132 Pull.png cumulative states
    - proof_state_assembler.py → 7-checkpoint proof_state_sheet.png
    - contact_sheet_renderer.py → all_blocks_contact_sheet.png (v13 replacement)
  ↓
  VALIDATE (6 hard gates):
    - plate_not_composite_score (reject if < 0.6)
    - role_purity_score (reject if < 0.7)
    - jigsaw_separation_score (reject if min < 5mm)
    - proof_progression_score (reject if any pair < N pixels shift)
    - underlayer_reversal_check (boolean)
    - final_match_score (advisory only)
  ↓
  SERVE preview UI (colorv2.reidsurmeier.wtf):
    - composite/target side-by-side
    - 27-block grid (primary review)
    - scrubber pull progression
    - sidebar: text prompt + interpretation panel + provenance per underlayer
  ↓
  on user sign-off:
    write ~/cnc-carving-jobs/emma-<date>/
      plates/, jig/, proofs/, pulls/, recipes/, meta/
```

## Open issues for downstream integration

1. **LLM-prompt-translation schema must include cell_id assignments.** The v2 LLM agent specced a schema for intent translation; the v3 MediaPipe removal means that schema needs to grow to include per-region `cell_ids` lists. Vocabulary anchors from `mediapipe-face-spatial/region_vocabulary.py` carry over (19 regions, ~55 synonyms).

2. **Validator implementations vs design-doc formulas.** The `plate_not_composite_score` formula direction was inverted in the original design doc; cell-zone-renderer and validators-reconstruction agents both implemented the design INTENT, not the literal formula. Doc is now corrected. Verify the other 5 validators don't have similar inversions.

3. **June 15 2026 Anthropic subscription/SDK credit split.** chuck-mcp must reassess `claude -p` transport when that lands. V2 REST migration likely.

4. **MediaPipe pipeline is shelf-ready.** If Opus 4.7 vision underperforms on production data, unshelving is a 2-day swap. Both vocabularies are aligned.

## Audit brief

For external LLM review of this design, see `docs/audit-brief-for-external-llm-2026-05-16.md` — 7 specific stress-test questions with response template.
