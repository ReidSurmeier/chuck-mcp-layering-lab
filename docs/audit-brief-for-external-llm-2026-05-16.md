# Audit Brief — chuck-mcp v2 Design (for external LLM review)

**Date:** 2026-05-16
**Reviewer instructions:** read this file end-to-end, then respond using the AUDIT RESPONSE section at the bottom. You're being asked to stress-test the design before implementation begins. Find logic errors, hidden assumptions, technical debt, and questions the author didn't think to ask. Be adversarial. The author wants to find problems NOW, not in week 4 of building.

---

## What chuck-mcp is

A tool that takes an input image (a portrait, specifically Chuck Close's "Emma" 2002 as the gold-standard test case) and produces a CNC-ready multi-impression mokuhanga (Japanese water-based woodblock) print plan. The artist (Reid Surmeier, RISD student) will carve the resulting 27 woodblocks on RISD's ShopBot and print an edition of 10 on washi by hand.

Reference example: Yasu Shibata printed Emma at Pace Editions using 27 carved blocks, 113 distinct ink colors, and 132 print pulls (multi-pull per block via reduction technique). The system aims to replicate that production methodology — not pixel-level reconstruction, but printmaking-aware plate planning.

**Critical visual reference:** The Examples folder contains the Emma reference + 8 photographed underlayer blocks (multi-color jigsaw regions on each block face). The progressive-proof series style (Hokusai Great Wave + Chuck Close 2015 self-portrait proof) is the acceptance test target.

## Key files for context (read in order)

1. `docs/v2-design-locked-2026-05-16.md` — full design contract (READ FIRST)
2. `docs/reconstruction-plan-2026-05-16.md` — diagnostic of v13's failure mode (alpha-maps masquerading as plates)
3. `CONTEXT.md` — domain glossary (Plate, Pull, Impression, Mask, Pigment, Order, Block, etc.)
4. `research/papers/INDEX.md` — first research swarm (71 artifacts, 6 domains: color science / inverse rendering / segmentation / mokuhanga methodology / vectorization / graph coloring)
5. `research/v2-implementation/INDEX.md` — second swarm (54 artifacts: binary masks STE / load-bearing / LLM prompt / Photoshop / calibration / CNC mokuhanga)
6. `research/v3-construction/` — third swarm (working code: frontend clone strategy / claude-p transport / cell-zone renderers / validators / mokuhanga rule classifier / MediaPipe parked)

## The locked design (28 grilling decisions, summarized)

### Product

- V1 scope: Chuck Close-style portraits only, Emma as test case. If Emma works, generalize later.
- **27 physical blocks** (matches Shibata reference; not a parameter, hardcoded)
- **3–7 jigsaw regions per block face** (binary masks, no gradient inside a region — relief carving is binary)
- **1–5 pulls per block** (multi-pull, overprint depth via re-inking same carved geometry)
- **~132 total pulls per print**, **edition of 10**
- Output: physically printable on RISD ShopBot + water-based ink + washi
- Pigment library: Reid's physical inventory (YAML/JSON), uncalibrated in V1, calibration-ready architecture
- Calibration deferred to V2 (DSLR rig + ColorChecker + Finlayson 2015 CCM researched but not built)

### Workflow

1. Upload image at `colorv2.reidsurmeier.wtf`
2. Auto-upscale to ≥2048px via color-separator's RealESRGAN over Tailscale if needed
3. Free-form text intent prompt (optional — empty prompt uses Emma defaults)
4. Single `claude -p` invocation: Opus 4.7 reads image + SNIC cell graph overlay + text prompt → outputs structured constraints + per-region cell ID lists for 4–9 underlayer plates
5. **MediaPipe REMOVED from V1** — Opus 4.7 vision replaces it. MediaPipe code parked as V2 escape hatch in `research/v3-construction/mediapipe-face-spatial/`
6. Algorithm proposes underlayer baseline from `mokuhanga_rule_classifier.underlayer_proposer` (94.4% match against Reid's annotated reference)
7. LLM-extracted text overrides apply on top of algorithmic baseline
8. JAX solver fills blocks 5–27 + pull order + dark/detail plates, with:
   - SNIC superpixels for cell graph (drop-in replacement for SLIC; deterministic, polygons native)
   - Hard-sigmoid STE for binary mask outputs (Bengio 2013, BinaryConnect)
   - Continuity loss term + load-bearing test per pull (gradient×mask + counterfactual ablation + SIM-Shapley audit)
   - Staged 3-batch outer loop (Wang 2024 simplified-target, Mehta 2023 topo-derivative, Worchel 2023 shrinking trust region)
9. Preview rendered: composite vs target + 27-block grid (primary review surface) + scrubber + interpretation panel + verification UI
10. Iteration: anchored via `previous_plan.json` warm-start, soft cap at 5 with diminishing-returns nag
11. Single-block patch via `regenerate_blocks: [N]` if mid-carving discovery
12. Sign off → `export_carving_files` writes to `~/cnc-carving-jobs/emma-<date>/`
13. Reid manually carves on RISD ShopBot (1/4" + 1/8" + 1/16" + 30° V-bit, ShopBot OpenSBP via Vectric VCarve), uses Mike Lyon soto-kento jig pattern
14. V1.0 ship criterion: engineering + one carved block + one proof pull

### Acceptance test

Per Reid: **overlap and continuity from pull 1 to last pull — buildup of tones from layer to layer**, matching Hokusai/Chuck Close progressive-proof reference structure. Final ΔE is advisory, not gating.

Pulls may be fully covered IF load-bearing via overprint physics: a pull "earns its place" when removing it shifts visible final-pixel color by > some-small-ΔE in some region. Wasted pulls (zero contribution) are rejected.

### Six hard validators (gate sign-off)

| Validator | Function | Threshold | Verified |
|---|---|---|---|
| `plate_not_composite_score` | Penalize blocks that look like final image | Reject if **< 0.6** (HIGH = good plate, LOW = bad residual). v13 blocks 24/25/26 score 0.999-1.000 badness (FAIL); blocks 01–23 score 0.40-0.46 (PASS). | ✅ |
| `role_purity_score` | Each block has clear print role | Reject if < 0.7 fraction sharing modal role | ✅ |
| `jigsaw_separation_score` | Zones on a block need brushable boundaries | Reject if any pair < 5mm physical separation | ✅ |
| `proof_progression_score` | Proof states add visible families over time | Reject if any consecutive pair < N pixels shift | ✅ |
| `underlayer_reversal_check` | Plates mirrored, pulls not | Boolean pass/fail | ✅ |
| `final_match_score` | Color/structure match | ΔE_2000 mean, advisory not gating | ✅ |

Smoking-gun proof in `research/v3-construction/validators-reconstruction/v13_smoking_gun_results.json`.

### Stack

- **Forward render:** t1_mixbox (binder-mixing approximation) for V1. t2_empirical (Curtis 1997 2-substrate inverse K-M) and t3_spectral (Curtis 1997 multilayer K-M + Saunderson 1942 surface correction + Berns 2016 36-wavelength K/S) deferred to V2 after calibration ships.
- **Block packing:** DSATUR + chordality cert + MaxRects face packer (Yekezare 2024 optimality proof for chordal graphs at chuck-mcp scale of 12–40 impressions).
- **Plate vectorization:** mill-radius-sized area-opening + opening-by-reconstruction (Vincent 1993) before Potrace (Selinger 2003). Horizontal flip on plate SVG export.
- **CNC stock:** 280×400×12mm hard maple plywood. 27 blocks + 1 soto-kento jig fit on 2 sheets of 4'×8'×12mm.
- **Frontend:** in-place addition `src/app/colorv2/` parallel to existing `src/app/color-separator/` in the same Next.js project. Shares deploy. Plate loading animation from `globals.css:1114-1188` + `PlatesGrid.tsx` reused verbatim.

### Deferred to V2 explicit

- Calibration (DSLR rig, swatch print + photo Lab extraction, `bootstrap_pigment` + `drift_check` MCP tools)
- t2/t3 spectral renderers
- Photoshop integration (psd-tools 1.17.0 sample shipped, parked as user-facing escape hatch)
- MediaPipe face landmarks (full working pipeline shipped to research, parked behind Opus 4.7 vision)
- Edition session tracking UI
- Auto-CNC dispatch
- Multi-image scope beyond Chuck Close portraits
- Multi-project gallery, auth/private mode

## What Reid wants you to check

These are the specific concerns Reid is asking the external reviewer to stress-test. Be ruthless.

### 1. Is Opus 4.7 vision actually capable of producing accurate cell-ID assignments?

The MediaPipe pipeline was replaced by Opus 4.7 vision in the same `claude -p` call that parses intent. The LLM sees the input image, a SNIC cell graph overlay with cell IDs labeled, and Reid's text prompt. It outputs per-region cell ID lists.

Concerns:
- **Accuracy:** can Opus 4.7 reliably identify which of ~1700 SNIC cells fall inside "the cheek region" from a labeled overlay? Sub-pixel landmarks were the MediaPipe path's advantage.
- **Determinism:** the design says "deterministic iteration achieved via cached cell assignments in `previous_plan.json`," but the FIRST call has no cache. If the first call produces wrong cells, every iteration starts from a bad anchor.
- **Latency:** ~36s per call already established. With image input, will tokens balloon and latency increase?
- **Cost:** $0.50/call × 5 iterations × N projects/mo — is the $100/mo Max plan budget sustainable?

**Question for reviewer:** can Opus 4.7 vision produce reliable cell-ID lists from a labeled SNIC overlay, OR should MediaPipe be unshelved? Cite specific Anthropic capability documentation if available.

### 2. Is the load-bearing test computationally correct?

The validator gates a "pull earns its place" by checking whether removing it shifts visible final-composite color by > threshold-ΔE in any region. Implementation:
- Inner loop: gradient×mask (SLA-style 1-step IG), O(1) cost
- Outer checkpoint: counterfactual ablation via `jax.vmap`, O(N)
- Offline audit: SIM-Shapley with 1000-coalition budget for cancellation pairs

Concerns:
- **Cancellation pairs:** two pulls may individually be load-bearing but cancel when both present (overprint to a color that's identical to skipping both). Gradient×mask misses this. SIM-Shapley catches it but only offline. Should the outer-loop counterfactual ablation cover pairs too, not just singletons?
- **Threshold setting:** "ΔE > some-small-threshold" — what's the actual threshold? ΔE 2? 5? The design doesn't say.
- **Reachability:** the load-bearing test runs on the FINAL composite. What if a pull is only load-bearing for a SPECIFIC region (e.g., the lip) but the rest of the composite is unchanged? Is the test region-specific or global?

**Question for reviewer:** is the load-bearing computation tight enough to gate physical carving decisions, or are there failure modes that would let bad pulls slip through?

### 3. Is the validator threshold direction correct?

The original design doc had an inverted threshold on `plate_not_composite_score` ("Reject if > 0.6" when it should have been "Reject if < 0.6"). Caught by the cell-zone-renderer agent. The doc has been corrected, but:

Concerns:
- Are the OTHER 5 validators' thresholds set correctly? Reviewer should check each.
- The reconstruction-doc framing is "v13's residual α-maps must FAIL `plate_not_composite_score`" — does the corrected threshold actually achieve this? (Verified empirically: blocks 24/25/26 score 0.999-1.000 badness with the implemented formula. But the implemented formula is `badness = (sim + spread_bad) / 2 > 0.6` which inverts to the design doc's `score = 1.0 - (sim + ...) / 2 < 0.6`. The reviewer should confirm both forms reject the same things.)

**Question for reviewer:** are all 6 validator thresholds correctly set such that they reject ONLY bad outputs and pass ONLY good ones?

### 4. The continuity criterion is the WHOLE acceptance test. Is the algorithmic model sufficient to satisfy it?

Reid's stated criterion: "overlap and continuity from pull 1 to the last pull — buildup of tones from one layer to the next." Reference: Hokusai progressive proof and Chuck Close 2015 self-portrait proof.

Continuity is enforced by:
- Staged 3-batch outer loop (light supports → mid → detail) with shrinking trust region
- Load-bearing test (every pull must be load-bearing OR directly visible)
- proof_progression_score validator (each checkpoint adds visible families)
- mokuhanga rule classifier producing underlayer roles with documented printmaker provenance

Concerns:
- **Staged 3-batch ≠ 132 pulls.** Reid's reference is 132-pull continuity, not 3-batch continuity. Does the 3-batch outer loop produce internally-continuous batches that look like a 132-pull sequence to the eye, or does it produce 3 stylistic jumps?
- **Re-inking same block at later pulls:** Shibata uses each block ~5 times. The current design says `(block_id, pass_index, mask)` triple with `mask[t+1] ⊆ mask[t]` reduction monotonicity. But the SOLVER objective only sees the final composite — does it have any term that rewards re-using the same block at later pulls vs producing a new block?
- **Hokusai vs Emma reference:** Hokusai shows ~8-pull buildup. Emma is ~132-pull buildup. The acceptance test was framed "matches Hokusai/Chuck Close style" — these are DIFFERENT scales. Is the system optimizing for Hokusai-style structural buildup or Shibata-style 132-pull accumulation?

**Question for reviewer:** does the design's continuity machinery actually produce 132-pull progressive proofs that look like Shibata's Emma, or does it produce 27 distinct plates that lack the Shibata depth-via-overprint quality?

### 5. The frontend in-place fork — risk of cross-contamination

The frontend strategy is to add `src/app/colorv2/` parallel to `src/app/color-separator/` in the same Next.js project. Shared deploy, shared globals.css, shared API proxy code.

Concerns:
- **Breaking color-separator:** any global CSS change for colorv2 risks regressions on color.reidsurmeier.wtf which is in active use.
- **Build complexity:** Two routes, two backend integrations, two SSE streams. If colorv2 has a perf bug, does it impact color-separator?
- **Domain routing:** `colorv2.reidsurmeier.wtf` and `color.reidsurmeier.wtf` need separate Cloudflare tunnel routes hitting the same Next.js instance with `src/middleware.ts` doing host-based routing.

**Question for reviewer:** is the in-place fork strategy actually safer than a true clone (separate repo), or does it accumulate hidden coupling risks that bite during V1.0 carving validation?

### 6. The 5-week build estimate — is it real?

| Week | Scope |
|---|---|
| 1 | Domain objects, SNIC drop-in, STE binarization, `plate_not_composite_score` validator, mill morphology gate, horizontal flip |
| 2 | Block partition + role assignment, mirrored plate rendering, 5 remaining validators, 5 review sheets |
| 3 | Continuity objective + load-bearing in solver loss, LLM (`claude -p`) prompt translation, anchored iteration, single-block patch |
| 4 | colorv2.reidsurmeier.wtf Next.js frontend, preview UI, verification UI, sign-off, carving export |
| 5 | One-block validation — Reid carves block_01, prints one proof pull, V1.0 tag |

Concerns:
- **The MediaPipe removal** (just decided) implies the LLM-prompt-translation work in Week 3 expands to include vision. Is that an extra 2-3 days? Or already absorbed?
- **The frontend** is estimated at 32 hours / 5 days. Week 4 might be tight if backend stubs aren't ready by Day 3.
- **Carving validation** requires CNC time at RISD. Studio access, ShopBot scheduling, end-mill purchases. Realistic for week 5?
- **V13 has working code** — but the design says "incremental refactor (B)." How much of v13's solver code actually gets reused vs replaced? Reid said: keep methodology generator, replace block_NN.png output, add new validators.

**Question for reviewer:** is 5 weeks credible, or is this a 10-week project being optimistically scheduled?

### 7. The pigment library — uncalibrated risk for edition-of-10

V1 ships without calibration. Pigment YAML uses catalog Lab values for store-bought + Reid-estimated values for handmade pigments. Edition of 10 means 10 prints must match within an acceptable variance.

Concerns:
- **Calibration was deferred AFTER edition-of-10 was locked.** Q9 locked edition=10, Q16 deferred calibration. These are inconsistent — edition-of-10 requires sub-ΔE-2 inter-print consistency per the calibration research; uncalibrated pigments produce ΔE 4-5 inter-print drift.
- **Reid's plan:** "trust digital approximations, calibrate at print time via notebook" (Q17). But that defers the edition-consistency problem to PHYSICAL printing without any system support.
- **What does V1.0 actually ship if uncalibrated?** A proof print that's "in the ballpark" of the digital plan, with all the inter-print variance to be discovered by Reid at the inking table.

**Question for reviewer:** is the calibration deferral compatible with the edition-of-10 commitment, or is Reid setting himself up for unsatisfying edition variance that V1.0 has no tools to fix?

---

## AUDIT RESPONSE (for the reviewing LLM to complete)

After reading this brief and the linked design docs, fill out each section:

### Response 1: Opus 4.7 vision for cell-ID assignment

**Verdict:** [SOUND / RISKY / UNSOUND]
**Reasoning:**
**Cite Anthropic documentation if available:**
**Recommended action:**

### Response 2: Load-bearing test correctness

**Verdict:**
**Failure modes identified:**
**Threshold recommendation (concrete ΔE value):**
**Should outer-loop ablation cover pull pairs as well as singletons?**

### Response 3: Validator threshold direction

**Verdict (per validator):**
- `plate_not_composite_score`:
- `role_purity_score`:
- `jigsaw_separation_score`:
- `proof_progression_score`:
- `underlayer_reversal_check`:
- `final_match_score`:

### Response 4: Continuity criterion satisfaction

**Verdict:**
**Will the 3-batch staged solver produce 132-pull continuity, or 3 visible stylistic jumps?**
**Does the design have a solver term rewarding re-inking the same block?**
**Hokusai-scale vs Shibata-scale: are we optimizing for the right scale?**

### Response 5: Frontend in-place fork risk

**Verdict:**
**Cross-contamination risks identified:**
**Recommendation: keep in-place or separate repo?**

### Response 6: 5-week build estimate realism

**Verdict:**
**Identified slippage risks:**
**Adjusted estimate:**

### Response 7: Calibration deferral vs edition-of-10

**Verdict:**
**Is this internally consistent?**
**What does V1.0 actually deliver as a "physical proof" if uncalibrated?**
**Recommended action:**

### Open questions the brief author DIDN'T ask but should have

List any concerns you have that aren't covered above. Especially anything that could derail V1.0 at week 4-5.

### Final summary

Give a one-paragraph adversarial summary of the design's overall soundness. Is this project shippable in 5 weeks? What's the single biggest risk?

---

**End of audit brief. Reviewer respond above.**
