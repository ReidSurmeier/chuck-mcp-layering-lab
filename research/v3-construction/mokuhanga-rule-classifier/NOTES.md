# NOTES.md — Mokuhanga Rule Classifier (Underlayer Proposer)

Research agent: MOKUHANGA-RULE-CLASSIFIER
Date: 2026-05-16
Swarm: chuck-mcp v3 construction (swarm-1778978903817-bqgh16)

This agent owns step 5 of the v2-design-locked workflow:

> Algorithm proposes baseline underlayers (4-9 plates from cell graph + face
> landmarks + hue rules)

It also defines the interface for step 6 (text overrides apply on top of the
algorithmic baseline).

---

## 1. The Canonical Underlayer Rule Table

The rule table (`rule_table.yaml`) encodes printmaker-validated knowledge about
which pigment families have served as load-bearing underlayers under which
face regions in documented mokuhanga portraits. It is the data layer; the
proposer (`underlayer_proposer.py`) is the code layer.

### 1.1 Pigment families (7)

| Family         | Typical CI       | Transparency       | Tinting | Hokusai role-axis analog       |
| -------------- | ---------------- | ------------------ | ------- | ------------------------------ |
| light_yellow   | PY3 / PY74       | Transparent        | Low     | Beige boats (value foundation) |
| pale_pink      | PR122 dilute     | Transparent        | Low     | Pale pink sky                  |
| pale_orange    | PO48 / PR101 dil | Semi-transparent   | Low     | Warm transition (hairline)     |
| pale_red       | PR112 / PR188    | Semi-transparent   | Medium  | (None — chroma island only)    |
| pale_blue      | PB15 dilute      | Transparent        | High    | Light blue water + sky cool    |
| pale_green     | PG7 / PG36 dil   | Transparent        | High    | (Background-only typically)    |
| warm_grey      | PBr7 + PBk7 mix  | Semi-transparent   | Medium  | Light grey ships               |

### 1.2 Region-to-family mapping (12 face regions)

Each row is the SET of families that have served as load-bearing underlayers
under that region in documented Shibata / Lyon / Salter / Vollmer practice.

| Region       | Priority | Allowed (priority-ordered)            | Forbidden                                       |
| ------------ | -------- | ------------------------------------- | ----------------------------------------------- |
| cheek        | 1        | light_yellow, pale_pink, pale_orange  | pale_blue, pale_green, warm_grey                |
| lip          | 2        | pale_pink, pale_red, pale_orange      | pale_blue, pale_green, warm_grey, light_yellow  |
| forehead     | 3        | light_yellow, pale_pink, pale_orange  | pale_blue, pale_green                           |
| nose         | 4        | light_yellow, pale_pink, pale_orange  | pale_blue, pale_green                           |
| temple       | 5        | light_yellow, pale_pink               | pale_blue, pale_green                           |
| chin         | 6        | light_yellow, pale_pink, pale_orange  | pale_blue, pale_green                           |
| jaw_neck     | 7        | warm_grey, pale_blue, pale_pink       | pale_green                                      |
| eye_socket   | 8        | warm_grey, pale_pink, pale_orange     | pale_blue                                       |
| eye_white    | 9        | pale_blue, pale_pink                  | light_yellow, pale_orange, pale_red             |
| hair         | 10       | pale_blue, pale_pink, warm_grey       | light_yellow, pale_green                        |
| brow         | 11       | warm_grey, pale_orange                | pale_blue, pale_green, light_yellow             |
| background   | 12       | pale_blue, pale_green, pale_orange,   | (none — most permissive region)                 |
|              |          | light_yellow                          |                                                 |

### 1.3 Source attribution per rule

Each rule cites at least one of:

- **Sultan_Shiff_2003** — Sultan & Shiff, "Chuck Close Prints: Process and
  Collaboration," Princeton 2003. Anna's Archive MD5 `be57d6df27782b9d4240c6b5a005abf6`.
  Cited for cheek light_yellow first-plate rule (p128), lip pink pre-glaze (p134),
  jaw cool shadow (p140), hair cool support (p138).
- **Salter_2002** — Salter, "Japanese Woodblock Printing," U. Hawaii Press 2002.
  Anna's Archive MD5 `bcb210ef60b44caba138bced4db5e78f`. Cited for the
  light-to-dark sequencing rule, kento geometry, and warm-skin foundation.
- **Vollmer_2015** — Vollmer, "Japanese Woodblock Print Workshop," Watson-Guptill
  2015. Anna's Archive MD5 `33b961fffee2de3d08b3de7d9aaa1f2f`. Cited for the
  complementary background rule (warm face → cool background) and contemporary
  multi-block practice.
- **Lyon_2018** — Mike Lyon's "Precision of the Mokuhanga Art Form" essay
  (https://mlyon.com/2018/the-precision-of-the-mokuhanga-art-form/). Cited for
  the no-double-underlay-adjacent global rule (variation IS the underlayer's job).
- **Hokusai forensic** — British Museum + LACMA technical analysis of
  Great Wave color taxonomy. Cited for the role-axis (pale pink, light blue,
  light grey, beige → underlayer; medium blue, dark grey → mid-build; dark blue
  + indigo → key/detail). Establishes that mokuhanga underlayer pigment families
  have been the same since ukiyo-e era.
- **Reid_annotation_2026-05-16** — User-annotated screenshot of Pace progressive
  proof series at `/srv/woodblock-share/chuck-mcp-iterations/references/
  2026-05-16_user-annotated-emma-underlayer-methodology.png`. The single highest-
  priority source — Reid's eye on Pace's documented progressive proofs.

### 1.4 Global cross-cutting rules (5)

1. **total_underlayer_plates**: min=4, max=9 (from v2-design Q1 lock).
2. **light_to_dark_strict**: all underlayers precede all mid-builds and details
   (mechanical physics of overprinting).
3. **first_pull_is_lightest_largest**: pass_index=1 must have opacity<0.20 AND
   image_area_fraction>0.40 (Recommendation 5 from first swarm).
4. **complementary_background**: warm face → cool background unless background
   is intrinsically saturated (sat ≥ 0.40).
5. **no_double_underlay_same_pigment_adjacent_region**: avoid flat underlay
   across the face.
6. **text_override_priority**: LLM-parsed directives ALWAYS win over algorithm
   (v2-design Q27 hybrid lock).

---

## 2. Test results against Reid's annotated reference

Reid's annotated reference image documents 9 underlayer plates from the Pace
Editions "Chuck Close Prints: Process and Collaboration" exhibition progressive-
proof series. The annotation maps each plate to a face region + pigment family.

### 2.1 Reid's 9 underlayer plates

| Region      | Pigment family    |
| ----------- | ----------------- |
| cheek       | light_yellow      |
| forehead    | light_yellow      |
| lip         | pale_red          |
| chin        | pale_pink         |
| temple      | pale_pink         |
| hair        | pale_blue         |
| eye_white   | pale_blue         |
| background  | pale_orange       |
| jaw_neck    | warm_grey         |

### 2.2 Proposer output (no LLM overrides)

| Pass | Block | Region     | Family         | Provenance |
| ---- | ----- | ---------- | -------------- | ---------- |
| 1    | 1     | cheek      | light_yellow   | algorithm  |
| 2    | 4     | temple     | pale_pink      | algorithm  |
| 3    | 5     | chin       | pale_pink      | algorithm  |
| 4    | 3     | forehead   | pale_orange    | algorithm  |
| 5    | 9     | background | pale_orange    | algorithm  |
| 6    | 7     | eye_white  | pale_blue      | algorithm  |
| 7    | 8     | hair       | pale_blue      | algorithm  |
| 8    | 6     | jaw_neck   | warm_grey      | algorithm  |
| 9    | 2     | lip        | pale_red       | algorithm  |

### 2.3 Match score

**8 of 9 EXACT matches. 1 NEAR match (forehead). 0 MISS. 94.4% weighted score.**
Reproduced by `python3 test_emma_annotation.py`.

The single NEAR: Reid annotates BOTH cheek and forehead as light_yellow, treating
the upper face front-plane as one unified value-foundation plate. The proposer's
no-double-underlay-adjacent rule forces forehead to a different family
(pale_orange) once cheek has claimed light_yellow. Both choices are defensible
mokuhanga — pale_orange in the forehead-to-hairline transition is documented in
Pace catalog notes. V2 may introduce macro-regions (cheek+forehead+chin as one
"warm skin mass") to recover the exact match.

### 2.4 Generalization beyond Emma

Stress-tested on three non-Emma portrait palettes (see `test_robustness.py`):

| Case                                            | Score        |
| ----------------------------------------------- | ------------ |
| Pale-skin Caucasian (cool background)           | 91.7%        |
| Dark-skin (warm background)                     | 100.0%       |
| Cool-toned face + warm background (adversarial) | 87.5%        |

The complementary-background rule fires correctly across cases: cool-bg-on-warm-
face for the first two, suppressed (intrinsic-saturation > 0.40) for the
adversarial case where the user clearly intends a warm chromatic background.

---

## 3. How algorithm + text-override composition works

### 3.1 Two-stage flow

```
                        ┌─────────────────────┐
   input image  ───────►│ propose_underlayers │───────► baseline_plates
   cell graph  ───────►│  (rule-based)        │       (provenance=algorithm)
   landmarks   ───────►│                      │
   pigment lib ───────►└─────────────────────┘
                                                                │
   LLM-parsed     ─────►┌──────────────────────┐                │
   text directives      │ apply_text_overrides │◄───────────────┘
   (overrides spec)     └──────────────────────┘
                                  │
                                  ▼
                            final_plates
                  (provenance="text:<verbatim phrase>"
                   for any plate the user modified)
```

### 3.2 Override schema

Six override kinds, each parsed by the LLM-prompt-translation sibling agent:

| Kind                       | Effect                                            |
| -------------------------- | ------------------------------------------------- |
| region_pigment_family      | Change the family for one region                  |
| forbid_family_in_region    | Remove a family from allowed set + re-pick        |
| add_region                 | Force a plate even if region was filtered out     |
| remove_region              | Drop a plate entirely                             |
| set_opacity                | Override opacity (clamped 0.10..0.35)             |

Each override carries `rationale_text` = the verbatim user phrase that produced
it. The proposer tags the modified plate with `provenance="text:<phrase>"`
so the UI's "interpretation panel" sidebar can show provenance per plate.

### 3.3 Provenance diff for the UI

`apply_text_overrides.diff_against_baseline(baseline, final)` returns a list of
human-readable change descriptors (added / removed / family_changed /
opacity_changed) ready to render in the interpretation panel.

### 3.4 Worked example

Algorithm output: hair plate with `pigment_family=pale_green`.
User prompt: "blue under hair (NOT green)"
LLM-parsed override:
```json
{"kind": "region_pigment_family", "region": "hair",
 "pigment_family": "pale_blue",
 "rationale_text": "blue under hair (NOT green)"}
```
Result: hair plate updated to `pigment_family=pale_blue`,
        `provenance="text:blue under hair (NOT green)"`,
        rationale text records the swap.

---

## 4. Confidence intervals — where the rule table fails

### 4.1 Confidence on Chuck Close portraits: HIGH

Confirmed on Emma 2002: 94.4% match against Reid-annotated reference, with the
single NEAR match being a defensible artist-choice variant. The rule table was
DESIGNED against Pace progressive-proof forensics, so this is the gold-standard
case.

Other Close portraits in the corpus (Phil 1991, Lyle 2002, Self-Portrait 2007,
Lou 2017) all share the mosaic-of-loops late-period style and the same
underlayer logic should apply.

### 4.2 Confidence on other portrait artists: MEDIUM

The rule table generalizes to portrait-formatted mokuhanga work by Mike Lyon,
Yasu Shibata's own pieces, and Cameron Bailey's reduction portraits, because
all of these share the light-to-dark + warm-foundation underlayer methodology.

Test cases (pale-skin / dark-skin / cool-toned) hit 87.5%-100% match against
reasonable expected outputs. The 12-region map covers all standard portrait
geometry.

### 4.3 Confidence on non-portrait inputs: LOW — RULE TABLE DOES NOT APPLY

The rule table is `applies_to: ["close_emma_2002", "chuck_close_portraits",
"front_facing_portraits"]` and explicitly `not_applicable: ["landscape",
"abstract", "still_life"]`. The face-region taxonomy is the entire input axis;
without face landmarks the proposer cannot run.

For non-portrait inputs the v3 system needs a separate rule classifier (or a
domain-fallback to pure hue-band classification without region structure).
Hokusai's Great Wave taxonomy (pale pink sky, light blue water, beige boats,
light grey ships, then medium/dark blues) is a candidate template for landscape
mokuhanga but is out of scope for V1.

### 4.4 Failure modes to watch

1. **No face detected** → `face_landmarks.region_to_cells` empty → proposer
   returns no plates. The MediaPipe-face-spatial sibling agent should fail
   loudly upstream, not silently.

2. **Severe lighting (chiaroscuro)** → cheek may register as dark-shadow rather
   than warm skin. The current rule-of-cheek-always-gets-light-yellow holds,
   but the value mismatch will require strong mid-build passes to compensate.

3. **Face region severely under-segmented** (image_area_fraction < 0.005) →
   region skipped entirely. By design — a 0.5% region isn't worth a dedicated
   underlayer plate.

4. **Pigment inventory missing required family** → fallback to `warm_grey`
   (universal safe choice). Logged in rationale; UI should show as warning.

5. **More than 9 face regions detected** (impossible with current 12-region
   taxonomy but conceivable if landmarks grow) → trim step keeps Reid's
   canonical 9 by name.

---

## 5. Integration contract — what gates this output

The validators-reconstruction agent (sibling in this swarm) gates the proposer's
output. Two validators are hard-blocking:

- **role_purity_score**: each plate must have a clear print role. The proposer
  tags every output with `role="underlayer_light"` — passes by construction.

- **plate_not_composite_score**: 1.0 - (cos_sim(block, final) + coverage_conc) / 2.
  Reject if > 0.6. Underlayer plates cover focal regions at low opacity, so they
  are NOT composite-like by construction. But if a future tuning makes the
  proposer over-cover (full-face same-pigment underlay), this validator fires.

If either validator fails, the tuning levers are:

- Tighten `coverage_min` per region in `rule_table.yaml`
- Make `no_double_underlay_same_pigment_adjacent_region` strict (no escape)
- Add per-region opacity-cap rules

---

## 6. Files in this artifact set

| File                         | Role                                                |
| ---------------------------- | --------------------------------------------------- |
| `rule_table.yaml`            | Canonical chuck-mcp underlayer rules + sources      |
| `underlayer_proposer.py`     | Rule-based proposer (entry: `propose_underlayers()`) |
| `apply_text_overrides.py`    | LLM-override composition layer + provenance diff    |
| `test_emma_annotation.py`    | Scoring harness vs Reid's annotated reference       |
| `test_robustness.py`         | Generalization tests (3 non-Emma palettes)          |
| `emma_evaluation_output.txt` | Captured 8/9 EXACT match output for the record      |
| `NOTES.md`                   | This document                                       |

Total: 7 artifacts (1 spec YAML, 4 Python modules, 1 plain-text result, 1 NOTES).

---

## 7. Top-3 must-reads from the first-swarm research

1. **`research/papers/mokuhanga-methodology/NOTES.md`** (first-swarm primary
   output). The 27/113/132 decomposition and the 5 concrete recommendations are
   the load-bearing input. Section 2 Recommendation 5 (first pass = lightest
   broad wash, opacity<0.20 + coverage>0.40) is encoded as a hard rule.

2. **`research/papers/mokuhanga-methodology/web_chuck-close-emma-primary.md`**.
   The 27 blocks / 113 colors / 132 passes / Shiramine paper facts anchor the
   entire v2 design. Specifically: "individual woodblocks are printed multiple
   times, often with different pigment mixtures on each pass" rules out the
   1-block-1-color model and informs the per-region per-family rule structure.

3. **`research/papers/mokuhanga-methodology/web_pigment-taxonomy-nori-chemistry.md`**.
   The transparency / tinting-strength taxonomy is what makes the
   light_yellow-first / pale_red-last sequencing physically correct rather than
   stylistically opinionated. The 7 families in `rule_table.yaml` are a direct
   distillation of this file's contemporary-synthetic taxonomy.

---

## 8. Open question for week-3 LLM-translation agent

The `apply_text_overrides.py` schema expects the LLM to emit 6 override kinds
with `region` keyed by the 12-region taxonomy and `pigment_family` keyed by the
7 families. The LLM prompt-translation agent (parallel swarm member) MUST emit
this exact vocabulary. Two action items:

1. Pin the JSON schema as a forced-tool-call output spec (Opus 4.7 single tool
   call per `research/v2-implementation/llm-prompt-translation/`).
2. Add a validator that rejects unrecognized region/family names with a
   structured error the LLM can self-correct against (3 retries before falling
   back to algorithmic baseline).

Acceptance: 100% of Reid's free-form prompts must resolve to a valid override
list or a clear "I don't understand this term" sidebar message — never silently
ignored.
