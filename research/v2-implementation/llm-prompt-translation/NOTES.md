# NOTES — LLM Prompt Translation for chuck-mcp v2

**Research domain:** LLM-driven translation of natural-language artistic intent into structured solver constraints.
**Swarm:** swarm-1778969836247-ys4o7z, agent LLM-PROMPT-TRANSLATION.
**Date:** 2026-05-16.
**Artifacts in this folder:**

| File | Topic |
|---|---|
| `arxiv_2509_10058_color_me_correctly.md` | LLM disambiguation of color terms → CIELab anchors (precedent) |
| `arxiv_2508_08987_colorgpt.md` | LLM as designer; hex>CIELAB; short profile; JSON > MLM; similarity-retrieved exemplars |
| `arxiv_2309_03409_opro.md` | LLM-as-optimizer (rejected pattern for V1; future v3 candidate) |
| `web_nl4opt_competition.md` | Canonical NL→optimization-problem entity types (NeurIPS 2022) |
| `web_anthropic_tool_use_spec.md` | Verbatim Anthropic recommendations for tool-call schema |
| `web_japanese_pigment_vocabulary.md` | 25-pigment anchor table for system prompt |
| `web_face_regions_portrait.md` | Overlay-label canonical vocabulary + fallback strategy |
| `web_negation_handling_llms.md` | Why LLMs miss "NOT magenta"; mitigation strategy |

---

## VERDICT

### Model choice: **Claude Opus 4.7** for prompt translation

- Anthropic explicitly recommends Opus for "complex tools and ambiguous queries" and notes Opus seeks clarification rather than silently guessing. Reid's prompts are inherently ambiguous (mokuhanga + nihonga vocabulary + artistic intent + Western color terminology mixed) — Sonnet/Haiku will silently make up answers.
- Cost is irrelevant — Reid's on Anthropic Max $100/mo flat-rate. Single translation per MCP tool call, ~3K input tokens system prompt + 2K user input + 1K output = ~6K tokens per call. Even at metered Opus rates (~$0.10/call) it would be fine. On Max it's free.
- Tool-use system-prompt overhead is **identical (346 tokens)** across Opus 4.7 / Sonnet 4.6 / Haiku 4.5. No reason to downsize.
- Reserve Sonnet 4.6 as a fast-path fallback for re-translation when user only changed a small part of the prompt and the previous Opus output is cached.

### Prompting pattern: **Strict tool-call with `tool_choice: tool` + structured JSON schema + `input_examples`**

Three Anthropic primitives, used together:

1. **`tool_choice: {"type": "tool", "name": "translate_artistic_intent"}`** — guarantees the LLM emits the structured object on every invocation. No risk of free-form chatter.
2. **`strict: true`** on the tool definition — constrained-decoding guarantees the JSON validates against the schema. No retry loop, no `JSON.parse()` failure path.
3. **`input_examples`** with 3–5 worked examples covering the prompt taxonomy (single-region, multi-region with negation, layer-order, signature-color, global-exclusivity). This is the most cost-effective robustness lever — examples literally show Claude the desired shape.

**Tools used: ONE tool, called `translate_artistic_intent`.** Not a multi-tool agentic loop. The whole translation is a single function call. ColorGPT confirmed this empirically — "short task profile" beats "long task profile" by 2x absolute accuracy. Don't over-engineer.

### Two stages, both at the same LLM call

Following NL4Opt's two-stage NER + logical-form split, but collapsed into one Opus 4.7 call:

```
PROMPT → Opus 4.7 + tool(translate_artistic_intent, strict=true) → Validated JSON → Solver
```

Stage-1 NER (find entities) and Stage-2 logical form (compose into constraints) are conceptually distinct but Opus 4.7 handles both in one shot. Output goes through a validator (Python) before reaching the JAX solver. Validator does the *hard* checks (pigment exists in YAML, no contradictions) that JSON schema can't express.

---

## JSON SCHEMA for `translate_artistic_intent`

```json
{
  "name": "translate_artistic_intent",
  "description": "Translate Reid's natural-language artistic intent into structured solver constraints for chuck-mcp v2 mokuhanga inverse-stack solver. The user prompt describes desired hues, tone ranges, pigment preferences, layer-order intentions, and exclusion constraints. Map the prompt onto the labeled regions present in the Photoshop overlay (provided as 'regions'), select pigments from the physical YAML (provided as 'pigment_library'), and surface every ambiguity in 'ambiguities_flagged'. Never invent pigment_ids; only emit IDs present in pigment_library. When the user uses Japanese pigment vocabulary (shu, ai, gunjo, etc.) match to pigment_id; when they use Western names (vermilion, indigo, cadmium red), normalize to the closest pigment_id in the library.",
  "strict": true,
  "input_schema": {
    "type": "object",
    "required": [
      "region_constraints",
      "global_preferences",
      "layer_order_constraints",
      "ambiguities_flagged"
    ],
    "properties": {
      "region_constraints": {
        "type": "array",
        "description": "One entry per labeled region the user constrained, OR one entry per overlay region if user gave only global directives. Skip regions the user did not address.",
        "items": {
          "type": "object",
          "required": [
            "region_label",
            "tone_anchors_hex",
            "preferred_pigment_ids",
            "preferred_pigment_families",
            "forbidden_pigment_ids",
            "forbidden_hue_families",
            "confidence"
          ],
          "properties": {
            "region_label": {
              "type": "string",
              "description": "Must match a label present in the input 'regions' array. If user named a region not in the overlay, choose the nearest match and add an ambiguity flag."
            },
            "tone_anchors_hex": {
              "type": "array",
              "description": "1-3 hex codes bounding the desired tone range. If user gave one hex, emit one. If user gave a range (e.g., 'between #f7d8c2 and #e0a888'), emit two. Use empty array if user gave no specific tone.",
              "items": {"type": "string", "description": "hex code in form #RRGGBB"}
            },
            "preferred_pigment_ids": {
              "type": "array",
              "description": "Pigment IDs from the YAML library the user explicitly or strongly implicitly preferred. Empty if user gave no preference.",
              "items": {"type": "string"}
            },
            "preferred_pigment_families": {
              "type": "array",
              "description": "Pigment family flags from the YAML: mineral_japanese, mineral_western, dye_organic_japanese, dye_organic_western, modern_synthetic. Empty if no preference.",
              "items": {"type": "string", "enum": ["mineral_japanese", "mineral_western", "dye_organic_japanese", "dye_organic_western", "modern_synthetic"]}
            },
            "forbidden_pigment_ids": {
              "type": "array",
              "description": "Pigment IDs the user explicitly forbade (e.g. via 'NOT magenta'). Empty if no exclusions.",
              "items": {"type": "string"}
            },
            "forbidden_hue_families": {
              "type": "array",
              "description": "Hue families to ban for this region. Used when user said e.g. 'NOT magenta' or 'no warm hues here'.",
              "items": {"type": "string", "enum": ["red", "orange", "yellow", "green", "blue", "violet", "neutral", "magenta"]}
            },
            "confidence": {
              "type": "string",
              "description": "How confident the LLM is that this region constraint reflects user intent.",
              "enum": ["high", "medium", "low"]
            }
          }
        }
      },
      "global_preferences": {
        "type": "object",
        "required": [
          "exclusive_hue_family",
          "high_chroma_regions",
          "low_chroma_regions",
          "overall_warmth"
        ],
        "properties": {
          "exclusive_hue_family": {
            "type": "string",
            "description": "If user said 'only X in the image', emit X. Otherwise 'none'.",
            "enum": ["red", "orange", "yellow", "green", "blue", "violet", "neutral", "none"]
          },
          "high_chroma_regions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Region labels where chroma should be maximized."
          },
          "low_chroma_regions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Region labels where chroma should be muted."
          },
          "overall_warmth": {
            "type": "string",
            "description": "Global warmth bias from user phrasing like 'warm-tonal' / 'cool-tonal' / 'neutral'.",
            "enum": ["warm", "neutral", "cool", "unspecified"]
          }
        }
      },
      "layer_order_constraints": {
        "type": "array",
        "description": "When user says 'X with Y underlayer', emit {region, top, under}. Order matters: 'top' is the surface impression, 'under' is the prior impression.",
        "items": {
          "type": "object",
          "required": ["region_label", "top_pigment_id", "under_pigment_id"],
          "properties": {
            "region_label": {"type": "string"},
            "top_pigment_id": {"type": "string"},
            "under_pigment_id": {"type": "string"}
          }
        }
      },
      "ambiguities_flagged": {
        "type": "array",
        "description": "Every interpretation the LLM had to guess on, in plain English. Empty array if no ambiguity. ALWAYS populate this for vague terms (warm-tonal, deep, soft, light, etc.) — describe the chosen default.",
        "items": {"type": "string"}
      }
    }
  },
  "input_examples": [
    /* see below */
  ]
}
```

The Python validator (between LLM and solver) additionally enforces:
- every `region_label` exists in the input overlay
- every `pigment_id` exists in the pigment library YAML
- no pigment appears in both `preferred_pigment_ids` and `forbidden_pigment_ids`
- if `exclusive_hue_family != "none"`, all preferred pigments fall in that hue family

Validation failure → re-call LLM with the failure message in the conversation; do **not** silently auto-fix.

---

## WORKED EXAMPLE — Reid's prompt

**Input prompt:**
> "Warm-tonal Emma. Skin between #f7d8c2 and #e0a888, lean toward Asian-mineral pigments. Hair is deep umber under indigo. Lip is vermilion with soft pink underlayer NOT magenta. Keep red high-chroma — only red in the image."

**Overlay regions (from Photoshop):** `skin_forehead, skin_cheek_l, skin_cheek_r, skin_chin, skin_neck, hair_main, hair_under, lip_outer, lip_inner, eye_iris_l, eye_iris_r, background`

**Pigment library (excerpt):** `shu, beni, bengara, taisha, kodaishu, ai, gunjo, ruri, sumi, gofun, taishikoh, suoh, kakishibu, ...`

**Expected LLM output:**

```json
{
  "region_constraints": [
    {
      "region_label": "skin_forehead",
      "tone_anchors_hex": ["#f7d8c2", "#e0a888"],
      "preferred_pigment_ids": ["taisha", "kodaishu", "bengara"],
      "preferred_pigment_families": ["mineral_japanese"],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": [],
      "confidence": "medium"
    },
    {
      "region_label": "skin_cheek_l",
      "tone_anchors_hex": ["#f7d8c2", "#e0a888"],
      "preferred_pigment_ids": ["taisha", "kodaishu", "bengara"],
      "preferred_pigment_families": ["mineral_japanese"],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": [],
      "confidence": "medium"
    },
    /* ... skin_cheek_r, skin_chin, skin_neck — all same tone band ... */
    {
      "region_label": "hair_main",
      "tone_anchors_hex": [],
      "preferred_pigment_ids": ["taishikoh", "sumi"],
      "preferred_pigment_families": ["mineral_japanese"],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": [],
      "confidence": "medium"
    },
    {
      "region_label": "hair_under",
      "tone_anchors_hex": [],
      "preferred_pigment_ids": ["ai"],
      "preferred_pigment_families": [],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": [],
      "confidence": "high"
    },
    {
      "region_label": "lip_outer",
      "tone_anchors_hex": [],
      "preferred_pigment_ids": ["shu"],
      "preferred_pigment_families": ["mineral_japanese"],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": ["magenta", "violet"],
      "confidence": "high"
    },
    {
      "region_label": "lip_inner",
      "tone_anchors_hex": [],
      "preferred_pigment_ids": ["beni"],
      "preferred_pigment_families": [],
      "forbidden_pigment_ids": [],
      "forbidden_hue_families": ["magenta", "violet"],
      "confidence": "medium"
    }
  ],
  "global_preferences": {
    "exclusive_hue_family": "red",
    "high_chroma_regions": ["lip_outer", "lip_inner"],
    "low_chroma_regions": [],
    "overall_warmth": "warm"
  },
  "layer_order_constraints": [
    {
      "region_label": "hair_main",
      "top_pigment_id": "taishikoh",
      "under_pigment_id": "ai"
    },
    {
      "region_label": "lip_outer",
      "top_pigment_id": "shu",
      "under_pigment_id": "beni"
    }
  ],
  "ambiguities_flagged": [
    "'Warm-tonal Emma' interpreted globally — applied the hex tone-band uniformly to all skin sub-regions (forehead, cheeks, chin, neck) rather than zone-differentiated (cheeks warmer, chin cooler). Confirm if zone-differentiation desired.",
    "'Asian-mineral pigments' resolved to family=mineral_japanese, members: shu, gunjo, ruri, roku, byakuroku, taisha, bengara, sumi, gofun, kodaishu, taishikoh.",
    "'deep umber under indigo' for hair — chose taishikoh (Japanese brown-ochre incense) for umber and ai for indigo. taishikoh has only borderline 'umber' character; if Reid wants a true umber consider bengara+sumi mix or accept that pigment library lacks PBr7 burnt umber.",
    "'vermilion' resolved to shu (HgS); not historically PR108 cadmium. If Reid wants the brighter cadmium hue, mark this as a swatch-calibration miss.",
    "'soft pink underlayer' resolved to beni (organic safflower carmine). beni's hue is borderline magenta in some lights — the 'NOT magenta' constraint may be violated by beni itself. Flagging for review; alternatives: dilute shu, dilute kodaishu.",
    "'only red in the image' applied globally — set exclusive_hue_family=red. This BANS ai (indigo) and taishikoh (brown). But user explicitly named ai for hair_under and taishikoh-like for hair_main. CONTRADICTION — solver will fail validation. Reid must clarify: is 'only red' a chroma directive (only red has high chroma; other hues are neutral-low-chroma support) or a literal exclusion?"
  ]
}
```

**Key behaviors demonstrated:**

1. **Region expansion** — "skin" → 5 sub-region constraints, not one.
2. **Family resolution** — "Asian-mineral pigments" → explicit member list in ambiguity note (auditable).
3. **Negation captured explicitly** — `forbidden_hue_families: ["magenta", "violet"]` on lip regions.
4. **Layer order parsed** — "X under Y" → `{region, top, under}`.
5. **Contradiction surfaced** — "only red" conflicts with named non-red pigments. LLM does NOT auto-fix. Surface and wait for user.
6. **Calibration boundary admitted** — "vermilion = shu, not PR108" — important for Reid's swatch sheets.

---

## AMBIGUITY HANDLING POLICY

LLM **never asks a clarifying question** mid-MCP-tool-call. The MCP protocol is request/response. Instead:

1. **Pick a default and proceed.** Default is encoded in the system prompt:
   - "warm-tonal" → `overall_warmth: "warm"`, unspecified tone bias
   - "deep X" → highest-chroma + lowest-luminance pigment in family X
   - "soft X" → mid-chroma + mid-luminance
   - "light X" → low-chroma + high-luminance (mixed with gofun if available)
   - "X NOT Y" → forbidden constraint on Y
   - "only X" → `exclusive_hue_family: X`
2. **Surface the default in `ambiguities_flagged`** so Reid can see the LLM's reasoning.
3. **Contradiction handling:** when constraints conflict (e.g. "only red" + "indigo hair"), the LLM emits both, flags the contradiction, and lets the solver fail loudly OR lets Reid review the output before invoking the solver.
4. **Iterate via second MCP call.** Reid can refine the prompt and call again. Each call is stateless; the previous output is cached client-side in chuck-mcp.

This is the same pattern as ColorGPT (LLM commits to a guess, surfaces it, user iterates) and avoids the multi-turn-clarification antipattern.

---

## ANCHOR PROMPT VOCABULARY (system-prompt boilerplate)

The system prompt embeds a controlled vocabulary the user is *encouraged* (not required) to use. This is the table from `web_japanese_pigment_vocabulary.md` plus the region labels from `web_face_regions_portrait.md` plus the modifier conventions:

**Modifiers and their semantics (taught in system prompt):**
- `light`, `pale`, `washed` → high luminance, lower chroma (mix with gofun)
- `dark`, `deep` → low luminance, often higher chroma
- `soft`, `muted`, `dusty` → mid luminance, low chroma
- `vivid`, `bright`, `pure`, `high-chroma` → max chroma
- `warm`, `warm-tonal` → bias toward yellow-orange-red hue families
- `cool`, `cool-tonal` → bias toward green-blue-violet hue families
- `under X` (where X is a pigment or color) → layer order: target pigment goes ON TOP of X
- `over X` → target pigment goes UNDER X (X overprints)
- `NOT X`, `no X`, `without X`, `avoid X` → exclusion (forbidden)
- `only X` → global exclusivity

**Region modifiers:**
- `entire skin` → all `skin_*` labels
- `cheek` / `cheeks` → both `skin_cheek_l` and `skin_cheek_r`
- `under the eye(s)` → both `eye_socket_l` and `eye_socket_r`
- `lip` / `lips` → both `lip_outer` and `lip_inner`
- `background` / `bg` / `behind` → `background`

---

## VALIDATION CHECKLIST (Python, runs before solver)

```python
def validate(llm_output: dict, overlay_regions: list, pigment_library: dict) -> list[str]:
    errors = []

    # 1. Every region_label exists in overlay
    overlay_labels = {r["label"] for r in overlay_regions}
    for rc in llm_output["region_constraints"]:
        if rc["region_label"] not in overlay_labels:
            errors.append(f"unknown region: {rc['region_label']}")

    # 2. Every pigment_id exists in library
    valid_ids = set(pigment_library.keys())
    for rc in llm_output["region_constraints"]:
        for pid in rc["preferred_pigment_ids"] + rc["forbidden_pigment_ids"]:
            if pid not in valid_ids:
                errors.append(f"unknown pigment_id: {pid}")
    for loc in llm_output["layer_order_constraints"]:
        for pid in (loc["top_pigment_id"], loc["under_pigment_id"]):
            if pid not in valid_ids:
                errors.append(f"unknown pigment_id in layer order: {pid}")

    # 3. No pigment in both preferred and forbidden of same region
    for rc in llm_output["region_constraints"]:
        overlap = set(rc["preferred_pigment_ids"]) & set(rc["forbidden_pigment_ids"])
        if overlap:
            errors.append(f"region {rc['region_label']} has conflicting prefs/forbids: {overlap}")

    # 4. If exclusive_hue, every preferred pigment must be in that family
    excl = llm_output["global_preferences"]["exclusive_hue_family"]
    if excl != "none":
        for rc in llm_output["region_constraints"]:
            for pid in rc["preferred_pigment_ids"]:
                pigment_hue = pigment_library[pid]["hue_family"]
                if pigment_hue != excl and pigment_hue != "neutral":
                    errors.append(
                        f"exclusive_hue={excl} but pigment {pid} (region {rc['region_label']}) "
                        f"has hue_family={pigment_hue}"
                    )

    # 5. Layer order constraints reference regions that have those pigments preferred
    for loc in llm_output["layer_order_constraints"]:
        rc = next((r for r in llm_output["region_constraints"] if r["region_label"] == loc["region_label"]), None)
        if rc and loc["top_pigment_id"] not in rc["preferred_pigment_ids"]:
            errors.append(
                f"layer order says {loc['top_pigment_id']} on top in {loc['region_label']}, "
                f"but it's not in that region's preferred_pigment_ids"
            )

    return errors
```

If `errors` is empty → ship to solver. If non-empty → re-call Opus with `errors` in the conversation as a tool_result, ask for a corrected output. Loop max 3 times before surfacing to user.

---

## EVALUATION

How to know the LLM translator is working:

1. **Gold-set regression** — Reid hand-writes 10–15 prompts + expected constraint outputs. Run translator nightly via cron; alert on field-level diff.
2. **Solver-feasibility rate** — fraction of LLM outputs that pass validation on the first call. Target: >95%. Below that, prompt engineer the system prompt.
3. **dE delta vs. naive baseline** — compare end-to-end solver dE on (a) full LLM-translation pipeline, (b) naive "translate prompt to just-named-pigments, no other structure". If (a) is not meaningfully better than (b), the structured-extraction effort is wasted.
4. **Ambiguity audit** — for each gold prompt, manually check that `ambiguities_flagged` lists every reasonable interpretation choice. Missed ambiguities are silent-failure-mode bugs.

---

## OUT OF SCOPE (deliberately)

- **Multi-turn clarification dialogue** — single MCP-call, single LLM-call, single response. No back-and-forth.
- **OPRO-style iterative refinement** — future v3. V2 ships one-shot translation.
- **Fine-tuning** — Anthropic doesn't offer it for Opus. And it would be premature optimization — Opus + good prompt is enough.
- **Vision input** — current MCP design takes the image as a file, but the LLM doesn't see it as visual content. Optional v2.5 enhancement: feed the image to Claude vision so it can ground the overlay labels by inspecting actual pixel regions. Adds tokens, may help with ambiguity ("which cheek is shadowed?"). Defer.
- **Pigment-mix recommendations** — V2 LLM emits single pigment IDs only. Mixing is the solver's job. Don't ask the LLM to do pigment proportions.

---

## SHIPPABLE V1 SPEC SUMMARY

| Component | Decision |
|---|---|
| Model | `claude-opus-4-7` (Anthropic Max plan) |
| Tool count | 1 tool: `translate_artistic_intent` |
| Tool call mode | `tool_choice: {"type": "tool", "name": "translate_artistic_intent"}` |
| Schema enforcement | `strict: true` |
| In-context examples | 3–5 via `input_examples` field |
| System prompt size | ~3K tokens (pigment table + region vocab + modifier conventions + few-shot) |
| User input | overlay regions JSON + pigment library JSON + user prompt string |
| Validator | Python, runs between LLM output and solver, 5-check list above |
| Retry policy | max 3 LLM re-calls on validation failure, then surface to Reid |
| Ambiguity disclosure | always populate `ambiguities_flagged`, never ask clarifying question |
| Fallback model | none for V1 (Opus is fast enough; ~3-5s per call) |
| Caching | client-side: cache LLM output keyed by `hash(overlay + library + prompt)` |
| Cost | ~$0.10/call metered; $0/call on Max plan |
