# NL4Opt Competition — Formulating Optimization Problems from Natural Language

**NeurIPS 2022 competition, arXiv 2303.08233.** Single most-relevant prior art for chuck-mcp v2 prompt translation.

## Why this is the canonical reference

NL4Opt is *exactly* the task chuck-mcp v2 needs: a natural-language description of an optimization problem → structured representation a solver can consume. The competition is for linear programming, not color, but the entity types translate one-to-one to chuck-mcp's pigment / region / constraint vocabulary.

## Two-stage architecture (recommended for chuck-mcp v2)

```
NL text  →  Stage 1: Named Entity Recognition  →  Stage 2: Logical-form generation  →  Solver
```

**Stage 1 (NER):** Tag every word in the prompt with an entity type.
**Stage 2 (Logical form):** Compose tagged entities into a structured constraint object.

For chuck-mcp v2 these can collapse into a single LLM call (Claude is much stronger than the 2022-era models the competition used). Output a single JSON object via tool-call / strict JSON schema. Conceptually still two stages.

## NL4Opt entity types — direct analogues to chuck-mcp

| NL4Opt entity | NL4Opt meaning | chuck-mcp v2 analogue |
|---|---|---|
| `B-OBJ_DIR` / `I-OBJ_DIR` | objective direction (min/max) | implicit ("minimize dE", always min) |
| `B-OBJ_NAME` / `I-OBJ_NAME` | objective name (cost / profit) | implicit (perceptual dE between target and rendered) |
| `B-VAR` / `I-VAR` | decision variable | region-mask → pigment-amount mapping |
| `B-PARAM` / `I-PARAM` | numerical parameter | hex value, dE threshold, Lab coordinates |
| `B-LIMIT` / `I-LIMIT` | constraint bound | tone range upper / lower bound |
| `B-CONST_DIR` / `I-CONST_DIR` | constraint direction (≤, ≥, =) | "between X and Y", "NOT magenta", "preferred" |

The B-/I- prefix is BIO tagging (begin/inside) used by 2022 sequence models. With Claude as the LLM you skip BIO entirely and just emit JSON.

## chuck-mcp v2 entity types (derived from NL4Opt + Reid's example prompt)

Taking Reid's example prompt:
> "Warm-tonal Emma. Skin between #f7d8c2 and #e0a888, lean toward Asian-mineral pigments. Hair is deep umber under indigo. Lip is vermilion with soft pink underlayer NOT magenta. Keep red high-chroma — only red in the image."

Entities to extract:

- **REGION** — "skin", "hair", "lip" — must match a Photoshop overlay label
- **TONE_RANGE** — "between #f7d8c2 and #e0a888" → two hex anchors → solver gets a Lab box
- **PIGMENT_FAMILY_PREF** — "Asian-mineral pigments" — maps to pigment YAML metadata
- **PIGMENT_NAME_POSITIVE** — "vermilion", "indigo" — direct pigment references
- **PIGMENT_NAME_NEGATIVE** — "NOT magenta" — exclusion
- **LAYER_ORDER** — "vermilion with soft pink underlayer" — pigment A goes *on top of* pigment B
- **REGION_RELATION** — "hair under indigo" — block-pass-mask ordering (mokuhanga reduction)
- **GLOBAL_CHROMA_TARGET** — "keep red high-chroma"
- **GLOBAL_EXCLUSIVITY** — "only red in the image" — no other hue families
- **CONTENT_TAG** — "Warm-tonal Emma" — image-level descriptor (becomes a default if other fields underspecified)

This is the schema the v2 LLM call needs to produce.

## Stage-2 logical form (what the solver receives)

NL4Opt converts NER output to a "canonical form" — equivalent to chuck-mcp's solver inputs. For chuck-mcp this is:

```json
{
  "regions": [
    {
      "label": "skin",
      "tone_range_lab": {"L":[78, 88], "a":[8, 18], "b":[16, 28]},
      "preferred_pigment_families": ["mineral_japanese"],
      "preferred_pigment_ids": [],
      "forbidden_pigment_ids": []
    }
  ],
  "layer_order_constraints": [
    {"region":"lip","top":"shu","under":"beni_pink"}
  ],
  "global": {
    "exclusive_chroma_hue": "red",
    "high_chroma_regions": ["lip"]
  }
}
```

## Winners' approach (2022)

The 2022 winners used T5 + BART fine-tuned on the NL4Opt dataset. **Skipping fine-tuning entirely** is now viable with Claude Sonnet 4.6 + JSON-schema-strict mode. The compute / cost saving vs. fine-tuning is huge.

## Sources

- [NL4Opt competition page](https://nl4opt.github.io/)
- [Competition paper (arXiv 2303.08233)](https://arxiv.org/abs/2303.08233)
- [Ner4Opt follow-up (Constraints 2024)](https://link.springer.com/article/10.1007/s10601-024-09376-5)
- [Holy Grail 2.0 (arXiv 2308.01589)](https://arxiv.org/abs/2308.01589) — same problem, LLM-based approach
