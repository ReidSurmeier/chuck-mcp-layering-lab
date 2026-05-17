# Negation Handling in LLM-Based Constraint Extraction

For chuck-mcp v2 negation matters: Reid's example prompt has "NOT magenta", "only red in the image" — both *exclusion* constraints that radically change the solver's feasible set.

## Findings from the negation literature

### LLMs are bad at negation as a class (still in 2026)

Multiple negation benchmarks since 2022 show LLMs systematically underweight `not`, `never`, and sub-clausal negation.

- **xNot360** (Nguyen 2023, arXiv 2306.16638): GPT-4 zero-shot negation detection is much better than GPT-3.5 but still well below human accuracy on sentence pairs where negation is the only difference.
- **Thunder-NUBench** (So et al. 2025, arXiv 2506.14397): even modern instruct-tuned LLMs confuse "standard negation" with "contradiction" and "paraphrase" — multiple-choice format reveals this.
- **NeIn** (Bui et al. 2024, arXiv 2409.06481): vision-language models given image-editing instructions with negation ("a sunset *without* clouds") consistently include the negated element. Direct analogue of "NOT magenta" failure mode.
- **This is not a Dataset** (García-Ferrero et al. 2023, arXiv 2310.15941): even fine-tuning doesn't close the gap.

### Implication for chuck-mcp v2

**Do not trust the LLM to internalize "NOT magenta" implicitly.** Force the LLM to emit explicit `forbidden_pigments` / `forbidden_hue_families` fields. The downstream solver enforces them as hard constraints.

## Recommended schema-level mitigations

1. **Mandatory `forbidden_pigments` field** — always present, even if empty. Forces the LLM to consider it on every prompt.
2. **Mandatory `forbidden_hue_families` field** — same. Hue families: red / orange / yellow / green / blue / violet / neutral.
3. **`global.exclusive_hue` field** — when prompt says "only red in the image", emit `exclusive_hue: "red"` and the solver hard-bans all other hue families globally.
4. **`ambiguities_flagged` field must surface negations the LLM was unsure about.** Example: "NOT magenta — assumed magenta = M30–M50 hue range, ban pigments with hue in that range".

## Recommended prompting mitigations

From the contrastive in-context literature (Gao & Das 2024, arXiv 2401.17390 — "Customizing Language Model Responses with Contrastive In-Context Learning"): **contrastive examples** where positive and negative versions of the same prompt sit side-by-side improve negation handling. For chuck-mcp v2's system prompt, include 2–3 `input_examples` like:

```json
// Example 1: positive only
{"user_prompt": "vermilion lip", ...}
// Example 2: with negation
{"user_prompt": "vermilion lip NOT magenta", ...}
// Example 3: global exclusivity
{"user_prompt": "vermilion lip, only red in image, no other hues", ...}
```

## Solver-side hard guarantee

Even if the LLM misses a negation, the solver should refuse to produce a result that violates `forbidden_pigments`. Two layers:

1. **Validation step (Python, runs between LLM output and solver):**
   - Check that every pigment in `preferred_pigments` exists in the YAML.
   - Check that no pigment appears in *both* preferred and forbidden lists.
   - If `exclusive_hue` set, compute every pigment's hue family and pre-filter the YAML to that subset.
2. **Solver constraint:** the inverse-stack solver's search space excludes forbidden pigments entirely. Not a soft penalty — a hard exclusion in the variable index.

This belt-and-suspenders pattern is the standard recommendation in the LLM-agent-reliability literature (e.g., Failure Modes in LLM Systems, arXiv 2511.19933 — flags "incorrect tool invocation" as a top-5 production failure mode).

## Sources

- [This is not a Dataset (arXiv 2310.15941)](https://arxiv.org/abs/2310.15941)
- [Thunder-NUBench (arXiv 2506.14397)](https://arxiv.org/abs/2506.14397)
- [NeIn — Telling What You Don't Want (arXiv 2409.06481)](https://arxiv.org/abs/2409.06481)
- [GPT negation detection xNot360 (arXiv 2306.16638)](https://arxiv.org/abs/2306.16638)
- [Contrastive In-Context Learning (arXiv 2401.17390)](https://arxiv.org/abs/2401.17390)
- [Failure Modes in LLM Systems (arXiv 2511.19933)](https://arxiv.org/abs/2511.19933)
