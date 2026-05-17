# ColorGPT — Leveraging LLMs for Multimodal Color Recommendation

**arXiv 2508.08987** (Aug 2025, U Tokyo + CyberAgent) — Xia, Inoue, Qiu, Kikuchi.

## TL;DR

LLMs (GPT-4o, GPT-3.5-turbo, Llama3-8B) are competitive-to-SOTA designers for color palette completion (mask 1–3 of 5 colors) and full palette generation given text. The pipeline is **training-free** — just prompt engineering plus a vector DB of in-context exemplars.

## Direct lessons for chuck-mcp v2 prompt translation

### 1. Color representation in prompts: hex > everything for precision tasks

For palette completion (precision-critical), accuracy ranked:

| Representation | 1-color acc | 2-color acc | 3-color acc |
|---|---|---|---|
| **Hexcode** | **52.60%** | **31.64%** | **22.61%** |
| RGB triplet | 42.86% | 26.02% | 19.18% |
| Word(Hex)-H | 42.86% | 28.31% | 17.72% |
| CIELAB | 38.78% | 25.12% | 15.23% |
| Word(Hex)-W | 18.47% | 7.67% | 2.70% |
| Word only | 17.06% | 6.23% | 2.71% |

CIELAB underperforms because LLMs see far less CIELAB in training corpora than hex. **For chuck-mcp v2 the prompt anchors should use hex (`#f7d8c2`) and let downstream solver convert to Lab.** This matches Reid's example prompt exactly (`#f7d8c2 to #e0a888`).

For palette *generation* (subjective), `Word(Hex)-H` (e.g. "lime green (#32cd32)") gave best similarity to ground-truth (26.09 DCCW). This is the right format for *output* when the LLM must invent a pigment color — pair name + hex.

### 2. Structured prompt format: JSON > MLM mask, always

Switching from inline mask format (`[red, _, blue]`) to explicit JSON with title/category/keywords/layout/type/text/palette gave **+22% absolute accuracy** (30.43% → 52.60% on 1-color task). For chuck-mcp v2 the LLM's input *must* be a structured JSON describing pigments + region masks + prompt, not flat text concatenation.

### 3. Short task profile beats long task profile

`short` profile: 52.60% 1-color acc. `long` profile: 23.40%. **More instructions → worse.** Counter-intuitive but consistent with Reid's "caveman brevity" principle. The LLM has the commonsense — don't over-specify the task.

### 4. Similarity-retrieved exemplars >> random exemplars

`similarity-based` few-shot: 52.60% acc. `random` few-shot: 31.60%. **+21% absolute.** Build a FAISS index of past (prompt, mask, output constraints) tuples; retrieve top-k by text embedding similarity for in-context exemplars.

### 5. Model size matters less than you'd think (but matters)

For full palette generation:
- GPT-4o: 26.09 similarity (best)
- GPT-3.5-turbo: 27.93
- Llama3-8B: 33.34

Gap between top closed model and open 8B model is real but not catastrophic. For chuck-mcp v2, Claude Sonnet 4.6 (cheaper) will likely match Opus 4.7 quality for constraint extraction. Reserve Opus for ambiguous cases / failures.

### 6. Failure modes seen in ColorGPT (will apply to chuck-mcp)

- Small / repetitive elements get confused contextually — solution: explicit "this is for region X under the eye" wording
- RGBA/alpha confusion — chuck-mcp doesn't have this, but be careful when v2 emits hex codes that the recipient knows they are 100% opaque pigment refs, not RGBA
- Creative deviations from ground truth that are still good — accept these; constrain only via solver, not LLM

## Architecture takeaway for chuck-mcp v2

```
INPUT (MCP tool args)
  ├── image (PNG/PSD)
  ├── overlay (PSD with 4-9 labeled masks: "skin", "hair_under_indigo", "lip_outer", ...)
  └── prompt_text (NL artistic intent)
        │
        ▼
LLM step 1: PROMPT TRANSLATION (this research domain)
  Input prompt (structured JSON):
    {
      "image_summary": "<vision summary or empty>",
      "regions": [{"label":"skin","area_px":12345}, ...],
      "pigment_library": [<15-25 pigments from YAML, each with hex + name>],
      "user_prompt": "Warm-tonal Emma. Skin between #f7d8c2 and #e0a888 ..."
    }
  Output (structured JSON constraints — schema in NOTES.md):
    {
      "region_constraints": [
        {"region":"skin","tone_range_lab":{...},"preferred_pigments":["sumi_yellow","bengara"],"forbidden_pigments":["magenta_lake"]}, ...
      ],
      "global_preferences": {"chroma_emphasis":"red","family":"mineral_japanese"},
      "ambiguities_flagged": ["warm-tonal — assumed YR hue family"]
    }
        │
        ▼
SOLVER step (existing JAX L-BFGS-B inverse-stack)
```

## Source

[ColorGPT paper on arXiv](https://arxiv.org/abs/2508.08987) — accepted to ECCV 2024 workshop equivalent (paper text suggests journal pipeline).
