# Anthropic Tool-Use Spec (Claude 4.x family)

Sourced from `platform.claude.com/docs/en/...` (current as of May 2026).

## Model recommendation, verbatim

> Use the latest Claude Opus (4.7) model for complex tools and ambiguous queries; it handles multiple tools better and seeks clarification when needed.
>
> Use Claude Haiku models for straightforward tools, but note they may infer missing parameters.

> If Claude Opus doesn't have enough context to fill in the required parameters, it is far more likely to respond with a clarifying question instead of making a tool call.

**Direct implication for chuck-mcp v2:** Use **Claude Opus 4.7** for the prompt-translation tool call. Reid's prompts are inherently ambiguous (artistic intent + Japanese aesthetic vocabulary). Haiku/Sonnet may silently guess; Opus will surface ambiguity. Cost is fine — Reid is on Anthropic Max $100/mo flat.

## Tool-definition structure

```json
{
  "name": "...",                       // ^[a-zA-Z0-9_-]{1,64}$
  "description": "detailed plaintext", // 3-4 sentences minimum
  "input_schema": { /* JSON Schema */ },
  "input_examples": [ /* optional, schema-valid */ ],
  "strict": true                       // guarantees schema validation
}
```

## Best-practice rules, verbatim

> **Provide extremely detailed descriptions.** This is by far the most important factor in tool performance.
>
> Aim for at least 3-4 sentences per tool description, more if the tool is complex.

> **Consolidate related operations into fewer tools.** Rather than creating a separate tool for every action … group them into a single tool with an `action` parameter.

> **Use meaningful namespacing in tool names.**

> Make parameters `required` where possible. Each optional parameter roughly doubles a portion of the grammar's state space.

> Mark only critical tools as strict. If you have many tools, reserve it for tools where schema violations cause real problems, and rely on Claude's natural adherence for simpler tools.

## Structured-output spec

Two complementary mechanisms:

1. **`output_config.format`** — controls the assistant's response format; ties to a JSON schema; gives **constrained-decoding-guaranteed** schema compliance.
2. **`strict: true` on a tool** — guarantees the tool-call args validate against the input_schema.

Verbatim:

> Structured outputs guarantee schema-compliant responses through constrained decoding:
> - **Always valid:** No more `JSON.parse()` errors
> - **Type safe:** Guaranteed field types and required fields
> - **Reliable:** No retries needed for schema violations

Limitations to be aware of for the chuck-mcp schema design:
- Max 20 strict tools per request
- Max 24 optional parameters total across all strict tools
- Max 16 parameters with union types
- **Not supported:** Recursive schemas, `$ref` to external files, numeric `minimum`/`maximum`, string `minLength`/`maxLength`
- → For chuck-mcp v2 we cannot rely on JSON Schema to bound `L*` to `[0, 100]` — must validate in solver-side Python.

## Pricing — same across all 4.x models

Tool use system prompt token count is **346 tokens** (or 313 for `any`/`tool` choice) for Opus 4.7, Sonnet 4.6, Haiku 4.5. Choice of model only affects per-token rates.

## Tool-choice control

- `auto` (default with tools) — Claude decides
- `any` — must use one of the tools
- `tool` (with `name`) — forces a specific tool
- `none` — no tool calls

For chuck-mcp v2: use `tool_choice: {"type": "tool", "name": "translate_artistic_intent"}` to guarantee the LLM emits the structured object on every call.

> Combine `tool_choice: {"type": "any"}` with strict tool use to guarantee both that one of your tools will be called AND that the tool inputs strictly follow your schema.

## input_examples is the underused power feature

```json
"input_examples": [
  { /* valid input matching the schema */ },
  { /* another, demonstrating optional fields */ }
]
```

Adds ~20–200 tokens per example, but is the most reliable way to teach Claude the *idiomatic* shape of constraint output. Use 3–5 examples covering: simple single-region prompt, multi-region with NOT constraint, layer-order specification, signature-color prompt ("Tiffany blue").

## Sources

- [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Implement tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
