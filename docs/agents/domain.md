# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root.
- `docs/adr/` entries that touch the area being changed.

If either path is missing in a future branch, proceed silently and avoid inventing domain language.

## Layout

This is a single-context repo:

```text
/
├── CONTEXT.md
└── docs/adr/
```

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, test name, or PRD, use the term as defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

For this repo, especially prefer **Block**, **Impression**, **Mask**, **Pigment**, **Order**, **Underprint**, **Review preview**, and **Validator truth**.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly instead of silently overriding it.
