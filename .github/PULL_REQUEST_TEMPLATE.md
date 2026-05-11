<!-- v23-MCP PR template — every section required. Drop none. -->

## Summary

<!-- 1-3 sentences. What changed, why, which user/plan section justifies it. -->

## Build sequence step

- **Step ID:** D<N>.<n>  <!-- e.g. D5.3 -->
- **Plan section:** <!-- /mnt/c/Users/reidsurmeier2/Books/printmaking/v23/research-v23-mcp-plan-v2.1.md §<n> -->
- **Reference:** /tmp/research-v23-mcp-build-sequence.md row matching `D<N>.<n>`

## TDD evidence

- **Test file(s):** `backend/tests/v23/<path>`
- **Test command:**
  ```bash
  pytest backend/tests/v23/ -q
  ```
- **Counts at HEAD of this branch:** `<green>/<total> green` (paste `pytest ... | tail -1` output)
- **Red → green commit pair (if separated):** <commit-sha-red> → <commit-sha-green>

## CONTEXT.md compliance

- [ ] No banned terms introduced (`plate`, `separator`, `layer`, `pass`, `underbase`, `underlayer`, `detect underlayer`, `recover underprint`, `true hidden block`)
- [ ] No "Mixbox predicts the print" without `as if pre-mixed` qualifier (WB-LANG-02)
- [ ] If user-facing string mentions Mixbox, paragraph also contains `mixing` or `pre-mixed`
- [ ] Domain terms used per glossary (Impression for pass, Mask for region, Block for physical wood, etc.)

## Banned-terms grep result

```bash
grep -E -i -f scripts/banned_terms.txt backend/
```

Paste output (must be empty):

```
<paste here>
```

## Manifest schema

- [ ] Schema version unchanged at `v23.0`, OR
- [ ] ADR added under `docs/adr/` justifying bump
- [ ] Change is additive-only (no field rename/removal pre-ship)

## Rollback

- **Pre-merge tag:** `pre-v23-D<N>`  <!-- create with `git tag pre-v23-D<N>` before merging -->
- **Rollback command:**
  ```bash
  git reset --hard pre-v23-D<N>
  ```

## Reviewer checklist

- [ ] Code under 400 LOC per file (specialist context cap)
- [ ] Ruff + mypy clean
- [ ] No code in `main.py` or v20 paths touched (v20 untouched policy)
- [ ] PR body links the binding plan/addendum section
