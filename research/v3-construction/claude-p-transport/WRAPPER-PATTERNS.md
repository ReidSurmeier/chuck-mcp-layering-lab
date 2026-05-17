# Python subprocess wrapper patterns for `claude -p` — survey + lessons

Survey of how existing projects shell out to `claude -p` from a Python (or
Node) long-running service, and what patterns we should adopt for
chuck-mcp.

## 1. ruflo / claude-flow `agent-execute-core.js` (Reid's local install)

Path: `/home/reidsurmeier/.npm-global/lib/node_modules/ruflo/node_modules/@claude-flow/cli/dist/src/mcp-tools/agent-execute-core.js`

**Stock pattern (Tier-3): direct REST to `api.anthropic.com/v1/messages`**.
Not subprocess. Requires `ANTHROPIC_API_KEY`.

```javascript
const res = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: { 'x-api-key': anthropicKey, 'anthropic-version': '2023-06-01' },
  body: JSON.stringify({ model, max_tokens, system, messages }),
  signal: controller.signal,
});
```

Has a `AbortController` timer (60s default), explicit fallback to Ollama
when no Anthropic key, normalized response shape.

**Reid's patch (per `reference_ruflo_claude_code_executor.md`):**
when `RUFLO_EXECUTOR=claude-code`, instead of REST it shells out to
`claude -p` and parses the result. Patch is on `agent-execute-core.js.bak`
and **wiped on `npm update`**. This is the precise mechanism we're
re-implementing for chuck-mcp, except for Python and not patched onto
someone else's code.

**Lessons applied to `claude_p.py`:**
* Same 60s default timeout (we use 120s because schema calls are slower).
* Same "normalize result shape" pattern → our `ClaudeResult` dataclass.
* Same provider-fallback shape → our v1→v2 swap.
* Avoid Reid's patch problem: chuck-mcp owns `claude_p.py` directly. No
  patches against vendor files.

## 2. avasdream blog — "Wrapping Claude CLI for Agentic Applications"

URL: https://avasdream.com/blog/claude-cli-agentic-wrapper

Key recommendations:
* Prefer subprocess over SDK: *"Subprocesses are debuggable. When something
  breaks, I can run the exact same command manually."*
* `--output-format json` + check `is_error` field (don't rely on rc).
* Use `--max-turns`, `--max-budget-usd`, permission modes to constrain.
* `--dangerously-skip-permissions` only in sandboxes.

Pitfalls called out:
* Session IDs MUST be valid UUIDs.
* `--output-format stream-json` requires `--verbose`.
* `--json-schema` requires `--output-format json` or the field is silently
  dropped.
* Tool patterns use prefix matching — `Bash(git diff *)` (with space)
  matches `git diff`, but `Bash(git diff*)` (no space) also matches
  `git diff-index`.

Omits: explicit timeout/retry implementation. Leaves it to the caller.

**Lessons applied:**
* We parse the envelope and check `is_error`, never trust rc.
* We use `--max-turns` (3) but skip `--max-budget-usd` (no-op for subs).
* `claude_p.py` is the explicit timeout/retry implementation the blog
  punts on.

## 3. `claude-headless-mode` Smithery skill

URL: https://smithery.ai/skills/majesticlabs-dev/claude-headless-mode

Confirms the canonical pattern:
```bash
claude -p "PROMPT" \
  --output-format json \
  --json-schema '<schema>' | jq '.structured_output'
```
Skill is a thin reference, not a production wrapper. Same gotchas: schema
requires JSON output format, exit codes are 0/1.

## 4. Anthropic's own `claude-agent-sdk-python` (the official path)

GitHub: https://github.com/anthropics/claude-agent-sdk-python

The Python SDK actually wraps the same `claude` binary internally (Node
SDK bundles it as an optional dep; Python SDK spawns the binary at
runtime). API surface:

```python
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async for message in query(
    prompt="...",
    options=ClaudeAgentOptions(
        output_format={"type": "json_schema", "schema": schema}
    ),
):
    if isinstance(message, ResultMessage) and message.structured_output:
        ...
```

Notable features we deliberately do NOT use for v1:
* Async iterator interface (we want blocking semantics inside the MCP tool
  handler — async would complicate the call site).
* `session_store_flush` for real-time transcript mirroring.
* Hook callbacks (`PreToolUse`, `PostToolUse`).
* Permission callbacks.
* Trio compatibility.

Notable features we'd love but can't use on Max subscription:
* `max_budget_usd` parameter (subscription billing ignores it).
* W3C `TRACEPARENT`/`TRACESTATE` distributed tracing.

**Why not just use this SDK?** Three reasons:
1. SDK calls go through `claude` binary the same way subprocess does, so
   subprocess is no worse and is fully debuggable from a shell prompt.
2. SDK pins specific `claude-agent-sdk-python` versions; chuck-mcp wants
   to be deployable without that dependency.
3. The v2 migration path (direct REST) is cleaner if v1 is also "thin
   subprocess wrapper" rather than "SDK abstraction". Less to unlearn.

If chuck-mcp grows to need streaming/hooks, swap to the SDK then.

## 5. GitHub issue #9058 — feature request: guaranteed JSON schema compliance

URL: https://github.com/anthropics/claude-code/issues/9058

User complaint: current `--json-schema` provides best-effort validation
with retry, not constrained decoding. Production users want 100%
guarantee (like OpenAI Structured Outputs).

**Status today (May 2026):** Anthropic's `--json-schema` retries internally
on schema-violation up to a built-in limit, then surfaces
`error_max_structured_output_retries`. We have observed zero violations in
chuck-mcp smoke tests but the failure mode is real for complex schemas.

**Mitigation in `claude_p.py`:** even on `success`, we re-validate
`structured_output` against the schema client-side. Catches the rare case
where Anthropic's validator and our local `jsonschema` disagree (oneOf,
$ref, format edge cases).

## 6. Reid's `feedback_subprocess_wrapper_observability.md` memory

The 4 rules Reid wants every subprocess wrapper to follow:

| Rule | Implementation in `claude_p.py` |
|---|---|
| explicit cwd | `_run_claude_subprocess(cwd=run_cwd or str(Path.cwd()))` |
| `stdin=DEVNULL` | `subprocess.run(..., stdin=subprocess.DEVNULL)` |
| tee output to persistent file | `_log_event()` writes JSON-line per call to `~/.chuck-mcp/claude-p-calls.log` |
| `subprocess_tail` event on rc != 0 | `event: "subprocess_failed"` log entry with `rc`, `stderr_head`, `stdout_head` |

## 7. Reid's `feedback_cron_claude_p_loop.md` memory

Reid already learned: *"Headless dispatchers MUST use
`--dangerously-skip-permissions` + `--max-budget-usd` + concrete
state-driven prompt + per-tick log. Vague prompts stall silently."*

That advice is for **autonomous loop dispatchers** (cron jobs running
`claude -p` repeatedly to make progress on a task). chuck-mcp is the
**opposite** — a single-shot, JSON-only call where:
* `--dangerously-skip-permissions` is unnecessary because we deny all
  tools anyway via `--permission-mode dontAsk`.
* `--max-budget-usd` is a no-op on subscription.
* The prompt is concrete by construction (user-supplied text + strict
  schema).
* Per-tick log → our `~/.chuck-mcp/claude-p-calls.log`.

We're applying the spirit (concrete prompt, per-call log, defensive
defaults), not the letter (different attack surface).

## 8. Consolidated production checklist

| Check | Status |
|---|---|
| Subprocess `stdin=DEVNULL`? | Yes |
| Explicit `cwd`? | Yes |
| Timeout enforced (Python-side)? | Yes (120s default) |
| `--output-format json`? | Yes |
| `--json-schema` validated client-side AND server-side? | Yes |
| Auth failures (`Not logged in`) detected and not retried? | Yes (via `is_error` + `result` substring) |
| Cost recorded per call? | Yes (audit log) |
| Concurrent-call lock? | Yes (single `threading.Lock`) |
| v2 swap path documented? | Yes (NOTES.md §6) |
| Module testable from CLI? | Yes (`python -m claude_p test`) |
