# claude -p transport for chuck-mcp — operations notes

Research agent: `CLAUDE-P-TRANSPORT` for swarm `swarm-1778978903817-bqgh16`.
Verified against Claude Code v2.1.129 (live `claude` binary at
`/home/reidsurmeier/.npm-global/bin/claude`), 2026-05-16.

Reference implementation: `./claude_p.py` (end-to-end tested — 2/2 prompts
returned schema-valid structured output, log captured, sub-$0.60/call).

---

## 1. Verified flag set

```bash
claude -p \
  --output-format json \
  --json-schema '<JSON_SCHEMA_STRING>' \
  --max-turns 3 \
  --no-session-persistence \
  --permission-mode dontAsk \
  --disallowedTools "Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep" \
  --append-system-prompt '<JSON-only constraint>' \
  '<USER_PROMPT>'
```

Why these and not others:

| Flag | Why |
|---|---|
| `-p` / `--print` | non-interactive; the only mode that supports JSON output / json-schema / max-budget. |
| `--output-format json` | single envelope on stdout. Without it `--json-schema` is silently dropped (verified). |
| `--json-schema <s>` | the harness validates output against the schema AND retries internally up to `error_max_structured_output_retries`. Top-level `structured_output` field is populated on success. |
| `--max-turns 3` | **MANDATORY ≥ 2** with `--json-schema`. Schema enforcement runs as an internal tool-use round; max-turns=1 immediately returns `error_max_turns`. 3 leaves one validator-retry budget. |
| `--no-session-persistence` | chuck-mcp is stateless per-call; no need to write JSONL transcripts under `~/.claude/projects/`. |
| `--permission-mode dontAsk` | deny anything not pre-allowed. Since chuck-mcp wants ONLY structured output, no tool execution is needed. |
| `--disallowedTools "..."` | belt-and-suspenders. Explicitly forbid Bash/Edit/Read/Write/WebFetch/WebSearch/Glob/Grep so the model can't accidentally try to "be helpful" by reading the cwd or hitting the web. |
| `--append-system-prompt` | re-emphasizes "JSON only, no prose" on top of Reid's default system prompt (which includes verbose CLAUDE.md context). |

**Flags we deliberately DO NOT use:**

| Flag | Reason |
|---|---|
| `--bare` | Skips OAuth and keychain reads. Forces `ANTHROPIC_API_KEY` or `apiKeyHelper`. **Breaks Reid's Max subscription billing.** Re-evaluate after 2026-06-15 when Anthropic splits Agent SDK credits. |
| `--dangerously-skip-permissions` | Pointless when no tools should run anyway. `--permission-mode dontAsk` + `--disallowedTools` is the safer pair. |
| `--max-budget-usd` | Currently a no-op for subscription users; budget enforcement is per-MAX-plan, not per-call. Re-add post 2026-06-15. |
| `--continue` / `--resume` | chuck-mcp is request/response, not multi-turn dialogue. |
| `--include-partial-messages` / `--output-format stream-json` | We want a single envelope, not streaming events. |
| `--verbose` / `--debug` | Adds noise to stdout. The CLI mixes verbose chatter with JSON in some debug modes. |

---

## 2. JSON envelope reality check (live capture)

The `--output-format json` envelope returns these fields (verified):

```json
{
  "type": "result",
  "subtype": "success",                  // or "error_max_turns", "error_max_structured_output_retries", "error_during_execution"
  "is_error": false,
  "api_error_status": null,
  "duration_ms": 35858,
  "duration_api_ms": 14755,
  "num_turns": 2,
  "result": "",                          // free-text body; empty when --json-schema is used and structured_output is populated
  "structured_output": {                 // only present with --json-schema
    "intent": "draw",
    "medium": "pencil",
    "confidence": 0.95
  },
  "stop_reason": "end_turn",
  "session_id": "1233aa7e-...",
  "total_cost_usd": 0.55,
  "usage": { "input_tokens": ..., "cache_creation_input_tokens": ..., "cache_read_input_tokens": ..., "output_tokens": ... },
  "modelUsage": {
    "claude-opus-4-7[1m]": { "inputTokens": ..., "costUSD": 0.43 },
    "claude-haiku-4-5":    { "inputTokens": ..., "costUSD": 0.20 }
  },
  "permission_denials": [],
  "terminal_reason": "completed",        // or "max_turns"
  "fast_mode_state": "off",
  "uuid": "57d49afc-..."
}
```

Subtype semantics (success vs error vs retriable):

| `subtype` | Meaning | Retriable? |
|---|---|---|
| `success` | `structured_output` is valid against schema | n/a |
| `error_max_turns` | hit `--max-turns` before producing a valid response | yes (raise `--max-turns`) |
| `error_max_structured_output_retries` | harness gave up after internal schema retries | yes once (model may pick up on lower temperature / better prompt) |
| `error_during_execution` | tool-use or API failure mid-turn | yes |
| `rate_limit`, `server_error`, `overloaded` (in `result` string) | upstream Anthropic transient | yes with backoff |
| `authentication_failed`, `oauth_org_not_allowed`, `invalid_request` | fatal | no |

`is_error: true` is also set on `result: "Not logged in · Please run /login"` — important: this is how OAuth-session expiry surfaces.

---

## 3. Retry strategy

`claude_p.py` does two layers of retry:

1. **Internal (CLI harness)** — `--json-schema` runs its own validator on the model output. We allocate `max_turns=3` (one tool-use round for validation + one retry budget). Default retry count inside the harness is controlled by the CLI; if it exhausts, we get `error_max_structured_output_retries`.

2. **External (Python)** — `translate_intent_prompt` retries on:
   * `ClaudePTimeoutError`
   * `result.is_error=true` with subtype in `{error_max_turns, error_max_structured_output_retries, error_during_execution}`
   * `rate_limit / server_error / overloaded` substrings in the error message
   * Empty stdout
   * Stage-1 JSON parse failure

   Default `max_retries=1` (2 attempts total). One retry is plenty — chuck-mcp's intent classification is structurally simple, and each retry costs ~$0.55. **DO NOT** crank this above 2 retries on Max subscription; runaway retry loops will eat the monthly cap.

   **No exponential backoff** between retries. The CLI already has its own backoff on transient API errors (emits `system/api_retry` events in stream-json mode). Retrying immediately is fine.

Schema-violation that gets through both layers is unrecoverable — raise
`ClaudePSchemaError` and let the MCP caller decide how to respond (return
an error to the MCP client, fallback to a default intent, etc.).

---

## 4. Subprocess timeout

**Recommended: 120s wall-clock per call.**

Live observations (subscription tier, 1M context, Opus 4.7):

| Quantile | Wall time | Reason |
|---|---|---|
| p50 | ~35–40s | Normal schema call |
| p90 | ~60s | Cold-cache call with full CLAUDE.md context (first call after restart) |
| p99 | ~75–90s | Tool-use retry or cache rebuild |

Notes:
* `duration_ms` (API time) is usually 15–20s; the extra wall-time is CLI
  bootstrap (plugin sync, settings load, MCP server probes). Adding `--bare`
  cuts this in half but breaks OAuth — accept the tax for v1.
* Hard ceiling 120s. Anything past that means the call is wedged; kill and
  retry. Python's `subprocess.run(timeout=120)` raises `TimeoutExpired`
  which we map to `ClaudePTimeoutError`.

---

## 5. Concurrent calls

chuck-mcp will rarely see parallel `translate_intent_prompt` calls (one MCP
client typically issues one at a time). The reference impl uses a single
module-level `threading.Lock` (`_CALL_LOCK`) to serialize subprocesses.

Reasons:
* Subscription-tier rate-limits are aggressive and per-second.
* Each call costs $0.40–$1.00. Two parallel calls = double burn for no
  user-visible win.
* Claude Code session JSONL files are per-cwd; concurrent calls in the
  same cwd would clobber each other (we use `--no-session-persistence`
  but the safety net is cheap).

If chuck-mcp ever needs parallel intent translation (batch processing),
swap the lock for an `asyncio.Semaphore(1)` and use `asyncio.create_subprocess_exec`
instead of `subprocess.run`. Don't raise concurrency above 2 — see the
ruflo memory note (`reference_ruflo_claude_code_executor.md`) where Reid
already pinned a single-flight pattern for the same reason.

---

## 6. V2 migration path — subprocess → `anthropic.Anthropic()`

The whole point of `translate_intent_prompt(text, schema) -> dict` having
a stable signature is the v2 swap is one file.

After 2026-06-15 (Anthropic splits subscription + Agent SDK credits) or
once chuck-mcp gets its own `ANTHROPIC_API_KEY` budget:

```python
# v2 claude_p.py — drop-in replacement
from anthropic import Anthropic

_CLIENT = Anthropic()  # picks up ANTHROPIC_API_KEY from env

def translate_intent_prompt(text, schema, **_kw) -> ClaudeResult:
    msg = _CLIENT.messages.create(
        model="claude-opus-4-7-20260101",  # or whatever the current id is
        max_tokens=512,
        system=_SCHEMA_SYSTEM_PROMPT,
        tools=[{
            "name": "emit_intent",
            "description": "Emit the structured intent.",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": "emit_intent"},
        messages=[{"role": "user", "content": text}],
    )
    tool_use = next(b for b in msg.content if b.type == "tool_use")
    return ClaudeResult(
        structured_output=tool_use.input,
        raw_envelope={
            "session_id": msg.id,
            "total_cost_usd": _estimate_cost(msg.usage),
            "duration_ms": 0,  # not reported by REST
            "num_turns": 1,
            "modelUsage": {"...": msg.usage.__dict__},
        },
    )
```

Differences callers should know about:
* No session JSONL persistence ever; no `--continue` semantics.
* `duration_ms` will be 0 (not tracked by REST). Use Python wall-time.
* `total_cost_usd` must be computed client-side from token counts × price.
* No CLAUDE.md auto-injection. The system prompt is exactly what you pass.
* Cost is **~30x lower** per call (~$0.02 vs ~$0.55) because no cached
  CLAUDE.md prefill. This is the main reason v2 exists.

---

## 7. Failure modes + mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| OAuth session expired | envelope `is_error: true, result: "Not logged in · Please run /login"` | Surface in audit log as `auth_expired`; ask user to run `claude` interactively to refresh. Not retriable. |
| `--bare` mistakenly added | Same `Not logged in` error in seconds | Don't pass `--bare`. Verified in this research. |
| `--max-turns 1` with `--json-schema` | `error_max_turns` immediately, no `structured_output` | Always set max-turns ≥ 2 (claude_p enforces this in `translate_intent_prompt`). |
| Schema too complex / required fields can't be filled | `error_max_structured_output_retries` after multiple internal retries | Simplify schema; mark uncertain fields as optional; lower nesting. |
| Model wanders into prose ("Sure! Here's the JSON: ```json...```") | `is_error: false` but `structured_output` is `null` and `result` has prose | We require `--json-schema` so this should never happen, but `_parse_envelope` + `_validate_against_schema` catch it. Raise `ClaudePSchemaError`. |
| Timeout / process hang | `ClaudePTimeoutError` after 120s | Kill, retry once, then bubble. |
| Cache eviction / cold start | First call after `claude` restart is ~60s; subsequent ones ~35s | Accept it. No fix without `--bare`. |
| stdin attached interactively | `claude -p` may hang reading TTY | We always pass `stdin=subprocess.DEVNULL`. |
| Rate limit on subscription tier | envelope error containing `rate_limit` | Retry once with no backoff (the CLI already retried internally). If second attempt also rate-limits, bubble. |
| Anthropic outage / `overloaded` | `server_error` or `overloaded` in error string | Retry once. If still down, return a 503-equivalent to the MCP client. |
| stdout corruption (rare) | `json.JSONDecodeError` in `_parse_envelope` | Log head-500-chars of stdout; retry once. |
| Audit log disk-full | Logged at WARN level, doesn't fail the call | Rotate `~/.chuck-mcp/claude-p-calls.log` weekly (cron or systemd-tmpfiles). |

---

## 8. Cost economics

Per-call cost on Reid's Max subscription, observed:

| Scenario | Cost | Driver |
|---|---|---|
| Simple intent classification | $0.43–$0.55 | Opus 4.7 base call + cache-creation tax for CLAUDE.md context |
| With cache hit (rapid consecutive calls) | $0.30–$0.45 | Cache reads dominate |
| With one schema-retry | $0.70–$1.00 | Doubled cache_creation |

**Subscription Max is $100/mo flat → ~180 calls/mo headroom at p50.**

For chuck-mcp this is fine — intent translation is rare (one per MCP
invocation, and MCP invocations are user-initiated). For high-volume
batched workloads, the v2 REST path (~$0.02/call, ~$5,000/mo headroom on
the same budget) is mandatory.

Effective levers if cost becomes a problem before v2:
1. `--bare` + a dedicated `ANTHROPIC_API_KEY` (no Max billing) → cuts cost
   ~30x but defeats the subscription point.
2. Add `claude_p` result memoization on `(text, schema_hash)`. Many MCP
   clients will issue the same intent translation repeatedly.
3. Move from Opus to Sonnet via `--model sonnet` (~6x cheaper, slightly
   worse JSON adherence).

---

## 9. Smoke test — how to verify

```bash
cd research/v3-construction/claude-p-transport/
python3 claude_p.py one "I want to sketch a cat with a pencil"
# expected: prints {"intent": "draw", "medium": "pencil", "subject": "cat", "confidence": 0.9x}

python3 claude_p.py test --count 2
# runs 2 of 4 built-in prompts. expected: failures=0/2.

tail -1 ~/.chuck-mcp/claude-p-calls.log | python3 -m json.tool
# verify audit log captured the call with session_id, cost_usd, duration_ms.
```

Live smoke results (2026-05-16):
```
--> 'I want to sketch a cat with a pencil'
    {'intent': 'draw', 'medium': 'pencil', 'subject': 'cat', 'confidence': 0.95}
    cost=$0.5494  dur=35858ms  turns=2

--> 'paint a sunset over the ocean in watercolor'
    {'intent': 'paint', 'medium': 'watercolor', 'subject': 'sunset over the ocean', 'confidence': 0.98}
    cost=$0.5682  dur=35495ms  turns=2
```

---

## 10. Open questions for the swarm

* Does chuck-mcp need to expose `total_cost_usd` and `session_id` back to
  the MCP client (for observability) or hide them inside the transport?
  Current `ClaudeResult` exposes both via properties.
* Should we add `~/.chuck-mcp/claude-p-calls.log` rotation to systemd
  tmpfiles, or punt to OpenClaw cron?
* Reid's `reference_ruflo_claude_code_executor.md` notes the ruflo patch
  resets on `npm update`. chuck-mcp's `claude_p.py` is self-contained so
  no patch dependency, but if we ever shell out via ruflo instead of
  directly, we inherit that fragility.
