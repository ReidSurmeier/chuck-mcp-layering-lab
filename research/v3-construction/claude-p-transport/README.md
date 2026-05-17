# claude-p-transport — research deliverables

Agent: `CLAUDE-P-TRANSPORT`
Swarm: `swarm-1778978903817-bqgh16`
Date: 2026-05-16
Status: complete, reference impl tested end-to-end

## Contents

| File | Purpose |
|---|---|
| `claude_p.py` | **Working reference implementation.** Subprocess wrapper exposing `translate_intent_prompt(text, schema) -> ClaudeResult`. End-to-end tested (2/2 smoke prompts returned valid structured output, audit log captured, ~$0.55/call). Runnable via `python3 -m claude_p test`. |
| `NOTES.md` | Operations notes: verified flag set, retry strategy, timeouts, concurrency, v2 migration, failure modes, cost economics. **Read this first.** |
| `CLI-FLAGS.md` | Full `claude -p` flag matrix with "chuck-mcp uses it?" column. Documents why each include/exclude. |
| `ENVELOPE-SCHEMA.md` | Live-verified JSON shape of `claude -p --output-format json` envelope. With 4 captured samples in `samples/`. |
| `WRAPPER-PATTERNS.md` | Survey of existing claude -p wrappers (ruflo, claude-agent-sdk-python, avasdream blog) and which patterns we adopted. |
| `INTEGRATION.md` | How to wire `claude_p.py` into the chuck-mcp Python MCP server. Schema authoring guidance. Operations runbook. |
| `samples/` | Live JSON envelope captures from `claude -p` invocations during research. |

## TL;DR

**Verified flag set chuck-mcp should use:**

```bash
claude -p \
  --output-format json \
  --json-schema '<schema>' \
  --max-turns 3 \
  --no-session-persistence \
  --permission-mode dontAsk \
  --disallowedTools "Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep" \
  --append-system-prompt 'JSON only, no prose, no fences.' \
  '<user_prompt>'
```

**Status of `claude_p.py`:** runs, parses JSON, validates against schema,
logs every call to `~/.chuck-mcp/claude-p-calls.log`, retries on
retriable errors. Smoke test:

```
--> 'I want to sketch a cat with a pencil'
    {'intent': 'draw', 'medium': 'pencil', 'subject': 'cat', 'confidence': 0.95}
    cost=$0.5494  dur=35858ms  turns=2

--> 'paint a sunset over the ocean in watercolor'
    {'intent': 'paint', 'medium': 'watercolor', 'subject': 'sunset over the ocean', 'confidence': 0.98}
    cost=$0.5682  dur=35495ms  turns=2

failures=0/2
```

**Top 3 must-reads (in order):**

1. `NOTES.md` — verified flag set, why each is set/unset, failure modes,
   v2 migration path.
2. `INTEGRATION.md` — concrete MCP server wiring sketch, schema authoring
   rules, operations runbook.
3. `ENVELOPE-SCHEMA.md` — exactly what `claude -p --output-format json`
   returns (with live captures).

## Critical findings

1. **DO NOT use `--bare`.** It forces `ANTHROPIC_API_KEY`/apiKeyHelper
   and breaks Reid's Max subscription billing. Verified live: returns
   "Not logged in · Please run /login" in 67ms. (`samples/02-bare-not-logged-in-error.json`)

2. **`--json-schema` REQUIRES `--max-turns ≥ 2`.** Schema validation
   consumes one tool-use round; max-turns=1 always returns
   `error_max_turns`. Default to 3 to leave one validator retry budget.

3. **`--json-schema` REQUIRES `--output-format json`.** Without it the
   schema is silently dropped and you get the model's raw text in
   `result` instead of a parsed object in `structured_output`.

4. **Cost is ~$0.50/call on Max subscription** because the non-bare path
   pays cache-creation tax for Reid's full CLAUDE.md + plugin context.
   ~180 calls/mo headroom at $100/mo. v2 REST swap drops this ~30x.

5. **Each call's envelope contains `is_error: bool`. Trust the envelope,
   not the exit code.** Auth failures, max-turns, and rate-limits all
   return rc=0 + `is_error: true`.

6. **June 15, 2026: Anthropic splits subscription + Agent SDK credits.**
   chuck-mcp will need to revisit this transport choice (or move to v2
   REST) when that happens. Document the date in any RFC.

## Sources

* `claude --help` v2.1.129 (local install, 2026-05-16)
* https://code.claude.com/docs/en/cli-reference
* https://code.claude.com/docs/en/headless
* https://code.claude.com/docs/en/agent-sdk
* https://code.claude.com/docs/en/agent-sdk/structured-outputs
* https://github.com/anthropics/claude-code/issues/9058
* https://github.com/anthropics/claude-agent-sdk-python (CHANGELOG)
* https://avasdream.com/blog/claude-cli-agentic-wrapper
* `/home/reidsurmeier/.npm-global/lib/node_modules/ruflo/.../agent-execute-core.js`
* Reid memory: `feedback_subprocess_wrapper_observability.md`, `feedback_cron_claude_p_loop.md`, `reference_ruflo_claude_code_executor.md`
