# Integrating `claude_p.py` into the chuck-mcp Python backend

This is the concrete plug-in spec. Other research agents in the swarm
need this to wire chuck-mcp's MCP tool handler to the LLM transport.

## Public surface

```python
from claude_p import translate_intent_prompt, ClaudeResult, ClaudePError

result: ClaudeResult = translate_intent_prompt(
    text="user's raw intent text",
    schema={                                  # JSON Schema dict
        "type": "object",
        "properties": {...},
        "required": [...],
        "additionalProperties": False,
    },
    timeout_s=120,                            # optional, default 120
    max_turns=3,                              # optional, default 3 (MUST be >= 2)
    max_retries=1,                            # optional, default 1
    cwd=None,                                 # optional, default cwd()
    extra_system_prompt=None,                 # optional, appended to JSON-only anchor
)

assert isinstance(result.structured_output, dict)
print(result.structured_output)               # the validated payload
print(result.session_id)                      # UUID
print(result.total_cost_usd)                  # float
print(result.duration_ms)                     # int
print(result.num_turns)                       # int
```

## Errors

```python
from claude_p import (
    ClaudePError,             # base
    ClaudePTimeoutError,      # wall-clock timeout exceeded
    ClaudePInvocationError,   # subprocess rc != 0 OR envelope is_error
    ClaudePSchemaError,       # envelope success but structured_output failed validation
)
```

All three are subclasses of `ClaudePError`. Default behavior is "raise" so
MCP tool handlers can convert to MCP `ErrorResponse` shapes:

```python
try:
    result = translate_intent_prompt(text, schema)
    return {"intent": result.structured_output, "session": result.session_id}
except ClaudePTimeoutError:
    return mcp_error("LLM transport timed out", retriable=True)
except ClaudePSchemaError as exc:
    return mcp_error(f"LLM produced unusable output: {exc}", retriable=False)
except ClaudePInvocationError as exc:
    return mcp_error(f"LLM transport failed: {exc}", retriable=False)
```

## MCP tool registration sketch

For an MCP tool whose contract is "translate this intent text to a
structured constraints JSON":

```python
# chuck_mcp/server.py
import json
from pathlib import Path
from claude_p import translate_intent_prompt, ClaudePError

# Load schema from disk so it's editable without code changes.
SCHEMA_PATH = Path(__file__).parent / "schemas" / "intent_constraints.json"
INTENT_SCHEMA = json.loads(SCHEMA_PATH.read_text())

@server.tool("translate_intent_prompt")
async def tool_translate_intent_prompt(text: str) -> dict:
    """Convert free-text user intent into ConstraintsJSON."""
    try:
        result = translate_intent_prompt(text, INTENT_SCHEMA)
    except ClaudePError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "ok": True,
        "constraints": result.structured_output,
        "transport_metadata": {
            "session_id": result.session_id,
            "cost_usd": result.total_cost_usd,
            "duration_ms": result.duration_ms,
            "num_turns": result.num_turns,
        },
    }
```

If the MCP server is async (likely; `mcp` Python SDK uses asyncio),
wrap the synchronous transport with `asyncio.to_thread`:

```python
import asyncio

@server.tool("translate_intent_prompt")
async def tool_translate_intent_prompt(text: str) -> dict:
    try:
        result = await asyncio.to_thread(
            translate_intent_prompt, text, INTENT_SCHEMA
        )
    except ClaudePError as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}
    return {"ok": True, "constraints": result.structured_output, ...}
```

Important: `claude_p` uses a module-level `threading.Lock` to serialize
calls. `asyncio.to_thread` is compatible — only one underlying subprocess
runs at a time regardless of how many MCP tool invocations are pending.

## Schema authoring guidance

For best `claude -p` reliability:

1. **Keep schemas flat.** chuck-mcp's intent constraints should be a
   single-level object. Deep nesting raises `error_max_structured_output_retries`.

2. **Use enums where possible.** `intent: {"enum": ["draw", "paint", ...]}`
   is more reliable than `intent: {"type": "string"}`.

3. **Mark optional fields optional.** If the user might say "draw a cat"
   without specifying medium, make `medium` non-required. Otherwise the
   model has to invent values and confidence drops.

4. **Always include `additionalProperties: false`.** Forces the model to
   stick to documented fields.

5. **Include a `confidence: {"type": "number", "minimum": 0, "maximum": 1}`
   field.** This gives the MCP client a signal for when to re-prompt the
   human user.

6. **Avoid `oneOf` / `anyOf` / `allOf` at the top level.** They work but
   significantly increase retry rate in our smoke tests.

Example minimum-viable schema:
```json
{
  "type": "object",
  "properties": {
    "intent": {"type": "string", "enum": ["draw", "paint", "sculpt", "print", "none"]},
    "medium": {"type": "string"},
    "subject": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["intent", "medium", "subject", "confidence"],
  "additionalProperties": false
}
```

## Dependencies

`claude_p.py` has zero hard Python dependencies. Optional:

* `jsonschema` (≥4.0) — if installed, used for full draft-2020-12
  validation. If not, falls back to a hand-rolled top-level validator
  that handles `required`, `type`, `enum`. chuck-mcp's flat schemas work
  without it; add to `pyproject.toml` if you want nested support.

External requirement:
* `claude` CLI binary at `/home/reidsurmeier/.npm-global/bin/claude`
  (auto-discovered via `shutil.which`).
* Valid Claude Code login (run `claude` once interactively and complete
  `/login` if it's not authenticated).

## Operations runbook

| Symptom | Check | Fix |
|---|---|---|
| Every call returns `ClaudePInvocationError: Not logged in` | `claude --version && claude doctor` | Run `claude` interactively, complete `/login` flow. OAuth refresh tokens last ~30 days. |
| First call slow (~60s), rest fast (~35s) | Normal | Cache warmup. No action. |
| `error_max_turns` even with `max_turns=3` | Schema too complex | Simplify schema (see §schema authoring) |
| Calls cost >$1.50 each | Probably cache-creation tax + retry | Verify only one retry happened; check `modelUsage` in audit log |
| Audit log growing large | After ~3 months | Rotate via systemd-tmpfiles or weekly cron |
| Subscription monthly cap hit | Check Anthropic console | Wait for reset; consider v2 swap (REST + dedicated key) |

## Test hooks

Built into `claude_p.py`:

```bash
# Single prompt
python3 -m claude_p one "I want to sketch a cat" --max-turns 3

# Custom schema
python3 -m claude_p one "..." --schema '{"type":"object", ...}'

# Full smoke test (4 prompts, ~$2 cost)
python3 -m claude_p test --count 4
```

Recommend wiring `python3 -m claude_p test --count 1` into chuck-mcp's
healthcheck so OAuth-expiry is detected before the first MCP client
attempts a real call.
