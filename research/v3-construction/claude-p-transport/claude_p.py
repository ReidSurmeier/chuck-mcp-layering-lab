"""
claude_p.py — chuck-mcp LLM transport.

Subprocess-based wrapper around the `claude -p` headless CLI. This is the
v1 transport for chuck-mcp: one LLM call per MCP invocation
(`translate_intent_prompt`). We deliberately shell out to `claude` instead
of calling the Anthropic REST API directly so the call is billed against
Reid's $100/mo Anthropic Max subscription (claude.ai OAuth credentials in
the user keychain) rather than against an API key, until June 15, 2026
when Anthropic splits subscription + Agent SDK credit pools.

V2 migration path (single-file swap): replace `_run_claude_subprocess` with
`anthropic.Anthropic().messages.create(..., tools=[{"input_schema": ...}])`
once chuck-mcp gets its own API key budget. The `translate_intent_prompt`
signature does not change.

Design rules (Reid's USER.md + memory):
  * absolute paths only — log goes to ~/.chuck-mcp/claude-p-calls.log
  * subprocess wrapper 4-rules: explicit cwd, stdin=DEVNULL, tee output to
    persistent file, structured event on rc!=0
  * verify, don't assume — schema-validate every result before returning
  * caveman: minimal hedging, clear failure modes

CLI flags chuck-mcp uses (verified 2026-05-16 against Claude Code v2.1.129):

    claude -p
        --output-format json           # single-shot JSON result envelope
        --json-schema <schema>         # harness validates + re-prompts on miss
        --max-turns 3                  # allow tool-use roundtrip + retry
        --no-session-persistence       # stateless service, no JSONL writes
        --append-system-prompt <text>  # constrain output to JSON
        --permission-mode dontAsk      # block any tool not in allow-list
        --disallowedTools "Bash,Edit,Write,WebFetch,WebSearch"  # belt + suspenders
        <prompt>

Notes on flags we DO NOT use:
  * --bare           — would force ANTHROPIC_API_KEY, breaks OAuth subscription
  * --dangerously-skip-permissions — pointless for a JSON-only schema call
  * --max-budget-usd — currently no-op for subscription users; reconsider post 2026-06-15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

CLAUDE_BIN = shutil.which("claude") or "/home/reidsurmeier/.npm-global/bin/claude"
LOG_DIR = Path.home() / ".chuck-mcp"
LOG_PATH = LOG_DIR / "claude-p-calls.log"
DEFAULT_TIMEOUT_S = 120  # observed: ~35s p50, ~75s p99 for schema calls
DEFAULT_MAX_TURNS = 3
DEFAULT_MAX_RETRIES = 1  # 1 retry = 2 attempts total; subscription cost matters

# JSON-schema-bound system-prompt anchor. Reid's CLAUDE.md is loaded into
# every non-bare call, so we re-emphasize JSON-only behavior here.
_SCHEMA_SYSTEM_PROMPT = (
    "You convert user text into a strictly-typed JSON object that matches the "
    "provided JSON schema. Output JSON only. No prose. No markdown fences. "
    "No commentary. If the user request is ambiguous, pick the most-likely "
    "interpretation and lower the `confidence` field accordingly. Never invoke "
    "tools other than what is strictly required to emit the structured output."
)

# A serialized lock so chuck-mcp never accidentally fires two `claude -p`
# processes at once — interactive subscription tier rate-limits hard, and
# parallel headless calls are not safe per the docs. The MCP backend should
# almost never see concurrent traffic, but this prevents the foot-gun.
_CALL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Logging — append-only JSON-lines to ~/.chuck-mcp/claude-p-calls.log
# ---------------------------------------------------------------------------


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_event(event: dict[str, Any]) -> None:
    """Append one JSON-line audit record per claude invocation."""
    _ensure_log_dir()
    event.setdefault("ts", time.time())
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
    except OSError as exc:  # pragma: no cover — disk-full / permissions
        logging.getLogger("claude_p").warning("audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClaudePError(Exception):
    """Base error from claude_p transport."""


class ClaudePTimeoutError(ClaudePError):
    """Subprocess exceeded wall-clock timeout."""


class ClaudePInvocationError(ClaudePError):
    """Subprocess returned a non-zero rc OR is_error=true in result envelope."""


class ClaudePSchemaError(ClaudePError):
    """Returned structured_output did not validate after retries."""


# ---------------------------------------------------------------------------
# Result envelope (typed subset of what `claude -p --output-format json` returns)
# ---------------------------------------------------------------------------


@dataclass
class ClaudeResult:
    structured_output: dict[str, Any]
    raw_envelope: dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str | None:
        return self.raw_envelope.get("session_id")

    @property
    def total_cost_usd(self) -> float:
        return float(self.raw_envelope.get("total_cost_usd") or 0.0)

    @property
    def duration_ms(self) -> int:
        return int(self.raw_envelope.get("duration_ms") or 0)

    @property
    def num_turns(self) -> int:
        return int(self.raw_envelope.get("num_turns") or 0)

    @property
    def model_usage(self) -> dict[str, Any]:
        return self.raw_envelope.get("modelUsage") or {}


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


def _build_argv(
    prompt: str,
    schema: dict[str, Any],
    *,
    max_turns: int,
    system_prompt: str,
) -> list[str]:
    """Build the argv vector for one `claude -p` call.

    Schema is passed as a JSON string via --json-schema. We pin
    --max-turns >= 2 because the schema validator runs as an internal
    tool-use round (verified: max_turns=1 + --json-schema → error_max_turns).
    """
    return [
        CLAUDE_BIN,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--max-turns",
        str(max_turns),
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--disallowedTools",
        "Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep",
        "--append-system-prompt",
        system_prompt,
        prompt,
    ]


def _run_claude_subprocess(
    argv: list[str],
    *,
    timeout_s: int,
    cwd: str | None,
) -> tuple[int, str, str]:
    """Run the subprocess with the four canonical safety rules.

    Returns (returncode, stdout, stderr).

    Rules (from feedback_subprocess_wrapper_observability):
      1. explicit cwd
      2. stdin=DEVNULL (claude -p reads stdin if a TTY would be attached)
      3. captured output tee'd to log on failure
      4. structured event on rc != 0 (handled by caller)
    """
    run_cwd = cwd or str(Path.cwd())
    try:
        proc = subprocess.run(
            argv,
            cwd=run_cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudePTimeoutError(
            f"claude -p timed out after {timeout_s}s"
        ) from exc
    return proc.returncode, proc.stdout, proc.stderr


def _parse_envelope(stdout: str) -> dict[str, Any]:
    """The CLI emits one JSON envelope on stdout for --output-format json."""
    stdout = stdout.strip()
    if not stdout:
        raise ClaudePInvocationError("claude -p returned empty stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        # Surface first 500 chars so the audit log captures the corruption mode.
        raise ClaudePInvocationError(
            f"failed to parse JSON envelope: {exc}; head={stdout[:500]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_intent_prompt(
    text: str,
    schema: dict[str, Any],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cwd: str | None = None,
    extra_system_prompt: str | None = None,
) -> ClaudeResult:
    """Translate free-text user intent into a schema-conformant JSON object.

    This is THE entry point chuck-mcp calls. Everything else in this module
    is implementation. Signature is stable across the v1→v2 transport swap.

    Args:
        text: user intent string (from MCP tool input)
        schema: JSON Schema describing required shape
        timeout_s: subprocess wall-clock timeout (default 120s)
        max_turns: must be >= 2 for --json-schema to work; default 3 gives
                   one validation retry inside the harness
        max_retries: number of *external* retries on parse/timeout/api failure
        cwd: working dir for the subprocess (affects which CLAUDE.md loads)
        extra_system_prompt: additional system prompt suffix (optional)

    Returns:
        ClaudeResult with .structured_output dict matching `schema`.

    Raises:
        ClaudePTimeoutError, ClaudePSchemaError, ClaudePInvocationError
    """
    if max_turns < 2:
        raise ValueError("max_turns must be >= 2 when using --json-schema")

    sys_prompt = _SCHEMA_SYSTEM_PROMPT
    if extra_system_prompt:
        sys_prompt = sys_prompt + "\n\n" + extra_system_prompt

    last_exc: Exception | None = None
    attempts = max_retries + 1

    for attempt in range(1, attempts + 1):
        argv = _build_argv(
            text, schema, max_turns=max_turns, system_prompt=sys_prompt
        )
        log_base: dict[str, Any] = {
            "attempt": attempt,
            "of": attempts,
            "prompt_head": text[:200],
            "schema_keys": list(schema.get("properties", {}).keys()),
            "timeout_s": timeout_s,
        }

        with _CALL_LOCK:
            t0 = time.time()
            try:
                rc, stdout, stderr = _run_claude_subprocess(
                    argv, timeout_s=timeout_s, cwd=cwd
                )
            except ClaudePTimeoutError as exc:
                last_exc = exc
                _log_event({**log_base, "event": "timeout", "wall_s": time.time() - t0})
                if not _is_retriable(exc) or attempt >= attempts:
                    raise
                continue

        wall_s = time.time() - t0

        if rc != 0:
            last_exc = ClaudePInvocationError(
                f"claude -p exited {rc}; stderr={stderr[:400]!r}"
            )
            _log_event(
                {
                    **log_base,
                    "event": "subprocess_failed",
                    "rc": rc,
                    "wall_s": wall_s,
                    "stderr_head": stderr[:400],
                    "stdout_head": stdout[:400],
                }
            )
            if not _is_retriable(last_exc) or attempt >= attempts:
                raise last_exc
            continue

        try:
            envelope = _parse_envelope(stdout)
        except ClaudePInvocationError as exc:
            last_exc = exc
            _log_event({**log_base, "event": "envelope_parse_failed", "wall_s": wall_s})
            if attempt >= attempts:
                raise
            continue

        # Result envelope semantics — verified live against v2.1.129:
        #   { is_error: bool, subtype: "success" | "error_*", result: str|null,
        #     structured_output: {...} | null, session_id, total_cost_usd,
        #     duration_ms, num_turns, usage, modelUsage, terminal_reason }
        if envelope.get("is_error"):
            sub = envelope.get("subtype")
            err_msg = envelope.get("result") or envelope.get("errors") or sub
            last_exc = ClaudePInvocationError(
                f"claude -p result error ({sub}): {err_msg!r}"
            )
            _log_event(
                {
                    **log_base,
                    "event": "result_is_error",
                    "subtype": sub,
                    "envelope_err": err_msg,
                    "session_id": envelope.get("session_id"),
                    "cost_usd": envelope.get("total_cost_usd"),
                    "wall_s": wall_s,
                }
            )
            if attempt >= attempts or not _is_retriable_subtype(sub):
                raise last_exc
            continue

        structured = envelope.get("structured_output")
        if not isinstance(structured, dict):
            last_exc = ClaudePSchemaError(
                "envelope.success but structured_output missing/non-dict; "
                "did you forget --json-schema?"
            )
            _log_event(
                {
                    **log_base,
                    "event": "structured_output_missing",
                    "envelope_keys": list(envelope.keys()),
                    "wall_s": wall_s,
                }
            )
            if attempt >= attempts:
                raise last_exc
            continue

        ok, err = _validate_against_schema(structured, schema)
        if not ok:
            last_exc = ClaudePSchemaError(f"schema validation failed: {err}")
            _log_event(
                {
                    **log_base,
                    "event": "client_validation_failed",
                    "err": err,
                    "structured_output": structured,
                    "wall_s": wall_s,
                }
            )
            if attempt >= attempts:
                raise last_exc
            continue

        _log_event(
            {
                **log_base,
                "event": "success",
                "session_id": envelope.get("session_id"),
                "cost_usd": envelope.get("total_cost_usd"),
                "duration_ms": envelope.get("duration_ms"),
                "num_turns": envelope.get("num_turns"),
                "wall_s": wall_s,
                "structured_output": structured,
            }
        )
        return ClaudeResult(structured_output=structured, raw_envelope=envelope)

    # Shouldn't reach here, but be explicit.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


_RETRIABLE_SUBTYPES = {
    "error_max_turns",
    "error_max_structured_output_retries",
    "error_during_execution",
}


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, ClaudePTimeoutError):
        return True
    if isinstance(exc, ClaudePInvocationError):
        s = str(exc).lower()
        # Transient API/rate signals
        return any(
            tok in s
            for tok in ("rate_limit", "server_error", "overloaded", "timeout")
        )
    return False


def _is_retriable_subtype(subtype: str | None) -> bool:
    return subtype in _RETRIABLE_SUBTYPES


# ---------------------------------------------------------------------------
# Client-side schema validation
# ---------------------------------------------------------------------------


def _validate_against_schema(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, str | None]:
    """Optional belt-and-suspenders validation.

    claude -p already validates against --json-schema and retries internally
    via error_max_structured_output_retries, but we re-validate here so the
    audit log captures the exact shape we returned to the MCP caller.

    Uses `jsonschema` if installed, otherwise a hand-rolled subset checker
    that covers the shapes chuck-mcp actually uses: required, type, enum.
    """
    try:
        import jsonschema  # type: ignore

        try:
            jsonschema.validate(data, schema)
            return True, None
        except jsonschema.ValidationError as exc:
            return False, exc.message
    except ImportError:
        return _fallback_validate(data, schema)


def _fallback_validate(
    data: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, str | None]:
    """Minimal validator — only handles top-level required/type/enum.

    chuck-mcp's intent schemas are flat (intent + medium + confidence). If we
    grow nested schemas we should add `jsonschema` to pyproject.toml.
    """
    if schema.get("type") == "object":
        for key in schema.get("required") or []:
            if key not in data:
                return False, f"missing required key: {key}"
        for key, sub in (schema.get("properties") or {}).items():
            if key not in data:
                continue
            v = data[key]
            t = sub.get("type")
            type_ok = {
                "string": isinstance(v, str),
                "number": isinstance(v, (int, float)) and not isinstance(v, bool),
                "integer": isinstance(v, int) and not isinstance(v, bool),
                "boolean": isinstance(v, bool),
                "array": isinstance(v, list),
                "object": isinstance(v, dict),
            }.get(t, True)
            if not type_ok:
                return False, f"{key}: expected {t}, got {type(v).__name__}"
            if "enum" in sub and v not in sub["enum"]:
                return False, f"{key}: value {v!r} not in enum {sub['enum']}"
    return True, None


# ---------------------------------------------------------------------------
# CLI entry — `python -m claude_p test`
# ---------------------------------------------------------------------------


_TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["draw", "paint", "sculpt", "print", "none"],
        },
        "medium": {"type": "string"},
        "subject": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["intent", "medium", "subject", "confidence"],
    "additionalProperties": False,
}


_TEST_PROMPTS = [
    "I want to sketch a cat with a pencil",
    "paint a sunset over the ocean in watercolor",
    "carve a wood block print of mount fuji",
    "the weather today",
]


def _cmd_test(args: argparse.Namespace) -> int:
    print(f"# claude_p smoke test (claude={CLAUDE_BIN})")
    print(f"# log -> {LOG_PATH}")
    failures = 0
    for prompt in _TEST_PROMPTS[: args.count]:
        print(f"\n--> prompt: {prompt!r}")
        try:
            result = translate_intent_prompt(
                prompt,
                _TEST_SCHEMA,
                timeout_s=args.timeout,
                max_turns=args.max_turns,
                max_retries=args.retries,
            )
            print(f"    structured: {result.structured_output}")
            print(
                f"    session={result.session_id}  "
                f"cost=${result.total_cost_usd:.4f}  "
                f"dur={result.duration_ms}ms  turns={result.num_turns}"
            )
        except ClaudePError as exc:
            failures += 1
            print(f"    FAIL: {type(exc).__name__}: {exc}")
    print(f"\n# done. failures={failures}/{args.count}")
    return 0 if failures == 0 else 1


def _cmd_one(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema) if args.schema else _TEST_SCHEMA
    try:
        result = translate_intent_prompt(
            args.prompt,
            schema,
            timeout_s=args.timeout,
            max_turns=args.max_turns,
            max_retries=args.retries,
        )
        print(json.dumps(result.structured_output, indent=2))
        return 0
    except ClaudePError as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claude_p", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_test = sub.add_parser("test", help="run built-in smoke prompts")
    p_test.add_argument("--count", type=int, default=len(_TEST_PROMPTS))
    p_test.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    p_test.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p_test.add_argument("--retries", type=int, default=0)  # smoke = no retry
    p_test.set_defaults(func=_cmd_test)

    p_one = sub.add_parser("one", help="run a single prompt")
    p_one.add_argument("prompt")
    p_one.add_argument("--schema", help="JSON-schema string, default = test schema")
    p_one.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    p_one.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p_one.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES)
    p_one.set_defaults(func=_cmd_one)

    args = parser.parse_args(argv)
    if not Path(CLAUDE_BIN).exists():
        print(
            f"ERROR: claude binary not found at {CLAUDE_BIN}. "
            "Install Claude Code or set PATH.",
            file=sys.stderr,
        )
        return 3
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
