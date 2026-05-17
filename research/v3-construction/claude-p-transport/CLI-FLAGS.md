# `claude -p` flag matrix — verified 2026-05-16 against v2.1.129

Every flag below was either run live against the local `claude` binary or
cross-referenced against `https://code.claude.com/docs/en/headless` and
`https://code.claude.com/docs/en/agent-sdk/structured-outputs`. Entries
marked "verified live" had an actual subprocess invocation behind them.

## Core output flags

| Flag | Type | chuck-mcp uses it? | Notes |
|---|---|---|---|
| `-p`, `--print` | bool | YES | Non-interactive mode. Hard requirement for everything below. |
| `--output-format <text\|json\|stream-json>` | enum | YES (`json`) | `json` = single envelope, `stream-json` = NDJSON events. Default `text`. Only json formats expose `structured_output`. |
| `--json-schema <schema>` | string (JSON-encoded) | YES | Enforces structured output. **Requires `--output-format json`** (silently dropped otherwise). Verified live: schema-violations retry inside harness up to `error_max_structured_output_retries`. |
| `--max-turns <int>` | int | YES (3) | **MUST be ≥ 2** with `--json-schema`. Verified: max-turns=1 immediately returns `error_max_turns` because schema validation consumes one tool-use round. |
| `--max-budget-usd <amount>` | float | NO | Only enforced with `--print`. **No-op for subscription users today** — re-evaluate post 2026-06-15. |
| `--include-partial-messages` | bool | NO | Only valid with `--output-format stream-json`. Streams token-by-token. |
| `--include-hook-events` | bool | NO | Only valid with `--output-format stream-json`. |
| `--input-format <text\|stream-json>` | enum | NO | Default `text`. `stream-json` is for two-way agent loops. |
| `--verbose` | bool | NO | Adds debug chatter to stderr. Required by docs for stream-json + partial messages. |

## Permission & tool flags

| Flag | chuck-mcp uses it? | Notes |
|---|---|---|
| `--permission-mode <mode>` | YES (`dontAsk`) | Modes: `acceptEdits`, `auto`, `bypassPermissions`, `default`, `dontAsk`, `plan`. `dontAsk` = deny anything not in allow-list — perfect for headless. |
| `--allowedTools` / `--allowed-tools` | NO | We want zero tools. Don't pre-approve any. |
| `--disallowedTools` / `--disallowed-tools` | YES | Belt-and-suspenders. Explicit deny of `Bash,Edit,Write,WebFetch,WebSearch,Read,Glob,Grep`. |
| `--dangerously-skip-permissions` | NO | Sandboxes only. We don't need it. |
| `--allow-dangerously-skip-permissions` | NO | Enables skip-perms as an option without making it default. Sandbox-only. |
| `--tools <list>` | NO | Restricts built-in tool set. Use empty string to disable all — could be an alternative to `--disallowedTools` but more error-prone. |

## Auth & context flags

| Flag | chuck-mcp uses it? | Notes |
|---|---|---|
| `--bare` | NO | **Breaks OAuth subscription.** Forces `ANTHROPIC_API_KEY` or `apiKeyHelper`. Skips hooks, plugins, MCP servers, CLAUDE.md, keychain. Verified live: returns `Not logged in · Please run /login` when used with Reid's Max subscription. |
| `--system-prompt <prompt>` | NO | Replaces default system prompt entirely. Heavy hammer. |
| `--append-system-prompt <prompt>` | YES | Appends JSON-only instruction on top of default. |
| `--append-system-prompt-file <path>` | NO | Same but from a file. |
| `--mcp-config <files>` | NO | Loads MCP servers from JSON. chuck-mcp's transport is just the LLM call — no MCP-in-MCP. |
| `--strict-mcp-config` | NO | Only relevant with `--mcp-config`. |
| `--settings <file-or-json>` | NO | Per-call settings override. Could be used for `apiKeyHelper` in v2 hybrid mode. |
| `--setting-sources <sources>` | NO | Defaults pick up `user, project, local`. Fine. |
| `--add-dir <dirs>` | NO | Expands tool access. Not needed when tools are denied. |
| `--agents <json>` / `--agent <name>` | NO | Custom agents — irrelevant for single-shot JSON call. |
| `--plugin-dir <path>` / `--plugin-url <url>` | NO | Per-session plugin load. Not needed. |
| `--disable-slash-commands` | NO | Already disabled in non-interactive mode anyway. |

## Session flags

| Flag | chuck-mcp uses it? | Notes |
|---|---|---|
| `--no-session-persistence` | YES | Don't write JSONL transcripts. Stateless service. |
| `-c`, `--continue` | NO | Continue last conversation in cwd. We don't have one. |
| `-r`, `--resume [id]` | NO | Resume by session ID. Stateless. |
| `--session-id <uuid>` | NO | Pin a session ID. Useful for trace correlation if we ever need it. |
| `--fork-session` | NO | Branch a session. |
| `-n`, `--name <name>` | NO | Display name. UI-only. |
| `--from-pr [value]` | NO | Resume from PR. |

## Model & effort flags

| Flag | chuck-mcp uses it? | Notes |
|---|---|---|
| `--model <model>` | NO (use default) | Reid's session defaults to `claude-opus-4-7[1m]`. Could pin to `sonnet` for cost. |
| `--fallback-model <model>` | NO | Only works with `--print`. Worth adding if overloads become common. |
| `--effort <level>` | NO | low/medium/high/xhigh/max. Default is fine for intent classification. |

## Debug / dev flags

| Flag | chuck-mcp uses it? | Notes |
|---|---|---|
| `-d`, `--debug` | NO | Pollutes stdout. |
| `--debug-file <path>` | NO | Use the audit log instead. |
| `--betas <list>` | NO | API key users only. |
| `--exclude-dynamic-system-prompt-sections` | NO | Improves cache reuse across users; we're a single-user service. |
| `--replay-user-messages` | NO | Only valid for stream-json. |
| `--brief` | NO | Enables `SendUserMessage` tool — we don't need agent-to-user comms. |

## Exit codes (verified live + per blog research)

| Exit | Meaning |
|---|---|
| 0 | Success — even with `is_error: true` envelope. Check envelope, not rc. |
| Non-zero | Process-level failure (binary missing, malformed CLI args, OOM-kill). |

Important: Exit code 0 + `is_error: true` is the common case for auth failures, max-turns, rate limits. **Always parse the envelope; never trust rc alone.**

## stderr semantics (verified live)

* Empty on success.
* Non-empty when CLI bootstrap fails (binary missing, args malformed,
  plugin sync errors).
* Hook lifecycle errors and MCP server probe failures are printed to
  stderr but don't fail the call.
* Audit log captures `stderr_head` (first 400 chars) on rc != 0.

## stream-json event types (from docs)

When `--output-format stream-json --verbose`:

| type / subtype | Meaning |
|---|---|
| `system / init` | First event. Reports model, tools, MCP servers, loaded plugins. |
| `system / plugin_install` | Marketplace plugin install progress (only when `CLAUDE_CODE_SYNC_PLUGIN_INSTALL=1`). |
| `system / api_retry` | Transient API error, about to retry. Fields: `attempt`, `max_retries`, `retry_delay_ms`, `error_status`, `error` (category). |
| `assistant` | Assistant message (chunked if `--include-partial-messages`). |
| `user` | User message echo (with `--replay-user-messages`). |
| `stream_event` | Token deltas. Filter on `event.delta.type == "text_delta"`. |
| `result` | Final envelope (same shape as `--output-format json` returns whole). |

chuck-mcp v1 does NOT use stream-json. If we ever want progress reporting
back to the MCP client, switch and parse `system/api_retry` to surface
"Anthropic is rate-limiting, retrying..." messages.

## Sources

* `claude --help` (v2.1.129 local, 2026-05-16)
* https://code.claude.com/docs/en/headless
* https://code.claude.com/docs/en/agent-sdk/structured-outputs
* https://code.claude.com/docs/en/cli-reference
* https://code.claude.com/docs/en/agent-sdk
* Live smoke results in `./samples/` and `~/.chuck-mcp/claude-p-calls.log`
