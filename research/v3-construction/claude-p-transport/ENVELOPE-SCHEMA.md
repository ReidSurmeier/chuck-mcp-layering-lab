# `claude -p --output-format json` envelope schema (verified live)

All field documentation is taken from live captures against the local
`claude` v2.1.129 binary, not from docs alone. Raw captures are stored in
`./samples/`.

## Top-level shape

```typescript
interface ResultEnvelope {
  type: "result";                            // always "result" for single-shot JSON
  subtype: Subtype;                          // see enum below
  is_error: boolean;                         // true if any error path
  api_error_status: number | null;           // HTTP status from upstream API (null if not API-level)
  duration_ms: number;                       // total wall time inside the agent loop
  duration_api_ms: number;                   // sum of upstream API time
  num_turns: number;                         // turns consumed (validation tool-use counts as a turn)
  result: string;                            // free-text body. May be empty when structured_output is set
  structured_output?: object;                // only with --json-schema; null if schema failed
  stop_reason: "end_turn" | "tool_use" | "stop_sequence" | "max_tokens";
  session_id: string;                        // UUID. Stable across --resume
  total_cost_usd: number;                    // 0 on errors that never hit the API
  usage: Usage;                              // token counts; see below
  modelUsage: { [model: string]: ModelUsage };  // per-model cost + token breakdown
  permission_denials: PermissionDenial[];    // any tool calls blocked by permission system
  terminal_reason: "completed" | "max_turns" | "max_budget" | "user_interrupt";
  fast_mode_state: "on" | "off";
  uuid: string;                              // unique invocation id (NOT the session id)
  errors?: string[];                         // present on errors; human-readable strings
}

type Subtype =
  | "success"
  | "error_max_turns"
  | "error_max_structured_output_retries"
  | "error_during_execution";

interface Usage {
  input_tokens: number;
  cache_creation_input_tokens: number;       // big driver of cost on non-bare calls
  cache_read_input_tokens: number;
  output_tokens: number;
  server_tool_use: { web_search_requests: number; web_fetch_requests: number };
  service_tier: "standard" | "priority";
  cache_creation: { ephemeral_1h_input_tokens: number; ephemeral_5m_input_tokens: number };
  iterations: Iteration[];
  speed: "standard" | "fast";
}

interface ModelUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadInputTokens: number;
  cacheCreationInputTokens: number;
  webSearchRequests: number;
  costUSD: number;
  contextWindow: number;
  maxOutputTokens: number;
}
```

## Captured examples

### 1. Success without `--json-schema` (text result in `.result`)

File: `./samples/01-success-text-result.json`

Key fields:
```json
{
  "subtype": "success",
  "is_error": false,
  "result": "{\"ok\": true, \"message\": \"hello\"}",
  "structured_output": <MISSING>,
  "session_id": "9d2e2bb8-863f-4561-a5f2-079711168063",
  "total_cost_usd": 0.7384
}
```
Note: `result` is a string with the JSON we asked the model to produce. We
then have to `json.loads(envelope["result"])` ourselves — **error-prone**.
Compare to sample 3 which has `structured_output` already parsed.

### 2. `--bare` mode failing with OAuth subscription

File: `./samples/02-bare-not-logged-in-error.json`

Key fields:
```json
{
  "subtype": "error",
  "is_error": true,
  "result": "Not logged in · Please run /login",
  "duration_ms": 67,
  "duration_api_ms": 0,
  "total_cost_usd": 0,
  "usage": { "input_tokens": 0, ... }
}
```
**Definitive proof that `--bare` is incompatible with claude.ai OAuth.**
Returns in 67ms (no API call). Cost is $0. Cannot be retried — only fix is
to remove `--bare` or set `ANTHROPIC_API_KEY`.

### 3. Success WITH `--json-schema` (structured_output populated)

File: `./samples/03-success-structured-output.json`

Key fields:
```json
{
  "subtype": "success",
  "is_error": false,
  "result": "",
  "structured_output": { "intent": "draw", "confidence": 0.95 },
  "num_turns": 2,
  "session_id": "0638ad8e-...",
  "total_cost_usd": 0.8221
}
```
`result` is empty; `structured_output` has the validated object. Two turns
consumed: turn 1 is the model emitting a tool-use, turn 2 is the harness
validating + accepting. Cost is higher than sample 1 because `num_turns=2`
and Haiku 4.5 was invoked alongside Opus 4.7 for validation.

### 4. Success with `--no-session-persistence` (stateless service mode)

File: `./samples/04-success-no-session-persistence.json`

Key fields:
```json
{
  "subtype": "success",
  "is_error": false,
  "structured_output": { "intent": "paint", "medium": "watercolor", "confidence": 0.98 },
  "num_turns": 2,
  "total_cost_usd": 0.7223
}
```
Identical shape to sample 3 but with `--no-session-persistence`. No JSONL
file is created under `~/.claude/projects/`. **This is the mode chuck-mcp
uses.**

## modelUsage breakdown (live data)

Reid's session uses two models per call when `--json-schema` is enabled:

| Model | Role | Avg input | Avg output | Avg cost/call |
|---|---|---|---|---|
| `claude-opus-4-7[1m]` | Main responder | 6–10 tokens | 100–300 tokens | $0.43–$0.51 |
| `claude-haiku-4-5` | Validator / sub-tasks | ~80 tokens | ~600 tokens | $0.20–$0.40 |

The Haiku usage explains why naive intent classification is not cheap on
this path. v2 (direct REST + Opus only, no validator subagent) should
drop costs by ~10x.

## Detecting cost overrun in the audit log

The audit log writes one record per call with `cost_usd` field. To watch
for runaway calls:

```bash
# Sum today's spend
jq -s 'map(.cost_usd // 0) | add' ~/.chuck-mcp/claude-p-calls.log
```

```bash
# Find slow calls (>60s wall time)
jq 'select(.wall_s > 60)' ~/.chuck-mcp/claude-p-calls.log
```

```bash
# Find failed calls
jq 'select(.event != "success")' ~/.chuck-mcp/claude-p-calls.log
```

## Stream-json event types (not used by chuck-mcp v1, but documented)

When invoked with `--output-format stream-json --verbose`, the CLI emits
NDJSON events. Useful event types per docs (not exhaustive):

| `type` / `subtype` | Purpose |
|---|---|
| `system / init` | Session metadata, model selection, plugin load results. |
| `system / plugin_install` | Marketplace plugin install progress. |
| `system / api_retry` | API call failed, about to retry. Contains `attempt`, `max_retries`, `retry_delay_ms`, `error_status`, `error` category. |
| `assistant` | Assistant turn (chunked if `--include-partial-messages`). |
| `stream_event` | Token-level deltas. Filter on `event.delta.type == "text_delta"`. |
| `user` | User turn echo (with `--replay-user-messages`). |
| `result` | Final envelope. Same shape as the `--output-format json` whole-response. |
