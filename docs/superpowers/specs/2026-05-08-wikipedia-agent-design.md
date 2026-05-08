# Wikipedia Tool-Use Agent — Design Spec

**Date:** 2026-05-08  
**Status:** Approved

---

## Overview

A tool-use agent backed by Claude Haiku 4.5 that answers user questions by searching Wikipedia. The system prioritizes observability: every critical backend lifecycle event is captured in a structured trace stored as a JSON file per session. A minimal frontend provides a chat interface and a trace viewer.

---

## Stack

| Layer | Choice |
|-------|--------|
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Backend | Python + FastAPI |
| Frontend | Plain HTML + vanilla JS |
| Trace storage | One JSON file per session in `backend/traces/` |
| Config | `.env` loaded via `python-dotenv` (not committed) |

---

## Project Structure

```
agent_eval/
├── backend/
│   ├── main.py            # FastAPI app; serves API + static frontend files
│   ├── agent.py           # Agentic loop (Claude tool-use)
│   ├── tools.py           # search_wikipedia(query: str) implementation
│   ├── guardrails.py      # Guardrail protocol + RuleBasedGuardrail
│   ├── tracer.py          # Tracer class; writes session JSON files
│   └── traces/            # One JSON file per session (gitignored contents)
├── frontend/
│   ├── index.html         # Chat UI
│   └── traces.html        # Trace viewer
├── .env                   # Secret keys (gitignored)
├── .env.example           # Key names only (committed)
└── .gitignore
```

---

## API Surface

### `POST /api/chat`

Request:
```json
{ "query": "Who invented the telephone?" }
```

Response:
```json
{
  "session_id": "uuid4",
  "answer": "...",
  "status": "success"
}
```

On guardrail block, returns HTTP 400 with:
```json
{
  "session_id": "uuid4",
  "status": "blocked",
  "blocked_at": "request" | "response",
  "reason": "blocked_keywords"
}
```

### `GET /api/traces`

Returns a list of sessions ordered by recency:
```json
[
  {
    "session_id": "uuid4",
    "started_at": "ISO8601",
    "status": "success" | "blocked" | "error",
    "total_duration_ms": 1768,
    "total_usage": { "input_tokens": 400, "output_tokens": 140 }
  }
]
```

### `GET /api/traces/{session_id}`

Returns the full session JSON (schema defined below).

---

## Backend Pipeline

```
POST /api/chat
  │
  ├─ Tracer.start_session(session_id, query)
  │
  ├─ RuleBasedGuardrail.check_request(query)
  │     → emit: request_safety_check
  │     → if blocked: write session, return 400
  │
  ├─ Agent loop (repeats until stop_reason == "end_turn"):
  │     ├─ emit: llm_request  (turn N)
  │     ├─ Claude Haiku call
  │     ├─ emit: llm_response (turn N, duration_ms, usage, content)
  │     └─ if tool_use:
  │           ├─ emit: tool_call   (turn N, tool, input)
  │           ├─ search_wikipedia(query)
  │           └─ emit: tool_result (turn N, output, output_length, duration_ms)
  │
  ├─ RuleBasedGuardrail.check_response(answer)
  │     → emit: response_safety_check
  │     → if blocked: write session, return 400
  │
  ├─ emit: final_answer
  ├─ Tracer.end_session(status, total_duration_ms, total_usage)
  └─ return answer + session_id
```

---

## Guardrail Architecture

```python
class GuardrailResult:
    passed: bool
    reason: str | None          # name of the blocking rule, or None
    rules_evaluated: list[dict] # [{rule, passed}]
    metadata: dict              # reserved for LLM-based guardrail output

class Guardrail(Protocol):
    def check_request(self, text: str) -> GuardrailResult: ...
    def check_response(self, text: str) -> GuardrailResult: ...
```

`RuleBasedGuardrail` implements this protocol. Rules are evaluated in order; first failure short-circuits. Rules at launch:
- `max_length` — input must be ≤ 500 characters
- `blocked_keywords` — configurable list of disallowed terms

A future `LLMGuardrail` can implement the same protocol and be composed or swapped without touching agent logic.

---

## Trace Schema

File path: `backend/traces/<session_id>.json`

```json
{
  "session_id": "uuid4",
  "started_at": "ISO8601",
  "user_query": "...",
  "events": [
    {
      "event": "request_safety_check",
      "timestamp": "ISO8601",
      "duration_ms": 2,
      "passed": true,
      "rules_evaluated": [
        { "rule": "max_length", "passed": true },
        { "rule": "blocked_keywords", "passed": true }
      ],
      "blocked_by": null
    },
    {
      "event": "llm_request",
      "turn": 1,
      "timestamp": "ISO8601",
      "messages": [...]
    },
    {
      "event": "llm_response",
      "turn": 1,
      "timestamp": "ISO8601",
      "duration_ms": 843,
      "stop_reason": "tool_use",
      "content": [...],
      "usage": { "input_tokens": 120, "output_tokens": 45 }
    },
    {
      "event": "tool_call",
      "turn": 1,
      "timestamp": "ISO8601",
      "tool": "search_wikipedia",
      "input": { "query": "..." }
    },
    {
      "event": "tool_result",
      "turn": 1,
      "timestamp": "ISO8601",
      "duration_ms": 312,
      "tool": "search_wikipedia",
      "output": "...",
      "output_length": 1842
    },
    {
      "event": "llm_response",
      "turn": 2,
      "timestamp": "ISO8601",
      "duration_ms": 610,
      "stop_reason": "end_turn",
      "content": [...],
      "usage": { "input_tokens": 280, "output_tokens": 95 }
    },
    {
      "event": "response_safety_check",
      "timestamp": "ISO8601",
      "duration_ms": 1,
      "passed": true,
      "rules_evaluated": [
        { "rule": "max_length", "passed": true },
        { "rule": "blocked_keywords", "passed": true }
      ],
      "blocked_by": null
    },
    {
      "event": "final_answer",
      "timestamp": "ISO8601",
      "text": "..."
    }
  ],
  "status": "success",
  "total_duration_ms": 1768,
  "total_usage": { "input_tokens": 400, "output_tokens": 140 },
  "final_answer": "...",
  "completed_at": "ISO8601"
}
```

Error events (tool failure, unexpected Claude stop reason, unhandled exception):
```json
{
  "event": "error",
  "turn": 1,
  "timestamp": "ISO8601",
  "source": "tool" | "llm" | "guardrail" | "agent",
  "message": "..."
}
```

---

## Frontend

### `index.html` — Chat UI

- Text input + Submit button
- Response area showing the agent's final answer
- Link: "View trace →" (opens `traces.html?session=<session_id>`) after each response
- Displays blocked/error status inline when the API returns non-success

### `traces.html` — Trace Viewer

- **Sidebar:** list of all sessions ordered by recency; shows session ID (truncated), timestamp, status badge, total duration
- **Main panel:** vertical event timeline for the selected session
  - Each event is a collapsible card: event name, timestamp, duration badge (where applicable)
  - Color-coded by type: safety checks (yellow), LLM calls (blue), tool calls (green), errors (red), final answer (grey)
  - Expandable raw JSON for any event
- **Summary bar:** total duration, total tokens (input + output), turn count, status

Traces are loaded via `GET /api/traces` (sidebar list) and `GET /api/traces/{session_id}` (event timeline). Session ID may be passed as a URL query param to deep-link directly from the chat UI.

---

## Environment Variables

`.env.example` (committed):
```
ANTHROPIC_API_KEY=your_key_here
```

`.env` (gitignored) — fill in actual values before running.

---

## Out of Scope

- Multi-turn conversation memory (each `POST /api/chat` is stateless)
- LLM-based guardrails (architecture supports it; not implemented in v1)
- Authentication / rate limiting
- Streaming responses
- Frontend-side observability
