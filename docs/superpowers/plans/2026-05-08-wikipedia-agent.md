# Wikipedia Tool-Use Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend with a Claude Haiku 4.5 tool-use agent that searches Wikipedia, with rule-based guardrails, per-session JSON tracing, and a plain HTML frontend for chat and trace viewing.

**Architecture:** A single FastAPI process serves the REST API and static frontend files. The agent loop calls Claude with `search_wikipedia` as a tool, emitting structured trace events via a `Tracer` instance on every lifecycle step. Guardrails implement a `Guardrail` protocol so rule-based and future LLM-based implementations are interchangeable.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Anthropic SDK (claude-haiku-4-5-20251001), requests (Wikipedia API), python-dotenv, pytest, httpx

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `backend/traces/.gitkeep`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write requirements.txt**

```
anthropic>=0.40.0
fastapi>=0.115.0
uvicorn>=0.32.0
python-dotenv>=1.0.0
requests>=2.32.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
```

- [ ] **Step 2: Write .gitignore**

```
.env
__pycache__/
*.pyc
.pytest_cache/
backend/traces/*.json
```

- [ ] **Step 3: Write .env.example**

```
ANTHROPIC_API_KEY=your_key_here
```

- [ ] **Step 4: Create directory structure and .gitkeep**

```bash
mkdir -p backend/traces frontend tests
touch backend/traces/.gitkeep
```

- [ ] **Step 5: Write tests/conftest.py**

```python
import os
import sys
from pathlib import Path

# Ensure backend modules are importable without installing as a package.
# Set a placeholder API key so main.py can be imported without a real key.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
```

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore .env.example backend/traces/.gitkeep tests/conftest.py
git commit -m "chore: project setup — deps, gitignore, env template"
```

---

### Task 2: Tracer

**Files:**
- Create: `backend/tracer.py`
- Create: `tests/test_tracer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracer.py
import json
from pathlib import Path
from tracer import Tracer


def test_emit_adds_event_with_timestamp(tmp_path):
    t = Tracer(session_id="abc", user_query="hello", traces_dir=tmp_path)
    t.emit({"event": "test_event", "value": 42})
    assert len(t.events) == 1
    ev = t.events[0]
    assert ev["event"] == "test_event"
    assert ev["value"] == 42
    assert "timestamp" in ev


def test_save_writes_json_file(tmp_path):
    t = Tracer(session_id="abc", user_query="hello", traces_dir=tmp_path)
    t.emit({"event": "final_answer", "text": "Paris"})
    t.save(status="success", final_answer="Paris", total_usage={"input_tokens": 10, "output_tokens": 5})
    path = tmp_path / "abc.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["session_id"] == "abc"
    assert data["user_query"] == "hello"
    assert data["status"] == "success"
    assert data["final_answer"] == "Paris"
    assert data["total_usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert "total_duration_ms" in data
    assert "started_at" in data
    assert "completed_at" in data
    assert len(data["events"]) == 1


def test_save_creates_traces_dir_if_missing(tmp_path):
    traces_dir = tmp_path / "nested" / "traces"
    t = Tracer(session_id="xyz", user_query="q", traces_dir=traces_dir)
    t.save(status="success", final_answer=None, total_usage={"input_tokens": 0, "output_tokens": 0})
    assert (traces_dir / "xyz.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tracer.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracer'`

- [ ] **Step 3: Write backend/tracer.py**

```python
import json
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TRACES_DIR = Path(__file__).parent / "traces"


class Tracer:
    def __init__(self, session_id: str, user_query: str, traces_dir: Path = DEFAULT_TRACES_DIR):
        self.session_id = session_id
        self.user_query = user_query
        self.traces_dir = traces_dir
        self.events: list[dict] = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start = time.monotonic()

    def emit(self, event: dict) -> None:
        self.events.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})

    def save(self, status: str, final_answer: str | None, total_usage: dict) -> None:
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        session = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "user_query": self.user_query,
            "events": self.events,
            "status": status,
            "total_duration_ms": int((time.monotonic() - self._start) * 1000),
            "total_usage": total_usage,
            "final_answer": final_answer,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.traces_dir / f"{self.session_id}.json").write_text(json.dumps(session, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tracer.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/tracer.py tests/test_tracer.py
git commit -m "feat: add Tracer with per-session JSON trace file output"
```

---

### Task 3: Wikipedia Tool

**Files:**
- Create: `backend/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
from unittest.mock import MagicMock, patch
from tools import search_wikipedia


def _mock_search_response(titles: list[str]):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "query": {"search": [{"title": t} for t in titles]}
    }
    return mock


def _mock_summary_response(extract: str):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"extract": extract}
    return mock


def test_search_wikipedia_returns_title_and_extract():
    with patch("tools.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_search_response(["Python (programming language)"]),
            _mock_summary_response("Python is a high-level language."),
        ]
        result = search_wikipedia("Python programming")
    assert "Python (programming language)" in result
    assert "Python is a high-level language." in result


def test_search_wikipedia_no_results():
    with patch("tools.requests.get") as mock_get:
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"query": {"search": []}}
        mock_get.return_value = mock
        result = search_wikipedia("xyzzy nonexistent topic 12345")
    assert "No Wikipedia article found" in result


def test_search_wikipedia_http_error_raises():
    import requests as req
    with patch("tools.requests.get") as mock_get:
        mock = MagicMock()
        mock.raise_for_status.side_effect = req.HTTPError("404")
        mock_get.return_value = mock
        try:
            search_wikipedia("anything")
            assert False, "Expected HTTPError"
        except req.HTTPError:
            pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Write backend/tools.py**

```python
import requests

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def search_wikipedia(query: str) -> str:
    search_resp = requests.get(
        _SEARCH_URL,
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        timeout=10,
    )
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia article found for '{query}'."

    title = results[0]["title"]
    summary_resp = requests.get(
        _SUMMARY_URL.format(title=requests.utils.quote(title, safe="")),
        timeout=10,
    )
    summary_resp.raise_for_status()
    extract = summary_resp.json().get("extract", "No summary available.")
    return f"{title}: {extract}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/tools.py tests/test_tools.py
git commit -m "feat: add search_wikipedia tool using Wikipedia REST API"
```

---

### Task 4: Guardrails

**Files:**
- Create: `backend/guardrails.py`
- Create: `tests/test_guardrails.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_guardrails.py
from guardrails import RuleBasedGuardrail


def test_valid_request_passes():
    g = RuleBasedGuardrail()
    result = g.check_request("Who invented the telephone?")
    assert result.passed is True
    assert result.reason is None
    assert all(r["passed"] for r in result.rules_evaluated)


def test_request_too_long_blocked():
    g = RuleBasedGuardrail(max_length=10)
    result = g.check_request("This string is definitely longer than ten characters.")
    assert result.passed is False
    assert result.reason == "max_length"
    rule = next(r for r in result.rules_evaluated if r["rule"] == "max_length")
    assert rule["passed"] is False


def test_blocked_keyword_rejected():
    g = RuleBasedGuardrail(blocked_keywords=["badword"])
    result = g.check_request("Please badword something.")
    assert result.passed is False
    assert result.reason == "blocked_keywords"
    rule = next(r for r in result.rules_evaluated if r["rule"] == "blocked_keywords")
    assert rule["passed"] is False


def test_blocked_keyword_case_insensitive():
    g = RuleBasedGuardrail(blocked_keywords=["badword"])
    result = g.check_request("BADWORD appears here.")
    assert result.passed is False


def test_check_response_same_rules():
    g = RuleBasedGuardrail(max_length=5)
    result = g.check_response("This is too long.")
    assert result.passed is False
    assert result.reason == "max_length"


def test_rules_evaluated_always_populated():
    g = RuleBasedGuardrail()
    result = g.check_request("hello")
    rule_names = [r["rule"] for r in result.rules_evaluated]
    assert "max_length" in rule_names
    assert "blocked_keywords" in rule_names
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_guardrails.py -v
```

Expected: `ModuleNotFoundError: No module named 'guardrails'`

- [ ] **Step 3: Write backend/guardrails.py**

```python
from dataclasses import dataclass, field
from typing import Protocol

_DEFAULT_BLOCKED_KEYWORDS = ["ignore all previous", "jailbreak", "system prompt", "ignore previous instructions"]
_DEFAULT_MAX_LENGTH = 500


@dataclass
class GuardrailResult:
    passed: bool
    reason: str | None
    rules_evaluated: list[dict]
    metadata: dict = field(default_factory=dict)


class Guardrail(Protocol):
    def check_request(self, text: str) -> GuardrailResult: ...
    def check_response(self, text: str) -> GuardrailResult: ...


class RuleBasedGuardrail:
    def __init__(
        self,
        max_length: int = _DEFAULT_MAX_LENGTH,
        blocked_keywords: list[str] | None = None,
    ):
        self.max_length = max_length
        self.blocked_keywords = blocked_keywords if blocked_keywords is not None else _DEFAULT_BLOCKED_KEYWORDS

    def _evaluate(self, text: str) -> GuardrailResult:
        rules: list[dict] = []
        blocked_by: str | None = None
        text_lower = text.lower()

        length_ok = len(text) <= self.max_length
        rules.append({"rule": "max_length", "passed": length_ok})
        if not length_ok and blocked_by is None:
            blocked_by = "max_length"

        keywords_ok = not any(kw in text_lower for kw in self.blocked_keywords)
        rules.append({"rule": "blocked_keywords", "passed": keywords_ok})
        if not keywords_ok and blocked_by is None:
            blocked_by = "blocked_keywords"

        return GuardrailResult(passed=blocked_by is None, reason=blocked_by, rules_evaluated=rules)

    def check_request(self, text: str) -> GuardrailResult:
        return self._evaluate(text)

    def check_response(self, text: str) -> GuardrailResult:
        return self._evaluate(text)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_guardrails.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/guardrails.py tests/test_guardrails.py
git commit -m "feat: add Guardrail protocol and RuleBasedGuardrail"
```

---

### Task 5: Agent Loop

**Files:**
- Create: `backend/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent.py
from pathlib import Path
from unittest.mock import MagicMock, patch
from tracer import Tracer
from agent import run_agent


def make_tracer(tmp_path: Path) -> Tracer:
    return Tracer(session_id="test-session", user_query="test query", traces_dir=tmp_path)


def make_text_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    return resp


def make_tool_use_response(tool_id: str, tool_name: str, tool_input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    resp.usage.input_tokens = 80
    resp.usage.output_tokens = 30
    return resp


def test_run_agent_direct_answer(tmp_path):
    client = MagicMock()
    client.messages.create.return_value = make_text_response("The answer is 42.")
    tracer = make_tracer(tmp_path)

    answer, usage = run_agent("What is the answer?", tracer, client)

    assert answer == "The answer is 42."
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    event_names = [e["event"] for e in tracer.events]
    assert event_names == ["llm_request", "llm_response", ]


def test_run_agent_with_tool_use(tmp_path):
    client = MagicMock()
    client.messages.create.side_effect = [
        make_tool_use_response("tu_1", "search_wikipedia", {"query": "telephone inventor"}),
        make_text_response("Alexander Graham Bell invented the telephone."),
    ]
    tracer = make_tracer(tmp_path)

    with patch("agent.search_wikipedia", return_value="Alexander Graham Bell: Scottish-American inventor."):
        answer, usage = run_agent("Who invented the telephone?", tracer, client)

    assert "Bell" in answer
    assert usage["input_tokens"] == 180
    assert usage["output_tokens"] == 80
    event_names = [e["event"] for e in tracer.events]
    assert event_names == [
        "llm_request", "llm_response",
        "tool_call", "tool_result",
        "llm_request", "llm_response",
    ]


def test_run_agent_emits_turn_numbers(tmp_path):
    client = MagicMock()
    client.messages.create.side_effect = [
        make_tool_use_response("tu_1", "search_wikipedia", {"query": "python"}),
        make_text_response("Python is a language."),
    ]
    tracer = make_tracer(tmp_path)

    with patch("agent.search_wikipedia", return_value="Python: a programming language."):
        run_agent("Tell me about Python", tracer, client)

    turn1_events = [e for e in tracer.events if e.get("turn") == 1]
    turn2_events = [e for e in tracer.events if e.get("turn") == 2]
    assert len(turn1_events) == 4  # llm_request, llm_response, tool_call, tool_result
    assert len(turn2_events) == 2  # llm_request, llm_response


def test_run_agent_tool_error_emits_error_event(tmp_path):
    client = MagicMock()
    client.messages.create.side_effect = [
        make_tool_use_response("tu_1", "search_wikipedia", {"query": "fail"}),
        make_text_response("I could not find information."),
    ]
    tracer = make_tracer(tmp_path)

    with patch("agent.search_wikipedia", side_effect=Exception("network error")):
        run_agent("search something", tracer, client)

    error_events = [e for e in tracer.events if e["event"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["source"] == "tool"
    assert "network error" in error_events[0]["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Write backend/agent.py**

```python
import time
import anthropic
from tools import search_wikipedia
from tracer import Tracer

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024
_TOOL_DEFINITION = {
    "name": "search_wikipedia",
    "description": "Search Wikipedia for information about a topic and return a summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search term to look up on Wikipedia."}
        },
        "required": ["query"],
    },
}


def _serialize_block(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


def run_agent(query: str, tracer: Tracer, client: anthropic.Anthropic) -> tuple[str, dict]:
    messages: list[dict] = [{"role": "user", "content": query}]
    total_input = 0
    total_output = 0
    turn = 0

    while True:
        turn += 1
        tracer.emit({"event": "llm_request", "turn": turn, "messages": messages})

        t0 = time.monotonic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            tools=[_TOOL_DEFINITION],
            messages=messages,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        tracer.emit({
            "event": "llm_response",
            "turn": turn,
            "duration_ms": duration_ms,
            "stop_reason": response.stop_reason,
            "content": [_serialize_block(b) for b in response.content],
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        })

        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return text, {"input_tokens": total_input, "output_tokens": total_output}

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": [_serialize_block(b) for b in response.content]})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tracer.emit({"event": "tool_call", "turn": turn, "tool": block.name, "input": block.input})
                t1 = time.monotonic()
                try:
                    result = search_wikipedia(block.input["query"])
                    tool_duration_ms = int((time.monotonic() - t1) * 1000)
                    tracer.emit({
                        "event": "tool_result",
                        "turn": turn,
                        "tool": block.name,
                        "duration_ms": tool_duration_ms,
                        "output": result,
                        "output_length": len(result),
                    })
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                except Exception as exc:
                    tool_duration_ms = int((time.monotonic() - t1) * 1000)
                    tracer.emit({"event": "error", "turn": turn, "source": "tool", "message": str(exc)})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"Error: {exc}", "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            tracer.emit({"event": "error", "turn": turn, "source": "llm",
                         "message": f"Unexpected stop_reason: {response.stop_reason}"})
            raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/agent.py tests/test_agent.py
git commit -m "feat: add agentic loop with tool-use and per-turn trace emission"
```

---

### Task 6: FastAPI Backend

**Files:**
- Create: `backend/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import main
    # Redirect traces to a temp directory so tests don't touch real trace files
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    return TestClient(main.app)


def test_chat_success(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    with patch("main.run_agent", return_value=("Alexander Graham Bell.", {"input_tokens": 100, "output_tokens": 50})):
        resp = client.post("/api/chat", json={"query": "Who invented the telephone?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["answer"] == "Alexander Graham Bell."
    assert "session_id" in data


def test_chat_blocked_request(client):
    resp = client.post("/api/chat", json={"query": "x" * 600})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "blocked"
    assert data["blocked_at"] == "request"
    assert data["reason"] == "max_length"


def test_chat_blocked_response(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    long_answer = "y" * 600
    with patch("main.run_agent", return_value=(long_answer, {"input_tokens": 50, "output_tokens": 200})):
        resp = client.post("/api/chat", json={"query": "short question"})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "blocked"
    assert data["blocked_at"] == "response"


def test_list_traces_empty(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_traces_returns_sessions(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    session = {
        "session_id": "abc123",
        "started_at": "2026-05-08T10:00:00Z",
        "status": "success",
        "total_duration_ms": 1000,
        "total_usage": {"input_tokens": 100, "output_tokens": 50},
    }
    (tmp_path / "abc123.json").write_text(json.dumps(session))
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["session_id"] == "abc123"


def test_get_trace_not_found(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    resp = client.get("/api/traces/doesnotexist")
    assert resp.status_code == 404


def test_get_trace_returns_session(client, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "TRACES_DIR", tmp_path)
    session = {"session_id": "abc", "events": [], "status": "success"}
    (tmp_path / "abc.json").write_text(json.dumps(session))
    resp = client.get("/api/traces/abc")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "abc"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write backend/main.py**

```python
import json
import os
import time
import uuid
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent
from guardrails import RuleBasedGuardrail
from tracer import Tracer

load_dotenv()

TRACES_DIR = Path(__file__).parent / "traces"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI()
guardrail = RuleBasedGuardrail()


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = str(uuid.uuid4())
    tracer = Tracer(session_id=session_id, user_query=request.query, traces_dir=TRACES_DIR)
    total_usage: dict = {"input_tokens": 0, "output_tokens": 0}

    t0 = time.monotonic()
    req_result = guardrail.check_request(request.query)
    tracer.emit({
        "event": "request_safety_check",
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "passed": req_result.passed,
        "rules_evaluated": req_result.rules_evaluated,
        "blocked_by": req_result.reason,
    })

    if not req_result.passed:
        tracer.save(status="blocked", final_answer=None, total_usage=total_usage)
        return JSONResponse(status_code=400, content={
            "session_id": session_id, "status": "blocked",
            "blocked_at": "request", "reason": req_result.reason,
        })

    try:
        answer, total_usage = run_agent(request.query, tracer, _get_client())
    except Exception as exc:
        tracer.emit({"event": "error", "source": "agent", "message": str(exc)})
        tracer.save(status="error", final_answer=None, total_usage=total_usage)
        raise HTTPException(status_code=500, detail=str(exc))

    t1 = time.monotonic()
    resp_result = guardrail.check_response(answer)
    tracer.emit({
        "event": "response_safety_check",
        "duration_ms": int((time.monotonic() - t1) * 1000),
        "passed": resp_result.passed,
        "rules_evaluated": resp_result.rules_evaluated,
        "blocked_by": resp_result.reason,
    })

    if not resp_result.passed:
        tracer.save(status="blocked", final_answer=None, total_usage=total_usage)
        return JSONResponse(status_code=400, content={
            "session_id": session_id, "status": "blocked",
            "blocked_at": "response", "reason": resp_result.reason,
        })

    tracer.emit({"event": "final_answer", "text": answer})
    tracer.save(status="success", final_answer=answer, total_usage=total_usage)
    return {"session_id": session_id, "answer": answer, "status": "success"}


@app.get("/api/traces")
async def list_traces():
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in sorted(TRACES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text())
        sessions.append({
            "session_id": data.get("session_id"),
            "started_at": data.get("started_at"),
            "status": data.get("status"),
            "total_duration_ms": data.get("total_duration_ms"),
            "total_usage": data.get("total_usage"),
        })
    return sessions


@app.get("/api/traces/{session_id}")
async def get_trace(session_id: str):
    path = TRACES_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads(path.read_text())


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/ -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat: add FastAPI backend with chat, trace list, and trace detail endpoints"
```

---

### Task 7: Frontend Chat UI

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Write frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wikipedia Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #111; }
    h1 { font-size: 1.4rem; margin-bottom: 24px; }
    #form { display: flex; gap: 8px; }
    #query { flex: 1; padding: 10px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 10px 18px; background: #1a73e8; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; }
    button:disabled { background: #aaa; cursor: default; }
    #result { margin-top: 24px; }
    #answer { background: #f4f4f4; border-radius: 6px; padding: 16px; white-space: pre-wrap; line-height: 1.6; }
    #trace-link { display: inline-block; margin-top: 12px; color: #1a73e8; text-decoration: none; font-size: 0.9rem; }
    #trace-link:hover { text-decoration: underline; }
    #error { background: #fdecea; border: 1px solid #f5c6cb; border-radius: 6px; padding: 16px; color: #721c24; }
    #status { margin-top: 8px; font-size: 0.85rem; color: #666; }
  </style>
</head>
<body>
  <h1>Wikipedia Agent</h1>
  <form id="form">
    <input id="query" type="text" placeholder="Ask a question…" autocomplete="off" required />
    <button id="submit-btn" type="submit">Ask</button>
  </form>
  <div id="status"></div>
  <div id="result" style="display:none;"></div>

  <script>
    const form = document.getElementById('form');
    const queryInput = document.getElementById('query');
    const submitBtn = document.getElementById('submit-btn');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;

      submitBtn.disabled = true;
      statusEl.textContent = 'Thinking…';
      resultEl.style.display = 'none';
      resultEl.innerHTML = '';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const data = await res.json();

        resultEl.style.display = 'block';
        statusEl.textContent = '';

        if (data.status === 'success') {
          resultEl.innerHTML = `
            <div id="answer">${escapeHtml(data.answer)}</div>
            <a id="trace-link" href="traces.html?session=${data.session_id}">View trace →</a>
          `;
        } else if (data.status === 'blocked') {
          resultEl.innerHTML = `<div id="error">Request blocked (${escapeHtml(data.blocked_at)}): ${escapeHtml(data.reason)}</div>`;
        } else {
          resultEl.innerHTML = `<div id="error">An error occurred. Check the server logs.</div>`;
        }
      } catch (err) {
        statusEl.textContent = '';
        resultEl.style.display = 'block';
        resultEl.innerHTML = `<div id="error">Network error: ${escapeHtml(String(err))}</div>`;
      } finally {
        submitBtn.disabled = false;
      }
    });

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add chat UI frontend"
```

---

### Task 8: Frontend Trace Viewer

**Files:**
- Create: `frontend/traces.html`

- [ ] **Step 1: Write frontend/traces.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trace Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; display: flex; height: 100vh; color: #111; }

    /* Sidebar */
    #sidebar { width: 280px; min-width: 220px; border-right: 1px solid #ddd; overflow-y: auto; background: #fafafa; }
    #sidebar h2 { padding: 16px; font-size: 0.95rem; border-bottom: 1px solid #ddd; }
    .session-item { padding: 12px 16px; border-bottom: 1px solid #eee; cursor: pointer; }
    .session-item:hover { background: #f0f0f0; }
    .session-item.active { background: #e8f0fe; }
    .session-id { font-size: 0.8rem; color: #555; font-family: monospace; }
    .session-meta { font-size: 0.75rem; color: #888; margin-top: 4px; }
    .badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-blocked { background: #fff3cd; color: #856404; }
    .badge-error { background: #fdecea; color: #721c24; }

    /* Main panel */
    #main { flex: 1; overflow-y: auto; padding: 24px; }
    #summary { background: #f4f4f4; border-radius: 8px; padding: 16px; margin-bottom: 24px; display: flex; gap: 24px; flex-wrap: wrap; }
    .summary-item { font-size: 0.85rem; }
    .summary-label { color: #666; }
    .summary-value { font-weight: 600; }

    /* Event cards */
    .event-card { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
    .event-header { padding: 10px 14px; display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; }
    .event-header:hover { filter: brightness(0.97); }
    .event-name { font-weight: 600; font-size: 0.9rem; }
    .event-turn { font-size: 0.75rem; color: #666; }
    .event-ts { font-size: 0.75rem; color: #999; margin-left: auto; }
    .duration-badge { font-size: 0.72rem; padding: 1px 7px; border-radius: 10px; background: rgba(0,0,0,0.08); }
    .event-body { padding: 0 14px 12px; display: none; }
    .event-body.open { display: block; }
    pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.78rem; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
    .toggle-raw { font-size: 0.75rem; color: #1a73e8; cursor: pointer; margin-top: 8px; display: inline-block; }

    /* Color coding */
    .color-safety { background: #fffde7; border-color: #f9a825; }
    .color-llm { background: #e3f2fd; border-color: #1565c0; }
    .color-tool { background: #e8f5e9; border-color: #2e7d32; }
    .color-error { background: #fdecea; border-color: #c62828; }
    .color-final { background: #f5f5f5; border-color: #9e9e9e; }

    #empty { color: #999; font-size: 0.9rem; padding: 40px; text-align: center; }
    #back-link { font-size: 0.85rem; color: #1a73e8; text-decoration: none; display: inline-block; margin-bottom: 16px; }
    #back-link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>Sessions</h2>
    <div id="session-list"><p style="padding:16px;font-size:0.85rem;color:#888;">Loading…</p></div>
  </div>
  <div id="main">
    <a href="/" id="back-link">← Back to chat</a>
    <div id="trace-view"><p id="empty">Select a session to view its trace.</p></div>
  </div>

  <script>
    const EVENT_COLORS = {
      request_safety_check: 'color-safety',
      response_safety_check: 'color-safety',
      llm_request: 'color-llm',
      llm_response: 'color-llm',
      tool_call: 'color-tool',
      tool_result: 'color-tool',
      error: 'color-error',
      final_answer: 'color-final',
    };

    async function loadSessions() {
      const res = await fetch('/api/traces');
      const sessions = await res.json();
      const list = document.getElementById('session-list');
      if (!sessions.length) {
        list.innerHTML = '<p style="padding:16px;font-size:0.85rem;color:#888;">No sessions yet.</p>';
        return;
      }
      list.innerHTML = sessions.map(s => `
        <div class="session-item" data-id="${s.session_id}" onclick="loadSession('${s.session_id}')">
          <div class="session-id">${s.session_id.slice(0, 16)}…
            <span class="badge badge-${s.status}">${s.status}</span>
          </div>
          <div class="session-meta">
            ${new Date(s.started_at).toLocaleString()} &bull;
            ${s.total_duration_ms != null ? s.total_duration_ms + 'ms' : '—'} &bull;
            ${s.total_usage ? (s.total_usage.input_tokens + s.total_usage.output_tokens) + ' tok' : '—'}
          </div>
        </div>
      `).join('');
    }

    async function loadSession(sessionId) {
      document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
      const activeEl = document.querySelector(`.session-item[data-id="${sessionId}"]`);
      if (activeEl) activeEl.classList.add('active');

      const res = await fetch(`/api/traces/${sessionId}`);
      if (!res.ok) { alert('Session not found'); return; }
      const session = await res.json();
      renderSession(session);
      history.replaceState(null, '', `?session=${sessionId}`);
    }

    function renderSession(session) {
      const turns = new Set(session.events.filter(e => e.turn).map(e => e.turn)).size;
      const view = document.getElementById('trace-view');
      view.innerHTML = `
        <div id="summary">
          <div class="summary-item"><div class="summary-label">Status</div><div class="summary-value">${session.status}</div></div>
          <div class="summary-item"><div class="summary-label">Duration</div><div class="summary-value">${session.total_duration_ms ?? '—'}ms</div></div>
          <div class="summary-item"><div class="summary-label">Turns</div><div class="summary-value">${turns}</div></div>
          <div class="summary-item"><div class="summary-label">Input tokens</div><div class="summary-value">${session.total_usage?.input_tokens ?? '—'}</div></div>
          <div class="summary-item"><div class="summary-label">Output tokens</div><div class="summary-value">${session.total_usage?.output_tokens ?? '—'}</div></div>
        </div>
        <div><strong>Query:</strong> ${escapeHtml(session.user_query)}</div>
        <br>
        ${session.events.map((ev, i) => renderEvent(ev, i)).join('')}
      `;
    }

    function renderEvent(ev, idx) {
      const color = EVENT_COLORS[ev.event] || 'color-final';
      const turn = ev.turn ? `<span class="event-turn">turn ${ev.turn}</span>` : '';
      const dur = ev.duration_ms != null ? `<span class="duration-badge">${ev.duration_ms}ms</span>` : '';
      const ts = ev.timestamp ? `<span class="event-ts">${new Date(ev.timestamp).toLocaleTimeString()}</span>` : '';
      return `
        <div class="event-card ${color}">
          <div class="event-header" onclick="toggleEvent(${idx})">
            <span class="event-name">${ev.event}</span>${turn}${dur}${ts}
          </div>
          <div class="event-body" id="ev-${idx}">
            <span class="toggle-raw" onclick="toggleRaw(${idx})">▶ raw JSON</span>
            <pre id="raw-${idx}" style="display:none;">${escapeHtml(JSON.stringify(ev, null, 2))}</pre>
          </div>
        </div>
      `;
    }

    function toggleEvent(idx) {
      const body = document.getElementById(`ev-${idx}`);
      body.classList.toggle('open');
    }

    function toggleRaw(idx) {
      const raw = document.getElementById(`raw-${idx}`);
      const toggler = raw.previousElementSibling;
      const open = raw.style.display === 'none';
      raw.style.display = open ? 'block' : 'none';
      toggler.textContent = open ? '▼ raw JSON' : '▶ raw JSON';
    }

    function escapeHtml(str) {
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Boot
    loadSessions().then(() => {
      const params = new URLSearchParams(location.search);
      const sessionId = params.get('session');
      if (sessionId) loadSession(sessionId);
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/traces.html
git commit -m "feat: add trace viewer frontend with event timeline"
```

---

### Task 9: End-to-End Smoke Test

**Files:** none (manual verification)

- [ ] **Step 1: Copy .env.example and fill in your API key**

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=<your actual key>
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASSED, no failures.

- [ ] **Step 3: Start the server**

```bash
cd backend && uvicorn main:app --reload --port 8000
```

Expected output includes: `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 4: Test the chat UI**

Open `http://localhost:8000` in a browser. Type "Who invented the telephone?" and click Ask.

Expected:
- Answer appears within a few seconds
- "View trace →" link appears below the answer

- [ ] **Step 5: Test the trace viewer**

Click "View trace →". Verify:
- Summary bar shows duration, token counts, turn count
- Event timeline shows `request_safety_check`, `llm_request`, `llm_response`, `tool_call`, `tool_result` (if tool was called), `response_safety_check`, `final_answer`
- Each event card expands and shows raw JSON

- [ ] **Step 6: Test a blocked request**

In the chat UI, paste a string longer than 500 characters and click Ask.

Expected: "Request blocked (request): max_length" error message.

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "chore: verify end-to-end smoke test passes"
```
