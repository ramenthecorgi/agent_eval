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
