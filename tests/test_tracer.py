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
