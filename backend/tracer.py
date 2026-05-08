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
