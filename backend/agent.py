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
