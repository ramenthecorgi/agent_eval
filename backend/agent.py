import time
import anthropic
from tools import search_wikipedia
from tracer import Tracer

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024
_TOOL_DEFINITION = {
    "name": "search_wikipedia",
    "description": (
        "Search Wikipedia to retrieve factual context when a query requires reasoning, synthesis, "
        "or nuanced detail that benefits from enriched information — for example: causal questions "
        "('what caused X'), analytical questions ('how does X work in detail'), multi-factor "
        "explanations, or questions where the accuracy of specific facts is uncertain. "
        "Do NOT search when Claude can answer accurately from training knowledge: well-known "
        "biographical facts, established scientific definitions and constants, widely-known "
        "historical summaries, or common medical definitions. "
        "Do NOT search for real-time data, events from the past few weeks, private individuals, "
        "or hyper-local topics. "
        "When search is warranted, entity types covered: people, places, events, concepts, "
        "works, organizations, species, medical topics. "
        "Returns the article title and a summary extract."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The entity to look up. Use the canonical name of the entity, not a question. "
                    "Entity types: Person, Place, Event, Concept, Work, Organization, Species, "
                    "Medical condition."
                ),
            }
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


_SYSTEM_PROMPT = """
<role>
You are a knowledgeable and warm research assistant. Your purpose is to help users find
accurate, well-sourced information by drawing on Wikipedia when it offers strong coverage,
and on your general knowledge when it does not.
</role>

<tool_use>
Use search_wikipedia when the question is about an entity covered by the tool. The tool
description specifies exactly which entity types and topics qualify.
</tool_use>

<format>
- Answer in clear, flowing prose. Be warm and direct.
- When you used Wikipedia: end your answer with a "Sources" line citing the article
  title(s) you drew from, e.g. "Sources: Eiffel Tower, Paris."
- When you did not use Wikipedia: answer from general knowledge and note briefly that
  this answer is based on general knowledge rather than a live lookup.
- Keep answers focused — enough detail to be genuinely useful, not exhaustive.
</format>
""".strip()


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
            system=_SYSTEM_PROMPT,
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
