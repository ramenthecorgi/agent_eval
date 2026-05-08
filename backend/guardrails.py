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
        text_lower = text.lower()
        keywords_ok = not any(kw in text_lower for kw in self.blocked_keywords)
        rules = [{"rule": "blocked_keywords", "passed": keywords_ok}]
        return GuardrailResult(
            passed=keywords_ok,
            reason="blocked_keywords" if not keywords_ok else None,
            rules_evaluated=rules,
        )
