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
