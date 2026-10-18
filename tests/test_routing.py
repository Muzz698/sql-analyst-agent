import pytest

from sql_analyst.graph.routing import (
    Policy,
    after_analyst,
    after_approval,
    after_executor,
    after_guard,
    after_router,
    needs_approval,
)

P = Policy(max_retries=3, require_approval=True)
fail = {"sql": "x", "error": "e"}


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("sql", "schema_agent"),
        ("clarify", "clarify"),
        ("out_of_scope", "out_of_scope"),
        ("??", "out_of_scope"),
    ],
)
def test_after_router(intent, expected):
    assert after_router({"intent": intent}) == expected


def test_guard_ok_goes_to_executor():
    assert after_guard({"guard_status": "ok", "guard_flags": []}, P) == "executor"


def test_guard_unsafe_is_never_retried():
    assert after_guard({"guard_status": "unsafe", "attempts": []}, P) == "blocked"


def test_guard_invalid_retries_until_cap():
    assert after_guard({"guard_status": "invalid", "attempts": [fail] * 3}, P) == "sql_writer"
    assert after_guard({"guard_status": "invalid", "attempts": [fail] * 4}, P) == "failed"


def test_guard_flags_trigger_approval():
    state = {"guard_status": "ok", "guard_flags": ["cartesian_join"]}
    assert after_guard(state, P) == "human_approval"
    assert after_guard(state, Policy(require_approval=False)) == "executor"
    assert after_guard({**state, "human_approved": True}, P) == "executor"


def test_low_router_confidence_needs_approval():
    assert needs_approval({"guard_flags": [], "router_confidence": 0.4}, P) == [
        "ambiguous_question"
    ]
    assert needs_approval({"guard_flags": [], "router_confidence": 0.9}, P) == []


def test_after_approval():
    assert after_approval({"status": "rejected"}) == "rejected"
    assert after_approval({"human_approved": True, "human_edited": True}) == "safety_guard"
    assert after_approval({"human_approved": True, "human_edited": False}) == "executor"


def test_executor_error_retries_then_fails():
    assert after_executor({"error": "boom", "attempts": [fail]}, P) == "sql_writer"
    assert after_executor({"error": "boom", "attempts": [fail] * 4}, P) == "failed"


def test_executor_empty_result_retries_then_answers():
    empty = {"result": {"row_count": 0}}
    assert after_executor({**empty, "attempts": [fail]}, P) == "sql_writer"
    # still empty after all retries: "no data found" is a legitimate answer
    assert after_executor({**empty, "attempts": [fail] * 4}, P) == "analyst"


def test_executor_success_goes_to_analyst():
    assert after_executor({"result": {"row_count": 3}, "attempts": []}, P) == "analyst"


def test_zero_retries_policy():
    assert after_executor({"error": "boom", "attempts": [fail]}, Policy(max_retries=0)) == "failed"


def test_after_analyst():
    assert after_analyst({"chart_requested": True}) == "chart_agent"
    assert after_analyst({"chart_requested": False}) == "finalize"
