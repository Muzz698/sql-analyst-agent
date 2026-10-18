"""Conditional-edge functions: pure functions of state, so every branch is unit-testable."""

from dataclasses import dataclass

from sql_analyst.graph.state import AgentState

# Router confidence below this counts as "ambiguous" and needs human approval.
LOW_CONFIDENCE = 0.6


@dataclass(frozen=True)
class Policy:
    max_retries: int = 3
    require_approval: bool = True


def after_router(state: AgentState) -> str:
    return {"sql": "schema_agent", "clarify": "clarify"}.get(
        state.get("intent", ""), "out_of_scope"
    )


def needs_approval(state: AgentState, policy: Policy) -> list[str]:
    """Reasons this query needs a human OK (empty list = run it)."""
    if not policy.require_approval:
        return []
    if state.get("human_approved"):
        return []
    reasons = list(state.get("guard_flags") or [])
    if state.get("router_confidence", 1.0) < LOW_CONFIDENCE:
        reasons.append("ambiguous_question")
    return reasons


def can_retry(state: AgentState, policy: Policy) -> bool:
    """Each failed attempt is recorded; allow up to ``max_retries`` rewrites after the first."""
    return len(state.get("attempts") or []) <= policy.max_retries


def after_guard(state: AgentState, policy: Policy) -> str:
    status = state.get("guard_status")
    if status == "unsafe":
        return "blocked"  # never retried: retries must not become a jailbreak loop
    if status == "invalid":
        return "sql_writer" if can_retry(state, policy) else "failed"
    return "human_approval" if needs_approval(state, policy) else "executor"


def after_approval(state: AgentState) -> str:
    if state.get("status") == "rejected":
        return "rejected"
    # Human-edited SQL must pass the safety guard again before it runs.
    return "safety_guard" if state.get("human_edited") else "executor"


def after_executor(state: AgentState, policy: Policy) -> str:
    if state.get("error"):
        return "sql_writer" if can_retry(state, policy) else "failed"
    result = state.get("result") or {}
    if result.get("row_count", 0) == 0 and can_retry(state, policy):
        return "sql_writer"  # empty results often mean a wrong filter value; try again
    return "analyst"  # includes "still empty after retries": a valid "no data" answer


def after_analyst(state: AgentState) -> str:
    return "chart_agent" if state.get("chart_requested") else "finalize"
