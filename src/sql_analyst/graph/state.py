"""Typed graph state.

Values are plain JSON-friendly types (dicts, not Pydantic objects) so the SQLite
checkpointer can persist them and the API can return them without conversion.

``history`` accumulates across turns of a thread (reducer: list concatenation);
everything else is per-turn and is reset by the ``start_turn`` node.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

Status = Literal["answered", "clarify", "out_of_scope", "blocked", "rejected", "failed"]


class Turn(TypedDict):
    question: str
    answer: str


class AgentState(TypedDict, total=False):
    # Input
    question: str

    # Memory across turns (append-only)
    history: Annotated[list[Turn], operator.add]

    # Router
    intent: str
    standalone_question: str
    router_confidence: float

    # Schema + SQL
    tables: list[str]
    draft_sql: str | None
    sql: str | None  # guarded, LIMIT-enforced SQL that was (or will be) executed
    guard_status: str | None
    guard_flags: list[str]
    attempts: list[dict[str, str]]  # [{"sql", "error"}] fed back for self-correction
    retries: int

    # Human-in-the-loop
    approval_reason: list[str]
    human_approved: bool  # reset whenever the writer produces new SQL
    human_edited: bool

    # Execution + analysis
    result: dict[str, Any] | None  # QueryResult.model_dump()
    error: str | None
    answer: str | None
    chart_requested: bool
    chart_spec: dict[str, Any] | None
    chart_json: str | None

    status: Status | None


PER_TURN_RESET: dict[str, Any] = {
    "intent": "",
    "standalone_question": "",
    "router_confidence": 1.0,
    "tables": [],
    "draft_sql": None,
    "sql": None,
    "guard_status": None,
    "guard_flags": [],
    "attempts": [],
    "retries": 0,
    "approval_reason": [],
    "human_approved": False,
    "human_edited": False,
    "result": None,
    "error": None,
    "answer": None,
    "chart_requested": False,
    "chart_spec": None,
    "chart_json": None,
    "status": None,
}
