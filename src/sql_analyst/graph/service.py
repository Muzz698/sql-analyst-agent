"""Thin service facade over the compiled graph.

The API, the Streamlit UI and the eval harness all go through this, so
"ask a question / approve a query / read history" behaves identically everywhere.
"""

import time
import uuid
from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel, Field

RECURSION_LIMIT = 50  # worst case: 4 SQL attempts x 3 nodes + approval + analysis


class ApprovalRequest(BaseModel):
    sql: str
    reasons: list[str]
    question: str | None = None


class TurnResult(BaseModel):
    thread_id: str
    status: Literal[
        "answered", "clarify", "out_of_scope", "blocked", "rejected", "failed",
        "awaiting_approval", "error",
    ]  # fmt: skip
    question: str
    standalone_question: str | None = None
    answer: str | None = None
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    chart_json: str | None = None
    retries: int = 0
    approval: ApprovalRequest | None = None
    error: str | None = None
    latency_ms: float = 0.0


class AnalystService:
    def __init__(self, graph):
        self.graph = graph

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    def _config(self, thread_id: str, run_name: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
            "run_name": run_name,
            "metadata": {"thread_id": thread_id},
        }

    def ask(self, thread_id: str, question: str) -> TurnResult:
        return self._run(thread_id, {"question": question}, question, "sql_analyst.ask")

    def resume(self, thread_id: str, approved: bool, sql: str | None = None) -> TurnResult:
        question = self.graph.get_state(self._config(thread_id, "")).values.get("question", "")
        command = Command(resume={"approved": approved, "sql": sql})
        return self._run(thread_id, command, question, "sql_analyst.resume")

    def history(self, thread_id: str) -> list[dict[str, str]]:
        return self.graph.get_state(self._config(thread_id, "")).values.get("history", [])

    def pending_approval(self, thread_id: str) -> ApprovalRequest | None:
        state = self.graph.get_state(self._config(thread_id, ""))
        for intr in state.interrupts:
            if isinstance(intr.value, dict) and intr.value.get("type") == "approval":
                return ApprovalRequest(**{k: intr.value[k] for k in ("sql", "reasons", "question")})
        return None

    def _run(self, thread_id: str, payload, question: str, run_name: str) -> TurnResult:
        start = time.monotonic()
        try:
            out = self.graph.invoke(payload, self._config(thread_id, run_name))
        except Exception as e:  # LLM outage after all retries, etc.
            return TurnResult(
                thread_id=thread_id, status="error", question=question,
                answer="Something went wrong while answering. Please try again.",
                error=f"{type(e).__name__}: {e}",
                latency_ms=round((time.monotonic() - start) * 1000, 1),
            )  # fmt: skip
        result = self._to_result(thread_id, question, out)
        result.latency_ms = round((time.monotonic() - start) * 1000, 1)
        return result

    def _to_result(self, thread_id: str, question: str, out: dict) -> TurnResult:
        res = out.get("result") or {}
        base = {
            "thread_id": thread_id,
            "question": out.get("question", question),
            "standalone_question": out.get("standalone_question") or None,
            "sql": out.get("sql"),
            "retries": out.get("retries", 0),
            "columns": res.get("columns", []),
            "rows": res.get("rows", []),
            "row_count": res.get("row_count", 0),
            "truncated": res.get("truncated", False),
        }
        interrupts = out.get("__interrupt__") or []
        if interrupts:
            v = interrupts[0].value
            return TurnResult(
                **base,
                status="awaiting_approval",
                approval=ApprovalRequest(
                    sql=v["sql"], reasons=v["reasons"], question=v.get("question")
                ),
            )
        return TurnResult(
            **base,
            status=out.get("status") or "answered",
            answer=out.get("answer"),
            chart_json=out.get("chart_json"),
            error=out.get("error"),
        )
