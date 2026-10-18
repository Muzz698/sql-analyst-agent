"""Graph nodes. Each takes the state and returns a partial update.

Nodes are thin: they adapt state to the agent/tool functions and back, so the
real logic stays independently testable.
"""

import logging

from langgraph.types import interrupt

from sql_analyst.agents.analyst import analyze
from sql_analyst.agents.chart import make_chart_spec
from sql_analyst.agents.router import route
from sql_analyst.agents.schema_agent import select_tables
from sql_analyst.agents.sql_writer import write_sql
from sql_analyst.config import Settings
from sql_analyst.graph.routing import Policy, needs_approval
from sql_analyst.graph.state import PER_TURN_RESET, AgentState
from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import Attempt, QueryResult
from sql_analyst.tools.charts import build_figure_json
from sql_analyst.tools.executor import QueryError, execute_sql
from sql_analyst.tools.safety_guard import validate_sql
from sql_analyst.tools.schema_introspect import get_schema, render_schema, table_summaries

log = logging.getLogger(__name__)

EMPTY_RESULT_HINT = (
    "Query ran but returned 0 rows. Check filter values (spelling, case, date format "
    "'YYYY-MM-DD HH:MM:SS') and join conditions."
)

MESSAGES = {
    "out_of_scope": (
        "I can only answer questions about the data in this database (music store sales, "
        "customers, tracks, artists, employees), using read-only queries."
    ),
    "blocked": "I can't run that: only read-only SELECT queries are allowed.",
    "rejected": "Okay, I didn't run that query. Feel free to rephrase the question.",
}


class Nodes:
    def __init__(self, llm: StructuredLLM, settings: Settings, policy: Policy):
        self.llm = llm
        self.s = settings
        self.policy = policy
        self.tables = get_schema(settings.db_path)
        self.summaries = table_summaries(self.tables)
        self.row_counts = {t.name: t.row_count for t in self.tables}

    # --- turn lifecycle ---------------------------------------------------

    def start_turn(self, state: AgentState) -> dict:
        return dict(PER_TURN_RESET)

    def finalize(self, state: AgentState) -> dict:
        status = state.get("status") or "answered"
        turn = {"question": state["question"], "answer": state.get("answer") or ""}
        return {"status": status, "history": [turn]}

    # --- agents -----------------------------------------------------------

    def router(self, state: AgentState) -> dict:
        d = route(self.llm, state["question"], self.summaries, state.get("history"))
        update = {
            "intent": d.intent,
            "standalone_question": d.standalone_question.strip() or state["question"],
            "router_confidence": d.confidence,
        }
        if d.intent == "clarify":
            update["answer"] = d.clarification_question or "Could you clarify what you mean?"
        return update

    def schema_agent(self, state: AgentState) -> dict:
        if not self.s.use_schema_pruning:
            return {"tables": [t.name for t in self.tables]}
        return {"tables": select_tables(self.llm, state["standalone_question"], self.tables)}

    def sql_writer(self, state: AgentState) -> dict:
        attempts = [Attempt(**a) for a in state.get("attempts") or []]
        draft = write_sql(
            self.llm,
            state["standalone_question"],
            render_schema(self.tables, state.get("tables")),
            few_shot=self.s.use_few_shot,
            attempts=attempts,
        )
        return {
            "draft_sql": draft.sql,
            "retries": len(attempts),
            "error": None,
            "result": None,
            "human_approved": False,
            "human_edited": False,
        }

    def safety_guard(self, state: AgentState) -> dict:
        g = validate_sql(
            state.get("draft_sql") or "", self.s.default_row_limit, table_rows=self.row_counts
        )
        update: dict = {"guard_status": g.status, "guard_flags": g.flags, "sql": g.sql}
        if g.status == "invalid":
            failed = {"sql": state.get("draft_sql") or "", "error": g.reason}
            update["attempts"] = [*(state.get("attempts") or []), failed]
        elif g.status == "unsafe":
            update["error"] = g.reason
        return update

    def human_approval(self, state: AgentState) -> dict:
        reasons = needs_approval(state, self.policy)
        # Pauses the graph; the checkpointer persists state until resumed with
        # Command(resume={"approved": bool, "sql": optional edited SQL}).
        decision = interrupt(
            {
                "type": "approval",
                "question": state.get("standalone_question"),
                "sql": state.get("sql"),
                "reasons": reasons,
            }
        )
        if not decision.get("approved"):
            return {
                "status": "rejected",
                "approval_reason": reasons,
                "answer": MESSAGES["rejected"],
            }
        edited = (decision.get("sql") or "").strip()
        if edited and edited != state.get("sql"):
            return {"draft_sql": edited, "human_approved": True, "human_edited": True}
        return {"human_approved": True, "human_edited": False, "approval_reason": reasons}

    def executor(self, state: AgentState) -> dict:
        sql = state["sql"]
        try:
            r = execute_sql(
                sql,
                self.s.db_path,
                timeout_ms=self.s.query_timeout_ms,
                max_rows=self.s.default_row_limit,
            )
        except QueryError as e:
            failed = {"sql": sql, "error": str(e)}
            return {"error": str(e), "attempts": [*(state.get("attempts") or []), failed]}
        update: dict = {"result": r.model_dump(), "error": None}
        if r.row_count == 0:
            update["attempts"] = [
                *(state.get("attempts") or []),
                {"sql": sql, "error": EMPTY_RESULT_HINT},
            ]
        return update

    def analyst(self, state: AgentState) -> dict:
        result = QueryResult(**state["result"])
        out = analyze(self.llm, state["standalone_question"], state["sql"], result)
        return {"answer": out.answer, "chart_requested": out.chart_recommended}

    def chart_agent(self, state: AgentState) -> dict:
        result = QueryResult(**state["result"])
        try:
            spec = make_chart_spec(self.llm, state["standalone_question"], result)
            if spec is None:
                return {}
            return {"chart_spec": spec.model_dump(), "chart_json": build_figure_json(spec, result)}
        except Exception:  # a chart is a nice-to-have; never fail the answer over it
            log.exception("chart generation failed")
            return {}

    # --- terminal responses -------------------------------------------------

    def clarify(self, state: AgentState) -> dict:
        return {"status": "clarify"}

    def out_of_scope(self, state: AgentState) -> dict:
        return {"status": "out_of_scope", "answer": MESSAGES["out_of_scope"]}

    def blocked(self, state: AgentState) -> dict:
        return {"status": "blocked", "answer": MESSAGES["blocked"]}

    def rejected(self, state: AgentState) -> dict:
        return {"status": "rejected", "answer": MESSAGES["rejected"]}

    def failed(self, state: AgentState) -> dict:
        attempts = state.get("attempts") or []
        last = attempts[-1]["error"] if attempts else state.get("error") or "unknown error"
        return {
            "status": "failed",
            "answer": f"I couldn't write a working query after {len(attempts)} attempts. "
            f"Last problem: {last}",
        }
