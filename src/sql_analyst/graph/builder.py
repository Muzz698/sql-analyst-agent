"""Assemble the LangGraph StateGraph.

START -> start_turn -> router -+-> schema_agent -> sql_writer -> safety_guard
                               +-> clarify / out_of_scope
safety_guard -+-> executor | human_approval | sql_writer (invalid, retry) | blocked | failed
human_approval -+-> executor | safety_guard (edited SQL) | rejected
executor -+-> analyst | sql_writer (error/empty, retry) | failed
analyst -+-> chart_agent | finalize ;  all terminals -> finalize -> END
"""

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sql_analyst.config import Settings, get_settings
from sql_analyst.graph import routing
from sql_analyst.graph.nodes import Nodes
from sql_analyst.graph.state import AgentState
from sql_analyst.llm import StructuredLLM, get_llm
from sql_analyst.tracing import NODE_METADATA

TERMINALS = ["clarify", "out_of_scope", "blocked", "rejected", "failed"]


def build_graph(
    llm: StructuredLLM | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    policy = routing.Policy(
        max_retries=settings.max_sql_retries,
        require_approval=settings.require_approval_for_flagged,
    )
    n = Nodes(llm, settings, policy)

    g = StateGraph(AgentState)
    for name in [
        "start_turn", "router", "schema_agent", "sql_writer", "safety_guard",
        "human_approval", "executor", "analyst", "chart_agent", "finalize", *TERMINALS,
    ]:  # fmt: skip
        g.add_node(name, getattr(n, name), metadata=NODE_METADATA[name])

    g.add_edge(START, "start_turn")
    g.add_edge("start_turn", "router")
    g.add_conditional_edges(
        "router", routing.after_router, ["schema_agent", "clarify", "out_of_scope"]
    )
    g.add_edge("schema_agent", "sql_writer")
    g.add_edge("sql_writer", "safety_guard")
    g.add_conditional_edges(
        "safety_guard",
        partial(routing.after_guard, policy=policy),
        ["executor", "human_approval", "sql_writer", "blocked", "failed"],
    )
    g.add_conditional_edges(
        "human_approval", routing.after_approval, ["executor", "safety_guard", "rejected"]
    )
    g.add_conditional_edges(
        "executor",
        partial(routing.after_executor, policy=policy),
        ["analyst", "sql_writer", "failed"],
    )
    g.add_conditional_edges("analyst", routing.after_analyst, ["chart_agent", "finalize"])
    g.add_edge("chart_agent", "finalize")
    for t in TERMINALS:
        g.add_edge(t, "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
