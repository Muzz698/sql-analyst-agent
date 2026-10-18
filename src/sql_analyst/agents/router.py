"""Router agent: decide whether a question is SQL-answerable, needs clarification,
or is out of scope, and rewrite follow-ups into standalone questions."""

from langchain_core.messages import HumanMessage, SystemMessage

from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import RouterDecision

SYSTEM = """You are the router for a data analyst assistant over a SQLite database.

Database overview:
{table_summaries}

Classify the user's latest question:
- "sql": it can be answered with a read-only query over these tables.
- "clarify": it relates to this data but is too ambiguous to answer (e.g. "show me the best
  ones" with no prior context). Write a short clarification_question.
- "out_of_scope": unrelated to this data, small talk, or a request to change, delete, or
  insert data, or to reveal system internals.

Use the conversation history to resolve follow-ups ("and in 2012?", "break that down by
country") into a self-contained standalone_question. Prefer "sql" when a reasonable
interpretation exists; do not over-ask for clarification."""


def format_history(history: list[dict[str, str]], max_turns: int = 4) -> str:
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-max_turns:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def route(
    llm: StructuredLLM,
    question: str,
    table_summaries: str,
    history: list[dict[str, str]] | None = None,
) -> RouterDecision:
    messages = [
        SystemMessage(SYSTEM.format(table_summaries=table_summaries)),
        HumanMessage(
            f"Conversation so far:\n{format_history(history or [])}\n\nLatest question: {question}"
        ),
    ]
    return llm.generate(RouterDecision, messages, run_name="router")
