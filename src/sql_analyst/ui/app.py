"""Streamlit chat UI.

Run:  uv run streamlit run src/sql_analyst/ui/app.py
"""

import pandas as pd
import plotly.io as pio
import streamlit as st

from sql_analyst.config import get_settings
from sql_analyst.graph.service import TurnResult
from sql_analyst.llm import LLMConfigError
from sql_analyst.ui import client as client_mod

EXAMPLES = [
    "Which 5 countries generated the most revenue?",
    "Show monthly revenue for 2013",
    "Who are the top 3 best-selling artists?",
    "Which sales agent made the most in sales in 2010?",
]

STATUS_STYLE = {
    "blocked": st.error,
    "failed": st.error,
    "error": st.error,
    "out_of_scope": st.info,
    "clarify": st.info,
    "rejected": st.info,
}

st.set_page_config(page_title="SQL Analyst Agent", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading agent…")
def get_client():
    return client_mod.get_client()


def init_session(client) -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = client.new_thread_id()
        # [{"role": "user", "content": str} | {"role": "assistant", "result": dict}]
        st.session_state.messages = []
        # Not reset by "New conversation", so the cap can't be bypassed that way.
        st.session_state.questions_asked = 0


def reset_conversation(client) -> None:
    st.session_state.thread_id = client.new_thread_id()
    st.session_state.messages = []


def send_feedback(client, run_id: str, key: str) -> None:
    value = st.session_state.get(key)
    if value is not None and run_id:
        client.feedback(run_id, float(value))


def render_approval(client, result: TurnResult) -> None:
    approval = result.approval
    st.warning(
        "This query needs your approval before it runs: "
        + ", ".join(r.replace("_", " ") for r in approval.reasons)
    )
    edited = st.text_area("SQL to run (you can edit it)", value=approval.sql, key="approval_sql")
    c1, c2 = st.columns(2)
    if c1.button("✅ Approve & run", key="approve", type="primary"):
        changed = edited.strip() != approval.sql.strip()
        resolve_approval(client, approved=True, sql=edited if changed else None)
    if c2.button("✖ Reject", key="reject"):
        resolve_approval(client, approved=False)


def resolve_approval(client, approved: bool, sql: str | None = None) -> None:
    with st.spinner("Running query…"):
        new = client.approve(st.session_state.thread_id, approved=approved, sql=sql)
    st.session_state.messages[-1] = {"role": "assistant", "result": new.model_dump()}
    st.rerun()


def render_result(client, result: TurnResult, index: int, is_last: bool) -> None:
    if result.status == "awaiting_approval":
        if is_last:
            render_approval(client, result)
        else:
            st.caption("(approval request superseded)")
        return

    show = STATUS_STYLE.get(result.status, st.markdown)
    show(result.answer or "")

    if result.chart_json:
        st.plotly_chart(pio.from_json(result.chart_json), width="stretch", key=f"chart_{index}")
    if result.rows:
        with st.expander(
            f"Data ({result.row_count} rows{', truncated' if result.truncated else ''})"
        ):
            st.dataframe(pd.DataFrame(result.rows, columns=result.columns), hide_index=True)
    if result.sql:
        with st.expander("SQL"):
            st.code(result.sql, language="sql")

    meta = [f"{result.latency_ms / 1000:.1f}s"]
    if result.retries:
        meta.append(f"self-corrected {result.retries}x")
    st.caption(" · ".join(meta))

    if result.status == "answered" and result.run_id:
        key = f"fb_{result.run_id}"
        st.feedback("thumbs", key=key, on_change=send_feedback, args=(client, result.run_id, key))


def questions_left() -> int | None:
    cap = get_settings().ui_max_questions_per_session
    return None if cap == 0 else max(0, cap - st.session_state.get("questions_asked", 0))


def ask(client, question: str) -> None:
    st.session_state.questions_asked = st.session_state.get("questions_asked", 0) + 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        result = client.ask(st.session_state.thread_id, question)
    st.session_state.messages.append({"role": "assistant", "result": result.model_dump()})
    st.rerun()


def sidebar(client) -> str | None:
    clicked = None
    with st.sidebar:
        st.header("📊 SQL Analyst Agent")
        st.caption("Ask questions about the Chinook music store database in plain English.")
        if st.button("New conversation", width="stretch"):
            reset_conversation(client)
            st.rerun()
        st.subheader("Try asking")
        for i, q in enumerate(EXAMPLES):
            if st.button(q, key=f"ex_{i}", width="stretch"):
                clicked = q
        with st.expander("Database tables"):
            for t in client.schema():
                st.markdown(f"**{t['name']}** ({t['row_count']} rows)  \n{', '.join(t['columns'])}")
        info = client.info()
        st.caption(
            f"Model: `{info.get('llm')}` · Tracing: {'on' if info.get('tracing') else 'off'}"
        )
        st.caption("Read-only: queries that modify data are blocked.")
    return clicked


def main() -> None:
    try:
        client = get_client()
    except LLMConfigError as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    init_session(client)
    example = sidebar(client)

    st.title("Ask your data")
    messages = st.session_state.messages
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(
                    client,
                    TurnResult.model_validate(msg["result"]),
                    i,
                    is_last=i == len(messages) - 1,
                )

    awaiting = (
        bool(messages) and messages[-1].get("result", {}).get("status") == "awaiting_approval"
    )
    left = questions_left()
    exhausted = left == 0
    if exhausted:
        st.info(
            "You've reached this demo's question limit for the session (it runs on a shared "
            "free-tier LLM quota). Run it locally for unlimited use; see the README."
        )
    placeholder = (
        "Approve or reject the query above first"
        if awaiting
        else "Question limit reached"
        if exhausted
        else "Ask a question about the data…"
    )
    prompt = st.chat_input(placeholder, disabled=awaiting or exhausted)
    question = prompt or (None if awaiting or exhausted else example)
    if question:
        ask(client, question)


main()
