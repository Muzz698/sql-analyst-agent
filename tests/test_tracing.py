import os

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import collect_runs
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.run_helpers import is_traceable_function

from sql_analyst.config import Settings
from sql_analyst.graph.service import AnalystService, create_service
from sql_analyst.llm import FakeLLM
from sql_analyst.tools.executor import execute_sql
from sql_analyst.tools.safety_guard import validate_sql
from sql_analyst.tracing import NODE_METADATA, configure_tracing, log_feedback, tracing_enabled


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    # monkeypatch restores these after each test even though configure_tracing writes os.environ
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    yield
    configure_tracing(Settings(langsmith_tracing=False))


def test_tracing_off_by_default():
    assert configure_tracing(Settings()) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert not tracing_enabled()


def test_flag_without_key_stays_off(caplog):
    assert configure_tracing(Settings(langsmith_tracing=True, langsmith_api_key="  ")) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "tracing disabled" in caplog.text


def test_flag_with_key_enables_and_exports_env():
    s = Settings(langsmith_tracing=True, langsmith_api_key="lsv2_test", langsmith_project="proj")
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"


def test_feedback_is_noop_when_tracing_off():
    configure_tracing(Settings())
    assert log_feedback("00000000-0000-0000-0000-000000000000", 1.0) is False


def test_tools_are_traceable():
    assert is_traceable_function(validate_sql)
    assert is_traceable_function(execute_sql)


# --- trace structure (captured locally, no LangSmith needed) ------------------


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.chains: list[tuple[str, dict]] = []

    def on_chain_start(self, serialized, inputs, *, metadata=None, name=None, **kw):
        self.chains.append((name or "", metadata or {}))


def _script():
    return {
        "RouterDecision": [
            {"reasoning": "r", "intent": "sql", "standalone_question": "q", "confidence": 0.9}
        ],
        "TableSelection": [{"reasoning": "r", "tables": ["Artist"]}],
        "SQLDraft": [{"reasoning": "r", "sql": "SELECT COUNT(*) FROM Artist"}],
        "AnalystOutput": [{"answer": "3", "chart_recommended": False}],
    }


@pytest.fixture
def service(tiny_db):
    settings = Settings(llm_provider="fake", db_path=tiny_db, use_few_shot=False)
    return create_service(settings, llm=FakeLLM(_script()), checkpointer=InMemorySaver())


def test_root_run_has_name_id_thread_and_config_labels(service):
    with collect_runs() as cb:
        r = service.ask("thread-42", "q")
    root = cb.traced_runs[0]
    assert root.name == "sql_analyst.ask"
    assert str(root.id) == r.run_id
    md = root.extra["metadata"]
    assert md["thread_id"] == "thread-42"
    assert md["llm"] == "fake"
    assert md["few_shot"] is False
    assert set(root.tags) >= {"fake", "zero_shot"}


def test_every_executed_node_is_a_named_run_with_kind_metadata(service):
    rec = Recorder()
    traced = AnalystService(service.graph, service.metadata, service.tags, callbacks=[rec])
    traced.ask("t1", "q")
    by_name = {name: md for name, md in rec.chains}
    for node in ["router", "schema_agent", "sql_writer", "safety_guard", "executor", "analyst"]:
        assert node in by_name, node
        assert by_name[node]["kind"] == NODE_METADATA[node]["kind"]
    assert by_name["executor"]["kind"] == "tool"
    assert by_name["sql_writer"]["thread_id"] == "t1"  # run metadata propagates to children


def test_every_graph_node_has_metadata():
    from sql_analyst.graph.builder import TERMINALS

    core = ["start_turn", "router", "schema_agent", "sql_writer", "safety_guard",
            "human_approval", "executor", "analyst", "chart_agent", "finalize"]  # fmt: skip
    assert set(NODE_METADATA) == set(core + TERMINALS)
