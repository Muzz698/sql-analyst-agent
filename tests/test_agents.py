import json

from sql_analyst.agents.analyst import analyze, format_result
from sql_analyst.agents.chart import make_chart_spec
from sql_analyst.agents.router import format_history, route
from sql_analyst.agents.schema_agent import select_tables
from sql_analyst.agents.sql_writer import load_few_shot, write_sql
from sql_analyst.llm import FakeLLM
from sql_analyst.schemas import Attempt, ChartSpec, QueryResult
from sql_analyst.tools.charts import build_figure_json, validate_spec
from sql_analyst.tools.schema_introspect import get_schema

REVENUE = QueryResult(
    columns=["year", "revenue"],
    rows=[["2009", 449.46], ["2010", 481.45], ["2011", 469.58]],
    row_count=3,
    elapsed_ms=1.0,
)
SINGLE = QueryResult(columns=["n"], rows=[[42]], row_count=1, elapsed_ms=1.0)


def text_of(call) -> str:
    return "\n".join(m.content for m in call["messages"])


# --- Router ------------------------------------------------------------------


def test_router_passes_history_and_returns_decision():
    llm = FakeLLM(
        {
            "RouterDecision": [
                {
                    "reasoning": "follow-up",
                    "intent": "sql",
                    "standalone_question": "What was revenue in 2012?",
                    "confidence": 0.9,
                }
            ]
        }
    )
    history = [{"question": "Revenue in 2011?", "answer": "$469.58"}]
    d = route(llm, "and 2012?", "- Invoice: Total", history)
    assert d.intent == "sql"
    prompt = text_of(llm.calls[0])
    assert "Revenue in 2011?" in prompt and "and 2012?" in prompt and "Invoice" in prompt


def test_format_history_keeps_recent_turns():
    turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
    text = format_history(turns, max_turns=2)
    assert "q9" in text and "q8" in text and "q7" not in text
    assert format_history([]) == "(no previous turns)"


# --- Schema agent ------------------------------------------------------------


def test_schema_agent_expands_join_path(chinook_db):
    tables = get_schema(chinook_db)
    llm = FakeLLM({"TableSelection": [{"reasoning": "r", "tables": ["Customer", "Genre"]}]})
    assert set(select_tables(llm, "genres per country", tables)) == {
        "Customer",
        "Invoice",
        "InvoiceLine",
        "Track",
        "Genre",
    }


def test_schema_agent_falls_back_to_all_tables_on_bad_names(tiny_db):
    tables = get_schema(tiny_db)
    llm = FakeLLM({"TableSelection": [{"reasoning": "r", "tables": ["Hallucinated"]}]})
    assert set(select_tables(llm, "q", tables)) == {"Artist", "Album"}


# --- SQL writer --------------------------------------------------------------


def _sql_llm():
    return FakeLLM({"SQLDraft": [{"reasoning": "r", "sql": "SELECT 1"}]})


def test_sql_writer_includes_schema_and_few_shot():
    llm = _sql_llm()
    write_sql(llm, "How many artists?", "CREATE TABLE Artist (...)", few_shot=True)
    prompt = text_of(llm.calls[0])
    assert "CREATE TABLE Artist" in prompt
    assert load_few_shot()[0]["question"] in prompt


def test_sql_writer_without_few_shot():
    llm = _sql_llm()
    write_sql(llm, "q", "schema", few_shot=False)
    assert "Examples of good queries" not in text_of(llm.calls[0])


def test_sql_writer_feeds_back_previous_errors():
    llm = _sql_llm()
    attempts = [Attempt(sql="SELECT Nme FROM Artist", error="no such column: Nme")]
    write_sql(llm, "q", "schema", attempts=attempts)
    prompt = text_of(llm.calls[0])
    assert "no such column: Nme" in prompt and "SELECT Nme FROM Artist" in prompt


def test_few_shot_examples_are_valid_safe_sql(chinook_db):
    from sql_analyst.tools.executor import execute_sql
    from sql_analyst.tools.safety_guard import validate_sql

    for ex in load_few_shot():
        guard = validate_sql(ex["sql"])
        assert guard.status == "ok", ex["question"]
        assert execute_sql(guard.sql, chinook_db).row_count > 0, ex["question"]


# --- Analyst -----------------------------------------------------------------


def test_analyst_never_recommends_chart_for_single_row():
    llm = FakeLLM({"AnalystOutput": [{"answer": "42 tracks.", "chart_recommended": True}]})
    out = analyze(llm, "How many?", "SELECT 42", SINGLE)
    assert out.answer == "42 tracks."
    assert out.chart_recommended is False


def test_format_result_truncates_rows():
    big = QueryResult(columns=["n"], rows=[[i] for i in range(50)], row_count=50, elapsed_ms=1)
    text = format_result(big, max_rows=5)
    assert "45 more rows not shown" in text
    assert "0 rows" in format_result(QueryResult(columns=["a"], rows=[], row_count=0, elapsed_ms=0))


# --- Chart -------------------------------------------------------------------


def test_chart_agent_returns_valid_spec_and_renders():
    llm = FakeLLM(
        {"ChartSpec": [{"chart_type": "line", "x": "year", "y": "revenue", "title": "Revenue"}]}
    )
    spec = make_chart_spec(llm, "revenue by year", REVENUE)
    fig = json.loads(build_figure_json(spec, REVENUE))
    assert fig["data"][0]["type"] == "scatter"  # plotly line traces are scatter/lines
    assert fig["layout"]["title"]["text"] == "Revenue"


def test_chart_agent_skips_llm_when_not_chartable():
    llm = FakeLLM()
    assert make_chart_spec(llm, "q", SINGLE) is None
    assert llm.calls == []


def test_validate_spec_rejects_hallucinated_or_non_numeric_columns():
    bad_col = ChartSpec(chart_type="bar", x="year", y="profit", title="t")
    non_numeric = ChartSpec(chart_type="bar", x="revenue", y="year", title="t")
    assert validate_spec(bad_col, REVENUE) is None
    assert validate_spec(non_numeric, REVENUE) is None


def test_validate_spec_downgrades_crowded_pie_and_drops_bad_color():
    many = QueryResult(
        columns=["g", "n"], rows=[[str(i), i] for i in range(10)], row_count=10, elapsed_ms=1
    )
    spec = validate_spec(ChartSpec(chart_type="pie", x="g", y="n", color="zz", title="t"), many)
    assert spec.chart_type == "bar"
    assert spec.color is None
