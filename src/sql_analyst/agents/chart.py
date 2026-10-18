"""Chart agent: choose a chart type and columns; code renders it.

The LLM only produces a small ``ChartSpec`` (type + column names). Plotly figure
JSON is then built deterministically, so the model can't produce invalid figures,
and column names are validated against the actual result.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from sql_analyst.agents.analyst import format_result
from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import ChartSpec, QueryResult
from sql_analyst.tools.charts import is_chartable, validate_spec

SYSTEM = """Choose the best simple chart for this query result.
- line: values over time (dates, months, years) on x.
- bar: comparing a numeric value across categories (default choice).
- pie: shares of a whole with at most 6 categories.
- scatter: relationship between two numeric columns.
x and y MUST be exact column names from the result; y must be numeric."""


def make_chart_spec(llm: StructuredLLM, question: str, result: QueryResult) -> ChartSpec | None:
    if not is_chartable(result):
        return None
    human = f"Question: {question}\n\nResult:\n{format_result(result, max_rows=10)}"
    spec = llm.generate(
        ChartSpec, [SystemMessage(SYSTEM), HumanMessage(human)], run_name="chart_agent"
    )
    return validate_spec(spec, result)
