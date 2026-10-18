"""Analyst agent: explain a query result in plain English and decide if a chart helps."""

from langchain_core.messages import HumanMessage, SystemMessage

from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import AnalystOutput, QueryResult

SYSTEM = """You are a business data analyst. Answer the user's question in 1-4 clear
sentences using ONLY the query result provided. Quote the key numbers. If the result is
empty, say no matching data was found. If the result was truncated, say the answer covers
the rows shown. Do not mention SQL unless asked.

Numbers: copy values exactly as they appear in the result. Never rescale them or add
suffixes like "k" or "M", never invent units, and never wrap numbers in quotes. Use a
currency symbol only for monetary columns (revenue, total, price).

Recommend a chart when the result has 3 or more rows with a numeric column AND it shows a
ranking/top-N, a comparison across categories, a trend over time, or a distribution.
Do not recommend one for a single number, a one-row result, or a plain list of names."""

MAX_ROWS_IN_PROMPT = 30


def format_result(result: QueryResult, max_rows: int = MAX_ROWS_IN_PROMPT) -> str:
    if result.row_count == 0:
        return f"Columns: {', '.join(result.columns)}\n(0 rows)"
    lines = [" | ".join(result.columns)]
    lines += [" | ".join(str(v) for v in row) for row in result.rows[:max_rows]]
    more = result.row_count - max_rows
    if more > 0:
        lines.append(f"... {more} more rows not shown")
    if result.truncated:
        lines.append("(result truncated by row limit)")
    return "\n".join(lines)


def analyze(llm: StructuredLLM, question: str, sql: str, result: QueryResult) -> AnalystOutput:
    human = f"Question: {question}\n\nSQL used:\n{sql}\n\nResult:\n{format_result(result)}"
    output = llm.generate(
        AnalystOutput, [SystemMessage(SYSTEM), HumanMessage(human)], run_name="analyst"
    )
    if result.row_count < 2:
        output.chart_recommended = False  # deterministic override: nothing to plot
    return output
