"""Deterministic Plotly rendering from a ChartSpec."""

import json
from numbers import Number

import pandas as pd
import plotly.express as px

from sql_analyst.schemas import ChartSpec, QueryResult


def _numeric_columns(result: QueryResult) -> list[str]:
    numeric = []
    for i, col in enumerate(result.columns):
        values = [r[i] for r in result.rows if r[i] is not None]
        if values and all(isinstance(v, Number) and not isinstance(v, bool) for v in values):
            numeric.append(col)
    return numeric


def is_chartable(result: QueryResult) -> bool:
    return result.row_count >= 2 and len(result.columns) >= 2 and bool(_numeric_columns(result))


def validate_spec(spec: ChartSpec, result: QueryResult) -> ChartSpec | None:
    """Return the spec if its columns exist and y is numeric, else None."""
    cols = set(result.columns)
    if spec.x not in cols or spec.y not in cols or spec.y not in _numeric_columns(result):
        return None
    if spec.color is not None and spec.color not in cols:
        spec = spec.model_copy(update={"color": None})
    if spec.chart_type == "pie" and result.row_count > 6:
        spec = spec.model_copy(update={"chart_type": "bar"})
    return spec


def build_figure_json(spec: ChartSpec, result: QueryResult) -> str:
    """Render to Plotly figure JSON (serialisable for the API and checkpointer)."""
    df = pd.DataFrame(result.rows, columns=result.columns)
    if spec.chart_type == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, title=spec.title)
    else:
        plot = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[spec.chart_type]
        fig = plot(df, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
    fig.update_layout(template="plotly_white", margin={"l": 40, "r": 20, "t": 60, "b": 40})
    return fig.to_json()


def figure_dict(figure_json: str) -> dict:
    return json.loads(figure_json)
