"""Pydantic models shared across tools, agents, the API and evals."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Safety guard ------------------------------------------------------------

GuardStatus = Literal["ok", "unsafe", "invalid"]


class GuardResult(BaseModel):
    """Outcome of validating LLM-generated SQL.

    ``unsafe`` ends the run (never retried); ``invalid`` is fed back to the SQL
    writer as an error so it can self-correct.
    """

    status: GuardStatus
    reason: str = ""
    sql: str | None = Field(default=None, description="Normalised SQL with LIMIT enforced")
    flags: list[str] = Field(
        default_factory=list, description="Cost/ambiguity hints for human approval"
    )
    tables: list[str] = Field(default_factory=list)


# --- Execution ---------------------------------------------------------------


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False
    elapsed_ms: float


# --- Schema introspection ----------------------------------------------------


class ColumnInfo(BaseModel):
    name: str
    type: str
    primary_key: bool = False
    nullable: bool = True


class ForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    row_count: int = 0
    sample_rows: list[list[Any]] = Field(default_factory=list)


# --- Agent outputs (LLM-facing: field descriptions are part of the prompt) ---


class RouterDecision(BaseModel):
    reasoning: str = Field(description="One short sentence explaining the decision.")
    intent: Literal["sql", "clarify", "out_of_scope"] = Field(
        description=(
            "sql: answerable by a read-only query on this database. "
            "clarify: plausibly answerable but too ambiguous to write SQL for. "
            "out_of_scope: unrelated to the data, or asks to modify/delete data."
        )
    )
    standalone_question: str = Field(
        description="The question rewritten to be self-contained, resolving references to "
        "earlier turns (e.g. 'what about 2012?' -> 'What was total revenue in 2012?')."
    )
    clarification_question: str | None = Field(
        default=None, description="If intent is clarify, the question to ask the user."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the chosen intent, 0-1.")


class TableSelection(BaseModel):
    reasoning: str = Field(description="One short sentence on why these tables.")
    tables: list[str] = Field(description="Exact names of the tables needed to answer.")


class SQLDraft(BaseModel):
    reasoning: str = Field(description="Brief plan: tables, joins, filters, aggregation.")
    sql: str = Field(description="A single read-only SQLite SELECT statement.")


class AnalystOutput(BaseModel):
    answer: str = Field(
        description="Plain-English answer to the question, citing the key numbers. "
        "Use only facts present in the result."
    )
    chart_recommended: bool = Field(
        description="True if a chart would genuinely help (comparisons, trends, distributions)."
    )


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    x: str = Field(description="Result column for the x-axis (or pie labels).")
    y: str = Field(description="Numeric result column for the y-axis (or pie values).")
    color: str | None = Field(default=None, description="Optional column to colour/group by.")
    title: str


class Attempt(BaseModel):
    """A previous SQL attempt and why it failed, fed back for self-correction."""

    sql: str
    error: str
