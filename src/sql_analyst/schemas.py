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
