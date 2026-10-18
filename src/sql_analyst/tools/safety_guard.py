"""Read-only SQL safety guard.

Parses SQL into an AST with sqlglot instead of regex-matching keywords, so:
  * comments, casing and whitespace tricks don't bypass it, and
  * write operations hidden inside a CTE (``WITH d AS (DELETE ...) SELECT ...``)
    are caught, because the whole tree is walked, not just the top node.

This is defence layer 1. Layer 2 is the executor opening SQLite read-only.
"""

import sqlglot
from langsmith import traceable
from sqlglot import exp
from sqlglot.errors import ParseError

from sql_analyst.schemas import GuardResult

DIALECT = "sqlite"

# Any of these node types anywhere in the tree means the query is not read-only.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Command,  # VACUUM, REINDEX, and anything else sqlglot can't model
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

# Functions that touch the filesystem or load native code.
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

# Heuristic thresholds for "expensive" queries that need human approval.
MAX_JOINS_BEFORE_FLAG = 3
LARGE_TABLE_ROWS = 1_000


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def _enforce_limit(query: exp.Query, row_limit: int) -> exp.Query:
    """Add a LIMIT, or cap an existing one that is too large / non-literal."""
    limit = query.args.get("limit")
    if limit is None:
        return query.limit(row_limit)
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int and int(value.this) <= row_limit:
        return query
    return query.limit(row_limit)


def _cost_flags(query: exp.Query, table_rows: dict[str, int]) -> list[str]:
    flags: list[str] = []
    joins = list(query.find_all(exp.Join))
    if len(joins) > MAX_JOINS_BEFORE_FLAG:
        flags.append(f"many_joins:{len(joins)}")
    # sqlglot normalises `FROM a, b` to CROSS JOIN, so judge by whether any
    # condition (ON / USING / WHERE) actually relates the tables.
    for join in joins:
        no_condition = not join.args.get("on") and not join.args.get("using")
        parent = join.find_ancestor(exp.Select)
        filtered = parent is not None and parent.args.get("where") is not None
        if no_condition and not filtered:
            flags.append("cartesian_join")
    for select in query.find_all(exp.Select):
        if select.args.get("where") or select.args.get("group"):
            continue
        for table in select.find_all(exp.Table):
            if table_rows.get(table.name.lower(), 0) >= LARGE_TABLE_ROWS and select.is_star:
                flags.append(f"full_scan:{table.name}")
    return sorted(set(flags))


@traceable(run_type="tool", name="validate_sql")
def validate_sql(
    sql: str,
    row_limit: int = 200,
    table_rows: dict[str, int] | None = None,
) -> GuardResult:
    """Validate that ``sql`` is a single read-only query and enforce a LIMIT."""
    if not sql or not sql.strip():
        return GuardResult(status="invalid", reason="Empty SQL.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except ParseError as e:
        return GuardResult(status="invalid", reason=f"SQL syntax error: {e}")

    if len(statements) != 1:
        # Multiple statements are a classic injection vector: reject outright.
        unsafe = any(isinstance(s, FORBIDDEN_NODES) for s in statements)
        return GuardResult(
            status="unsafe" if unsafe else "invalid",
            reason=f"Expected exactly one statement, got {len(statements)}.",
        )

    tree = statements[0]

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return GuardResult(
                status="unsafe",
                reason=f"Only read-only SELECT queries are allowed (found {type(node).__name__}).",
            )
        if isinstance(node, exp.Func) and _function_name(node) in FORBIDDEN_FUNCTIONS:
            return GuardResult(
                status="unsafe", reason=f"Function '{_function_name(node)}' is not allowed."
            )

    if not isinstance(tree, exp.Query):
        return GuardResult(
            status="invalid", reason="Not a SELECT query (could not parse as a query)."
        )

    tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    limited = _enforce_limit(tree, row_limit)
    rows = {k.lower(): v for k, v in (table_rows or {}).items()}
    return GuardResult(
        status="ok",
        sql=limited.sql(dialect=DIALECT),
        flags=_cost_flags(limited, rows),
        tables=tables,
    )
