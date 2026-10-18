"""Read-only, time-limited SQL execution against SQLite.

Defence in depth: even if a write slipped past the safety guard, the connection
is opened with ``mode=ro`` and ``PRAGMA query_only``, so SQLite itself refuses.
"""

import sqlite3
import time
from pathlib import Path

from sql_analyst.schemas import QueryResult

# How many SQLite VM instructions between timeout checks.
_PROGRESS_INTERVAL = 10_000


class QueryError(Exception):
    """Raised for any execution failure; the message is fed back to the SQL writer."""


class QueryTimeoutError(QueryError):
    pass


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise QueryError(f"Database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute_sql(
    sql: str,
    db_path: Path | str,
    timeout_ms: int = 5_000,
    max_rows: int = 200,
) -> QueryResult:
    conn = connect_readonly(db_path)
    start = time.monotonic()
    deadline = start + timeout_ms / 1000

    # Returning non-zero from the progress handler makes SQLite abort the query.
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), _PROGRESS_INTERVAL)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        fetched = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise QueryTimeoutError(f"Query exceeded timeout of {timeout_ms} ms.") from e
        raise QueryError(str(e)) from e
    except sqlite3.Error as e:
        raise QueryError(str(e)) from e
    finally:
        conn.close()

    truncated = len(fetched) > max_rows
    rows = [list(r) for r in fetched[:max_rows]]
    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_ms=round((time.monotonic() - start) * 1000, 2),
    )
