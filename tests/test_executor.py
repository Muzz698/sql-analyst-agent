import pytest

from sql_analyst.tools.executor import QueryError, QueryTimeoutError, execute_sql


def test_executes_select(tiny_db):
    r = execute_sql("SELECT Name FROM Artist ORDER BY ArtistId", tiny_db)
    assert r.columns == ["Name"]
    assert r.rows == [["AC/DC"], ["Accept"], ["Aerosmith"]]
    assert r.row_count == 3
    assert not r.truncated


def test_truncates_to_max_rows(tiny_db):
    r = execute_sql("SELECT * FROM Artist", tiny_db, max_rows=2)
    assert r.row_count == 2
    assert r.truncated


def test_empty_result(tiny_db):
    r = execute_sql("SELECT * FROM Artist WHERE 1 = 0", tiny_db)
    assert r.row_count == 0
    assert r.columns == ["ArtistId", "Name"]


def test_sql_error_raises_query_error_with_message(tiny_db):
    with pytest.raises(QueryError, match="no such column"):
        execute_sql("SELECT Nme FROM Artist", tiny_db)


def test_connection_is_read_only_even_if_guard_bypassed(tiny_db):
    """Defence in depth: the DB itself refuses writes."""
    with pytest.raises(QueryError, match=r"readonly|read-only|query_only"):
        execute_sql("DELETE FROM Artist", tiny_db)
    assert execute_sql("SELECT count(*) FROM Artist", tiny_db).rows == [[3]]


def test_timeout_aborts_long_query(tiny_db):
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    with pytest.raises(QueryTimeoutError):
        execute_sql(endless, tiny_db, timeout_ms=100)


def test_missing_database(tmp_path):
    with pytest.raises(QueryError, match="not found"):
        execute_sql("SELECT 1", tmp_path / "nope.sqlite")


def test_chinook_smoke(chinook_db):
    assert execute_sql("SELECT count(*) FROM Invoice", chinook_db).rows == [[412]]
