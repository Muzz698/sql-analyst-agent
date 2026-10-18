import pytest

from sql_analyst.tools.safety_guard import validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT Name FROM Artist",
        "select name from artist where artistid = 1",
        "WITH t AS (SELECT 1 AS a) SELECT a FROM t",
        "SELECT Name FROM Artist UNION SELECT Name FROM Genre",
        "SELECT a.Name, count(*) FROM Artist a JOIN Album b ON a.ArtistId = b.ArtistId GROUP BY 1",
        "SELECT strftime('%Y', InvoiceDate) AS y, sum(Total) FROM Invoice GROUP BY y",
    ],
)
def test_allows_read_only_queries(sql):
    result = validate_sql(sql)
    assert result.status == "ok", result.reason
    assert "LIMIT" in result.sql.upper()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Track",
        "INSERT INTO Artist (Name) VALUES ('x')",
        "UPDATE Customer SET Email = NULL",
        "DROP TABLE Invoice",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE Artist ADD COLUMN x INT",
        "PRAGMA table_info(Track)",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "VACUUM",
        "SELECT 1; DROP TABLE Track",
        "SELECT * FROM Artist; DELETE FROM Artist",
        # Write hidden inside a CTE: top-level node is a SELECT.
        "WITH d AS (DELETE FROM Track RETURNING *) SELECT * FROM d",
        "SELECT load_extension('/tmp/evil.so')",
        "/* harmless */ DrOp /* sneaky */ TABLE Track",
    ],
)
def test_blocks_unsafe_queries(sql):
    result = validate_sql(sql)
    assert result.status == "unsafe", f"{sql!r} -> {result}"
    assert result.sql is None


@pytest.mark.parametrize("sql", ["", "   ", "SELEC * FRM Track", "SELECT FROM WHERE"])
def test_invalid_sql_is_retryable_not_unsafe(sql):
    assert validate_sql(sql).status == "invalid"


def test_multiple_selects_rejected_as_invalid():
    assert validate_sql("SELECT 1; SELECT 2").status == "invalid"


def test_adds_limit_when_missing():
    assert validate_sql("SELECT Name FROM Artist", row_limit=50).sql.endswith("LIMIT 50")


def test_keeps_small_limit():
    assert validate_sql("SELECT Name FROM Artist LIMIT 5", row_limit=50).sql.endswith("LIMIT 5")


def test_caps_large_limit():
    assert validate_sql("SELECT Name FROM Artist LIMIT 99999", row_limit=50).sql.endswith(
        "LIMIT 50"
    )


def test_limit_on_union_applies_to_whole_query():
    sql = validate_sql("SELECT Name FROM Artist UNION SELECT Name FROM Genre", row_limit=10).sql
    assert sql.endswith("LIMIT 10")
    assert sql.upper().count("LIMIT") == 1


def test_reports_tables():
    r = validate_sql("SELECT * FROM Artist a JOIN Album b ON a.ArtistId = b.ArtistId")
    assert r.tables == ["Album", "Artist"]


def test_flags_many_joins():
    sql = (
        "SELECT * FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId "
        "JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId "
        "JOIN Track t ON il.TrackId = t.TrackId "
        "JOIN Album a ON t.AlbumId = a.AlbumId"
    )
    assert "many_joins:4" in validate_sql(sql).flags


def test_flags_cartesian_join():
    assert "cartesian_join" in validate_sql("SELECT * FROM Artist CROSS JOIN Track").flags
    assert "cartesian_join" in validate_sql("SELECT * FROM Artist, Track").flags


def test_implicit_join_with_where_not_flagged():
    sql = "SELECT * FROM Artist a, Album b WHERE a.ArtistId = b.ArtistId"
    assert "cartesian_join" not in validate_sql(sql).flags


def test_flags_full_scan_of_large_table_only():
    rows = {"Track": 3503, "Genre": 25}
    assert "full_scan:Track" in validate_sql("SELECT * FROM Track", table_rows=rows).flags
    assert validate_sql("SELECT * FROM Genre", table_rows=rows).flags == []
    filtered = validate_sql("SELECT * FROM Track WHERE GenreId = 1", table_rows=rows)
    assert filtered.flags == []
