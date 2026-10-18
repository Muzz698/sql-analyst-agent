from sql_analyst.tools.schema_introspect import (
    expand_with_join_path,
    get_schema,
    render_schema,
    table_row_counts,
    table_summaries,
)


def test_reads_tables_columns_and_fks(tiny_db):
    tables = {t.name: t for t in get_schema(tiny_db)}
    assert set(tables) == {"Artist", "Album"}
    album = tables["Album"]
    assert [c.name for c in album.columns] == ["AlbumId", "Title", "ArtistId"]
    assert album.columns[0].primary_key
    assert album.foreign_keys[0].ref_table == "Artist"
    assert tables["Artist"].row_count == 3


def test_render_schema_filters_and_includes_fks(tiny_db):
    text = render_schema(get_schema(tiny_db), include_names=["album"])
    assert "CREATE TABLE Album" in text
    assert "CREATE TABLE Artist" not in text
    assert "REFERENCES Artist(ArtistId)" in text
    assert "sample rows" in text


def test_table_summaries(tiny_db):
    assert "- Artist: ArtistId, Name" in table_summaries(get_schema(tiny_db))


def test_join_path_adds_bridge_tables(chinook_db):
    tables = get_schema(chinook_db)
    result = expand_with_join_path(["Customer", "Genre"], tables)
    assert set(result) == {"Customer", "Invoice", "InvoiceLine", "Track", "Genre"}


def test_join_path_ignores_unknown_and_single(chinook_db):
    tables = get_schema(chinook_db)
    assert expand_with_join_path(["artist", "Nope"], tables) == ["Artist"]


def test_row_counts(chinook_db):
    counts = table_row_counts(chinook_db)
    assert counts["Invoice"] == 412
    assert counts["Track"] == 3503
