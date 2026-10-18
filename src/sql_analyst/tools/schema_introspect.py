"""Database schema introspection and prompt rendering.

The Schema Agent picks relevant tables; these helpers supply the raw metadata,
add the tables needed to join them (via foreign keys), and render compact
DDL-style text for the SQL writer prompt.
"""

from collections import deque
from functools import lru_cache
from pathlib import Path

from sql_analyst.schemas import ColumnInfo, ForeignKey, TableInfo
from sql_analyst.tools.executor import connect_readonly


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=8)
def _load_schema(db_path: str, sample_rows: int) -> tuple[TableInfo, ...]:
    conn = connect_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables = []
        for name in names:
            q = _quote(name)
            cols = [
                ColumnInfo(name=c[1], type=c[2] or "", primary_key=bool(c[5]), nullable=not c[3])
                for c in conn.execute(f"PRAGMA table_info({q})")
            ]
            fks = [
                ForeignKey(column=f[3], ref_table=f[2], ref_column=f[4])
                for f in conn.execute(f"PRAGMA foreign_key_list({q})")
            ]
            count = conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0]
            samples = [list(r) for r in conn.execute(f"SELECT * FROM {q} LIMIT {int(sample_rows)}")]
            tables.append(
                TableInfo(
                    name=name, columns=cols, foreign_keys=fks, row_count=count, sample_rows=samples
                )
            )
        return tuple(tables)
    finally:
        conn.close()


def get_schema(db_path: Path | str, sample_rows: int = 3) -> list[TableInfo]:
    return list(_load_schema(str(Path(db_path).resolve()), sample_rows))


def table_row_counts(db_path: Path | str) -> dict[str, int]:
    return {t.name: t.row_count for t in get_schema(db_path)}


def _adjacency(tables: list[TableInfo]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {t.name: set() for t in tables}
    for t in tables:
        for fk in t.foreign_keys:
            if fk.ref_table in graph:
                graph[t.name].add(fk.ref_table)
                graph[fk.ref_table].add(t.name)
    return graph


def expand_with_join_path(selected: list[str], tables: list[TableInfo]) -> list[str]:
    """Add bridge tables so every selected table is reachable via foreign keys.

    E.g. {Customer, Track} -> {Customer, Invoice, InvoiceLine, Track}. Uses BFS
    shortest paths from the first selected table to each other one.
    """
    by_lower = {t.name.lower(): t.name for t in tables}
    wanted = [by_lower[s.lower()] for s in selected if s.lower() in by_lower]
    if len(wanted) <= 1:
        return wanted
    graph = _adjacency(tables)
    result = set(wanted)
    root = wanted[0]
    for target in wanted[1:]:
        parent: dict[str, str | None] = {root: None}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for nxt in sorted(graph[node]):
                if nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        step: str | None = target if target in parent else None
        while step is not None:
            result.add(step)
            step = parent[step]
    return [t.name for t in tables if t.name in result]


def render_schema(
    tables: list[TableInfo], include_names: list[str] | None = None, with_samples: bool = True
) -> str:
    """Render compact CREATE TABLE-style text for an LLM prompt."""
    keep = {n.lower() for n in include_names} if include_names is not None else None
    blocks = []
    for t in tables:
        if keep is not None and t.name.lower() not in keep:
            continue
        lines = []
        for c in t.columns:
            extra = " PRIMARY KEY" if c.primary_key else ""
            lines.append(f"  {c.name} {c.type}{extra}")
        for fk in t.foreign_keys:
            lines.append(f"  FOREIGN KEY ({fk.column}) REFERENCES {fk.ref_table}({fk.ref_column})")
        block = f"CREATE TABLE {t.name} (  -- {t.row_count} rows\n" + ",\n".join(lines) + "\n);"
        if with_samples and t.sample_rows:
            header = ", ".join(c.name for c in t.columns)
            rows = "\n".join(f"--   {tuple(r)}" for r in t.sample_rows)
            block += f"\n-- sample rows ({header}):\n{rows}"
        blocks.append(block)
    return "\n\n".join(blocks)


def table_summaries(tables: list[TableInfo]) -> str:
    """One line per table (name + columns), used by the Schema Agent to choose tables."""
    return "\n".join(f"- {t.name}: {', '.join(c.name for c in t.columns)}" for t in tables)
