"""SQLite checkpointer: persists every thread's state so conversations survive
restarts, follow-ups have memory, and interrupted (awaiting-approval) runs can resume."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def sqlite_checkpointer(path: Path | str) -> SqliteSaver:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI/Streamlit may call from worker threads.
    # SqliteSaver serialises access with its own lock.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
