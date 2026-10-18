import sqlite3
from pathlib import Path

import pytest

from sql_analyst.config import Settings, get_settings

CHINOOK = Path(__file__).resolve().parents[1] / "data" / "chinook.sqlite"


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Force the fake LLM and ignore any developer .env so tests are hermetic and free."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def chinook_db() -> Path:
    if not CHINOOK.exists():
        pytest.skip("Chinook not downloaded; run `uv run python data/download_chinook.py`")
    return CHINOOK


@pytest.fixture
def tiny_db(tmp_path) -> Path:
    """Small self-contained DB so core tests don't depend on the download."""
    path = tmp_path / "tiny.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT);
            CREATE TABLE Album (
                AlbumId INTEGER PRIMARY KEY, Title TEXT,
                ArtistId INTEGER REFERENCES Artist(ArtistId)
            );
            INSERT INTO Artist VALUES (1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith');
            INSERT INTO Album VALUES (1, 'For Those About To Rock', 1), (2, 'Balls to the Wall', 2);
            """
        )
    return path


# --- Shared fake-LLM service builder (API + UI tests) ------------------------


def turn_script(
    sql: str = "SELECT COUNT(*) AS n FROM Artist", answer: str = "There are 3 artists."
):
    """FakeLLM script for one successful SQL turn."""
    return {
        "RouterDecision": [
            {"reasoning": "r", "intent": "sql", "standalone_question": "q", "confidence": 0.95}
        ],
        "TableSelection": [{"reasoning": "r", "tables": ["Artist"]}],
        "SQLDraft": [{"reasoning": "r", "sql": sql}],
        "AnalystOutput": [{"answer": answer, "chart_recommended": False}],
    }


@pytest.fixture
def fake_service(tiny_db):
    from langgraph.checkpoint.memory import InMemorySaver

    from sql_analyst.graph.service import create_service
    from sql_analyst.llm import FakeLLM

    def _make(script):
        settings = Settings(llm_provider="fake", db_path=tiny_db)
        return create_service(settings, llm=FakeLLM(script), checkpointer=InMemorySaver())

    return _make
