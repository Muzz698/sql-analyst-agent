import pytest

from sql_analyst.config import Settings, get_settings


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
