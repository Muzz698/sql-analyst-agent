import pytest
from pydantic import ValidationError

from sql_analyst.config import Settings, get_settings


def test_defaults_are_safe_and_free():
    s = get_settings()
    assert s.llm_provider == "fake"
    assert s.groq_api_key is None
    assert s.max_sql_retries == 3
    assert s.groq_model == "llama-3.3-70b-versatile"


def test_secret_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecret")
    s = Settings()
    assert "gsk_supersecret" not in repr(s)
    assert s.groq_api_key.get_secret_value() == "gsk_supersecret"


def test_invalid_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        Settings()
