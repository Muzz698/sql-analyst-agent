"""LangSmith tracing setup.

The LangSmith SDK reads ``LANGSMITH_*`` from ``os.environ``, but pydantic-settings
loads ``.env`` without exporting it. ``configure_tracing`` bridges the two and
guarantees tracing is *off* unless both the flag and a key are present, so a
missing key never breaks the app (or leaks traces to a default project).
"""

import logging
import os
from typing import Any

from sql_analyst.config import Settings, get_settings

log = logging.getLogger(__name__)

_state = {"enabled": False}


def _has_key(settings: Settings) -> bool:
    key = settings.langsmith_api_key
    return bool(key and key.get_secret_value().strip())


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing if configured. Returns whether tracing is on."""
    settings = settings or get_settings()
    enabled = settings.langsmith_tracing and _has_key(settings)
    if settings.langsmith_tracing and not enabled:
        log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing disabled.")
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    _state["enabled"] = enabled
    return enabled


def tracing_enabled() -> bool:
    return _state["enabled"]


# Per-node trace metadata: lets you filter LangSmith runs by kind (agent / tool / control).
NODE_METADATA: dict[str, dict[str, Any]] = {
    "start_turn": {"kind": "control"},
    "router": {"kind": "agent", "uses_llm": True},
    "schema_agent": {"kind": "agent", "uses_llm": True},
    "sql_writer": {"kind": "agent", "uses_llm": True},
    "safety_guard": {"kind": "tool"},
    "human_approval": {"kind": "control", "interrupts": True},
    "executor": {"kind": "tool"},
    "analyst": {"kind": "agent", "uses_llm": True},
    "chart_agent": {"kind": "agent", "uses_llm": True},
    "finalize": {"kind": "control"},
    "clarify": {"kind": "terminal"},
    "out_of_scope": {"kind": "terminal"},
    "blocked": {"kind": "terminal"},
    "rejected": {"kind": "terminal"},
    "failed": {"kind": "terminal"},
}


def log_feedback(run_id: str, score: float, comment: str | None = None) -> bool:
    """Attach user feedback (e.g. thumbs up/down) to a traced run. No-op if tracing is off."""
    if not tracing_enabled():
        return False
    from langsmith import Client

    try:
        Client().create_feedback(run_id, key="user_score", score=score, comment=comment)
        return True
    except Exception:  # feedback is best-effort; never break the user flow
        log.exception("failed to send feedback to LangSmith")
        return False
