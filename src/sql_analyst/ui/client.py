"""How the Streamlit UI talks to the agent.

* ``LocalClient`` runs the graph in-process: one container, ideal for free
  Hugging Face Spaces / Streamlit Community Cloud deployments.
* ``HttpClient`` calls the FastAPI backend: used by docker-compose, where UI
  and API run as separate services.

Set ``API_URL`` to use the HTTP client; otherwise the UI runs locally.
"""

import os
from typing import Protocol

import httpx

from sql_analyst.graph.service import AnalystService, TurnResult


class AnalystClient(Protocol):
    def new_thread_id(self) -> str: ...
    def ask(self, thread_id: str, question: str) -> TurnResult: ...
    def approve(self, thread_id: str, approved: bool, sql: str | None = None) -> TurnResult: ...
    def feedback(self, run_id: str, score: float) -> bool: ...
    def schema(self) -> list[dict]: ...
    def info(self) -> dict: ...


class LocalClient:
    def __init__(self, service: AnalystService):
        self.service = service

    def new_thread_id(self) -> str:
        return self.service.new_thread_id()

    def ask(self, thread_id, question):
        return self.service.ask(thread_id, question)

    def approve(self, thread_id, approved, sql=None):
        return self.service.resume(thread_id, approved=approved, sql=sql)

    def feedback(self, run_id, score):
        from sql_analyst.tracing import log_feedback

        return log_feedback(run_id, score)

    def schema(self):
        from sql_analyst.tools.schema_introspect import get_schema

        if self.service.db_path is None:
            return []
        return [
            {"name": t.name, "columns": [c.name for c in t.columns], "row_count": t.row_count}
            for t in get_schema(self.service.db_path)
        ]

    def info(self):
        from sql_analyst.tracing import tracing_enabled

        return {
            "llm": self.service.metadata.get("llm"),
            "tracing": tracing_enabled(),
            "mode": "local",
        }


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        # Long timeout: a turn may wait out Groq rate-limit backoff.
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def new_thread_id(self) -> str:
        return self.http.post("/threads").raise_for_status().json()["thread_id"]

    def ask(self, thread_id, question):
        r = self.http.post("/ask", json={"question": question, "thread_id": thread_id})
        return TurnResult.model_validate(r.raise_for_status().json())

    def approve(self, thread_id, approved, sql=None):
        r = self.http.post(
            f"/threads/{thread_id}/approval", json={"approved": approved, "sql": sql}
        )
        return TurnResult.model_validate(r.raise_for_status().json())

    def feedback(self, run_id, score):
        r = self.http.post("/feedback", json={"run_id": run_id, "score": score})
        return r.raise_for_status().json()["recorded"]

    def schema(self):
        return self.http.get("/schema").raise_for_status().json()

    def info(self):
        return {**self.http.get("/health").raise_for_status().json(), "mode": "api"}


def get_client() -> AnalystClient:
    api_url = os.environ.get("API_URL")
    if api_url:
        return HttpClient(api_url)
    from sql_analyst.graph.service import create_service

    return LocalClient(create_service())
