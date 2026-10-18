"""FastAPI backend.

Run:  uv run uvicorn sql_analyst.api.app:app --reload
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from pydantic import BaseModel, Field

from sql_analyst.graph.service import AnalystService, ApprovalRequest, TurnResult, create_service
from sql_analyst.tools.schema_introspect import get_schema
from sql_analyst.tracing import log_feedback, tracing_enabled

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN)


class ApprovalDecision(BaseModel):
    approved: bool
    sql: str | None = Field(default=None, max_length=10_000, description="Optional edited SQL")


class FeedbackRequest(BaseModel):
    run_id: str = Field(max_length=64)
    score: float = Field(ge=0, le=1, description="1 = helpful, 0 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)


class TableSummary(BaseModel):
    name: str
    columns: list[str]
    row_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build once at startup (loads schema, opens checkpointer). Tests pre-set app.state.service.
    if getattr(app.state, "service", None) is None:
        app.state.service = create_service()
    yield


app = FastAPI(
    title="SQL Analyst Agent",
    description="Ask business questions in plain English; get safe SQL, answers and charts.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> AnalystService:
    return request.app.state.service


Service = Annotated[AnalystService, Depends(get_service)]
ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]


@app.get("/health")
def health(service: Service) -> dict:
    return {"status": "ok", "llm": service.metadata.get("llm"), "tracing": tracing_enabled()}


@app.post("/threads")
def new_thread(service: Service) -> dict:
    return {"thread_id": service.new_thread_id()}


@app.post("/ask", response_model=TurnResult)
def ask(body: AskRequest, service: Service) -> TurnResult:
    thread_id = body.thread_id or service.new_thread_id()
    return service.ask(thread_id, body.question.strip())


@app.get("/threads/{thread_id}/pending", response_model=ApprovalRequest | None)
def pending(thread_id: ThreadId, service: Service) -> ApprovalRequest | None:
    return service.pending_approval(thread_id)


@app.post("/threads/{thread_id}/approval", response_model=TurnResult)
def approve(thread_id: ThreadId, body: ApprovalDecision, service: Service) -> TurnResult:
    if service.pending_approval(thread_id) is None:
        raise HTTPException(status_code=409, detail="No query is awaiting approval on this thread")
    return service.resume(thread_id, approved=body.approved, sql=body.sql)


@app.get("/threads/{thread_id}/history")
def history(thread_id: ThreadId, service: Service) -> list[dict]:
    return service.history(thread_id)


@app.post("/feedback")
def feedback(body: FeedbackRequest) -> dict:
    return {"recorded": log_feedback(body.run_id, body.score, body.comment)}


@app.get("/schema", response_model=list[TableSummary])
def schema(service: Service) -> list[TableSummary]:
    return [
        TableSummary(name=t.name, columns=[c.name for c in t.columns], row_count=t.row_count)
        for t in get_schema(service.db_path)
    ]
