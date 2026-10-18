import pytest
from fastapi.testclient import TestClient

from conftest import turn_script
from sql_analyst.api.app import app

FLAGGED = "SELECT * FROM Artist, Album"


@pytest.fixture
def client_for(fake_service):
    clients = []

    def _make(script):
        app.state.service = fake_service(script)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
    app.state.service = None


def test_health(client_for):
    r = client_for({}).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": "fake", "tracing": False}


def test_ask_creates_thread_and_answers(client_for):
    c = client_for(turn_script())
    body = c.post("/ask", json={"question": "How many artists?"}).json()
    assert body["status"] == "answered"
    assert body["answer"] == "There are 3 artists."
    assert body["rows"] == [[3]]
    assert body["thread_id"] and body["run_id"]
    history = c.get(f"/threads/{body['thread_id']}/history").json()
    assert history == [{"question": "How many artists?", "answer": "There are 3 artists."}]


def test_ask_validates_input(client_for):
    c = client_for({})
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "x" * 1001}).status_code == 422
    assert c.post("/ask", json={"question": "q", "thread_id": "../etc"}).status_code == 422


def test_approval_flow(client_for):
    c = client_for(turn_script(sql=FLAGGED, answer="Rows shown."))
    tid = c.post("/threads").json()["thread_id"]
    first = c.post("/ask", json={"question": "q", "thread_id": tid}).json()
    assert first["status"] == "awaiting_approval"
    assert "cartesian_join" in first["approval"]["reasons"]
    assert c.get(f"/threads/{tid}/pending").json()["sql"].startswith("SELECT *")

    done = c.post(f"/threads/{tid}/approval", json={"approved": True}).json()
    assert done["status"] == "answered"
    assert done["row_count"] == 6
    assert c.get(f"/threads/{tid}/pending").json() is None


def test_approval_without_pending_is_conflict(client_for):
    c = client_for({})
    r = c.post("/threads/abc/approval", json={"approved": True})
    assert r.status_code == 409


def test_invalid_thread_id_in_path(client_for):
    assert client_for({}).get("/threads/bad.id!/history").status_code == 422


def test_feedback_noop_without_tracing(client_for):
    r = client_for({}).post("/feedback", json={"run_id": "abc", "score": 1})
    assert r.json() == {"recorded": False}
    assert client_for({}).post("/feedback", json={"run_id": "a", "score": 5}).status_code == 422


def test_schema(client_for):
    tables = {t["name"]: t for t in client_for({}).get("/schema").json()}
    assert tables["Artist"] == {"name": "Artist", "columns": ["ArtistId", "Name"], "row_count": 3}
