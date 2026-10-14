import pytest
from fastapi.testclient import TestClient

from docanswer.api import create_app


@pytest.fixture(scope="module")
def client(settings, index):
    return TestClient(create_app(settings))


def test_ask_returns_answer_with_citations(client):
    res = client.post("/ask", json={"question": "How many remote days per week are allowed?", "role": "employee"})
    body = res.json()
    assert res.status_code == 200 and not body["refused"]
    assert body["citations"][0] == {**body["citations"][0], "doc": "Employee Handbook", "page": 3}
    assert 0 < body["confidence"] <= 1


def test_ask_respects_role(client):
    q = {"question": "What is the annual bonus target percentage?"}
    employee = client.post("/ask", json={**q, "role": "employee"}).json()
    manager = client.post("/ask", json={**q, "role": "manager"}).json()
    assert employee["refused"] and employee["citations"] == []
    assert not manager["refused"] and manager["citations"][0]["doc"] == "Manager Compensation Guidelines"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi there", "role": "root"}).status_code == 422
    assert client.post("/ask", json={"question": "", "role": "employee"}).status_code == 422


def test_ingest_requires_token_when_configured(settings, index):
    from dataclasses import replace

    secured = TestClient(create_app(replace(settings, ingest_token="s3cret")))
    assert secured.post("/ingest").status_code == 401
    ok = secured.post("/ingest", headers={"X-Ingest-Token": "s3cret"})
    assert ok.status_code == 200 and ok.json()["documents"] == 6


def test_ui_and_health(client):
    assert "DocAnswer" in client.get("/").text
    assert client.get("/health").json()["llm"] == "extractive"
