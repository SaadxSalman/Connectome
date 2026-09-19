"""HTTP gateway tests via the FastAPI TestClient (lifespan included)."""

from tests.conftest import SAMPLE_MD


def test_health_and_system(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "online"
    assert body["components"]["vector_store"] == "memory"
    assert body["components"]["embedder"]["provider"] == "hash"

    r2 = client.get("/api/system")
    assert r2.status_code == 200
    assert "gating_threshold" in r2.json()["settings"]


def test_full_lifecycle(client):
    # 1 · ingest a markdown file (multipart)
    r = client.post(
        "/api/ingest",
        files={"files": ("field_guide.md", SAMPLE_MD.encode(), "text/markdown")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["errors"] == []
    assert data["ingested"][0]["chunks"] > 0
    doc_id = data["ingested"][0]["doc_id"]

    # 2 · documents inventory
    docs = client.get("/api/documents").json()["documents"]
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3 · graph snapshot renders nodes + edges + agents
    snap = client.get("/api/graph", params={"max_chunks": 120}).json()
    types = {n["type"] for n in snap["nodes"]}
    assert {"agent", "doc", "chunk"} <= types
    assert snap["edges"]

    # 4 · query fires the nervous system
    q = client.post("/api/query", json={"query": "T4 neurons motion optic lobe"})
    assert q.status_code == 200, q.text
    body = q.json()
    assert body["answer"]
    assert body["fired"]
    assert body["grounded"] is True

    # 5 · stats updated
    stats = client.get("/api/health").json()["stats"]
    assert stats["queries"] >= 1
    assert stats["ingestions"] >= 1

    # 6 · delete the document again
    d = client.delete(f"/api/documents/{doc_id}")
    assert d.status_code == 200
    assert client.get("/api/documents").json()["documents"] == []


def test_query_validation(client):
    assert client.post("/api/query", json={}).status_code == 400
    assert client.post("/api/query", json={"query": "   "}).status_code == 400


def test_unknown_document_404(client):
    assert client.delete("/api/documents/nope").status_code == 404


def test_ingest_url_rejects_bad_scheme(client):
    r = client.post("/api/ingest/url", json={"url": "ftp://example.com"})
    assert r.status_code == 400


def test_websocket_neural_stream(client):
    client.post(
        "/api/ingest",
        files={"files": ("guide.md", SAMPLE_MD.encode(), "text/markdown")},
    )
    with client.websocket_connect("/ws/neural-activity") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        q = client.post("/api/query", json={"query": "optic lobe motion"})
        assert q.status_code == 200
        seen = {hello["type"]}
        for _ in range(80):
            evt = ws.receive_json()
            seen.add(evt["type"])
            if evt["type"] == "run_end":
                break
        assert "run_start" in seen and "run_end" in seen
