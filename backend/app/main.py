"""SynapseCraft neural gateway — FastAPI + WebSocket cockpit server.

REST surface
────────────
    GET    /api/health              liveness + component inventory
    GET    /api/system              full system telemetry snapshot
    GET    /api/graph               connectome graph snapshot (cockpit render)
    GET    /api/documents           ingested document inventory
    DELETE /api/documents/{doc_id}  prune a document's subgraph + vectors
    POST   /api/ingest              multipart upload → ingestion pipeline
    POST   /api/ingest/url          web scout: fetch + ingest a live URL
    POST   /api/query               fire one query through the nervous system
    POST   /api/reset               wipe the connectome (graph + vectors + stats)
    WS     /ws/neural-activity      live spike stream for the visualiser
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agents.cognition import NervousSystem
from .agents.tools import scout_fetch, scout_preview
from .config import Settings
from .embeddings.embedder import build_embedder
from .events import EventBus, HELLO
from .ingestion.pipeline import ingest_document
from .llm.provider import LLMProvider
from .stats import Stats
from .stores.graph_store import ConnectomeGraph
from .stores.neo4j_mirror import Neo4jMirror
from .stores.vector_store import resolve_vector_store
from .textutils import truncate
from .types import make_spike

MAX_DOCS = 200


class Gateway:
    """Composition root: wires every neural subsystem together once."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.bus = EventBus()
        self.stats = Stats(self.settings.data_path).load()
        self.embedder = None
        self.store = None
        self.graph = None
        self.llm = None
        self.brain = None
        self.mirror = None
        self._ingest_lock = asyncio.Lock()

    async def startup(self) -> None:
        s = self.settings
        s.data_path.mkdir(parents=True, exist_ok=True)

        if s.neo4j_mirror == "on" or (s.neo4j_mirror == "auto" and s.neo4j_uri):
            self.mirror = Neo4jMirror(s)
        else:
            self.mirror = None

        self.graph = ConnectomeGraph(s, mirror=self.mirror)
        self.graph.load()
        self.embedder = await build_embedder(s)
        self.store = await resolve_vector_store(s, self.embedder)
        await self.store.load(self.embedder)
        self.llm = LLMProvider(s)
        self.brain = NervousSystem(s, self.embedder, self.store, self.graph,
                                   self.llm, self.bus)

    async def shutdown(self) -> None:
        if self.embedder:
            await self.embedder.aclose()
        if self.llm:
            await self.llm.aclose()
        if self.mirror:
            self.mirror.close()


gw = Gateway()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await gw.startup()
    yield
    await gw.shutdown()


app = FastAPI(title="SynapseCraft", version="1.0.0",
              description="Connectome-inspired agentic RAG nervous system",
              lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=gw.settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────
def _require_brain() -> NervousSystem:
    if gw.brain is None:
        raise HTTPException(503, "nervous system still booting")
    return gw.brain


# ── health & telemetry ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> dict:
    counts = gw.graph.counts() if gw.graph else {}
    return {
        "status": "online",
        "components": {
            "embedder": gw.embedder.describe() if gw.embedder else None,
            "vector_store": gw.store.mode if gw.store else None,
            "llm": gw.llm.describe() if gw.llm else None,
            "neo4j_mirror": bool(gw.mirror),
            "graph": counts,
        },
        "stats": gw.stats.snapshot(),
    }


@app.get("/api/system")
async def system() -> dict:
    _require_brain()
    return {
        "settings": {
            "gating_threshold": gw.settings.gating_threshold,
            "recall_k": gw.settings.recall_k,
            "top_k": gw.settings.top_k,
            "lateral_inhibition": gw.settings.lateral_inhibition,
            "propagation_decay": gw.settings.propagation_decay,
            "max_reflection_loops": gw.settings.max_reflection_loops,
            "critic_mode": gw.settings.critic_mode,
        },
        "components": {
            "embedder": gw.embedder.describe(),
            "vector_store": gw.store.mode,
            "llm": gw.llm.describe(),
            "neo4j_mirror": bool(gw.mirror),
        },
        "graph": gw.graph.counts(),
        "stats": gw.stats.snapshot(),
        "recent_events": gw.bus.recent(40),
    }


# ── connectome graph & documents ───────────────────────────────────────────
@app.get("/api/graph")
async def graph_snapshot(
    types: str | None = None,
    hub: str | None = None,
    max_chunks: int = 180,
) -> dict:
    _require_brain()
    include = types.split(",") if types else None
    return gw.graph.snapshot(include_types=include, hub=hub or None,
                             max_chunks=max(20, min(max_chunks, 600)))


@app.get("/api/documents")
async def documents() -> dict:
    _require_brain()
    docs = []
    for n, d in gw.graph.gx.nodes(data=True):
        if d.get("type") != "doc":
            continue
        nbrs = [(c, gw.graph.node(c) or {}) for c, _ in gw.graph.neighbors(n)]
        docs.append({
            "doc_id": d.get("doc_id", n.removeprefix("doc:")),
            "title": d.get("label", n),
            "chunks": sum(1 for _, nd in nbrs if nd.get("type") == "chunk"),
            "sections": sum(1 for _, nd in nbrs if nd.get("type") == "section"),
        })
    docs.sort(key=lambda d: d["title"].lower())
    return {"documents": docs, "total_chunks": gw.store.count()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    _require_brain()
    if not gw.graph.has_doc(doc_id):
        raise HTTPException(404, f"unknown document: {doc_id}")
    await gw.store.remove_document(doc_id)
    gw.graph.remove_document(doc_id)
    if gw.mirror:
        gw.mirror.remove_document(doc_id)
    gw.store.persist(gw.embedder.kind)
    gw.graph.persist()
    gw.bus.publish(make_spike(
        "ingest", node=f"doc:{doc_id}", polarity="inhibitory",
        message=f"pruned document {doc_id} from the connectome",
    ))
    return {"removed": doc_id, "graph": gw.graph.counts()}

# ── ingestion ──────────────────────────────────────────────────────────────
@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> JSONResponse:
    _require_brain()
    if not files:
        raise HTTPException(400, "no files supplied")
    if gw.graph.counts().get("documents", 0) >= MAX_DOCS:
        raise HTTPException(413, f"connectome capacity reached ({MAX_DOCS} docs)")

    max_bytes = gw.settings.max_upload_mb * 1024 * 1024
    results: list[dict] = []
    errors: list[dict] = []
    async with gw._ingest_lock:
        for f in files:
            try:
                content = await f.read()
            except Exception as exc:
                errors.append({"file": f.filename, "error": f"read failed: {exc}"})
                continue
            if len(content) > max_bytes:
                errors.append({
                    "file": f.filename,
                    "error": f"exceeds {gw.settings.max_upload_mb} MB limit",
                })
                continue
            if not content.strip():
                errors.append({"file": f.filename, "error": "empty file"})
                continue
            try:
                res = await ingest_document(
                    f.filename or "upload.bin", content,
                    gw.settings, gw.embedder, gw.store, gw.graph, gw.bus, "",
                )
                data = res.dict()
                gw.stats.observe_ingest(data.get("chunks") or 0)
                results.append(data)
            except Exception as exc:
                errors.append({"file": f.filename, "error": str(exc)[:200]})
    return JSONResponse({"ingested": results, "errors": errors,
                         "graph": gw.graph.counts()})

# ── ingestion: web scout ───────────────────────────────────────────────────
@app.post("/api/ingest/url")
async def ingest_url(body: dict) -> dict:
    _require_brain()
    url = str(body.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, 'body must be {"url": "https://…"}')
    if body.get("preview_only"):
        try:
            return await asyncio.to_thread(scout_preview, url)
        except Exception as exc:
            raise HTTPException(422, f"scout failed: {exc}")
    try:
        filename, content = await asyncio.to_thread(scout_fetch, url)
    except Exception as exc:
        raise HTTPException(422, f"scout failed: {exc}")
    async with gw._ingest_lock:
        res = await ingest_document(
            filename, content, gw.settings, gw.embedder,
            gw.store, gw.graph, gw.bus, url,
        )
    data = res.dict()
    gw.stats.observe_ingest(data.get("chunks") or 0)
    return {"ingested": [data], "errors": [], "graph": gw.graph.counts()}


# ── query: fire the nervous system ─────────────────────────────────────────
@app.post("/api/query")
async def query(body: dict) -> dict:
    brain = _require_brain()
    q = str(body.get("query") or "").strip()
    if not q:
        raise HTTPException(400, 'body must be {"query": "…"}')
    if len(q) > 2000:
        raise HTTPException(413, "query too long (max 2000 chars)")
    result = await brain.run(q)
    gw.stats.observe_query(
        latency_ms=result["latency_ms"],
        suppressed=result["suppressed"],
        tokens_saved=result["tokens_saved"],
        loops=result["loops"],
    )
    return result


# ── reset ──────────────────────────────────────────────────────────────────
@app.post("/api/reset")
async def reset() -> dict:
    _require_brain()
    async with gw._ingest_lock:
        await gw.store.clear()
        gw.store.persist(gw.embedder.kind)
        gw.graph.clear()
        if gw.mirror:
            gw.mirror.clear()
        gw.stats.__init__(gw.settings.data_path)
        gw.stats._persist()
    gw.bus.publish(make_spike("ingest", node="system", polarity="inhibitory",
                              message="connectome wiped — clean slate"))
    return {"reset": True, "graph": gw.graph.counts()}

# ── live neural activity stream ────────────────────────────────────────────
@app.websocket("/ws/neural-activity")
async def neural_activity(ws: WebSocket) -> None:
    await ws.accept()
    sub_id, queue = gw.bus.subscribe()
    try:
        await ws.send_json(make_spike(
            HELLO, message="neural link established",
            graph=gw.graph.counts() if gw.graph else {},
            recent=gw.bus.recent(30),
        ))
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                await ws.send_json(make_spike("heartbeat"))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        gw.bus.unsubscribe(sub_id)
        with contextlib.suppress(Exception):
            await ws.close()




