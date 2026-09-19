"""Vector memory: in-process NumPy store (default) or Qdrant via REST.

Both backends speak the same tiny protocol, so SynapseCraft can run:

• ``memory`` — zero-dependency, persisted to ``DATA_DIR/vector_store.npz``
  + ``vector_meta.json``. Perfect for local demos and CI.
• ``qdrant`` — Qdrant Cloud / self-hosted via plain HTTPX REST calls
  (no heavy client SDK required).

Chunks themselves live in the vector payloads — the vector store is the
single source of truth for chunk bodies and metadata.
"""

from __future__ import annotations

import json
import uuid
from typing import NamedTuple

import httpx
import numpy as np

from ..config import Settings
from ..types import Chunk


class SearchHit(NamedTuple):
    id: str
    score: float


# ════════════════════════════════════════════════════════════════════════════
# Shared protocol
# ════════════════════════════════════════════════════════════════════════════
class BaseVectorStore:
    mode = "base"

    async def upsert(self, chunks: list[dict], vectors) -> None:
        raise NotImplementedError

    async def search(self, vector: list[float], k: int) -> list[SearchHit]:
        raise NotImplementedError

    async def get_vector(self, cid: str) -> list[float] | None:
        raise NotImplementedError

    async def get(self, cid: str) -> dict | None:
        raise NotImplementedError

    async def all_chunks(self) -> list[dict]:
        raise NotImplementedError

    async def remove_document(self, doc_id: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    async def load(self, embedder) -> None:
        raise NotImplementedError

    def persist(self, embed_kind: str | None = None) -> None:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# In-process NumPy store (default — zero-dependency, persisted to disk)
# ════════════════════════════════════════════════════════════════════════════
class MemoryVectorStore(BaseVectorStore):
    mode = "memory"
    kind = "memory"  # embed-kind tag used by persist() when none is supplied

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._meta: dict[str, dict] = {}
        self._vecs: np.ndarray | None = None
        self._dim: int | None = None
        self._file_npz = settings.data_path / "vector_store.npz"
        self._file_meta = settings.data_path / "vector_meta.json"

    # ── writes ─────────────────────────────────────────────────────────────
    async def upsert(self, chunks: list[dict], vectors) -> None:
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        incoming = [c["id"] for c in chunks]
        dim = vectors.shape[1]
        self._dim = dim
        keep = [cid for cid in self._ids if cid not in set(incoming)]
        if keep and self._vecs is not None and self._vecs.size:
            rows = self._vecs[[self._pos[cid] for cid in keep]]
        else:
            rows = np.zeros((0, dim), dtype=np.float32)
        self._rebuild(keep + incoming, np.vstack([rows, vectors]))
        for c in chunks:
            self._meta[c["id"]] = c

    def _rebuild(self, ids: list[str], rows: np.ndarray) -> None:
        self._ids = list(ids)
        self._vecs = rows
        self._pos = {cid: i for i, cid in enumerate(self._ids)}

    async def remove_document(self, doc_id: str) -> None:
        victims = {cid for cid, m in self._meta.items() if m.get("doc_id") == doc_id}
        if not victims:
            return
        keep = [cid for cid in self._ids if cid not in victims]
        if keep and self._vecs is not None and self._vecs.size:
            rows = self._vecs[[self._pos[c] for c in keep]]
        else:
            rows = np.zeros((0, self._dim or 0), dtype=np.float32)
        for v in victims:
            self._meta.pop(v, None)
        self._rebuild(keep, rows)

    async def clear(self) -> None:
        self._ids, self._pos, self._meta = [], {}, {}
        self._vecs, self._dim = None, None

    # ── reads ──────────────────────────────────────────────────────────────
    async def search(self, vector: list[float], k: int) -> list[SearchHit]:
        if self._vecs is None or not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        sims = self._vecs @ q
        kk = min(k, len(self._ids))
        idx = np.argsort(-sims)[:kk]
        return [SearchHit(self._ids[int(i)], float(sims[int(i)])) for i in idx]

    async def get_vector(self, cid: str) -> list[float] | None:
        i = self._pos.get(cid)
        if i is None or self._vecs is None:
            return None
        return self._vecs[i].tolist()

    async def get(self, cid: str) -> dict | None:
        return self._meta.get(cid)

    async def all_chunks(self) -> list[dict]:
        return list(self._meta.values())

    def count(self) -> int:
        return len(self._ids)

    # ── persistence ────────────────────────────────────────────────────────
    def persist(self, embed_kind: str | None = None) -> None:
        try:
            self.s.data_path.mkdir(parents=True, exist_ok=True)
            if self._vecs is not None and self._ids:
                np.savez_compressed(
                    self._file_npz, ids=np.array(self._ids), vectors=self._vecs
                )
            elif self._file_npz.exists():
                self._file_npz.unlink()
            self._file_meta.write_text(
                json.dumps(
                    {"embed_kind": embed_kind or self.kind, "chunks": self._meta},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    async def load(self, embedder) -> None:
        try:
            if not (self._file_npz.exists() and self._file_meta.exists()):
                return
            data = np.load(self._file_npz, allow_pickle=False)
            ids = [str(x) for x in data["ids"].tolist()]
            vecs = np.asarray(data["vectors"], dtype=np.float32)
            meta_raw = json.loads(self._file_meta.read_text(encoding="utf-8"))
            meta = meta_raw.get("chunks", {})
            self._meta = {k: v for k, v in meta.items() if k in set(ids)}
            self._rebuild(ids, vecs)
            # If the embedding cortex changed since the last run, re-embed the
            # whole corpus so the vector space stays coherent.
            if meta_raw.get("embed_kind") and meta_raw["embed_kind"] != embedder.kind and ids:
                texts = [
                    Chunk(**self._meta[cid]).embed_text()
                    for cid in self._ids
                    if cid in self._meta
                ]
                if texts:
                    self._vecs = np.asarray(await embedder.embed(texts), dtype=np.float32)
                    self._dim = self._vecs.shape[1]
        except Exception:
            # A corrupted store must never brick the gateway — start clean.
            await self.clear()


# ════════════════════════════════════════════════════════════════════════════
# Qdrant store (Cloud / self-hosted) via plain REST — no client SDK needed
# ════════════════════════════════════════════════════════════════════════════
class QdrantStore(BaseVectorStore):
    mode = "qdrant"

    def __init__(self, settings: Settings, embedder) -> None:
        self.s = settings
        self.embedder = embedder
        self.collection = settings.qdrant_collection or "synapsecraft"
        headers: dict[str, str] = {}
        if settings.qdrant_api_key:
            headers["api-key"] = settings.qdrant_api_key
        self._client = httpx.AsyncClient(
            base_url=settings.qdrant_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    # ── plumbing ───────────────────────────────────────────────────────────
    async def _req(self, method: str, path: str, json_body: dict | None = None,
                   ok: tuple[int, ...] = (200, 201)) -> dict:
        r = await self._client.request(method, path, json=json_body)
        if r.status_code not in ok:
            raise RuntimeError(f"qdrant {method} {path} → {r.status_code}: {r.text[:200]}")
        return r.json() if r.text else {}

    async def probe(self) -> None:
        await self._req("GET", "/collections", ok=(200,))

    async def _ensure_collection(self) -> None:
        try:
            await self._req(
                "PUT",
                f"/collections/{self.collection}",
                json_body={"vectors": {"size": self.embedder.dim, "distance": "Cosine"}},
            )
        except RuntimeError:
            pass  # 409 → collection already exists with that config

    @staticmethod
    def _point_id(cid: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://synapsecraft.local/{cid}"))

    # ── protocol ───────────────────────────────────────────────────────────
    async def upsert(self, chunks: list[dict], vectors) -> None:
        await self._ensure_collection()
        vectors = np.asarray(vectors, dtype=np.float32)
        points = [
            {
                "id": self._point_id(c["id"]),
                "vector": [float(x) for x in vectors[i]],
                "payload": {"chunk": c},
            }
            for i, c in enumerate(chunks)
        ]
        for i in range(0, len(points), 64):
            await self._req(
                "PUT",
                f"/collections/{self.collection}/points?wait=true",
                json_body={"points": points[i : i + 64]},
            )

    async def search(self, vector: list[float], k: int) -> list[SearchHit]:
        data = await self._req(
            "POST",
            f"/collections/{self.collection}/points/search",
            json_body={
                "vector": [float(x) for x in vector],
                "limit": k,
                "with_payload": True,
            },
        )
        hits: list[SearchHit] = []
        for p in (data.get("result") or []):
            payload = (p.get("payload") or {}).get("chunk") or {}
            if payload.get("id"):
                hits.append(SearchHit(payload["id"], float(p.get("score", 0.0))))
        return hits

    async def get_vector(self, cid: str) -> list[float] | None:
        try:
            data = await self._req(
                "POST",
                f"/collections/{self.collection}/points",
                json_body={"ids": [self._point_id(cid)], "with_vector": True, "limit": 1},
            )
            recs = data.get("result") or []
            if recs and recs[0].get("vector"):
                return [float(x) for x in recs[0]["vector"]]
        except Exception:
            return None
        return None

    async def get(self, cid: str) -> dict | None:
        try:
            data = await self._req(
                "POST",
                f"/collections/{self.collection}/points",
                json_body={"ids": [self._point_id(cid)], "limit": 1},
            )
            recs = data.get("result") or []
            if recs:
                return (recs[0].get("payload") or {}).get("chunk")
        except Exception:
            return None
        return None

    async def all_chunks(self) -> list[dict]:
        out: list[dict] = []
        offset = None
        while True:
            body: dict = {"limit": 256, "with_payload": True}
            if offset:
                body["offset"] = offset
            data = await self._req(
                "POST", f"/collections/{self.collection}/points/scroll", json_body=body
            )
            for p in (data.get("result", {}).get("points") or []):
                chunk = (p.get("payload") or {}).get("chunk")
                if chunk:
                    out.append(chunk)
            offset = data.get("result", {}).get("next_page_offset")
            if offset is None:
                break
        return out

    async def remove_document(self, doc_id: str) -> None:
        await self._req(
            "POST",
            f"/collections/{self.collection}/points/delete?wait=true",
            json_body={
                "filter": {
                    "must": [{"key": "chunk.doc_id", "match": {"value": doc_id}}]
                }
            },
        )

    async def clear(self) -> None:
        try:
            await self._req("DELETE", f"/collections/{self.collection}", ok=(200,))
        except Exception:
            pass

    def count(self) -> int:
        return 0  # use count_async() for live totals

    async def count_async(self) -> int:
        try:
            data = await self._req("GET", f"/collections/{self.collection}")
            return int((data.get("result") or {}).get("points_count") or 0)
        except Exception:
            return 0

    async def load(self, embedder) -> None:
        return  # Qdrant persists itself

    def persist(self, embed_kind: str | None = None) -> None:
        return


# ════════════════════════════════════════════════════════════════════════════
# Resolution: auto-probe Qdrant, gracefully degrade to in-process memory
# ════════════════════════════════════════════════════════════════════════════
async def resolve_vector_store(settings: Settings, embedder) -> BaseVectorStore:
    if settings.vector_provider in ("qdrant", "auto") and settings.qdrant_url:
        try:
            store = QdrantStore(settings, embedder)
            await store.probe()
            return store
        except Exception:
            # The connectome never goes down: degrade to the in-process store.
            pass
    return MemoryVectorStore(settings)



