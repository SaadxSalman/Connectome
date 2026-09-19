"""Connectome-aware ingestion pipeline: parse → chunk → embed → wire.

Preserves structural topology (tables, hierarchical outlines, and
cross-references) and transforms documents into the node-link matrix the
connectome graph stores.
"""

from __future__ import annotations

import time

from ..config import Settings
from ..embeddings.embedder import EmbeddingEngine
from ..events import EventBus, make_spike, INGEST
from ..stores.graph_store import ConnectomeGraph
from ..stores.vector_store import BaseVectorStore
from ..textutils import truncate
from ..types import Chunk
from .chunker import chunk_document
from .parser import ParsedDocument, flatten, parse_source
from ..connectome.builder import wire_documents


class IngestionResult:
    def __init__(self, doc_id: str, title: str, n_chunks: int, n_sections: int,
                 n_entities: int, warnings: list[str], ms: int) -> None:
        self.doc_id = doc_id
        self.title = title
        self.n_chunks = n_chunks
        self.n_sections = n_sections
        self.n_entities = n_entities
        self.warnings = warnings
        self.ms = ms

    def dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title,
            "chunks": self.n_chunks, "sections": self.n_sections,
            "entities": self.n_entities, "warnings": self.warnings, "ms": self.ms,
        }


async def ingest_document(
    name: str,
    content: bytes,
    settings: Settings,
    embedder: EmbeddingEngine,
    store: BaseVectorStore,
    graph: ConnectomeGraph,
    bus: EventBus | None = None,
    source_url: str = "",
) -> IngestionResult:
    """Full ingestion arc for one document: parse → chunk → embed → wire."""
    t0 = time.time()
    doc: ParsedDocument = parse_source(name, content, source_url)
    raw = flatten(doc)
    if not raw.strip():
        return IngestionResult(
            "", truncate(doc.title, 80), 0, 0, 0,
            ["document contained no extractable text"],
            int((time.time() - t0) * 1000),
        )

    doc_id = (
        "d" + __import__("hashlib").blake2b(
            (doc.title + "|" + raw[:2000]).encode("utf-8"), digest_size=6
        ).hexdigest()
    )

    # dedupe: identical doc content → no-op re-ingest
    if graph.has_doc(doc_id):
        return IngestionResult(
            doc_id, truncate(doc.title, 80), graph.counts().get("chunks", 0),
            0, 0, ["duplicate content — already in the connectome"],
            int((time.time() - t0) * 1000),
        )

    chunks = chunk_document(doc, doc_id, settings.chunk_max_chars, settings.chunk_overlap_sentences)
    if not chunks:
        return IngestionResult(
            doc_id, truncate(doc.title, 80), 0, 0, 0, ["no chunks produced"],
            int((time.time() - t0) * 1000),
        )

    # entity fan-out (union across chunk entities, capped)
    entities: list[str] = []
    for c in chunks:
        for e in c.entities:
            if e not in entities:
                entities.append(e)
        if len(entities) >= 200:
            break

    vectors = await embedder.embed([c.embed_text() for c in chunks])
    await store.upsert([c.model_dump() for c in chunks], vectors)
    wire_documents(graph, chunks, vectors, settings.relate_threshold)
    graph.rebuild_hubs()
    graph.update_positions()
    store.persist(embedder.kind)
    graph.persist()

    if bus:
        bus.publish(make_spike(
            INGEST, node=f"doc:{doc_id}", label=truncate(doc.title, 48),
            chunks=len(chunks), polarity="signal",
            message=f"ingested '{truncate(doc.title, 40)}' → {len(chunks)} chunks",
        ))

    from ..connectome.builder import count_sections

    return IngestionResult(
        doc_id,
        truncate(doc.title, 80),
        len(chunks),
        count_sections(doc),
        len(entities),
        doc.warnings,
        int((time.time() - t0) * 1000),
    )
