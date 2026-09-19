"""Storage layer tests: memory vector store + connectome graph."""

import asyncio

import numpy as np

from app.types import Chunk


def _chunk(i: int, doc_id: str = "d1", text: str = "") -> Chunk:
    return Chunk(
        id=f"chunk:{doc_id}:{i:04d}", doc_id=doc_id, doc_title="Guide",
        section_path=["Motion"], section_title="Motion",
        text=text or f"chunk {i} about T4 neurons and motion", kind="text",
        token_estimate=10, entities=["T4"], order=i, content_hash=f"h{i}",
    )


def test_memory_store_upsert_search(tmp_path, settings):
    from app.stores.vector_store import MemoryVectorStore

    async def flow():
        store = MemoryVectorStore(settings)
        chunks = [_chunk(i) for i in range(4)]
        vecs = np.random.default_rng(0).normal(size=(4, 64)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        await store.upsert([c.model_dump() for c in chunks], vecs)
        hits = await store.search(vecs[0].tolist(), 2)
        assert hits[0].id == chunks[0].id
        got = await store.get(chunks[1].id)
        assert got["doc_id"] == "d1"
        assert store.count() == 4
        await store.remove_document("d1")
        assert store.count() == 0
        return True

    assert asyncio.run(flow())


def test_memory_store_persist_reload_roundtrip(tmp_path, settings):
    from app.embeddings.embedder import EmbeddingEngine
    from app.stores.vector_store import MemoryVectorStore

    async def flow():
        embedder = await EmbeddingEngine(settings).probe()
        store = MemoryVectorStore(settings)
        chunks = [_chunk(i) for i in range(3)]
        vecs = await embedder.embed([c.embed_text() for c in chunks])
        await store.upsert([c.model_dump() for c in chunks], vecs)
        store.persist(embedder.kind)

        store2 = MemoryVectorStore(settings)
        await store2.load(embedder)
        assert store2.count() == 3
        hits = await store2.search((await embedder.embed(["T4 neurons motion"]))[0], 1)
        assert hits[0].id in {c.id for c in chunks}

        await store2.remove_document("d1")
        assert store2.count() == 0
        return True

    assert asyncio.run(flow())


def test_graph_wiring_and_snapshot(settings):
    from app.connectome.builder import wire_documents
    from app.embeddings.embedder import EmbeddingEngine
    from app.ingestion.chunker import chunk_document
    from app.ingestion.parser import parse_markdown
    from app.stores.graph_store import ConnectomeGraph

    md = ("# T\n\n## A\nT4 neurons in the Lobula Plate detect motion.\n\n"
          "## B\nKenyon cells in the Mushroom Body sparsen odor.\n")
    doc = parse_markdown(md, "T")
    chunks = chunk_document(doc, "d1", 400, 1)

    async def flow():
        embedder = await EmbeddingEngine(settings).probe()
        vecs = np.asarray(await embedder.embed([c.embed_text() for c in chunks]))
        graph = ConnectomeGraph(settings)
        wire_documents(graph, chunks, vecs, 0.32)
        graph.rebuild_hubs()
        graph.update_positions()
        counts = graph.counts()
        snap = graph.snapshot()
        graph.persist()
        graph2 = ConnectomeGraph(settings)
        graph2.load()
        return counts, snap, graph2.counts()

    counts, snap, counts2 = asyncio.run(flow())
    assert counts["documents"] == 1
    assert counts["chunks"] == len(chunks)
    assert counts["sections"] >= 2
    assert counts["entities"] >= 2  # T4 + Kenyon at least
    assert counts["hubs"] >= 1
    assert any(n["type"] == "agent" for n in snap["nodes"])
    assert counts2["chunks"] == counts["chunks"]


def test_graph_references_and_removal(settings):
    from app.connectome.builder import wire_documents
    from app.embeddings.embedder import EmbeddingEngine
    from app.ingestion.chunker import chunk_document
    from app.ingestion.parser import parse_markdown
    from app.stores.graph_store import ConnectomeGraph

    md = ("# T\n\n## Odor\nKenyon cells sparsen odor codes in the Mushroom Body.\n\n"
          "## Visual\nKenyon cells also sparsen odor codes. See section Odor.\n")
    doc = parse_markdown(md, "T")
    chunks = chunk_document(doc, "d1", 400, 1)

    async def flow():
        embedder = await EmbeddingEngine(settings).probe()
        vecs = np.asarray(await embedder.embed([c.embed_text() for c in chunks]))
        graph = ConnectomeGraph(settings)
        wire_documents(graph, chunks, vecs, 0.32)
        kinds = {d["kind"] for _, _, d in graph.gx.edges(data=True)}
        graph.rebuild_hubs()
        graph.remove_document("d1")
        return kinds, graph.counts()

    kinds, counts = asyncio.run(flow())
    assert "REFERENCES" in kinds or "RELATES_TO" in kinds
    assert counts["documents"] == 0 and counts["chunks"] == 0
