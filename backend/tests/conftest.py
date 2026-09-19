"""Shared fixtures: a fully offline, deterministic SynapseCraft runtime.

Every test runs in Reflex-Arc mode (hash embeddings + extractive synthesis +
heuristic critics) so the suite needs no network, no LLM key and no services.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# ── force the offline configuration before any app import ───────────────────
os.environ["LLM_PROVIDER"] = "reflex"
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["VECTOR_PROVIDER"] = "memory"
os.environ["CRITIC_MODE"] = "heuristic"
os.environ["NEO4J_MIRROR"] = "off"
os.environ["GATING_THRESHOLD"] = "0.40"
os.environ["MAX_REFLECTION_LOOPS"] = "2"
os.environ.setdefault("DATA_DIR", "data")

import pytest  # noqa: E402

from app.config import Settings  # noqa: E402

SAMPLE_MD = """# Fruit Fly Connectome Field Guide

## Optic Lobe Circuits
The fly optic lobe processes motion in four nested neuropils: Lamina,
Medulla, Lobula and Lobula Plate. Local T4 neurons compute elementary
motion detectors and project their outputs to the Lobula Plate.
T4 neurons fire at 120 spikes per second under full-field drift.

## Odor Coding
Antennal lobe glomeruli converge onto projection neurons; Kenyon cells in
the mushroom body sparsen the odor code dramatically.
Kenyon cells respond at 15 spikes per second to their best odor.
See section Optic Lobe Circuits for the visual pathway counterpart.

## Connectome Statistics

| Neuropil | Neurons | Synapses |
| --- | --- | --- |
| Optic Lobe | 92001 | 12000000 |
| Mushroom Body | 4000 | 2500000 |

## Central Complex
The fan-shaped body maintains a 60 degree heading map used for navigation.
The ring neurons encode visual features in the central complex.
"""


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Offline settings pointed at a throw-away data directory."""
    return Settings(
        data_dir=str(tmp_path),
        llm_provider="reflex",
        embed_provider="hash",
        vector_provider="memory",
        critic_mode="heuristic",
        neo4j_mirror="off",
        gating_threshold=0.40,
        recall_k=12,
        top_k=4,
        max_reflection_loops=2,
        relate_threshold=0.32,
    )


class Harness:
    """A complete, wired-up nervous system for tests."""

    def __init__(self, settings: Settings) -> None:
        from app.agents.cognition import NervousSystem
        from app.embeddings.embedder import EmbeddingEngine
        from app.events import EventBus
        from app.llm.provider import LLMProvider
        from app.stores.graph_store import ConnectomeGraph
        from app.stores.vector_store import MemoryVectorStore

        self.settings = settings
        self.embedder = asyncio.run(EmbeddingEngine(settings).probe())
        self.store = MemoryVectorStore(settings)
        self.graph = ConnectomeGraph(settings)
        self.llm = LLMProvider(settings)
        self.bus = EventBus()
        self.brain = NervousSystem(settings, self.embedder, self.store,
                                   self.graph, self.llm, self.bus)

    async def ingest_md(self, text: str = SAMPLE_MD, name: str = "guide.md") -> dict:
        from app.ingestion.pipeline import ingest_document

        res = await ingest_document(name, text.encode("utf-8"),
                                    self.settings, self.embedder,
                                    self.store, self.graph, self.bus)
        return res.dict()

    async def query(self, q: str) -> dict:
        return await self.brain.run(q)


@pytest.fixture()
def harness(settings) -> Harness:
    return Harness(settings)


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with the full lifespan (offline components)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

