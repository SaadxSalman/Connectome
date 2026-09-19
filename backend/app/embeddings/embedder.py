"""Embedding engine: Ollama dense vectors or deterministic feature hashing.

Two interchangeable "sensory cortices":

• ``ollama`` — dense open-weight embeddings (``nomic-embed-text`` by default)
  for state-of-the-art semantic matching.
• ``hash``   — deterministic n-gram feature hashing (a la sklearn's
  HashingVectorizer): zero external dependencies, fully offline, stable
  across restarts, and strong enough for lexical/phrase-level retrieval.

The engine is chosen once at startup (``EMBED_PROVIDER=auto`` probes Ollama
and falls back to hashing). Vectors are always L2-normalised so cosine
similarity reduces to a dot product everywhere else in the codebase.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import Counter

import httpx
import numpy as np

from ..config import Settings
from ..textutils import tokenize


class EmbeddingEngine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.kind = "hash"
        self.model = "feature-hashing-v1"
        self.dim = max(64, settings.embed_dim)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
        self._sem = asyncio.Semaphore(6)

    # ── startup probe ──────────────────────────────────────────────────────
    async def probe(self) -> "EmbeddingEngine":
        if self.s.embed_provider in ("auto", "ollama"):
            try:
                vec = await self._ollama_one("connectome probe")
                if vec:
                    self.kind = "ollama"
                    self.model = self.s.ollama_embed_model
                    self.dim = len(vec)
            except Exception:
                pass  # stay on the hashing cortex — fully offline mode
        return self

    def describe(self) -> dict:
        return {"provider": self.kind, "model": self.model, "dim": self.dim}

    # ── public API ─────────────────────────────────────────────────────────
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.kind == "ollama":
            try:
                return await self._embed_ollama(texts)
            except Exception:
                # Mixing vector spaces mid-session would corrupt retrieval —
                # surface the failure so the operator can fix the backend.
                raise
        return self._embed_hash(texts)

    # ── ollama cortex ──────────────────────────────────────────────────────
    async def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        async def one(t: str) -> list[float]:
            async with self._sem:
                vec = await self._ollama_one(t)
            return _normalise(vec)

        return list(await asyncio.gather(*(one(t) for t in texts)))

    async def _ollama_one(self, text: str) -> list[float]:
        r = await self._client.post(
            f"{self.s.ollama_base_url.rstrip('/')}/api/embeddings",
            json={"model": self.s.ollama_embed_model, "prompt": text[:4000]},
        )
        r.raise_for_status()
        return r.json().get("embedding") or []

    # ── hashing cortex ─────────────────────────────────────────────────────
    def _embed_hash(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_one(t) for t in texts]

    def _hash_one(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in Counter(feats).items():
            w = (1.0 + math.log(tf)) * (1.5 if "_" in feat else 1.0)
            idx = int.from_bytes(
                hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "big"
            ) % self.dim
            vec[idx] += w
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            idx = int.from_bytes(
                hashlib.blake2b((text or "empty").encode("utf-8"), digest_size=8).digest(),
                "big",
            ) % self.dim
            vec[idx] = 1.0
            norm = 1.0
        return (vec / norm).tolist()

    async def aclose(self) -> None:
        await self._client.aclose()


def _normalise(vec: list[float]) -> list[float]:
    if not vec:
        return [0.0]
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return vec
    return (arr / norm).tolist()


async def build_embedder(settings: Settings) -> EmbeddingEngine:
    return await EmbeddingEngine(settings).probe()
