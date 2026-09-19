"""Core domain types: Chunks, spike events and event-kind constants."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """An addressable packet of knowledge flowing through the connectome."""

    id: str
    doc_id: str
    doc_title: str
    section_path: list[str] = Field(default_factory=list)
    section_title: str = ""
    text: str
    kind: str = "text"  # text | table | list | code
    token_estimate: int = 0
    entities: list[str] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list)
    hub: str = ""
    order: int = 0
    content_hash: str = ""

    def embed_text(self) -> str:
        """Structural context is prepended so embeddings remember where a
        chunk lives inside the document hierarchy (connectome-aware retrieval)."""
        path = " › ".join([self.doc_title, *self.section_path])
        return f"{path}. {self.text}"


# ── spike-event kinds streamed over /ws/neural-activity ────────────────────
RUN_START = "run_start"
RUN_END = "run_end"
SPIKE = "spike"
GATE = "gate"
INGEST = "ingest"
HEARTBEAT = "heartbeat"
HELLO = "hello"


def make_spike(event_type: str = SPIKE, **fields) -> dict:
    """Build a JSON-safe neural event. `polarity` colours the impulse in the
    cockpit: excitatory (cyan) / inhibitory (rose) / signal (violet)."""
    evt: dict = {"type": event_type, "polarity": "excitatory", "ts": time.time()}
    for key, value in fields.items():
        if value is not None:
            evt[key] = value
    return evt
