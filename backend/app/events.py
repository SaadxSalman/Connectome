"""Event bus: fans spike events out to every subscribed WebSocket cockpit."""

from __future__ import annotations

import asyncio
import time
from collections import deque

# Re-export the canonical event-kind constants for every neural module.
from .types import (  # noqa: F401
    GATE, HELLO, INGEST, RUN_END, RUN_START, SPIKE, make_spike,
)


class EventBus:
    """A lightweight pub/sub hub.

    Every neural agent node publishes events (spikes, gate verdicts, ingest
    pulses); each connected WebSocket cockpit owns one bounded queue. Slow
    consumers simply drop overflow — the visualiser is best-effort and must
    never block the neural pipeline.
    """

    def __init__(self, buffer_size: int = 1200, queue_size: int = 900) -> None:
        self.buffer: deque[dict] = deque(maxlen=buffer_size)
        self.total = 0
        self._queue_size = queue_size
        self._subs: dict[int, asyncio.Queue] = {}
        self._next_id = 0

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        self._next_id += 1
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subs[self._next_id] = q
        return self._next_id, q

    def unsubscribe(self, sub_id: int) -> None:
        self._subs.pop(sub_id, None)

    def publish(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        self.buffer.append(event)
        self.total += 1
        for q in list(self._subs.values()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Visualiser lag must never throttle the nervous system.
                pass

    def recent(self, n: int = 60) -> list[dict]:
        return list(self.buffer)[-n:]
