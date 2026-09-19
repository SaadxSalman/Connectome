"""Runtime telemetry persisted under DATA_DIR/stats.json."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Stats:
    def __init__(self, data_path: Path) -> None:
        self.path = data_path / "stats.json"
        self.started_at = time.time()
        self.queries = 0
        self.ingestions = 0
        self.spikes_emitted = 0
        self.gates_suppressed = 0
        self.tokens_saved = 0
        self.reflection_loops = 0
        self.total_latency_ms = 0
        self.last_latency_ms = 0

    # ── observers ──────────────────────────────────────────────────────────
    def observe_query(self, latency_ms: int, suppressed: int, tokens_saved: int, loops: int) -> None:
        self.queries += 1
        self.gates_suppressed += suppressed
        self.tokens_saved += tokens_saved
        self.reflection_loops += loops
        self.total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms
        self._persist()

    def observe_ingest(self, spikes: int) -> None:
        self.ingestions += 1
        self.spikes_emitted += spikes
        self._persist()

    def observe_spikes(self, count: int = 1) -> None:
        self.spikes_emitted += count

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.queries, 1) if self.queries else 0.0

    def snapshot(self) -> dict:
        return {
            "uptime_s": int(time.time() - self.started_at),
            "queries": self.queries,
            "ingestions": self.ingestions,
            "spikes_emitted": self.spikes_emitted,
            "gates_suppressed": self.gates_suppressed,
            "tokens_saved": self.tokens_saved,
            "reflection_loops": self.reflection_loops,
            "avg_latency_ms": self.avg_latency_ms,
            "last_latency_ms": self.last_latency_ms,
        }

    # ── persistence ────────────────────────────────────────────────────────
    def load(self) -> "Stats":
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for key in (
                    "queries", "ingestions", "spikes_emitted", "gates_suppressed",
                    "tokens_saved", "reflection_loops", "total_latency_ms", "last_latency_ms",
                ):
                    if key in raw:
                        setattr(self, key, int(raw.get(key) or 0))
        except Exception:
            pass
        return self

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        except Exception:
            pass
