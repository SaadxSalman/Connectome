"""The structural connectome graph (NetworkX engine).

Nodes
─────
    doc      a whole ingested document
    section  a hierarchical heading path inside a document
    chunk    an addressable packet of knowledge (also in the vector store)
    entity   a named concept mentioned by chunks (acetylcholine, V1, …)
    hub      a neuropil: a community of densely interlinked chunks
    agent    a fixed agent population (sensory, circuits, synthesist, critic)

Edges (kind, weight)
────────────────────
    CONTAINS    doc → section → chunk hierarchy
    MENTIONS    chunk → entity
    RELATES_TO  chunk ↔ chunk semantic neighbours (cosine ≥ threshold)
    REFERENCES  chunk → section/doc cross-references
    BELONGS     chunk → neuropil hub
    SENSES / GATES / DRIVES / REFLECTS   agent wiring
"""

from __future__ import annotations

import json
from collections import Counter

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from ..config import Settings

GREEK = "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω".split()

AGENT_NODES = [
    ("agent:sensory", "Sensory Neuron", "⚡"),
    ("agent:circuits", "Local Circuits", "◈"),
    ("agent:synthesist", "Synthesist", "✦"),
    ("agent:critic", "Reflex Critic", "⟳"),
]

MAX_HUBS = 24


class ConnectomeGraph:
    """Documents, sections, chunks, entities, neuropil hubs and agents wired
    as one weighted graph — the wiring diagram the propagation engine fires
    across."""

    def __init__(self, settings: Settings, mirror=None) -> None:
        self.s = settings
        self.gx = nx.Graph()
        self.positions: dict[str, list[float]] = {}
        self.hubs: dict[str, list[str]] = {}
        self.mirror = mirror
        self._ensure_agents()

    # ── construction ───────────────────────────────────────────────────────
    def _ensure_agents(self) -> None:
        for nid, label, glyph in AGENT_NODES:
            self.add_node(nid, "agent", label, glyph=glyph)

    def add_node(self, nid: str, ntype: str, label: str, **attrs) -> None:
        data = {"type": ntype, "label": label}
        data.update({k: v for k, v in attrs.items() if v is not None})
        if self.gx.has_node(nid):
            self.gx.nodes[nid].update(data)
        else:
            self.gx.add_node(nid, **data)
        if self.mirror:
            self.mirror.upsert_node(nid, data)

    def add_edge(self, a: str, b: str, kind: str, weight: float = 1.0) -> None:
        if a == b or not self.gx.has_node(a) or not self.gx.has_node(b):
            return
        if self.gx.has_edge(a, b):
            ed = self.gx[a][b]
            ed["weight"] = max(float(ed.get("weight", 0)), float(weight))
            ed.setdefault("kind", kind)
        else:
            self.gx.add_edge(a, b, kind=kind, weight=float(weight))
        if self.mirror:
            self.mirror.upsert_edge(a, b, kind, weight)

    # ── document lifecycle ─────────────────────────────────────────────────
    def has_doc(self, doc_id: str) -> bool:
        return self.gx.has_node(f"doc:{doc_id}")

    def remove_document(self, doc_id: str) -> None:
        victims = [n for n, d in self.gx.nodes(data=True) if d.get("doc_id") == doc_id]
        if self.gx.has_node(f"doc:{doc_id}"):
            victims.append(f"doc:{doc_id}")
        self.gx.remove_nodes_from([v for v in victims if self.gx.has_node(v)])
        # prune entities that lost every mention
        orphans = [
            n for n, d in self.gx.nodes(data=True)
            if d.get("type") == "entity" and self.gx.degree(n) == 0
        ]
        self.gx.remove_nodes_from(orphans)
        self.rebuild_hubs()
        self.update_positions()
        self.persist()

        # ── neuropil hubs (community detection) ────────────────────────────────
    def rebuild_hubs(self) -> None:
        """Recompute neuropil hubs: greedy modularity communities over the
        chunk + entity subgraph, then wire the agent population to them."""
        for h in [n for n, d in self.gx.nodes(data=True) if d.get("type") == "hub"]:
            self.gx.remove_node(h)
        chunk_ids = [n for n, d in self.gx.nodes(data=True) if d.get("type") == "chunk"]
        self.hubs = {}
        if not chunk_ids:
            self._wire_agents()
            return
        entity_ids = [n for n, d in self.gx.nodes(data=True) if d.get("type") == "entity"]
        sub = self.gx.subgraph(chunk_ids + entity_ids)
        try:
            comms = (
                greedy_modularity_communities(sub, weight="weight")
                if sub.number_of_edges() > 0
                else []
            )
        except Exception:
            comms = []
        communities = [set(c) & set(chunk_ids) for c in comms]
        communities = [c for c in communities if c]
        assigned = set().union(*communities) if communities else set()
        for cid in chunk_ids:
            if cid not in assigned:
                communities.append({cid})
        # merge the smallest hubs into their most-connected neighbour until
        # the population fits within MAX_HUBS
        changed = True
        while changed and len(communities) > MAX_HUBS:
            changed = False
            communities.sort(key=len, reverse=True)
            small = communities.pop()
            best_i, best_score = 0, -1
            for i, c in enumerate(communities):
                score = sum(1 for m in small for nb in self.gx[m] if nb in c)
                if score > best_score:
                    best_score, best_i = score, i
            communities[best_i] |= small
            changed = True
        communities.sort(key=len, reverse=True)
        for i, members in enumerate(communities):
            hub_id = f"hub:{i + 1}"
            self.add_node(hub_id, "hub", f"Neuropil Hub {GREEK[i % len(GREEK)]}",
                          size=len(members))
            for cid in members:
                self.add_edge(cid, hub_id, "BELONGS", 1.0)
                if "hub" in self.gx.nodes[cid]:
                    self.gx.nodes[cid]["hub"] = hub_id
            self.hubs[hub_id] = sorted(members)
        self._wire_agents()

    def _wire_agents(self) -> None:
        for hub_id in list(self.hubs):
            self.add_edge("agent:sensory", hub_id, "SENSES", 0.6)
            self.add_edge("agent:circuits", hub_id, "GATES", 0.6)
        self.add_edge("agent:circuits", "agent:synthesist", "DRIVES", 0.9)
        self.add_edge("agent:synthesist", "agent:critic", "REFLECTS", 0.9)

    # ── layout ─────────────────────────────────────────────────────────────
    def update_positions(self, iterations: int = 60) -> None:
        if self.gx.number_of_nodes() == 0:
            return
        try:
            pos = nx.spring_layout(self.gx, seed=42, iterations=iterations,
                                   weight="weight")
        except Exception:
            return
        xs = [p[0] for p in pos.values()] or [0.0]
        ys = [p[1] for p in pos.values()] or [0.0]
        mx = max(abs(min(xs)), abs(max(xs))) or 1.0
        my = max(abs(min(ys)), abs(max(ys))) or 1.0
        self.positions = {
            n: [round(float(x) / mx * 640.0, 1), round(float(y) / my * 440.0, 1)]
            for n, (x, y) in pos.items()
        }

    # ── reads ──────────────────────────────────────────────────────────────
    def node(self, nid: str):
        return self.gx.nodes[nid] if self.gx.has_node(nid) else None

    def neighbors(self, nid: str) -> list[tuple[str, dict]]:
        if not self.gx.has_node(nid):
            return []
        return [(nb, dict(ed)) for nb, ed in self.gx[nid].items()]

    def counts(self) -> dict:
        c = Counter(d.get("type", "?") for _, d in self.gx.nodes(data=True))
        return {
            "nodes": self.gx.number_of_nodes(),
            "edges": self.gx.number_of_edges(),
            "documents": c.get("doc", 0),
            "sections": c.get("section", 0),
            "chunks": c.get("chunk", 0),
            "entities": c.get("entity", 0),
            "hubs": len(self.hubs),
            "agents": c.get("agent", 0),
        }

    def snapshot(self, include_types=None, max_chunks: int = 180,
                 hub: str | None = None) -> dict:
        """Serialize a cockpit-renderable subgraph (chunk nodes are capped by
        degree so browsers stay smooth even with huge corpora)."""
        allowed = set(include_types) if include_types else None
        nodes_out: list[dict] = []
        chunks: list[tuple[int, dict]] = []
        for n, d in self.gx.nodes(data=True):
            t = d.get("type")
            if allowed and t not in allowed:
                continue
            pos = self.positions.get(n, [0.0, 0.0])
            entry = {"id": n, "type": t, "label": d.get("label", n),
                     "x": pos[0], "y": pos[1]}
            for key in ("doc_id", "doc_title", "kind", "glyph", "hub", "section"):
                if d.get(key):
                    entry[key] = d.get(key)
            if t == "chunk":
                chunks.append((self.gx.degree(n), entry))
            else:
                nodes_out.append(entry)
        chunks.sort(key=lambda x: -x[0])
        limited = [e for _, e in chunks[:max_chunks] if not hub or e.get("hub") == hub]
        nodes_out.extend(limited)
        ids = {e["id"] for e in nodes_out}
        edges = [
            {"id": f"{u}->{v}", "source": u, "target": v,
             "kind": d.get("kind", ""), "weight": round(float(d.get("weight", 1)), 3)}
            for u, v, d in self.gx.edges(data=True)
            if u in ids and v in ids
        ]
        hubs_out = [
            {"id": h, "label": (self.node(h) or {}).get("label", h), "size": len(m)}
            for h, m in self.hubs.items()
        ]
        return {
            "nodes": nodes_out,
            "edges": edges,
            "hubs": hubs_out,
            "counts": self.counts(),
            "total_chunks": len(chunks),
            "rendered_chunks": len(limited),
        }

    # ── persistence ────────────────────────────────────────────────────────
    def persist(self) -> None:
        try:
            path = self.s.data_path / "connectome.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            nodes = [{"id": n, **d} for n, d in self.gx.nodes(data=True)]
            edges = [
                [u, v, d.get("kind", ""), float(d.get("weight", 1))]
                for u, v, d in self.gx.edges(data=True)
            ]
            path.write_text(
                json.dumps(
                    {
                        "nodes": nodes,
                        "edges": edges,
                        "hubs": self.hubs,
                        "positions": self.positions,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def load(self) -> None:
        try:
            path = self.s.data_path / "connectome.json"
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            self.gx.clear()
            for nd in data.get("nodes", []):
                nd = dict(nd)
                nid = nd.pop("id", None)
                if nid:
                    self.gx.add_node(nid, **nd)
            for u, v, kind, w in data.get("edges", []):
                if self.gx.has_node(u) and self.gx.has_node(v):
                    self.gx.add_edge(u, v, kind=kind, weight=float(w))
            self.hubs = {h: list(m) for h, m in (data.get("hubs") or {}).items()}
            self.positions = data.get("positions") or {}
            self._ensure_agents()
        except Exception:
            self.gx.clear()
            self._ensure_agents()

    def clear(self) -> None:
        self.gx.clear()
        self.hubs = {}
        self.positions = {}
        self._ensure_agents()
        try:
            path = self.s.data_path / "connectome.json"
            if path.exists():
                path.unlink()
        except Exception:
            pass



