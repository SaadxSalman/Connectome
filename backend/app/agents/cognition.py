"""The nervous system: a LangGraph-orchestrated, cyclic multi-agent pipeline.

Agent population (fixed wiring, local processing circuits)
──────────────────────────────────────────────────────────
    sensory neuron   tokenise + embed the query, pull vector candidates
    local circuits   Connectome-Aware Ingestion-style spreading activation
                     across the structural graph, then membrane-potential
                     gating per chunk (lateral inhibition + consistency
                     micro-critics) — relevance decided *locally*, not by a
                     monolith reranker
    synthesist       grounded generation (LLM or offline Reflex-Arc) with
                     [S#] citations and structural transparency
    reflex critic    grounding check: are all claims supported by the
                     evidence? cycles back to the circuits for one more
                     propagation pass if not (reflection loop, bounded)

The graph topology:  sensory → circuits → synthesist → critic
                          ↑______________________|   (reflection cycle)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..connectome.activation import (
    GateVerdict,
    membrane_potential,
    numeric_conflicts,
    spreading_activation,
)
from ..embeddings.embedder import EmbeddingEngine
from ..events import EventBus, GATE, RUN_END, RUN_START, SPIKE, make_spike
from ..llm.provider import LLMProvider, parse_json_loose
from ..llm.reflex import reflex_expand, reflex_synthesize
from ..stores.graph_store import ConnectomeGraph
from ..stores.vector_store import BaseVectorStore
from ..textutils import lexical_overlap, sentence_split, token_set, tokenize, truncate
from ..types import Chunk

_TAG_RE = re.compile(r"\[S(\d+)\]")


# ── pipeline state (flows through every node) ──────────────────────────────
@dataclass
class NeuralState:
    query: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    loop: int = 0
    query_vector: list[float] | None = None
    query_tokens: set[str] = field(default_factory=set)
    candidates: list[dict] = field(default_factory=list)
    potentials: dict[str, float] = field(default_factory=dict)
    fired: list[dict] = field(default_factory=list)
    verdicts: list[GateVerdict] = field(default_factory=list)
    suppressed: int = 0
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    tags_used: list[str] = field(default_factory=list)
    grounded: bool = False
    critic_note: str = ""
    expansion: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    error: str = ""
    tokens_saved: int = 0
    latency_ms: int = 0
    provider: str = ""


class NervousSystem:
    """Wires the four agent populations into a cyclic LangGraph state machine."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingEngine,
        store: BaseVectorStore,
        graph: ConnectomeGraph,
        llm: LLMProvider,
        bus: EventBus,
    ) -> None:
        self.s = settings
        self.embedder = embedder
        self.store = store
        self.graph = graph
        self.llm = llm
        self.bus = bus
        self._build_graph()

    # ── LangGraph wiring ───────────────────────────────────────────────────
    def _build_graph(self) -> None:
        from langgraph.graph import StateGraph, END

        g = StateGraph(dict)
        g.add_node("sensory", self._node_sensory)
        g.add_node("circuits", self._node_circuits)
        g.add_node("synthesist", self._node_synthesist)
        g.add_node("critic", self._node_critic)
        g.set_entry_point("sensory")
        g.add_edge("sensory", "circuits")
        g.add_edge("circuits", "synthesist")
        g.add_edge("synthesist", "critic")
        g.add_conditional_edges(
            "critic",
            self._after_critic,
            {"reflect": "circuits", "emit": END},
        )
        self.app = g.compile()

    def _after_critic(self, state: dict) -> str:
        if state.get("grounded") or state.get("loop", 0) >= self.s.max_reflection_loops:
            return "emit"
        return "reflect"

    # ── event helpers ──────────────────────────────────────────────────────
    def _emit(self, state: NeuralState, **kw) -> None:
        """Publish a neural event, capped per run so a huge corpus can never
        flood the live visualiser (WS_MAX_EVENTS safety fuse)."""
        self._run_events = getattr(self, "_run_events", 0) + 1
        if self._run_events > self.s.ws_max_events:
            return
        self.bus.publish(make_spike(run_id=state.run_id, **kw))

    def _spike_path(self, state: NeuralState, path: list[str],
                    polarity: str = "signal") -> None:
        """Fire a visible impulse along an agent path in the cockpit."""
        self._emit(state, type=SPIKE, path=path, polarity=polarity)

    # ── SENSORY NEURON: capture & tokenise the incoming signal ─────────────
    async def _node_sensory(self, state: dict) -> dict:
        st = NeuralState(query=state["query"], loop=state.get("loop", 0),
                         run_id=state.get("run_id") or uuid.uuid4().hex[:12])
        if st.loop > 0:
            self._emit(st, type=RUN_START, loop=st.loop,
                       message=f"reflection loop {st.loop}: re-propagating signal")
        else:
            self._emit(st, type=RUN_START, loop=0,
                       query=st.query, message=f"sensory neuron captured query: '{truncate(st.query, 60)}'")
        self._spike_path(st, ["agent:sensory"], "excitatory")

        vec = await self.embedder.embed([st.query])
        st.query_vector = vec[0]
        st.query_tokens = token_set(st.query)
        self._last_query_tokens = st.query_tokens

        k = max(6, self.s.recall_k)
        hits = await self.store.search(st.query_vector, k)
        for hit in hits:
            raw = await self.store.get(hit.id)
            if not raw:
                continue
            try:
                chunk = Chunk(**raw)
            except Exception:
                continue
            st.candidates.append({
                "chunk": chunk, "sim": hit.score,
                "lexical": lexical_overlap(st.query_tokens, chunk.text),
                "graph_boost": 0.0,
            })
        self._emit(st, type=SPIKE, node="agent:sensory",
                   candidates=len(st.candidates),
                   message=f"recalled {len(st.candidates)} candidate chunks "
                           f"(pool={self.store.count()}, k={k})")
        return {"query": st.query, "run_id": st.run_id, "loop": st.loop,
                "state": st}

    # ── LOCAL PROCESSING CIRCUITS: spread, gate, inhibit ───────────────────
    async def _node_circuits(self, state: dict) -> dict:
        st: NeuralState = state["state"]
        self._spike_path(st, ["agent:sensory", "agent:circuits"], "signal")
        theta = self.s.gating_threshold

        # 1 · spreading activation across the structural connectome
        cand_ids = [c["chunk"].id for c in st.candidates]
        seeds = {cid: max(0.05, c["sim"]) for cid, c in zip(cand_ids, st.candidates)}
        boost, prop_events = spreading_activation(
            self.graph, seeds, self.s.propagation_decay, skip_kinds={"CONTAINS", "BELONGS"}
        )
        st.potentials = {cid: round(gain, 4) for cid, gain in boost.items()}
        for c in st.candidates:
            c["graph_boost"] = boost.get(c["chunk"].id, 0.0)
        # visualise the strongest propagation paths only (bounded stream)
        for src, tgt, gain, _ in sorted(prop_events, key=lambda e: -e[2])[:12]:
            self._emit(st, type=SPIKE, path=[src, tgt], gain=round(gain, 3),
                       polarity="signal", message=None)

        # 2 · expected-reward expansion (reflection loops sharpen the query)
        if st.loop > 0 and not st.expansion and st.candidates:
            st.expansion = reflex_expand(
                st.query, [c["chunk"].text for c in st.candidates[:8]]
            )
            if st.expansion:
                self._emit(st, type=SPIKE, node="agent:circuits",
                           expansion=st.expansion,
                           message=f"expected-reward signals: {', '.join(st.expansion)}")

        # 2b · Hebbian second pass: re-probe the vector store with the
        # sharpened (expanded) signal so a reflection loop can recover
        # evidence the original sensory pass missed.
        if st.expansion:
            sharpened = f"{st.query} {' '.join(st.expansion)}"
            vec2 = (await self.embedder.embed([sharpened]))[0]
            self._last_query_tokens = token_set(sharpened)
            have = {c["chunk"].id for c in st.candidates}
            recovered = 0
            for hit in await self.store.search(vec2, max(6, self.s.recall_k)):
                if hit.id in have:
                    continue
                raw = await self.store.get(hit.id)
                if not raw:
                    continue
                try:
                    chunk = Chunk(**raw)
                except Exception:
                    continue
                st.candidates.append({
                    "chunk": chunk, "sim": hit.score,
                    "lexical": lexical_overlap(self._last_query_tokens, chunk.text),
                    "graph_boost": 0.0,
                })
                have.add(hit.id)
                recovered += 1
            if recovered:
                self._emit(st, type=SPIKE, node="agent:circuits", recovered=recovered,
                           polarity="signal",
                           message=f"second pass recovered {recovered} additional candidates")

        # 3 · membrane-potential gating, strongest candidates first
        st.candidates.sort(key=lambda c: -(c["sim"] + c["graph_boost"]))
        fired_texts: list[str] = []
        st.fired, st.verdicts = [], []
        st.suppressed = 0
        lateral_ref: list[float] = []
        for c in st.candidates:
            chunk: Chunk = c["chunk"]
            lateral = max(lateral_ref) if lateral_ref else 0.0
            entity_score = self._entity_congruence(chunk)
            conflict = numeric_conflicts(chunk.text, fired_texts)
            v: GateVerdict = membrane_potential(
                sim=c["sim"], lexical=c["lexical"],
                entity_score=entity_score, lateral=lateral,
                conflict=conflict, lateral_lambda=self.s.lateral_inhibition,
                threshold=theta,
            )
            if v.fired:
                lateral_ref.append(c["sim"])
                idx = len(st.fired) + 1
                st.fired.append({
                    "tag": f"S{idx}", "id": chunk.id, "text": chunk.text,
                    "doc_title": chunk.doc_title, "section": " › ".join(chunk.section_path)
                    or chunk.doc_title, "kind": chunk.kind, "order": chunk.order,
                    "score": round(v.potential, 3), "sim": round(c["sim"], 3),
                })
            else:
                st.suppressed += 1
            st.verdicts.append(v)
            self._emit(st, type=GATE, node=chunk.id, verdict="fire" if v.fired else "suppress",
                       potential=v.potential, threshold=theta,
                       components=v.components, reason=v.reason,
                       polarity="excitatory" if v.fired else "inhibitory",
                       label=chunk.section_title or chunk.doc_title,
                       message=("fired" if v.fired else v.reason) + f" · V={v.potential}")
            if len(st.fired) >= self.s.top_k:
                # everyone after this is suppressed by cap
                rest = st.candidates[len(st.fired) + st.suppressed:]
                st.suppressed += len(rest)
                for c2 in rest:
                    st.verdicts.append(GateVerdict(
                        potential=0.0, fired=False,
                        threshold=theta, reason="suppressed: context capacity reached"))
                    self._emit(st, type=GATE, node=c2["chunk"].id, verdict="suppress",
                               potential=0.0, threshold=theta, reason="capacity cap",
                               polarity="inhibitory",
                               label=c2["chunk"].section_title or c2["chunk"].doc_title,
                               message="suppressed: context capacity reached")
                break

        if not st.fired:
            st.error = "no signal crossed the firing threshold — the connectome holds no sufficiently related knowledge"
            self._emit(st, type=SPIKE, node="agent:circuits",
                       polarity="inhibitory", message=st.error)
            return {"state": st, "no_evidence": True}

        self._emit(st, type=SPIKE, node="agent:circuits",
                   fired=len(st.fired), suppressed=st.suppressed,
                   polarity="excitatory",
                   message=f"gating complete: {len(st.fired)} chunks fired, "
                           f"{st.suppressed} suppressed at θ={theta}")
        return {"state": st}

    def _entity_congruence(self, chunk: Chunk) -> float:
        q_tokens = self._last_query_tokens
        if not q_tokens or not chunk.entities:
            return 0.0
        hits = sum(
            1 for e in chunk.entities
            if any(tok in tokenize(e) for tok in q_tokens)
        )
        if not hits:
            return 0.0
        return min(1.0, hits / (len(chunk.entities) ** 0.5))

    _last_query_tokens: set[str] = set()

    # ══════════════════════════════════════════════════════════════════════
    # MOTOR OUTPUT LAYER (Synthesist): grounded generation
    # ══════════════════════════════════════════════════════════════════════
    _CTX_CHAR_BUDGET = 6000  # hard cap on context chars sent to any model

    def _build_context(self, st: NeuralState) -> str:
        """Serialise the fired evidence pool into a bounded citation-ready
        context block. Only gate-survivors ever reach this string — that is
        the compute-cost win the inhibitory circuitry buys."""
        parts: list[str] = []
        budget = self._CTX_CHAR_BUDGET
        for ev in st.fired:
            block = f"[{ev['tag']}] {ev['doc_title']} › {ev['section']}\n{ev['text']}"
            if budget - len(block) < 0:
                break
            parts.append(block)
            budget -= len(block)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_tags(answer: str, n_fired: int) -> list[str]:
        """Validated [S#] citation tags actually used in the answer."""
        tags: list[str] = []
        for m in _TAG_RE.finditer(answer):
            idx = int(m.group(1))
            if 1 <= idx <= n_fired:
                tag = f"S{idx}"
                if tag not in tags:
                    tags.append(tag)
        return tags

    async def _node_synthesist(self, state: dict) -> dict:
        st: NeuralState = state["state"]
        self._spike_path(st, ["agent:circuits", "agent:synthesist"], "signal")
        st.provider = self.llm.provider_name

        if state.get("no_evidence") or not st.fired:
            st.answer, st.citations, st.tags_used = "", [], []
            st.grounded = True  # nothing to critique — end the run cleanly
            st.error = st.error or (
                "no grounded evidence reached the motor layer — the connectome "
                "holds no relevant knowledge for this signal"
            )
            self._emit(st, type=RUN_END, node="agent:synthesist", answer="",
                       polarity="inhibitory", message=st.error)
            return {"state": st, "grounded": True, "loop": st.loop}

        context = self._build_context(st)
        prompt = (
            f"QUERY: {st.query}\n\n"
            "VERIFIED SOURCES (pre-filtered by the connectome's inhibitory gates):\n"
            f"{context}\n\n"
            "Write a concise, information-dense answer to the query. Every "
            "factual claim must carry an inline citation tag like [S1] naming "
            "the source it came from. Use ONLY the facts in the sources above — "
            "never invent numbers, names or claims. If the sources do not fully "
            "answer the query, state exactly what is missing."
        )
        raw = await self.llm.complete(
            prompt,
            system=(
                "You are the Synthesist — the motor output layer of a "
                "connectome-inspired RAG engine. You receive only evidence "
                "that survived bio-inspired inhibitory gating. Answer with "
                "surgical grounding: inline [S#] citations after every claim, "
                "no hedging, no filler, no invented facts."
            ),
            max_tokens=self.s.llm_max_tokens,
        )
        if raw and raw.strip():
            st.answer = raw.strip()
            mode = f"llm:{self.llm.model_name}"
        else:
            answer, _tags = reflex_synthesize(st.query, st.fired)
            st.answer = answer or (
                "The verified sources survived the gate but contain no "
                "statements relevant enough to synthesise an answer."
            )
            mode = "reflex-arc (offline extractive)"

        st.tags_used = self._extract_tags(st.answer, len(st.fired))
        st.citations = [
            {"tag": ev["tag"], "id": ev["id"], "doc_title": ev["doc_title"],
             "section": ev["section"], "kind": ev["kind"], "score": ev["score"],
             "text": truncate(ev["text"], 220)}
            for ev in st.fired
        ]
        # Compute-cost accounting: context the inhibitory gates kept out of
        # the generation layer = every suppressed chunk's token estimate.
        fired_ids = {ev["id"] for ev in st.fired}
        kept = sum(c["chunk"].token_estimate for c in st.candidates
                   if c["chunk"].id in fired_ids)
        total = sum(c["chunk"].token_estimate for c in st.candidates)
        st.tokens_saved = max(0, total - kept)

        self._emit(st, type=SPIKE, node="agent:synthesist", mode=mode,
                   chars=len(st.answer), tags=st.tags_used,
                   message=f"motor output drafted via {mode} · "
                           f"{len(st.tags_used)}/{len(st.fired)} sources cited")
        return {"state": st}

    # ══════════════════════════════════════════════════════════════════════
    # REFLEX CRITIC: grounding verification (reflection-loop entry point)
    # ══════════════════════════════════════════════════════════════════════
    def _heuristic_grounding(self, st: NeuralState) -> tuple[bool, str]:
        """Deterministic micro-critic circuit (fully offline).

        Three independent checks vote on the motor output:
        1. citation coverage — share of sentences carrying a valid [S#] tag;
        2. lexical support  — share of answer tokens grounded in fired evidence;
        3. numeric fidelity — every asserted number must exist in the evidence.
        """
        raw_sentences = [s.strip() for s in sentence_split(st.answer) if s.strip()]
        if not raw_sentences:
            return False, "motor output has no content"
        cited = sum(1 for s in raw_sentences if _TAG_RE.search(s))
        cite_ratio = cited / len(raw_sentences)

        ev_tokens: set[str] = set()
        for ev in st.fired:
            ev_tokens |= token_set(ev["text"])
        ans_tokens = token_set(_TAG_RE.sub(" ", st.answer))
        support = (len(ans_tokens & ev_tokens) / len(ans_tokens)) if ans_tokens else 0.0

        # numeric fidelity: catch fabricated statistics before they escape
        answer_bare = _TAG_RE.sub(" ", st.answer)
        ev_text = " ".join(ev["text"] for ev in st.fired)
        nums = set(re.findall(r"\d[\d,]*(?:\.\d+)?", answer_bare))
        ev_flat = ev_text.replace(",", "")
        fabricated = [n for n in nums
                      if n not in ev_text and n.replace(",", "") not in ev_flat]
        if fabricated:
            return False, ("numeric hallucination: "
                           + ", ".join(fabricated[:4]) + " not present in evidence")

        score = 0.5 * cite_ratio + 0.5 * support
        grounded = score >= 0.55
        note = (f"citation coverage {cite_ratio:.0%} · "
                f"lexical support {support:.0%} · grounding score {score:.2f}")
        return grounded, note

    async def _llm_grounding_check(self, st: NeuralState) -> tuple[bool, str]:
        """CRITIC_MODE=llm: an auditor model grades the motor output."""
        evidence = "\n".join(f"[{ev['tag']}] {ev['text']}" for ev in st.fired)
        prompt = (
            f"ANSWER UNDER AUDIT:\n{st.answer}\n\n"
            f"VERIFIED EVIDENCE:\n{evidence}\n\n"
            "Is every claim in the ANSWER supported by the EVIDENCE? Reply with "
            'JSON only: {"grounded": true|false, "note": "<one short sentence>"}'
        )
        raw = await self.llm.complete(
            prompt,
            system=("You are a strict grounding auditor. Flag unsupported "
                    "claims, numeric fabrications and invented entities."),
            json_mode=True, max_tokens=256, temperature=0.0,
        )
        data = parse_json_loose(raw)
        if data and isinstance(data.get("grounded"), bool):
            return data["grounded"], truncate(str(data.get("note") or ""), 160)
        # The auditor itself failed — fall back to the deterministic circuit.
        return self._heuristic_grounding(st)

    async def _node_critic(self, state: dict) -> dict:
        st: NeuralState = state["state"]
        self._spike_path(st, ["agent:synthesist", "agent:critic"], "signal")

        if not st.answer:
            st.grounded = True
            self._emit(st, type=RUN_END, node="agent:critic", answer="",
                       polarity="inhibitory",
                       message=st.error or "empty motor output")
            return {"state": st, "grounded": True, "loop": st.loop}

        if self.s.critic_mode == "llm" and self.llm.provider_name != "reflex":
            grounded, note = await self._llm_grounding_check(st)
        else:
            grounded, note = self._heuristic_grounding(st)

        st.grounded = grounded
        st.critic_note = note
        self._emit(st, type=GATE, node="agent:critic",
                   verdict="pass" if grounded else "fail",
                   polarity="excitatory" if grounded else "inhibitory",
                   reason=note,
                   message=f"grounding check {'passed' if grounded else 'failed'}: {note}")

        if grounded or st.loop >= self.s.max_reflection_loops:
            self._emit(
                st, type=RUN_END, node="agent:critic",
                answer=truncate(st.answer, 240), grounded=grounded,
                loops=st.loop, suppressed=st.suppressed,
                tokens_saved=st.tokens_saved,
                polarity="excitatory" if grounded else "signal",
                message=("answer emitted · grounding verified"
                         if grounded else
                         f"answer emitted after {st.loop} reflection loop(s) · "
                         f"grounding unverified"),
            )
            return {"state": st, "grounded": grounded, "loop": st.loop}

        # → reflection: cycle back to the local circuits for one more
        #   propagation pass with a sharpened (expanded) signal.
        st.loop += 1
        self._emit(st, type=SPIKE, node="agent:critic", loop=st.loop,
                   polarity="signal",
                   message=f"reflection loop {st.loop}: re-propagating through the circuits")
        return {"state": st, "grounded": False, "loop": st.loop}

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — one full neural query, from sensory capture to emission
    # ══════════════════════════════════════════════════════════════════════
    async def run(self, query: str) -> dict:
        """Fire one query through the whole nervous system and return the
        grounded answer plus full neuro-telemetry for the cockpit."""
        self._run_events = 0
        t0 = time.perf_counter()
        try:
            final = await self.app.ainvoke({"query": query})
            st: NeuralState = final.get("state") or NeuralState(query=query)
        except Exception as exc:  # a faulting neuron must never kill the run
            st = NeuralState(query=query)
            st.error = f"neural pipeline fault: {exc}"
            self._emit(st, type=RUN_END, polarity="inhibitory", message=st.error)
        st.latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "run_id": st.run_id,
            "query": st.query,
            "answer": st.answer,
            "citations": st.citations,
            "tags_used": st.tags_used,
            "grounded": st.grounded,
            "critic_note": st.critic_note,
            "loops": st.loop,
            "fired": st.fired,
            "suppressed": st.suppressed,
            "candidates": len(st.candidates),
            "expansion": st.expansion,
            "tokens_saved": st.tokens_saved,
            "latency_ms": st.latency_ms,
            "error": st.error,
            "provider": st.provider,
            "verdicts": [
                {"potential": v.potential, "fired": v.fired,
                 "threshold": v.threshold, "reason": v.reason,
                 "components": v.components}
                for v in st.verdicts
            ],
            "potentials": {
                k: v for k, v in sorted(
                    st.potentials.items(), key=lambda kv: -kv[1]
                )[:12]
            },
            "llm": self.llm.describe(),
        }






