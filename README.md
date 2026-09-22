# 🧠 SynapseCraft — A Connectome-Inspired Agentic RAG Engine

> **A full-stack, biologically inspired Agentic RAG engine that models information flow via distributed connectome-style routing — mimicking the local circuit processing found in the fruit fly nervous system — instead of centralized bottlenecks.**

SynapseCraft flips standard retrieval-augmented-generation design. Instead of routing everything through a single massive model core, it structures your document knowledge base and multi-agent network like an insect nervous system: queries are handled by localized, specialized micro-agents, information propagates dynamically along weighted synaptic paths, and irrelevant context is suppressed by inhibitory gating *before* it ever consumes a token of generation compute.

The entire pipeline is **fully observable**: a real-time cockpit renders the connectome wiring diagram and lights up every active neural pathway — spikes, gate verdicts, propagation pulses and reflection loops — as your query travels through the system.

---

## Table of Contents

1. [Why a Connectome? — Core Concept & Inspiration](#1-why-a-connectome--core-concept--inspiration)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [The Neural Agents — Multi-Agent Neuropil Hubs](#5-the-neural-agents--multi-agent-neuropil-hubs)
6. [Connectome-Aware Ingestion Pipeline](#6-connectome-aware-ingestion-pipeline)
7. [Neurodynamics — Spreading Activation & Spike-Timing Gating](#7-neurodynamics--spreading-activation--spike-timing-gating)
8. [The Reflection Loop — LangGraph Cyclic Workflow](#8-the-reflection-loop--langgraph-cyclic-workflow)
9. [Storage Layer — Vector Memory & Structural Graph](#9-storage-layer--vector-memory--structural-graph)
10. [Inference Layer — Groq, Ollama & the Offline Reflex-Arc](#10-inference-layer--groq-ollama--the-offline-reflex-arc)
11. [The Cockpit — Interactive Visual Dashboard](#11-the-cockpit--interactive-visual-dashboard)
12. [Quick Start](#12-quick-start)
13. [Configuration Reference (the single .env)](#13-configuration-reference-the-single-env)
14. [API Reference](#14-api-reference)
15. [WebSocket Neural-Activity Protocol](#15-websocket-neural-activity-protocol)
16. [Testing](#16-testing)
17. [Design Principles & Engineering Notes](#17-design-principles--engineering-notes)
18. [Performance Model — Why Gating Saves Compute](#18-performance-model--why-gating-saves-compute)
19. [Extending SynapseCraft](#19-extending-synapsecraft)
20. [Troubleshooting](#20-troubleshooting)
21. [Roadmap & The Science Behind It](#21-roadmap--the-science-behind-it)
22. [License](#22-license)

## 1. Why a Connectome? — Core Concept & Inspiration

In 2024–2025, the **FlyWire consortium** completed the mapping of an entire adult *Drosophila melanogaster* (fruit fly) brain: **~140,000 neurons and >50 million synapses** — the most complete connectome of any animal to date. The most striking discovery was not the size but the *wiring philosophy*:

- **Local processing circuits dominate.** Most of a fly's computation happens in dense, specialized local microcircuits (neuropils like the optic lobe, antennal lobe, mushroom body) — not in a single centralized "brain core."
- **Sparse, gated long-range communication.** Signals that travel between regions are heavily filtered; inhibitory neurons suppress noise long before signals reach motor outputs.
- **Lateral inhibition everywhere.** Neurons suppress their overlapping neighbours, sharpening contrast and keeping representations small and diverse.
- **Cyclic reflex arcs.** Sensory → local circuit → motor is not a straight line; feedback loops constantly re-verify and refine signals.

### Traditional RAG vs. SynapseCraft

| Concern | Traditional RAG | SynapseCraft |
|---|---|---|
| **Routing** | One monolithic LLM chain; every query funnels through a single "brain" | Distributed local circuits; specialized micro-agents each own one decision |
| **Context selection** | Top-k similarity dump; rerankers re-read *everything* | Membrane-potential gating: only chunks crossing firing threshold θ reach the LLM |
| **Noise handling** | Prompt-engineering ("only use relevant context…") | Biophysical suppression: lateral inhibition + numeric-conflict critics physically remove the noise |
| **Document structure** | Flat chunk soup; headings/tables lost at parse time | Structure-preserving parsers; the hierarchy itself becomes graph wiring (CONTAINS synapses) |
| **Cross-document links** | None | Entity (MENTIONS), semantic (RELATES_TO) and cross-reference (REFERENCES) synapses; spreading activation recovers context vector search misses |
| **Self-correction** | Single-pass generation; hallucination goes out the door | Reflex critic verifies grounding; ungrounded answers **cycle back** for another propagation pass (bounded) |
| **Observability** | Token stream at best | Every spike, gate verdict and potential is streamed live and renderable on a wiring diagram |
| **Failure mode** | Provider outage = dead product | Degradation ladder: Groq → Ollama → fully-offline deterministic Reflex-Arc |

The result: **smaller context windows, lower hallucination rates, cheaper inference, and a system you can literally watch think.**

---

## 2. Architecture Overview

```
                              ┌──────────────────────────────────────────────┐
                              │              COCKPIT (Next.js 15)            │
                              │  ┌────────────┐  ┌───────────────────────┐   │
                              │  │ Query      │  │ Connectome Graph      │   │
                              │  │ Console    │  │ Visualiser (React     │   │
                              │  │ + Answer   │  │  Flow) live glow      │   │
                              │  └─────┬──────┘  └───────────▲───────────┘   │
                              │  ┌─────┴──────┐  ┌───────────┴───────────┐   │
                              │  │ Ingestion  │  │ Neural Activity Feed  │   │
                              │  │ Panel      │  │ (WebSocket stream)    │   │
                              │  └─────┬──────┘  └───────────▲───────────┘   │
                              └────────┼─────────────────────┼───────────────┘
                                       │ REST                │ WS /ws/neural-activity
                                       ▼                     │
   ┌─────────────────────────────────────────────────────────┴───────────────┐
   │                    NEURAL GATEWAY (FastAPI + LangGraph)                 │
   │                                                                         │
   │   ┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐  │
   │   │ SENSORY  │───▶│ LOCAL    │───▶│ SYNTHESIST  │───▶│ REFLEX       │  │
   │   │ NEURON   │    │ CIRCUITS │    │ (motor out) │    │ CRITIC       │  │
   │   └──────────┘    └────┬─────┘    └─────────────┘    └──────┬───────┘  │
   │                        │        ▲                           │          │
   │                        │        └─────── reflection ────────┘          │
   │                        ▼  (ungrounded → re-propagate)                   │
   │   ┌──────────────────────────────────────────────────────────────────┐  │
   │   │ NEURODYNAMICS   spreading activation · membrane potentials ·     │  │
   │   │                 lateral inhibition · conflict micro-critics      │  │
   │   └──────────────────────────────────────────────────────────────────┘  │
   └──────┬──────────────────────┬───────────────────────┬──────────────────┘
          ▼                      ▼                       ▼
   ┌────────────┐        ┌──────────────┐        ┌──────────────────┐
   │ VECTOR     │        │ CONNECTOME   │        │ INFERENCE        │
   │ STORE      │        │ GRAPH        │        │ Groq → Ollama →  │
   │ NumPy /    │        │ NetworkX     │        │ Reflex-Arc (off.)│
   │ Qdrant     │        │ (+Neo4j mir.)│        │                  │
   └────────────┘        └──────────────┘        └──────────────────┘
```

**One query, four stations:**

1. **Sensory Neuron** — tokenises + embeds the query, pulls the candidate pool (`RECALL_K` chunks) from the vector store.
2. **Local Circuits** — spreads activation across the structural graph, then gates every candidate through a membrane-potential vote (semantic + lexical + entity congruence − lateral inhibition − conflict). Only gate-survivors (≤ `TOP_K`) advance.
3. **Synthesist** — grounded generation over *only* the fired evidence, with inline `[S#]` citations. Uses the LLM if available; otherwise the deterministic Reflex-Arc.
4. **Reflex Critic** — audits the answer (citation coverage, lexical support, numeric fidelity — or an LLM auditor). Ungrounded answers loop back to the circuits with a sharpened signal; loops are bounded by `MAX_REFLECTION_LOOPS`.

---

## 3. Tech Stack

### Backend (Python 3.11+)

| Layer | Technology | Role |
|---|---|---|
| Gateway | **FastAPI** + Uvicorn | REST + WebSocket server, CORS, multipart uploads |
| Orchestration | **LangGraph** | Cyclic, stateful agent workflow with conditional reflection edges |
| Graph engine | **NetworkX** | The structural connectome; community detection builds neuropil hubs; spring-layout positions |
| Optional mirror | **Neo4j** (Desktop / Aura) | Write-only, explorable twin of every node and synapse |
| Vector store | **NumPy in-process store** (default) or **Qdrant Cloud** (plain REST) | Dense retrieval; cosine similarity as dot product (vectors always L2-normalised) |
| Embeddings | **Ollama** (`nomic-embed-text`) or **feature-hashing** (offline) | Two interchangeable sensory cortices |
| Inference | **Groq**, **Ollama** (`llama3.1` etc.), or **Reflex-Arc** | Graceful degradation ladder |
| Parsing | pypdf, BeautifulSoup4, csv/json built-ins | Structure-preserving: headings, tables, lists, code fences, cross-refs |
| Testing | pytest (+ FastAPI TestClient) | 28 tests, fully offline |

### Frontend (Node 18+)

| Layer | Technology | Role |
|---|---|---|
| Framework | **Next.js 15 (App Router)** + React 19 + TypeScript | The cockpit |
| Styling | **Tailwind CSS** | Dark neural theme with custom `signal.*` palette |
| Graph visualisation | **@xyflow/react (React Flow 12)** | Interactive wiring diagram with live pathway glow |

### Zero-lock-in guarantees

- Every external service is **optional**. With an empty `.env`, SynapseCraft runs 100% offline in Reflex-Arc mode (hash embeddings + extractive synthesis + heuristic critics).
- The single root `.env` feeds **both** backend and frontend (the frontend's `next.config.mjs` loads it via dotenv and inlines `NEXT_PUBLIC_*` at build time).
- Data persists locally under `backend/data/` — vector store, graph, and telemetry all survive restarts.

---

## 4. Project Structure

```
Connectome/
├── .env                        ← THE single configuration file (gitignored, never committed)
├── .gitignore                  ← .env, data dirs, node_modules, .next, venvs…
│
├── backend/
│   ├── run.py                  ← uvicorn launcher (reads APP_HOST / APP_PORT)
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── main.py             ← FastAPI gateway: REST routes + /ws/neural-activity
│   │   ├── config.py           ← pydantic-settings bound to the root .env
│   │   ├── types.py            ← Chunk model + spike-event constructors
│   │   ├── events.py           ← EventBus: fan-out pub/sub for live spikes
│   │   ├── stats.py            ← persisted telemetry (queries, tokens_saved, …)
│   │   ├── textutils.py        ← tokeniser, sentence splitter, idf, lexical overlap
│   │   ├── agents/
│   │   │   ├── cognition.py    ← THE NERVOUS SYSTEM: 4 LangGraph nodes + run()
│   │   │   └── tools.py        ← web scout (fetch + parse live URLs)
│   │   ├── connectome/
│   │   │   ├── activation.py   ← spreading activation, membrane potentials,
│   │   │   │                     lateral inhibition, conflict micro-critic
│   │   │   └── builder.py      ← wires parsed docs into the graph (all synapse kinds)
│   │   ├── embeddings/
│   │   │   └── embedder.py     ← Ollama cortex | deterministic hashing cortex
│   │   ├── ingestion/
│   │   │   ├── parser.py       ← Markdown + plain-text structure-preserving parsers
│   │   │   ├── parser_web.py   ← CSV / JSON / PDF / HTML parsers
│   │   │   ├── chunker.py      ← structure-aware chunking + entity/ref extraction
│   │   │   └── pipeline.py     ← parse → chunk → embed → wire (async arc)
│   │   ├── llm/
│   │   │   ├── provider.py     ← Groq/Ollama client + tolerant JSON parsing
│   │   │   └── reflex.py       ← offline extractive synthesis + query expansion
│   │   └── stores/
│   │       ├── vector_store.py ← NumPy memory store + Qdrant REST store
│   │       ├── graph_store.py  ← ConnectomeGraph (NetworkX): hubs, layout, persistence
│   │       └── neo4j_mirror.py ← optional write-only Neo4j twin
│   └── tests/                  ← 28 offline pytest tests (conftest builds a Harness)
│
├── frontend/
│   ├── package.json            ← Next 15 · React 19 · @xyflow/react 12 · Tailwind 3
│   ├── next.config.mjs         ← loads the ROOT .env (dotenv), inlines NEXT_PUBLIC_*
│   ├── tailwind.config.ts      ← neuron.* / signal.* theme tokens
│   ├── app/
│   │   ├── layout.tsx          ← dark shell
│   │   ├── page.tsx            ← cockpit dashboard layout
│   │   └── globals.css         ← tailwind + React Flow dark tuning
│   ├── components/
│   │   ├── ConnectomeGraph.tsx ← React Flow visualiser + live pathway glow
│   │   ├── QueryConsole.tsx    ← query box + grounded answer + citations
│   │   ├── EventFeed.tsx       ← live electrophysiology trace
│   │   ├── IngestPanel.tsx     ← uploads, web scout, doc inventory, wipe
│   │   └── SystemBar.tsx       ← telemetry header (provider, spikes, latency…)
│   └── lib/
│       ├── api.ts              ← typed REST client
│       ├── types.ts            ← wire-format types (mirror of backend)
│       └── useNeuralStream.ts  ← auto-reconnecting WebSocket hook
│
└── README.md                   ← you are here
```

Runtime artefacts live under `backend/data/` (gitignored): `vector_store.npz`, `vector_meta.json`, `connectome.json`, `stats.json`.

---

## 5. The Neural Agents — Multi-Agent Neuropil Hubs

All four agent populations live in `backend/app/agents/cognition.py` and are wired into a LangGraph `StateGraph`:

```
sensory ──▶ circuits ──▶ synthesist ──▶ critic ──┐
                ▲                                │
                └────── reflection (bounded) ◀───┘
```

### ⚡ Sensory Neuron (`_node_sensory`)
- Tokenises the query, embeds it (Ollama or hashing cortex).
- Pulls the top `RECALL_K` candidates from the vector store..
- Attaches per-candidate lexical-overlap features.
- Emits `run_start` + `spike` events so the cockpit lights the agent up.
- Re-fires on every reflection loop with a note that the signal is being re-sensed.

### ◈ Local Processing Circuits (`_node_circuits`)
The decentralised heart of the system — relevance is decided *locally*, per chunk, by multiple small votes rather than one reranker model:

1. **Spreading activation** (Collins–Loftus): energy leaves every candidate along `RELATES_TO` / `MENTIONS` / `REFERENCES` synapses, decaying by `PROPAGATION_DECAY` per hop. The resulting potentials become **graph boosts** for each chunk. The strongest propagation paths are streamed to the visualiser.
2. **Expected-reward expansion** (reflection loops only): pseudo-relevance feedback (Rocchio-style) mines recurring terms from the top candidates, appends them to the signal, then re-probes the vector store to recover evidence the first pass missed (a "Hebbian second pass").
3. **Membrane-potential gating** (see §7): each candidate accumulates excitatory and inhibitory postsynaptic potentials; it fires only if `V ≥ θ`. An already-firing chunk **laterally inhibits** overlapping competitors, and a **numeric-conflict micro-critic** suppresses chunks contradicting already-fired evidence.
4. Gating emits a `gate` event per chunk with the full component breakdown — the cockpit shows exactly *why* each chunk fired or was suppressed.
5. A hard **capacity cap**: once `TOP_K` chunks have fired, the remainder are suppressed with reason *"context capacity reached"*.

### ✦ Synthesist (`_node_synthesist`) — Motor Output Layer
- Serialises fired evidence into a bounded (6000-char) citation-ready context block.
- Prompts the LLM for an answer where **every claim must carry an inline `[S#]` tag** naming its source. Only gate-survivors ever enter this prompt — the compute win of the whole architecture.
- If no LLM is available (or the endpoint fails), the **Reflex-Arc synthesiser** takes over: idf-weighted extractive summarisation with sentence-level lateral inhibition (near-duplicate suppression), breadth-first across sources, and `[S#]` tags preserved.
- Computes **tokens_saved**: the token estimate of every suppressed chunk that never reached the generation layer.
- Emits `spike` with mode (`llm:<model>` or `reflex-arc`), answer length and cited tags.

### ⟳ Reflex Critic (`_node_critic`)
Grounding verification with two interchangeable auditor cortices:

- **Heuristic (default — offline, deterministic):** three independent checks vote on the answer —
  1. *citation coverage*: share of sentences carrying a valid `[S#]` tag;
  2. *lexical support*: share of answer tokens grounded in fired evidence;
  3. *numeric fidelity*: every number asserted in the answer must literally exist in the evidence — otherwise it is flagged as a **numeric hallucination** and rejected outright.
- **LLM (`CRITIC_MODE=llm`):** an auditor model returns strict `{"grounded": bool, "note": str}` JSON; falls back to the heuristic circuit if the auditor itself errors.

Verdict `fail` + loops remaining → the state cycles back to **circuits** with `loop+1` (reflection). Verdict `pass` (or loops exhausted) → `run_end` and the answer is emitted. A special **no-evidence** path ends the run cleanly with an inhibitory event instead of pretending to answer — the system would rather say "I hold no relevant knowledge" than fabricate.

---

## 6. Connectome-Aware Ingestion Pipeline

`backend/app/ingestion/pipeline.py` — one async arc per document:

```
upload / url ──▶ parse ──▶ chunk ──▶ embed ──▶ wire graph ──▶ persist ──▶ ingest spike
```

### 6.1 Structure-preserving parsing
Unlike naive text splitters, the parsers keep the document's skeleton:

| Format | Parser | Structure retained |
|---|---|---|
| Markdown | `parse_markdown` | Heading hierarchy (h1–h6 → nested `Section` tree), pipe tables, lists, fenced code |
| Plain text | `parse_text` | ALL-CAPS / numbered heading heuristics (`CHAPTER x`, `1.2 …`) |
| PDF | `parse_pdf` (pypdf) | Page sections, inferred headings, paragraphs |
| HTML | `parse_html` (BeautifulSoup) | h1–h6 nesting, `<table>` rows, `<li>` lists; strips scripts/nav/footer |
| CSV | `parse_csv` | Rows as one table block |
| JSON | `parse_json` | Object keys become nested sections; arrays become list blocks |

Every `Block` carries a `kind` (`text | table | list | code`) and lives inside a hierarchical `Section` tree. Content sniffing (`%PDF`, `<html`) dispatches even when the extension lies. Parse failures degrade to a raw-text section with a warning — never an exception.

### 6.2 Structure-aware chunking
`chunker.chunk_document` walks the section tree and produces `Chunk` objects that remember **where they live**:

- `section_path` — e.g. `["Fruit Fly Connectome Field Guide", "Optic Lobe Circuits"]`
- Sentence-window packing with configurable overlap (`CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_SENTENCES`)
- Tables / lists / code kept whole where possible, windowed only if oversized
- **Heuristic NER**: capitalised multi-word phrases + acronyms + markdown link labels → `entities` (these become `ent:` graph nodes)
- **Cross-reference extraction**: `see section X`, `cf. Y`, markdown anchors → `refs`
- Stable ids + blake2b content hashes for deduplication

Crucially, `Chunk.embed_text()` **prepends the section path** to the body text before embedding — the embedding remembers its position in the document hierarchy. That is the "connectome-aware" half of retrieval.

### 6.3 Wiring the connectome
`connectome/builder.wire_documents` transforms the chunk list into a node-link matrix:

| Synapse | From → To | Meaning |
|---|---|---|
| `CONTAINS` | doc → section → chunk | structural hierarchy (weight 1.0) |
| `MENTIONS` | chunk → entity | the named concepts a chunk discusses |
| `RELATES_TO` | chunk ↔ chunk | cosine neighbours (≥ `RELATE_THRESHOLD`), top-2 per chunk |
| `REFERENCES` | chunk → section/doc | resolved cross-references ("see section X") |

After wiring, `rebuild_hubs()` runs **greedy modularity community detection** over the chunk+entity subgraph to carve **neuropil hubs** (≤ 24, named after Greek letters: *Neuropil Hub α*, *β*, …), each chunk getting a `BELONGS` synapse. The fixed agent population is then wired to every hub (`SENSES`, `GATES`) and to each other (`DRIVES`, `REFLECTS`). Finally `update_positions()` runs a seeded spring layout, normalised to a 640×440 viewport — the exact coordinates the cockpit renders.

### 6.4 Deduplication & lifecycle
- Document identity = blake2b(title + first 2 KB of text). Re-ingesting the same content is a **no-op** with a `duplicate` warning.
- `DELETE /api/documents/{doc_id}` prunes the doc's chunks, sections, orphaned entities, vectors and hub memberships, then persists.
- Everything survives restarts: vectors (`.npz`), graph (`connectome.json`), stats (`stats.json`).

---

## 7. Neurodynamics — Spreading Activation & Spike-Timing Gating

`backend/app/connectome/activation.py` implements the biophysical metaphors that give SynapseCraft its noise immunity. Everything is pure math — deterministic, testable, dependency-free.

### 7.1 Spreading activation

```
E(t+1, target) += E(t, source) · w(source→target) · decay^hop
```

- Energy leaves seed chunks along semantic synapses only (`RELATES_TO`, `MENTIONS`, `REFERENCES`). Structural synapses (`CONTAINS`, `BELONGS`, agent wiring) never carry retrieval signal.
- Signals below `MIN_GAIN = 0.02` die out silently — sub-threshold activity doesn't waste cycles.
- Returns per-node potentials (the graph boosts) plus `(source, target, gain)` triples that the cockpit animates as travelling impulses.

### 7.2 The membrane potential

Every candidate chunk's fate is a weighted vote:

```
V = 0.55·semantic + 0.25·lexical + 0.20·entity      (excitatory PSPs)
  − λ·lateral − 0.30·conflict                       (inhibitory PSPs)

fired  ⟺  V ≥ θ          (θ = GATING_THRESHOLD, default 0.55)
```

| Component | Source | Weight |
|---|---|---|
| `semantic` | cosine similarity to the (propagation-boosted) query | `W_SEMANTIC = 0.55` |
| `lexical` | √-normalised idf-weighted token overlap | `W_LEXICAL = 0.25` |
| `entity` | named-concept congruence between query tokens and chunk entities | `W_ENTITY = 0.20` |
| `lateral` | max similarity among *already-firing* chunks | `LATERAL_INHIBITION = 0.35` |
| `conflict` | numeric contradiction with fired evidence | `W_CONFLICT = 0.30` |

### 7.3 Lateral inhibition
Once a chunk fires, its similarity becomes a reference potential: later candidates are penalised by `λ × lateral`. Overlapping / redundant context is suppressed in favour of diverse evidence — the classic contrast-sharpening mechanism of early sensory systems. The cockpit displays the exact reason: *"suppressed: lateral inhibition (redundant context)"*.

### 7.4 Numeric-conflict micro-critic
A regex micro-critic extracts `(subject-key, number)` claims from each chunk and the already-fired set. If a chunk asserts a different value for a key (>15% relative difference) that another fired chunk already claimed — e.g. *"120 spikes per second"* vs *"80 spikes per second"* — it is suppressed as a numeric contradiction. Contradictory sources never reach the generator, so the model can't be forced to arbitrate between them.

### 7.5 Capacity cap
Even past θ, at most `TOP_K` chunks fire; the remainder are suppressed with reason *"context capacity reached"*. The context window is a hard budget, not a suggestion.

---

## 8. The Reflection Loop — LangGraph Cyclic Workflow

The agent graph is a true **cyclic** state machine (LangGraph `StateGraph` + conditional edges):

```python
g.add_node("sensory",     self._node_sensory)
g.add_node("circuits",    self._node_circuits)
g.add_node("synthesist",  self._node_synthesist)
g.add_node("critic",      self._node_critic)
g.set_entry_point("sensory")
g.add_edge("sensory", "circuits")
g.add_edge("circuits", "synthesist")
g.add_edge("synthesist", "critic")
g.add_conditional_edges("critic",
    self._after_critic, {"reflect": "circuits", "emit": END})
```

- `_after_critic` returns `reflect` while the answer is ungrounded and `loop < MAX_REFLECTION_LOOPS` (default 2), else `emit`.
- Each reflection pass: the circuits run **expected-reward expansion** (recurring terms from the top candidates), re-probe the vector store with the sharpened signal, recover new candidates, and re-gate everything from scratch.
- The full `NeuralState` dataclass (query vector, candidates, potentials, fired evidence, verdicts, citations, telemetry) flows through every node, so each reflection pass *accumulates* rather than restarts.
- All errors are contained: a faulting node produces an inhibitory `run_end` event — the pipeline never crashes a run.

This is the fruit-fly reflex arc as software: sense → local processing → motor output → **reflex check** → re-sense if the output was wrong, with a hard bound so it always terminates.

---

## 9. Storage Layer — Vector Memory & Structural Graph

### 9.1 Vector stores (`stores/vector_store.py`)

**`MemoryVectorStore` (default)** — a zero-dependency in-process NumPy matrix:
- Vectors kept L2-normalised → cosine similarity is a single matmul.
- Payload = full chunk records (the store is the source of truth for chunk bodies).
- Persisted to `data/vector_store.npz` + `data/vector_meta.json`; reloaded on boot.
- **Embedding-cortex migration**: if the persisted `embed_kind` differs from the active embedder (e.g. you switched from hashing to Ollama), the whole corpus is transparently re-embedded on load so the vector space never mixes coordinate systems.
- Corrupted store files degrade to a clean slate instead of bricking the gateway.

**`QdrantStore`** — Qdrant Cloud / self-hosted via plain HTTPX REST (no heavy SDK):
- Auto-creates the collection with the embedder's dimension + Cosine distance.
- UUID5-deterministic point ids → idempotent upserts.
- Scroll-paginated `all_chunks`, filtered delete by `doc_id`, batched upserts (64/batch).
- Resolution order (`VECTOR_PROVIDER=auto`): Qdrant if `QDRANT_URL` is set **and** the probe succeeds, else in-memory. The connectome never goes down.

### 9.2 The structural graph (`stores/graph_store.py`)
A NetworkX `Graph` whose node types are `doc | section | chunk | entity | hub | agent`, with typed, weighted edges (§6.3). It provides:

- **Neuropil hubs** via greedy-modularity communities, with a merge-smallest loop bounding the population at 24.
- **Spring-layout positions** normalised to cockpit coordinates, seeded for stable re-layouts.
- **Snapshot API** for the cockpit: renderable subgraph with chunks capped by degree (default 180) so a 100k-chunk corpus still draws smoothly, plus counts, hubs and edges.
- **Persistence** to `connectome.json` and (optionally) a **Neo4j mirror** — every node/edge upserted as `(:Synapse)` with `SYNAPSE` relationships, best-effort: mirror failures are logged once and swallowed. The mirror activates when `NEO4J_MIRROR=on`, or `auto` + credentials present.

### 9.3 Telemetry (`stats.py`)
Persisted counters — queries, ingestions, spikes emitted, gates suppressed, tokens saved, reflection loops, average/last latency — served from `/api/health` and rendered live in the cockpit's System Bar.

---

## 10. Inference Layer — Groq, Ollama & the Offline Reflex-Arc

### 10.1 Provider ladder (`llm/provider.py`)

```
LLM_PROVIDER=auto ──▶ Groq (if GROQ_API_KEY set) ──▶ Ollama (if reachable) ──▶ Reflex-Arc
```

- OpenAI-compatible chat completions for Groq (`GROQ_BASE_URL` overridable), `/api/chat` for Ollama.
- Per-engine failure isolation with a **30-second Ollama cooldown** after a failure (no per-request hammering of a dead service).
- `complete()` returns `None` when every engine fails — callers transparently fall back to the Reflex-Arc. **The nervous system never goes dark.**

### 10.2 The Reflex-Arc (`llm/reflex.py`)
A deterministic, fully offline generation layer (pure functions — same inputs, same spikes, every time):

- **`reflex_synthesize`** — scores every candidate sentence with idf-weighted query overlap + gate score, suppresses near-duplicates (Jaccard ≥ 0.7), picks the best sentence **per source first** (breadth across evidence), then fills remaining slots by global salience, restores narrative order, and appends `[S#]` tags.
- **`reflex_expand`** — Rocchio-style pseudo-relevance feedback powering the reflection loop's second pass.

The Reflex-Arc is not a demo stub: it is the same mechanism that handles LLM outages in production mode, and it is what all 28 tests assert against.

---

## 11. The Cockpit — Interactive Visual Dashboard

The Next.js 15 cockpit (`frontend/`) is the observability surface of the nervous system:

### Connectome Graph Visualiser (`components/ConnectomeGraph.tsx`)
- React Flow 12 canvas rendering the graph snapshot: colour-coded node types — agents ◈ violet, docs ▤ sky, sections § light-blue, chunks ▪ slate, entities ◆ amber, hubs ◎ mint.
- **Live pathway glow**: the `/ws/neural-activity` stream drives a 2.6-second activation window — nodes get cyan glow halos and edges turn animated/flowing as impulses traverse them. Excitatory / inhibitory / signal polarities are colour-coded (cyan / rose / violet).
- Drag nodes, zoom 0.08×–2.2×, dot-grid background, dark-tuned controls; chunk count capped by the snapshot API so browsers stay smooth on huge corpora.
- Click any node → its id appears in the footer for inspection.

### Query Console (`components/QueryConsole.tsx`)
- Fire queries; the grounded answer returns with badges: grounding verdict, provider, latency, candidates, fired vs suppressed counts, reflection loops, **tokens gated out**.
- Collapsible **fired evidence** list — every `[S#]` source with its document › section path, membrane potential and full text.

### Neural Activity Feed (`components/EventFeed.tsx`)
- A live electrophysiology trace: every spike, gate verdict, propagation pulse, ingest pulse, run start/end and heartbeat, colour-coded by polarity, with the WebSocket link state (`connecting / live / down`).

### Ingestion Panel (`components/IngestPanel.tsx`)
- Multi-file uploads (md / txt / pdf / csv / json / html), **web-scout** URL ingestion, per-document pruning, and a full connectome wipe — each triggering graph refreshes through the REST API.

### System Bar (`components/SystemBar.tsx`)
- Live telemetry: gateway status, active LLM provider, embedding cortex + dimension, vector-store mode, Neo4j mirror state, query / gating / token-savings counters, and a pulsing spike ticker.

---

## 12. Quick Start

### Prerequisites
- **Python 3.11+** and **Node.js 18+** (Node 20+ recommended).
- Nothing else. Every cloud service is optional.

### Option A — fully offline (zero keys, zero services)

```bash
# 1 · Backend  (repo root)
python -m venv .venv
.venv\Scripts\activate                     # Windows — `source .venv/bin/activate` on macOS/Linux
pip install -r backend/requirements.txt
cd backend && ..\.venv\Scripts\python run.py     # gateway → http://localhost:8000

# 2 · Frontend  (new terminal, repo root)
cd frontend
npm install
npm run dev                                # cockpit → http://localhost:3000
```

Open **http://localhost:3000**. Upload a Markdown/PDF/TXT file in the right-hand Ingestion Panel (or `POST /api/ingest`), then fire a query in the Query Console. Watch the impulse travel: sensory → hubs → circuits → synthesist → critic, gates suppressing noise in rose, fired evidence glowing cyan.

No Ollama installed? The gateway boots straight into offline **Reflex-Arc** mode — everything still works, deterministically.

### Option B — upgrade individual layers

Edit **one file** — the root `.env`:

```ini
LLM_PROVIDER=auto
GROQ_API_KEY=gsk_...                       # cloud inference (https://console.groq.com/keys)
GROQ_MODEL=llama-3.3-70b-versatile

EMBED_PROVIDER=auto                        # Ollama if running, else hashing
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text

VECTOR_PROVIDER=auto                       # Qdrant if URL given, else in-process
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=...

NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io   # optional mirror (Aura / Desktop)
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

Restart the backend — `/api/health` shows which cortex each layer resolved to. Switching embedding cortex re-embeds the corpus automatically on the next boot.

### Option C — run the test suite

```bash
.venv\Scripts\python -m pytest backend/tests -q        # 28 passed, fully offline
```

### Option D — production build of the cockpit

```bash
cd frontend && npm run build && npm start
```

(`npm run dev` = hot reload; `npm start` = optimised build.)

---

## 13. Configuration Reference (the single `.env`)

Everything lives in **`<repo-root>/.env`** (gitignored). Both servers read it; there is deliberately **no** `.env.example` — the file ships with working defaults and every secret empty. Leave everything blank and SynapseCraft runs fully offline.

| Section | Key | Default | Notes |
|---|---|---|---|
| **Gateway** | `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | uvicorn bind |
| | `CORS_ORIGINS` | `http://localhost:3000,…` | comma-separated browser origins |
| | `DATA_DIR` | `data` | persistence dir (relative → `backend/`) |
| | `MAX_UPLOAD_MB` | `15` | per-file upload cap |
| **LLM** | `LLM_PROVIDER` | `auto` | `auto \| groq \| ollama \| reflex` |
| | `GROQ_API_KEY` | *(empty)* | enables the Groq engine |
| | `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| | `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible |
| | `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| | `OLLAMA_LLM_MODEL` | `llama3.1` | |
| | `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.2` / `1024` | |
| **Embeddings** | `EMBED_PROVIDER` | `auto` | `auto \| ollama \| hash` |
| | `EMBED_DIM` | `512` | hashing-cortex dimension |
| | `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | |
| **Vector store** | `VECTOR_PROVIDER` | `auto` | `auto \| memory \| qdrant` |
| | `QDRANT_URL` / `QDRANT_API_KEY` | *(empty)* | Cloud or self-hosted |
| | `QDRANT_COLLECTION` | `synapsecraft` | |
| **Graph mirror** | `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | *(empty)* | optional |
| | `NEO4J_MIRROR` | `auto` | `auto \| on \| off` |
| **Neurodynamics** | `GATING_THRESHOLD` | `0.55` | firing threshold θ (0–1) |
| | `RECALL_K` | `24` | candidate pool per pass |
| | `TOP_K` | `6` | max chunks past the gate |
| | `MAX_REFLECTION_LOOPS` | `2` | critic retry bound |
| | `LATERAL_INHIBITION` | `0.35` | λ — overlap suppression strength |
| | `PROPAGATION_DECAY` | `0.6` | per-hop spreading-activation decay |
| | `CRITIC_MODE` | `heuristic` | `heuristic \| llm` |
| | `RELATE_THRESHOLD` | `0.32` | cosine cut-off for `RELATES_TO` |
| | `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_SENTENCES` | `1000` / `1` | structure-aware chunking |
| | `WS_MAX_EVENTS` | `240` | per-run spike-stream safety cap |
| **Cockpit** | `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | inlined at build time |
| | `NEXT_PUBLIC_WS_URL` | *(empty)* | blank → auto-derive `ws://<API host>/ws/neural-activity` |

> ⚠️ **Security note:** the `.env` is gitignored (root `.gitignore`). Never commit it; if you fork this repo, re-create the file locally and populate your own keys.

---

## 14. API Reference

Base URL: `http://localhost:8000` (interactive Swagger UI at `/docs`).

| Method | Path | Body / Params | Response |
|---|---|---|---|
| `GET` | `/api/health` | — | status + every component's resolved cortex + persisted stats |
| `GET` | `/api/system` | — | live neurodynamics settings, graph counts, last 40 events |
| `GET` | `/api/graph` | `?types=doc,chunk&hub=hub:1&max_chunks=200` | renderable graph snapshot (nodes/edges/hubs/counts) |
| `GET` | `/api/documents` | — | document inventory with per-doc chunk/section counts |
| `DELETE` | `/api/documents/{doc_id}` | — | prunes vectors + graph subgraph + mirror |
| `POST` | `/api/ingest` | multipart `files[]` | per-doc results (chunks/sections/entities/warnings/ms) + errors |
| `POST` | `/api/ingest/url` | `{"url": "https://…" [, "preview_only": true]}` | scout fetch → full ingest (or a dry-run preview) |
| `POST` | `/api/query` | `{"query": "…"}` (≤ 2000 chars) | full query telemetry (see below) |
| `POST` | `/api/reset` | — | wipes vectors, graph, mirror, stats |
| `WS` | `/ws/neural-activity` | — | live spike stream (§15) |

Example session:

```bash
curl -F files=@field_guide.md http://localhost:8000/api/ingest
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How do T4 neurons detect motion?"}'
```

The `/api/query` response carries the full neuro-telemetry synchronously:

```json
{
  "run_id": "ab12cd34ef56",
  "answer": "T4 neurons compute elementary motion … [S1]",
  "tags_used": ["S1", "S3"],
  "grounded": true,
  "critic_note": "citation coverage 100% · lexical support 71% · grounding score 0.86",
  "loops": 0,
  "fired": [{"tag": "S1", "doc_title": "…", "section": "…", "score": 0.71, "text": "…"}],
  "suppressed": 18,
  "candidates": 24,
  "tokens_saved": 2710,
  "latency_ms": 812,
  "provider": "groq",
  "verdicts": [{"potential": 0.71, "fired": true, "components": {"semantic": 0.44, "lexical": 0.15, "entity": 0.12, "lateral_inhibition": 0.0, "conflict": 0.0}}],
  "potentials": {"chunk:d1:0003": 0.341, "…": 0.0}
}
```

---

## 15. WebSocket Neural-Activity Protocol

Connect: `ws://localhost:8000/ws/neural-activity`. Every frame is JSON with a common envelope:

```json
{ "type": "spike", "polarity": "excitatory", "ts": 1760000000.123, "run_id": "ab12cd34", "…": "…" }
```

`polarity` colours the impulse in the cockpit — `excitatory` (cyan), `inhibitory` (rose), `signal` (violet). Frame kinds:

| `type` | Emitted by | Key fields |
|---|---|---|
| `hello` | on connect | link confirmation, graph counts, 30 replayed recent events |
| `run_start` | sensory neuron | `query`, `loop` |
| `spike` | everywhere | `node` and/or `path` (impulse route), `message`; optional `fired/suppressed/mode/tags/gain/expansion` |
| `gate` | circuits + critic | `verdict` (fire/suppress/pass/fail), `potential`, `threshold`, `components` (full PSP breakdown), `reason` |
| `ingest` | ingestion pipeline | `node: doc:<id>`, `chunks`, label |
| `run_end` | critic / error paths | `answer` (truncated), `grounded`, `loops`, `tokens_saved` |
| `heartbeat` | keep-alive | every 15 s of silence |

Guarantees: per-subscriber bounded queues (slow clients drop frames — the visualiser can never throttle the pipeline); per-run event cap (`WS_MAX_EVENTS`) so a huge corpus can't flood the link; auto-reconnect with exponential backoff in the cockpit hook.

---

## 16. Testing

28 tests in `backend/tests/`, all **fully offline and deterministic** (Reflex-Arc mode, temp data dirs):

| File | Coverage |
|---|---|
| `test_textutils.py` | tokeniser, sentence split, lexical-overlap bounds, slugify/truncate |
| `test_activation.py` | membrane firing/suppression, lateral-inhibition reason, numeric-conflict detection, spreading-activation decay & die-out |
| `test_ingestion_parsing.py` | markdown structure (sections/tables/code), section-path-preserving chunking, entity/ref extraction, multi-format dispatch |
| `test_stores.py` | vector upsert/search/delete, disk persistence + reload roundtrip, graph wiring (counts/entities/hubs/agents), REFERENCES/RELATES_TO synapses, document removal, restart reload |
| `test_nervous_system.py` | **end-to-end**: ingest → query → grounded answer with valid `[S#]` citations; unrelated query → clean no-evidence error; live event stream (run_start/gate/run_end); duplicate-ingest no-op |
| `test_api.py` | full HTTP lifecycle via TestClient: health → multipart ingest → documents → graph snapshot → query → stats → delete; validation errors; **WebSocket stream test** |

```bash
.venv\Scripts\python -m pytest backend/tests -q
# ................................            [100%]   28 passed
```

The conftest forces offline env vars *before* any app import, so CI needs no keys and no network.

---

## 17. Design Principles & Engineering Notes

1. **Local decisions over central bottlenecks.** No reranker model, no single "router" agent: each chunk's fate is decided by a small arithmetic vote it can lose locally. This is the fly-brain thesis translated to middleware.
2. **Suppression before generation.** Hallucination and cost are both functions of context size. SynapseCraft spends its effort *before* the LLM call: laterally-inhibited, conflict-checked, capacity-capped evidence.
3. **Structure is signal.** Tables, heading paths and cross-references aren't formatting trivia — they become graph wiring that propagation exploits at query time.
4. **Never go dark.** Every external dependency has a fallback (Qdrant→memory, Ollama→hashing, Groq→Ollama→Reflex, Neo4j→no-op). Every subsystem catches, degrades, persists.
5. **Observability is a feature, not a plugin.** The event bus is a first-class subsystem; the visualiser is best-effort by design (bounded queues, drop-on-overflow) so a laggy browser can never slow cognition.
6. **Determinism where it counts.** Gating math, reflex synthesis and heuristic critique are pure functions — reproducible runs, testable offline, no temperature flakiness in the safety-critical path.
7. **One config file.** A single `.env`, read by both servers, with no example duplicates — the file itself documents every key.

---

## 18. Performance Model — Why Gating Saves Compute

Consider a corpus of 10,000 chunks (≈150-token average, `RECALL_K=24`, `TOP_K=6`):

- A naive "stuff everything" RAG pass might push **8,000+ tokens** of context per query.
- SynapseCraft's gate sends at most 6 chunks ≈ **900 tokens** (capped again by a 6,000-char context budget) — a **~90% context reduction**, while the numeric-conflict critic and lateral inhibition actively *improve* evidence quality.
- The Synthesist reports `tokens_saved` per query — the sum of suppressed chunks' token estimates that never entered the prompt — and the System Bar accumulates it live.
- Reflection loops are bounded (default 2) and only trigger on ungrounded answers, so worst-case cost is `1 + MAX_REFLECTION_LOOPS` propagation passes — each pass being cheap NumPy/graph arithmetic, not model calls.

---

## 19. Extending SynapseCraft

- **New parser** → add `parse_<fmt>` in `ingestion/`, dispatch in `parser.parse_source`, return a `ParsedDocument`.
- **New synapse kind** → add the edge in `connectome/builder.py`; include it in (or exclude from) `spreading_activation`'s skip-set depending on whether it should carry retrieval signal.
- **New agent** → add a node method + `g.add_node(...)` + edges in `NervousSystem._build_graph`; emit events with `self._emit(st, ...)`.
- **New LLM engine** → extend `LLMProvider._engines()` / add an engine method; order in the ladder is the only wiring.
- **New gating term** → add a weight constant in `activation.py` and a component in `membrane_potential` — it appears automatically in every gate event and in the cockpit.
- **Cockpit** → all wire types live in `frontend/lib/types.ts`; components consume the auto-reconnecting `useNeuralStream` hook.

---

## 20. Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| Cockpit shows "gateway offline" | backend not started / CORS | start `backend/run.py`; check `CORS_ORIGINS` includes your origin |
| Query returns *"no signal crossed the firing threshold"* | θ too high for your corpus, or corpus irrelevant | lower `GATING_THRESHOLD` (try 0.35), or ingest related documents |
| Everything fires / noise in answers | θ too low | raise θ toward 0.7, or raise `LATERAL_INHIBITION` |
| Ollama detected but slow first query | model cold load | pre-warm: `ollama run llama3.1 "hi"`; or set `LLM_PROVIDER=reflex` |
| Cockpit graph empty after restart | data didn't persist | check `backend/data/` exists & is writable; `DATA_DIR` points there |
| Neo4j mirror silent | no driver / bad creds | best-effort by design; check the startup warning line |
| Windows console `UnicodeEncodeError` on boot | legacy code page | fixed in `run.py` (stdout reconfigured to UTF-8); also set `PYTHONIOENCODING=utf-8` |
| WebSocket keeps reconnecting | proxy killing idle sockets | heartbeats fire every 15 s; set `NEXT_PUBLIC_WS_URL` explicitly behind proxies |

---

## 21. Roadmap & The Science Behind It

**Inspiration sources**

- *FlyWire Consortium* — complete adult *Drosophila* connectome (~140k neurons / ~50M synapses); local-circuit dominance in neuropils.
- *Collins & Loftus (1975)* — spreading-activation theory of semantic processing (our graph propagation).
- *Lateral inhibition* — the canonical contrast-sharpening mechanism of early sensory systems (Limulus eye → retina → our gate).
- *Reflex arcs (Sherrington)* — sense → integrate → act → verify loops (our critic cycle).

**Near-term roadmap**

- [ ] Multi-hop query decomposition — the circuits spawning sub-signals for compound questions.
- [ ] Spike-timing-dependent plasticity: reweighting `RELATES_TO` synapses from query feedback (the graph learning which paths deliver value).
- [ ] Streaming token responses over a second WebSocket channel.
- [ ] Neuropil-specific subgraph views + hub drill-down in the cockpit.
- [ ] D3 force-directed alternative layout with full propagation-tree animation.
- [ ] Docker compose: gateway + cockpit + Qdrant + Neo4j in one command.

---

## 22. License

MIT — see the repository. Built as a demonstration that **biology's wiring tricks — local circuits, inhibitory gating, reflex verification — are executable software architecture**, not just metaphors.

*SynapseCraft · the connectome is the router.* 🪰⚡












