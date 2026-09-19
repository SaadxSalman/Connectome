/** Shared domain types — mirror of the backend's wire format. */

export type Polarity = "excitatory" | "inhibitory" | "signal";

export type NodeType =
  | "agent"
  | "doc"
  | "section"
  | "chunk"
  | "entity"
  | "hub";

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  x: number;
  y: number;
  doc_id?: string;
  doc_title?: string;
  kind?: string;
  glyph?: string;
  hub?: string;
  section?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  weight: number;
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
  hubs: { id: string; label: string; size: number }[];
  counts: Record<string, number>;
  total_chunks: number;
  rendered_chunks: number;
}

export interface Citation {
  tag: string;
  id: string;
  doc_title: string;
  section: string;
  kind: string;
  score: number;
  text: string;
}

export interface GateVerdict {
  potential: number;
  fired: boolean;
  threshold: number;
  reason: string;
  components: Record<string, number>;
}

export interface QueryResult {
  run_id: string;
  query: string;
  answer: string;
  citations: Citation[];
  tags_used: string[];
  grounded: boolean;
  critic_note: string;
  loops: number;
  fired: {
    tag: string;
    id: string;
    text: string;
    doc_title: string;
    section: string;
    kind: string;
    order: number;
    score: number;
    sim: number;
  }[];
  suppressed: number;
  candidates: number;
  expansion: string[];
  tokens_saved: number;
  latency_ms: number;
  error: string;
  provider: string;
  verdicts: GateVerdict[];
  potentials: Record<string, number>;
  llm: { provider: string; model: string; fallback?: string };
}

export type EventKind =
  | "run_start"
  | "run_end"
  | "spike"
  | "gate"
  | "ingest"
  | "heartbeat"
  | "hello";

export interface NeuralEvent {
  type: EventKind;
  polarity?: Polarity;
  ts: number;
  run_id?: string;
  node?: string;
  path?: string[];
  message?: string | null;
  label?: string;
  verdict?: string;
  potential?: number;
  threshold?: number;
  components?: Record<string, number>;
  reason?: string;
  candidates?: number;
  fired?: number;
  suppressed?: number;
  loops?: number;
  grounded?: boolean;
  tokens_saved?: number;
  answer?: string;
  gain?: number;
  mode?: string;
  tags?: string[];
  expansion?: string[];
  recovered?: number;
  query?: string;
  chunks?: number;
  graph?: Record<string, number>;
  recent?: NeuralEvent[];
}

export interface HealthReport {
  status: string;
  components: {
    embedder: { provider: string; model: string; dim: number } | null;
    vector_store: string;
    llm: { provider: string; model: string; fallback?: string } | null;
    neo4j_mirror: boolean;
    graph: Record<string, number>;
  };
  stats: {
    uptime_s: number;
    queries: number;
    ingestions: number;
    spikes_emitted: number;
    gates_suppressed: number;
    tokens_saved: number;
    reflection_loops: number;
    avg_latency_ms: number;
    last_latency_ms: number;
  };
}

export interface IngestResult {
  doc_id: string;
  title: string;
  chunks: number;
  sections: number;
  entities: number;
  warnings: string[];
  ms: number;
}

export interface DocumentRecord {
  doc_id: string;
  title: string;
  chunks: number;
  sections: number;
}
