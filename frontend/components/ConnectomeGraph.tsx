"use client";

/** Interactive Connectome Graph Visualiser — React Flow rendering of the
 * structural connectome, lit up by the live /ws/neural-activity spike stream. */

import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  Background, BackgroundVariant, Controls, ReactFlow,
  useEdgesState, useNodesState, type Edge, type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { GraphSnapshot, NeuralEvent } from "@/lib/types";

const NODE_STYLE: Record<string, { color: string; border: string; glyph: string }> = {
  agent: { color: "#a78bfa", border: "#7c3aed", glyph: "◈" },
  doc: { color: "#38bdf8", border: "#0284c7", glyph: "▤" },
  section: { color: "#7dd3fc", border: "#0369a1", glyph: "§" },
  chunk: { color: "#94a3b8", border: "#334155", glyph: "▪" },
  entity: { color: "#fbbf24", border: "#b45309", glyph: "◆" },
  hub: { color: "#34d399", border: "#059669", glyph: "◎" },
};

const POLARITY_GLOW: Record<string, string> = {
  excitatory: "#22d3ee",
  inhibitory: "#fb7185",
  signal: "#a78bfa",
};

export default function ConnectomeGraphView({
  snapshot,
  events,
  onSelectNode,
}: {
  snapshot: GraphSnapshot | null;
  events: NeuralEvent[];
  onSelectNode: (id: string | null) => void;
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const activeRef = useRef<Map<string, number>>(new Map());

  // Rebuild the base graph whenever a new snapshot arrives.
  useEffect(() => {
    if (!snapshot) return;
    setNodes(snapshot.nodes.map((n) => {
      const st = NODE_STYLE[n.type] ?? NODE_STYLE.chunk;
      return {
        id: n.id,
        position: { x: n.x, y: n.y },
        data: { label: `${st.glyph} ${n.label}`.slice(0, 64), nodeType: n.type },
        style: {
          background: "#0a0f1c",
          border: `1px solid ${st.border}`,
          borderRadius: n.type === "hub" ? 999 : n.type === "agent" ? 10 : 6,
          color: st.color,
          fontSize: 10,
          padding: n.type === "chunk" ? "3px 7px" : "6px 10px",
        },
      };
    }));
    setEdges(snapshot.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      animated: false,
      style: {
        stroke: e.kind === "RELATES_TO" ? "#1e3a5f" : e.kind === "MENTIONS" ? "#3a2f14" : "#14213a",
        strokeWidth: Math.max(0.6, Math.min(2.2, e.weight * 1.6)),
      },
      label: e.kind === "RELATES_TO" || e.kind === "REFERENCES" ? e.kind : undefined,
      labelStyle: { fontSize: 8, fill: "#475569" },
    })));
  }, [snapshot, setNodes, setEdges]);

  // Light up pathways from the live spike stream.
  useEffect(() => {
    if (!events.length) return;
    const now = Date.now();
    const active = activeRef.current;
    for (const evt of events.slice(-40)) {
      if (evt.path) for (const t of evt.path) active.set(t, now);
      if (evt.node) active.set(evt.node, now);
      if (evt.path && evt.path.length === 2) active.set(`${evt.path[0]}->${evt.path[1]}`, now);
    }
    const cutoff = now - 2600;
    for (const [k, ts] of active) if (ts < cutoff) active.delete(k);

    setNodes((prev) => prev.map((n) => {
      const hot = active.has(n.id);
      const glow = hot
        ? `0 0 16px ${POLARITY_GLOW.excite}, 0 0 4px ${POLARITY_GLOW.excite}`
        : n.data.nodeType === "agent" ? "0 0 14px #7c3aed55" : undefined;
      if (n.style?.boxShadow === glow) return n;
      return { ...n, style: { ...n.style, boxShadow: glow } };
    }));
    setEdges((prev) => prev.map((e) => {
      const hot = active.has(e.id);
      if (hot === e.animated) return e;
      return hot
        ? { ...e, animated: true, style: { ...e.style, stroke: POLARITY_GLOW.excite, strokeWidth: 2.4 } }
        : { ...e, animated: false, style: { ...e.style, strokeWidth: 1 } };
    }));
  }, [events, setNodes, setEdges]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => onSelectNode(node.id),
    [onSelectNode],
  );

  const countsLine = useMemo(() => {
    if (!snapshot) return "";
    const c = snapshot.counts;
    return `${c.documents ?? 0} docs · ${c.sections ?? 0} sections · ${c.chunks ?? 0} chunks · ${c.entities ?? 0} entities · ${c.hubs ?? 0} hubs · ${c.edges ?? 0} synapses`;
  }, [snapshot]);

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        fitView
        minZoom={0.08}
        maxZoom={2.2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={26} size={1} color="#13203a" />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-neuron-line bg-neuron-panel/90 px-3 py-1.5 text-[11px] text-slate-400">
        {countsLine || "connectome offline"}
      </div>
      <div className="pointer-events-none absolute right-3 top-3 flex gap-3 text-[10px]">
        <Legend color="#22d3ee" label="excitatory" />
        <Legend color="#fb7185" label="inhibitory" />
        <Legend color="#a78bfa" label="signal" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-neuron-line bg-neuron-panel/90 px-2 py-0.5 text-slate-400">
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
