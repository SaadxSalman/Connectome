"use client";

/** Top status bar: component inventory + live nervous-system telemetry. */

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { HealthReport } from "@/lib/types";

export default function SystemBar({ pulse }: { pulse: number }) {
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const h = await api.health();
        if (!stop) {
          setHealth(h);
          setOnline(true);
        }
      } catch {
        if (!stop) setOnline(false);
      }
    };
    tick();
    const t = setInterval(tick, 6000);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  const c = health?.components;
  const s = health?.stats;

  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-1 border-b border-neuron-line bg-neuron-panel px-4 py-2 text-[11px]">
      <div className="flex items-center gap-2">
        <span className={`text-base ${pulse % 2 ? "opacity-60" : "opacity-100"}`}>🧠</span>
        <span className="font-semibold tracking-wide text-slate-200">SYNAPSECRAFT</span>
        <span className="text-slate-600">connectome cockpit</span>
      </div>

      <StatusDot ok={online} label={online ? "gateway online" : "gateway offline"} />

      {c && (
        <>
          <Stat label="llm" value={c.llm ? `${c.llm.provider}` : "reflex"} />
          <Stat label="embed" value={c.embedder ? `${c.embedder.provider}·${c.embedder.dim}d` : "—"} />
          <Stat label="store" value={c.vector_store} />
          {c.neo4j_mirror && <Stat label="neo4j" value="mirrored" />}
        </>
      )}

      {s && (
        <div className="ml-auto flex items-center gap-4 text-slate-500">
          <Stat label="queries" value={String(s.queries)} />
          <Stat label="gated out" value={String(s.gates_suppressed)} />
          <Stat label="tokens saved" value={String(s.tokens_saved)} />
          <Stat label="avg" value={`${s.avg_latency_ms}ms`} />
          <span className="flex items-center gap-1 text-signal-excite">
            <span className="inline-block h-1.5 w-1.5 animate-pulse-fast rounded-full bg-signal-excite" />
            {s.spikes_emitted} spikes
          </span>
        </div>
      )}
    </header>
  );
}

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`flex items-center gap-1.5 ${ok ? "text-signal-fire" : "text-signal-inhibit"}`}>
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${ok ? "bg-signal-fire" : "bg-signal-inhibit"}`} />
      {label}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex gap-1.5">
      <span className="text-slate-600">{label}</span>
      <span className="text-slate-300">{value}</span>
    </span>
  );
}
