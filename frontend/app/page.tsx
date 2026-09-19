"use client";

/** SynapseCraft cockpit — the main dashboard shell.
 *
 * Layout: system telemetry bar on top; the interactive connectome graph in
 * the centre; query console + answer panel on the left; live neural-activity
 * feed and ingestion controls on the right.
 */

import { useCallback, useEffect, useState } from "react";

import ConnectomeGraphView from "@/components/ConnectomeGraph";
import EventFeed from "@/components/EventFeed";
import IngestPanel from "@/components/IngestPanel";
import QueryConsole from "@/components/QueryConsole";
import SystemBar from "@/components/SystemBar";
import { api } from "@/lib/api";
import { useNeuralStream } from "@/lib/useNeuralStream";
import type { GraphSnapshot } from "@/lib/types";

export default function CockpitPage() {
  const { events, link, pulse } = useNeuralStream(220);
  const [snapshot, setSnapshot] = useState<GraphSnapshot | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const refreshGraph = useCallback(async () => {
    try {
      setSnapshot(await api.graph(200));
    } catch {
      /* gateway offline — keep the last snapshot on screen */
    }
  }, []);

  useEffect(() => {
    refreshGraph();
    const t = setInterval(refreshGraph, 15000);
    return () => clearInterval(t);
  }, [refreshGraph]);

  // Refresh the graph a moment after each ingestion pulse.
  useEffect(() => {
    const ingest = events.filter((e) => e.type === "ingest");
    if (ingest.length) {
      const last = ingest[ingest.length - 1];
      const t = setTimeout(refreshGraph, 1200);
      return () => clearTimeout(t);
    }
  }, [events, refreshGraph]);

  return (
    <div className="flex h-screen flex-col">
      <SystemBar pulse={pulse} />

      <div className="grid min-h-0 flex-1 grid-cols-[340px_1fr_320px]">
        {/* left rail: query + answer */}
        <aside className="flex min-h-0 flex-col border-r border-neuron-line bg-neuron-panel">
          <QueryConsole onResult={() => undefined} />
        </aside>

        {/* centre: the living connectome */}
        <main className="min-h-0">
          <ConnectomeGraphView
            snapshot={snapshot}
            events={events}
            onSelectNode={setSelected}
          />
        </main>

        {/* right rail: neural feed + ingestion */}
        <aside className="flex min-h-0 flex-col border-l border-neuron-line bg-neuron-panel">
          <div className="min-h-0 flex-[3] border-b border-neuron-line">
            <EventFeed events={events} link={link} />
          </div>
          <div className="min-h-0 flex-[2]">
            <IngestPanel onChanged={refreshGraph} />
          </div>
        </aside>
      </div>

      <footer className="flex items-center justify-between border-t border-neuron-line bg-neuron-panel px-4 py-1.5 text-[10px] text-slate-600">
        <span>
          sensory → circuits → synthesist → critic · reflection loop bounded · inhibitory gating θ active
        </span>
        {selected && (
          <span className="max-w-[40ch] truncate text-slate-500">selected: {selected}</span>
        )}
      </footer>
    </div>
  );
}
