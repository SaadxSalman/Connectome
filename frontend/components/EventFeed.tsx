"use client";

/** Live neural-event feed — the cockpit's "electrophysiology trace". */

import type { NeuralEvent } from "@/lib/types";

const COLORS: Record<string, string> = {
  excitatory: "text-signal-excite",
  inhibitory: "text-signal-inhibit",
  signal: "text-signal-info",
};

const KIND_GLYPH: Record<string, string> = {
  run_start: "▶",
  run_end: "■",
  spike: "⚡",
  gate: "⧨",
  ingest: "⇣",
  hello: "◉",
};

function lineFor(evt: NeuralEvent): { color: string; text: string } {
  const color = COLORS[evt.polarity ?? "signal"] ?? "text-slate-400";
  const node = (evt.node ?? evt.path?.join("→") ?? "").replace(/^(chunk|doc|sec|ent|hub):/, "").slice(0, 34);
  return { color, text: `${KIND_GLYPH[evt.type] ?? "•"} ${evt.message ?? evt.type}${node ? ` · ${node}` : ""}` };
}

export default function EventFeed({ events, link }: { events: NeuralEvent[]; link: string }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-neuron-line px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
        <span>Neural Activity</span>
        <span className={link === "live" ? "text-signal-fire" : link === "down" ? "text-signal-inhibit" : "text-slate-500"}>
          ● {link}
        </span>
      </div>
      <div className="flex-1 space-y-0.5 overflow-y-auto p-2">
        {events.length === 0 && (
          <p className="px-1 py-6 text-center text-[11px] text-slate-600">
            waiting for spikes… ingest a document and fire a query
          </p>
        )}
        {[...events].reverse().map((evt, i) => {
          const { color, text } = lineFor(evt);
          return (
            <div key={`${evt.ts}-${i}`} className={`truncate rounded px-1.5 py-0.5 text-[11px] hover:bg-neuron-edge ${color}`}>
              <span className="mr-1.5 text-slate-600">
                {new Date(evt.ts * 1000).toLocaleTimeString([], { hour12: false })}
              </span>
              {text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
