"use client";

/** Query console + grounded answer panel with structural transparency. */

import { useState } from "react";

import { api } from "@/lib/api";
import type { QueryResult } from "@/lib/types";

export default function QueryConsole({
  onResult,
  disabled,
}: {
  onResult: (r: QueryResult) => void;
  disabled?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);

  async function fire() {
    if (!query.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.query(query);
      setResult(data);
      onResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-neuron-line px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
        Query the Connectome
      </div>
      <div className="flex gap-2 p-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && fire()}
          placeholder="e.g. How do T4 neurons detect motion?"
          className="min-w-0 flex-1 rounded-md border border-neuron-line bg-neuron-bg px-3 py-2 text-[13px] text-slate-200 placeholder:text-slate-600 focus:border-signal-info focus:outline-none"
          disabled={disabled || busy}
        />
        <button
          onClick={fire}
          disabled={disabled || busy || !query.trim()}
          className="rounded-md border border-signal-info/40 bg-signal-info/10 px-4 py-2 text-[12px] font-medium text-signal-info hover:bg-signal-info/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "propagating…" : "fire ⚡"}
        </button>
      </div>
      {error && (
        <p className="mx-3 mb-2 rounded border border-signal-inhibit/30 bg-signal-inhibit/10 px-3 py-1.5 text-[11px] text-signal-inhibit">
          {error}
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {!result && !busy && (
          <p className="py-6 text-center text-[11px] text-slate-600">
            answer + citations will appear here
          </p>
        )}
        {result && <AnswerView r={result} />}
      </div>
    </div>
  );
}

function AnswerView({ r }: { r: QueryResult }) {
  return (
    <div className="space-y-3">
      {r.error && !r.answer ? (
        <p className="rounded border border-signal-inhibit/30 bg-signal-inhibit/10 px-3 py-2 text-[12px] text-signal-inhibit">
          {r.error}
        </p>
      ) : (
        <div className="rounded-md border border-neuron-line bg-neuron-panel p-3">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[10px]">
            <Badge tone={r.grounded ? "fire" : "info"}>{r.grounded ? "grounding verified" : "unverified"}</Badge>
            <Badge tone="info">{r.provider}</Badge>
            <span className="text-slate-500">{r.latency_ms} ms</span>
            <span className="text-slate-500">{r.candidates} candidates</span>
            <span className="text-signal-fire">{r.fired.length} fired</span>
            <span className="text-signal-inhibit">{r.suppressed} suppressed</span>
            {r.loops > 0 && <span className="text-signal-info">⟳ {r.loops} reflection loops</span>}
            {r.tokens_saved > 0 && <span className="text-slate-500">~{r.tokens_saved} tokens gated out</span>}
          </div>
          <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-slate-200">{r.answer}</p>
          {r.critic_note && (
            <p className="mt-2 border-t border-neuron-line pt-2 text-[10px] text-slate-500">
              reflex critic: {r.critic_note}
            </p>
          )}
        </div>
      )}

      {r.citations.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">fired evidence</p>
          {r.citations.map((c) => (
            <details key={c.tag} className="rounded-md border border-neuron-line bg-neuron-panel px-3 py-2">
              <summary className="cursor-pointer text-[11px] text-slate-400">
                <span className="text-signal-excite">[{c.tag}]</span> {c.doc_title} › {c.section}
                <span className="ml-2 text-slate-600">V={c.score}</span>
              </summary>
              <p className="mt-1.5 whitespace-pre-wrap border-t border-neuron-line pt-1.5 text-[11px] leading-relaxed text-slate-400">
                {c.text}
              </p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

function Badge({ tone, children }: { tone: "fire" | "info" | "inhibit"; children: React.ReactNode }) {
  const cls = {
    fire: "border-signal-fire/40 text-signal-fire bg-signal-fire/10",
    info: "border-signal-info/40 text-signal-info bg-signal-info/10",
    inhibit: "border-signal-inhibit/40 text-signal-inhibit bg-signal-inhibit/10",
  }[tone];
  return <span className={`rounded-full border px-2 py-0.5 ${cls}`}>{children}</span>;
}
