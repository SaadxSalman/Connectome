"use client";

/** Ingestion control panel: uploads, web-scout, document inventory. */

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { DocumentRecord, GraphSnapshot, IngestResult } from "@/lib/types";

export default function IngestPanel({
  onChanged,
}: {
  onChanged: (snapshot?: GraphSnapshot) => void;
}) {
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [url, setUrl] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const d = await api.documents();
      setDocs(d.documents);
    } catch {
      /* gateway offline */
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, [refresh]);

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setNotice("");
    try {
      const res = await api.ingestFiles(Array.from(files));
      const n = res.ingested.reduce((a, r) => a + r.chunks, 0);
      const errs = res.errors.map((e) => `${e.file}: ${e.error}`).join(" · ");
      setNotice(
        `${res.ingested.length} doc(s) → ${n} chunks wired${errs ? ` · errors: ${errs}` : ""}`,
      );
      onChanged();
      refresh();
    } catch (e) {
      setNotice(`ingest failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handleUrl() {
    if (!url.trim() || busy) return;
    setBusy(true);
    setNotice("");
    try {
      const res = await api.ingestUrl(url.trim());
      const ing = res.ingested?.[0];
      setNotice(ing ? `scouted '${ing.title}' → ${ing.chunks} chunks` : "no result");
      setUrl("");
      onChanged();
      refresh();
    } catch (e) {
      setNotice(`scout failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(docId: string) {
    setBusy(true);
    try {
      await api.deleteDocument(docId);
      onChanged();
      refresh();
    } finally {
      setBusy(false);
    }
  }

  async function handleReset() {
    if (!confirm("Wipe the entire connectome?")) return;
    setBusy(true);
    try {
      await api.reset();
      setNotice("connectome wiped");
      onChanged();
      refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-neuron-line px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500">
        <span>Ingestion</span>
        <button
          onClick={handleReset}
          disabled={busy || docs.length === 0}
          className="rounded border border-signal-inhibit/30 px-1.5 py-0.5 text-[9px] normal-case text-signal-inhibit hover:bg-signal-inhibit/10 disabled:opacity-30"
        >
          wipe
        </button>
      </div>

      <div className="space-y-2 p-3">
        <label
          className={`block cursor-pointer rounded-md border border-dashed border-neuron-line px-3 py-3 text-center text-[11px] text-slate-500 hover:border-signal-info/40 hover:text-signal-info ${busy ? "opacity-40" : ""}`}
        >
          <input
            ref={fileRef}
            type="file"
            multiple
            hidden
            accept=".md,.markdown,.txt,.pdf,.csv,.json,.html,.htm,.log"
            onChange={(e) => handleUpload(e.target.files)}
            disabled={busy}
          />
          {busy ? "wiring synapses…" : "⇡ drop files / click to upload (md · pdf · txt · csv · json · html)"}
        </label>

        <div className="flex gap-1.5">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleUrl()}
            placeholder="https:// — send the web scout"
            className="min-w-0 flex-1 rounded border border-neuron-line bg-neuron-bg px-2 py-1.5 text-[11px] placeholder:text-slate-600 focus:border-signal-info focus:outline-none"
            disabled={busy}
          />
          <button
            onClick={handleUrl}
            disabled={busy || !url.trim()}
            className="rounded border border-signal-info/40 bg-signal-info/10 px-2 py-1 text-[10px] text-signal-info disabled:opacity-30"
          >
            scout
          </button>
        </div>

        {notice && <p className="text-[10px] leading-relaxed text-slate-400">{notice}</p>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-neuron-line p-2">
        {docs.length === 0 ? (
          <p className="py-4 text-center text-[10px] text-slate-600">no documents wired yet</p>
        ) : (
          docs.map((d) => (
            <div key={d.doc_id} className="group flex items-center gap-2 rounded px-1.5 py-1 hover:bg-neuron-edge">
              <span className="text-signal-excite">▤</span>
              <span className="min-w-0 flex-1 truncate text-[11px] text-slate-300" title={d.title}>
                {d.title}
              </span>
              <span className="text-[9px] text-slate-600">{d.chunks}c</span>
              <button
                onClick={() => handleDelete(d.doc_id)}
                className="text-[10px] text-slate-700 opacity-0 transition group-hover:opacity-100 hover:text-signal-inhibit"
                title="prune this document"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
