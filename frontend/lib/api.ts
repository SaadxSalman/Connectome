/** Cockpit → gateway API client. */

import type {
  DocumentRecord,
  GraphSnapshot,
  HealthReport,
  IngestResult,
  QueryResult,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export function wsUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit.replace(/\/$/, "");
  return API_URL.replace(/^http/, "ws") + "/ws/neural-activity";
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${API_URL}/api/health`, { cache: "no-store" }).then((r) => json<HealthReport>(r)),

  graph: (maxChunks = 180) =>
    fetch(`${API_URL}/api/graph?max_chunks=${maxChunks}`, { cache: "no-store" }).then((r) => json<GraphSnapshot>(r)),

  documents: () =>
    fetch(`${API_URL}/api/documents`, { cache: "no-store" }).then((r) =>
      json<{ documents: DocumentRecord[]; total_chunks: number }>(r),
    ),

  query: (query: string) =>
    fetch(`${API_URL}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }).then((r) => json<QueryResult>(r)),

  ingestFiles: async (files: File[]): Promise<{ ingested: IngestResult[]; errors: { file: string; error: string }[] }> => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch(`${API_URL}/api/ingest`, { method: "POST", body: form });
    return json(res);
  },

  ingestUrl: (url: string, previewOnly = false) =>
    fetch(`${API_URL}/api/ingest/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, preview_only: previewOnly }),
    }).then((r) => json<{ ingested?: IngestResult[]; preview?: { title: string; chars: number; preview: string; warnings: string[] } }>(r)),

  deleteDocument: (docId: string) =>
    fetch(`${API_URL}/api/documents/${docId}`, { method: "DELETE" }).then((r) => json<{ removed: string }>(r)),

  reset: () => fetch(`${API_URL}/api/reset`, { method: "POST" }).then((r) => json<{ reset: boolean }>(r)),
};
