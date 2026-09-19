"""Sensory cortex tools: web scout (fetch + parse live feeds into docs).

The Sensory Neuron captures and tokenises incoming queries or live web feeds.
``scout_url`` downloads a page and returns (filename, bytes) ready for the
standard ingestion pipeline; it silently refuses non-HTML / non-text payloads
and oversized bodies.
"""

from __future__ import annotations

import httpx

from ..ingestion.parser import parse_source
from ..textutils import truncate

MAX_SCOUT_BYTES = 3_000_000
UA = "SynapseCraft-Scout/1.0 (+https://github.com/saadxsalman/connectome)"


def scout_fetch(url: str, *, timeout: float = 20.0) -> tuple[str, bytes]:
    """Download a URL with a browser-ish UA and size cap. Raises on failure."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=8.0),
        follow_redirects=True,
        headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
    ) as client:
        r = client.get(url)
        r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if any(bad in ctype for bad in ("image/", "video/", "audio/", "application/zip", "application/octet-stream")):
        raise ValueError(f"unsupported content-type: {ctype}")
    body = r.content[:MAX_SCOUT_BYTES]
    if not body.strip():
        raise ValueError("empty response body")
    return url, body


def scout_preview(url: str, max_chars: int = 400) -> dict:
    """Fetch + parse + preview: what *would* the connectome ingest?"""
    filename, body = scout_fetch(url)
    doc = parse_source(filename, body, url)
    from ..ingestion.parser import flatten

    text = flatten(doc).strip()
    if not text:
        raise ValueError("no extractable text at that url")
    return {
        "url": url,
        "title": truncate(doc.title, 120),
        "kind": doc.kind,
        "chars": len(text),
        "preview": truncate(text, max_chars),
        "warnings": doc.warnings,
    }
