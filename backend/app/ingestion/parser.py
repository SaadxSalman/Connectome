"""Document parsers that retain structural topology.

Unlike naive text splitters, these parsers keep the document's skeleton —
headings (hierarchical outlines), tables, lists, code fences and
cross-references — so the connectome can wire CONTAINS / REFERENCES synapses
exactly like the biological wiring map they imitate.

Core parsers (Markdown, plain text) live here; CSV/JSON/PDF/HTML live in
``parser_web``. ``parse_source`` dispatches by extension and content sniffing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Block:
    kind: str  # text | table | list | code
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class Section:
    title: str = ""
    level: int = 0
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str
    source: str
    kind: str
    root: Section = field(default_factory=lambda: Section("", 0))
    warnings: list[str] = field(default_factory=list)


def flatten(doc: ParsedDocument) -> str:
    """Full raw text of a parsed document (used for content hashing)."""
    parts: list[str] = []

    def walk(sec: Section) -> None:
        for b in sec.blocks:
            parts.append(b.text if b.kind != "table" else _table_text(b.rows))
        for c in sec.children:
            walk(c)

    walk(doc.root)
    return "\n".join(parts)


def _table_text(rows: list[list[str]]) -> str:
    lines = [" | ".join(rows[0])] if rows else []
    for r in rows[1:]:
        lines.append(" | ".join(r))
    return "\n".join(lines)


# ── Markdown ───────────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}[\s:|-]*\|?\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _split_row(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def parse_markdown(text: str, fallback_title: str) -> ParsedDocument:
    root = Section("", 0)
    doc = ParsedDocument(fallback_title, fallback_title, "markdown", root)
    stack = [root]
    para: list[str] = []
    list_buf: list[str] = []
    table_rows: list[list[str]] = []
    code_buf: list[str] = []
    in_code = False

    def flush_para() -> None:
        nonlocal para
        if para:
            stack[-1].blocks.append(Block("text", "\n".join(para).strip()))
            para = []

    def flush_list() -> None:
        nonlocal list_buf
        if list_buf:
            stack[-1].blocks.append(Block("list", "\n".join(list_buf).strip()))
            list_buf = []

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            stack[-1].blocks.append(Block("table", "", table_rows))
            table_rows = []

    def flush_code() -> None:
        nonlocal code_buf
        if code_buf:
            stack[-1].blocks.append(Block("code", "\n".join(code_buf).strip()))
            code_buf = []

    def flush_all() -> None:
        flush_para()
        flush_list()
        flush_table()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_all()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        if list_buf and not _LIST_RE.match(line):
            flush_list()
        m = _HEADING_RE.match(line)
        if m:
            flush_all()
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1 and doc.title == fallback_title:
                doc.title = title
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            sec = Section(title, level)
            stack[-1].children.append(sec)
            stack.append(sec)
            i += 1
            continue
        if "|" in stripped and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_all()
            table_rows = [_split_row(stripped)]
            i += 2
            while (
                i < len(lines)
                and "|" in lines[i]
                and lines[i].strip()
                and not _HEADING_RE.match(lines[i])
                and not lines[i].strip().startswith("```")
            ):
                table_rows.append(_split_row(lines[i].strip()))
                i += 1
            flush_table()
            continue
        if _LIST_RE.match(line):
            flush_para()
            flush_table()
            list_buf.append(stripped)
            i += 1
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        para.append(line)
        i += 1
    flush_all()
    flush_code()
    if doc.title == fallback_title and root.children:
        doc.title = root.children[0].title or fallback_title
    return doc

# ── Plain text ─────────────────────────────────────────────────────────────
_TXT_HEADING_RE = re.compile(
    r"^(?:(?:chapter|section|part|appendix)\s+[\dIVXivx]+\b.*|\d+(?:\.\d+)*\.?\s+\S.*)$"
)


def parse_text(text: str, fallback_title: str) -> ParsedDocument:
    root = Section("", 0)
    doc = ParsedDocument(fallback_title, fallback_title, "text", root)
    stack = [root]
    para: list[str] = []

    def flush() -> None:
        nonlocal para
        if para:
            stack[-1].blocks.append(Block("text", " ".join(para).strip()))
            para = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        s = line.strip()
        if len(s) <= 80 and (s.isupper() or _TXT_HEADING_RE.match(s)):
            flush()
            level = 2 if _TXT_HEADING_RE.match(s) else 1
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            sec = Section(s.rstrip("."), level)
            stack[-1].children.append(sec)
            stack.append(sec)
            if len(stack) == 2 and doc.title == fallback_title:
                doc.title = s.rstrip(".")
            continue
        para.append(s)
    flush()
    if doc.title == fallback_title and root.children:
        doc.title = root.children[0].title or fallback_title
    return doc

# ── dispatch ───────────────────────────────────────────────────────────────
def parse_source(name: str, content: bytes, url: str = "") -> ParsedDocument:
    lower = (name or "").lower()
    fallback = (name or url or "document").rsplit("/", 1)[-1] or "document"
    stem = fallback.rsplit(".", 1)[0] if "." in fallback else fallback
    try:
        if lower.endswith((".md", ".markdown")):
            return parse_markdown(content.decode("utf-8", errors="replace"), stem)
        if lower.endswith((".txt", ".text", ".log")):
            return parse_text(content.decode("utf-8", errors="replace"), stem)
        if lower.endswith((".csv", ".json", ".pdf", ".html", ".htm")):
            from .parser_web import parse_csv, parse_html, parse_json, parse_pdf

            if lower.endswith(".csv"):
                return parse_csv(content, stem)
            if lower.endswith(".json"):
                return parse_json(content, stem)
            if lower.endswith(".pdf"):
                return parse_pdf(content, stem)
            return parse_html(content.decode("utf-8", errors="replace"), stem, url)
        head = content[:512].lstrip().lower()
        if head.startswith(b"<!doctype html") or b"<html" in head:
            from .parser_web import parse_html

            return parse_html(content.decode("utf-8", errors="replace"), stem, url)
        if head.startswith(b"%pdf"):
            from .parser_web import parse_pdf

            return parse_pdf(content, stem)
        return parse_text(content.decode("utf-8", errors="replace"), stem)
    except Exception as exc:
        doc = ParsedDocument(stem, url or stem, "raw", Section("", 0))
        doc.warnings.append(f"parser fallback for {name}: {exc}")
        doc.root.children.append(
            Section(stem, 1, [Block("text", content.decode("utf-8", errors="replace")[:20000])])
        )
        return doc


