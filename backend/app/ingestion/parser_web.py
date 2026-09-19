"""Extended format parsers: CSV, JSON, PDF, HTML."""

from __future__ import annotations

import csv
import io
import json as jsonlib

from .parser import Block, ParsedDocument, Section, _table_text  # noqa: F401


# ── CSV ────────────────────────────────────────────────────────────────────
def parse_csv(content: bytes, fallback_title: str) -> ParsedDocument:
    text = content.decode("utf-8-sig", errors="replace")
    rows = [r for r in csv.reader(io.StringIO(text)) if any((c or "").strip() for c in r)]
    doc = ParsedDocument(fallback_title, fallback_title, "csv", Section("", 0))
    sec = Section(fallback_title, 1)
    sec.blocks.append(Block("table", "", rows[:2000]))
    doc.root.children.append(sec)
    return doc


# ── JSON ───────────────────────────────────────────────────────────────────
def parse_json(content: bytes, fallback_title: str) -> ParsedDocument:
    doc = ParsedDocument(fallback_title, fallback_title, "json", Section("", 0))
    raw = content.decode("utf-8-sig", errors="replace")
    try:
        data = jsonlib.loads(raw)
    except Exception as exc:
        doc.warnings.append(f"invalid JSON: {exc}")
        sec = Section(fallback_title, 1, [Block("text", raw[:20000])])
        doc.root.children.append(sec)
        return doc

    def emit(node, sec: Section, depth: int) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                child = Section(str(k), depth + 1)
                sec.children.append(child)
                emit(v, child, depth + 1)
        elif isinstance(node, list):
            child = Section(f"{sec.title} (list, {len(node)} items)", depth + 1)
            child.blocks.append(
                Block("text", jsonlib.dumps(node, ensure_ascii=False, indent=1)[:6000])
            )
            sec.children.append(child)
        else:
            sec.blocks.append(Block("text", str(node)))

    emit(data, doc.root, 0)
    return doc


# ── PDF ────────────────────────────────────────────────────────────────────
def parse_pdf(content: bytes, fallback_title: str) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    title = fallback_title
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            title = str(meta.title)
    except Exception:
        pass
    doc = ParsedDocument(title, fallback_title, "pdf", Section("", 0))
    for pageno, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        sec = Section(f"Page {pageno}", 2)
        para: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                if para:
                    sec.blocks.append(Block("text", " ".join(para)))
                    para = []
                continue
            if (
                len(line) < 70
                and (line.isupper() or (
                    line[:1].isupper()
                    and not line.endswith((".", ",", ";", ":"))
                    and len(line.split()) <= 10
                ))
            ):
                if para:
                    sec.blocks.append(Block("text", " ".join(para)))
                    para = []
                sec.blocks.append(Block("text", line))
            else:
                para.append(line)
        if para:
            sec.blocks.append(Block("text", " ".join(para)))
        doc.root.children.append(sec)
    return doc


# ── HTML ───────────────────────────────────────────────────────────────────
def parse_html(html: str, fallback_title: str, source_url: str = "") -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    title = fallback_title
    if soup.title and soup.title.string:
        title = soup.title.string.strip() or fallback_title
    doc = ParsedDocument(title, source_url or fallback_title, "html", Section("", 0))
    stack = [doc.root]
    body = soup.body or soup
    prev_li = False
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li"]):
        name = el.name
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if name.startswith("h"):
            level = min(6, int(name[1]))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            sec = Section(text[:120], level)
            stack[-1].children.append(sec)
            stack.append(sec)
            prev_li = False
        elif name == "table":
            rows: list[list[str]] = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                stack[-1].blocks.append(Block("table", "", rows[:500]))
            prev_li = False
        elif name == "li":
            blocks = stack[-1].blocks
            if prev_li and blocks and blocks[-1].kind == "list":
                blocks[-1].text = f"{blocks[-1].text}\n{text}"
            else:
                blocks.append(Block("list", text))
            prev_li = True
        else:
            stack[-1].blocks.append(Block("text", text))
            prev_li = False
    return doc
