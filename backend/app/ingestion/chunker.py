"""Structure-aware chunking + lightweight entity & cross-reference extraction."""

from __future__ import annotations

import hashlib
import re

from ..types import Chunk
from ..textutils import sentence_split, slugify, truncate
from .parser import Block, ParsedDocument, Section

_CAP_SEQ_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:\s+(?:of|de|for|and|in)?\s*[A-Z][a-zA-Z0-9]*)+)\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_LINK_RE = re.compile(r"\[([^\]]{3,60})\]\(([^)\s]+)\)")
_SEE_RE = re.compile(
    r"(?:see|cf\.?|ref(?:er)?\.?|discussed in)\s+(?:the\s+)?"
    r"(?:section\s+|chapter\s+|figure\s+|table\s+|doc(?:ument)?\s+)?"
    r"([A-Za-z0-9][\w \-:/]{2,48})",
    re.I,
)

_ENTITY_STOP = {
    "The", "This", "That", "These", "Those", "We", "Our", "It", "Its",
    "Figure", "Table", "Section", "Chapter", "Note", "Source", "However",
    "When", "Where", "While", "Because", "Their", "There", "Then", "Thus",
    "First", "Second", "Next", "Finally", "Such", "Some", "Each", "Every",
}


def extract_entities(text: str, cap: int = 8) -> list[str]:
    """Heuristic NER: capitalised multi-word phrases + acronyms + quotes."""
    found: list[str] = []
    for m in _CAP_SEQ_RE.finditer(text):
        phrase = m.group(1).strip()
        if phrase.split(" ")[0] in _ENTITY_STOP:
            continue
        if phrase not in found:
            found.append(phrase)
    for m in _ACRONYM_RE.finditer(text):
        tok = m.group(0)
        if tok not in found and tok not in {"II", "III", "IV", "OK", "ID"}:
            found.append(tok)
    for m in _LINK_RE.finditer(text):
        label = m.group(1).strip()
        if label and label[:1].isupper() and label not in found:
            found.append(label)
    return found[:cap]


def extract_refs(text: str, cap: int = 6) -> list[str]:
    """Cross-reference candidates: 'see Section X', markdown links, 'cf. Y'."""
    refs: list[str] = []
    for m in _SEE_RE.finditer(text):
        ref = m.group(1).strip(" .,;:")
        if ref and ref.lower() not in {"below", "above", "next", "figure"}:
            refs.append(ref)
    for m in _LINK_RE.finditer(text):
        target = m.group(2).strip()
        if target.startswith("#"):
            refs.append(target[1:].replace("-", " "))
    out: list[str] = []
    for r in refs:
        if r and r not in out:
            out.append(r)
    return out[:cap]


def walk_sections(section: Section, path: list[str]):
    """Yield (path_titles, section) for every section in the tree."""
    here = [*path, section.title] if section.title else path
    yield here, section
    for child in section.children:
        yield from walk_sections(child, here)


def _window_sentences(text: str, max_chars: int, overlap: int) -> list[str]:
    sents = sentence_split(text)
    windows: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if cur and size + len(s) > max_chars:
            windows.append(" ".join(cur))
            tail = cur[-overlap:] if overlap else []
            cur = [*tail, s]
            size = sum(len(x) for x in cur) + len(cur)
        else:
            cur.append(s)
            size += len(s)
    if cur:
        windows.append(" ".join(cur))
    return windows or ([text[:max_chars]] if text else [])


def _table_block_text(block: Block) -> str:
    if not block.rows:
        return block.text
    lines = [" | ".join(block.rows[0])]
    for r in block.rows[1:]:
        lines.append(" | ".join(r))
    return "\n".join(lines)


def chunk_document(
    doc: ParsedDocument,
    doc_id: str,
    max_chars: int = 1000,
    overlap_sents: int = 1,
) -> list[Chunk]:
    """Turn one parsed document into addressable chunks while keeping the
    hierarchical path of every chunk — the topology the connectome wires."""
    chunks: list[Chunk] = []
    order = 0

    def add(text: str, kind: str, path: list[str]) -> None:
        nonlocal order
        text = text.strip()
        if not text:
            return
        chunks.append(
            Chunk(
                id=f"chunk:{doc_id}:{order:04d}",
                doc_id=doc_id,
                doc_title=doc.title,
                section_path=list(path),
                section_title=path[-1] if path else doc.title,
                text=text,
                kind=kind,
                token_estimate=int(len(text.split()) * 1.3) + 1,
                entities=extract_entities(text),
                refs=extract_refs(text),
                order=order,
                content_hash=hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
            )
        )
        order += 1

    for path, section in walk_sections(doc.root, []):
        joined = [t for t in path if t]
        for block in section.blocks:
            if block.kind == "table":
                text = _table_block_text(block)
                if len(text) > max_chars * 2:
                    for part in _window_sentences(text, max_chars, 0):
                        add(part, "table", joined)
                else:
                    add(text, "table", joined)
            elif block.kind in ("list", "code"):
                text = block.text
                if len(text) > max_chars * 1.5:
                    for part in _window_sentences(text, max_chars, 0):
                        add(part, block.kind, joined)
                else:
                    add(text, block.kind, joined)
            else:
                text = block.text
                if len(text) <= max_chars:
                    add(text, "text", joined)
                else:
                    for part in _window_sentences(text, max_chars, overlap_sents):
                        add(part, "text", joined)
    return chunks
