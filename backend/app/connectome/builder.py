"""Wires parsed documents into the structural connectome graph."""

from __future__ import annotations

import numpy as np

from ..ingestion.parser import ParsedDocument
from ..textutils import slugify, truncate
from ..types import Chunk


def count_sections(doc: ParsedDocument) -> int:
    n = 0
    stack = [doc.root]
    while stack:
        sec = stack.pop()
        stack.extend(sec.children)
        n += len(sec.children)
    return n


def wire_documents(
    graph,
    chunks: list[Chunk],
    vectors: np.ndarray,
    relate_threshold: float,
) -> None:
    """Attach a batch of freshly chunked documents to the connectome.

    Builds the node-link matrix: doc → section → chunk containment chains,
    chunk → entity MENTIONS synapses, chunk ↔ chunk semantic RELATES_TO
    synapses and chunk → target REFERENCES synapses for cross-references.
    """
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c.doc_id, []).append(i)

    if vectors is not None:
        vectors = np.asarray(vectors, dtype=np.float32)

    section_title_index: dict[str, list[str]] = {}
    doc_title_index: dict[str, str] = {}

    for doc_id, idxs in by_doc.items():
        first = chunks[idxs[0]]
        doc_node = f"doc:{doc_id}"
        graph.add_node(doc_node, "doc", first.doc_title, doc_id=doc_id)
        doc_title_index[first.doc_title.strip().lower()] = doc_node
        seen_sections: dict[tuple, str] = {}

        for i in idxs:
            c = chunks[i]
            # nested section chain: doc › chapter › section › chunk
            parent = doc_node
            for depth in range(1, len(c.section_path) + 1):
                path_key = tuple(c.section_path[:depth])
                sec_node = seen_sections.get(path_key)
                if sec_node is None:
                    sec_node = f"sec:{doc_id}:{depth}:{slugify(' '.join(path_key), 50)}"
                    graph.add_node(
                        sec_node, "section", path_key[-1], doc_id=doc_id,
                        section=" › ".join(path_key),
                    )
                    seen_sections[path_key] = sec_node
                graph.add_edge(parent, sec_node, "CONTAINS", 1.0)
                parent = sec_node

            graph.add_node(
                c.id, "chunk", truncate(c.text, 56), doc_id=doc_id,
                doc_title=c.doc_title, kind=c.kind,
                section=" › ".join(c.section_path), order=c.order,
            )
            graph.add_edge(parent, c.id, "CONTAINS", 1.0)

            for ent in c.entities:
                ent_node = f"ent:{slugify(ent, 50)}"
                graph.add_node(ent_node, "entity", ent)
                graph.add_edge(c.id, ent_node, "MENTIONS", 1.0)

            key = c.section_title.strip().lower()
            if key:
                section_title_index.setdefault(key, []).append(c.id)

    # semantic RELATES_TO synapses between batch chunks (cosine neighbours)
    if len(chunks) > 1 and vectors is not None and len(vectors) == len(chunks):
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        for i in range(len(chunks)):
            order = np.argsort(-sims[i])[:2]
            for j in order:
                j = int(j)
                s = float(sims[i][j])
                if s >= relate_threshold:
                    a, b = sorted([chunks[i].id, chunks[j].id])
                    graph.add_edge(a, b, "RELATES_TO", s)

    # REFERENCES: resolve "see §X" style cross-references to real targets
    for c in chunks:
        for ref in c.refs:
            key = ref.strip().lower()
            if not key:
                continue
            targets = section_title_index.get(key, [])
            if not targets:
                for title, ids in section_title_index.items():
                    if key in title:
                        targets = ids
                        break
            if not targets:
                for title, doc_node in doc_title_index.items():
                    if key in title:
                        targets = [doc_node]
                        break
            for t in targets[:2]:
                if t != c.id:
                    graph.add_edge(c.id, t, "REFERENCES", 0.6)
