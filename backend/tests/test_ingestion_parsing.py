"""Structure-aware parser + chunker tests."""

from app.ingestion.chunker import chunk_document, extract_entities, extract_refs
from app.ingestion.parser import parse_markdown, parse_source

MD = """# Guide

## Motion
T4 neurons detect motion. They project to the Lobula Plate.
See section Odor for the counterpart pathway.

| Neuropil | Neurons |
| --- | --- |
| Medulla | 60000 |

## Odor
Kenyon cells sparsen odor codes.

```python
print("hello")
```
"""


def test_markdown_structure_preserved():
    doc = parse_markdown(MD, "guide")
    assert doc.title == "Guide"
    guide = doc.root.children[0]
    assert guide.title == "Guide"
    names = [c.title for c in guide.children]
    assert "Motion" in names and "Odor" in names
    motion = next(c for c in guide.children if c.title == "Motion")
    kinds = [b.kind for b in motion.blocks]
    assert "text" in kinds and "table" in kinds
    odor = next(c for c in guide.children if c.title == "Odor")
    assert any(b.kind == "code" for b in odor.blocks)


def test_chunking_keeps_section_paths():
    doc = parse_markdown(MD, "guide")
    chunks = chunk_document(doc, "d1", max_chars=400, overlap_sents=1)
    assert chunks
    motion = [c for c in chunks if "Motion" in c.section_path]
    assert motion, "chunk lost its hierarchical path"
    assert motion[0].kind == "text"
    tables = [c for c in chunks if c.kind == "table"]
    assert tables and "60000" in tables[0].text
    assert all(c.doc_id == "d1" for c in chunks)
    assert len({c.content_hash for c in chunks}) == len(chunks)


def test_entity_and_ref_extraction():
    text = ("T4 neurons and the Lobula Plate are involved. "
            "The mushroom body matters. See section Odor for details. "
            "Read [Connectome Stats](#connectome-stats) too.")
    ents = extract_entities(text)
    assert any("Lobula Plate" in e for e in ents)
    refs = extract_refs(text)
    assert any("odor" in r.lower() for r in refs)


def test_parse_source_dispatch():
    assert parse_source("x.md", MD.encode()).kind == "markdown"
    assert parse_source("y.csv", b"a,b\n1,2\n").kind == "csv"
    doc = parse_source("z.json", b'{"k": [1, 2]}')
    assert doc.kind == "json"
    assert parse_source("plain", b"Just some plain text.\n").kind == "text"
