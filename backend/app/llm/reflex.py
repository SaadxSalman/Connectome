"""The Reflex-Arc: a deterministic, fully offline fallback nervous system.

When no LLM endpoint is reachable, these functions keep SynapseCraft
completely functional:

• ``reflex_synthesize`` — idf-weighted extractive motor output with citations.
• ``reflex_expand``     — pseudo-relevance feedback for query expansion.

Both are pure functions: same inputs, same spikes, every time.
"""

from __future__ import annotations

from collections import Counter
from math import log

from ..textutils import sentence_split, token_set, tokenize


def reflex_synthesize(
    question: str,
    evidences: list[dict],
    max_sentences: int = 5,
) -> tuple[str, list[str]]:
    """Extractive motor output.

    Picks the most query-salient sentences from the verified evidence pool
    (spreading one sentence per source first, then filling remaining slots),
    suppresses near-duplicates (lateral inhibition at the sentence level),
    preserves narrative order and appends ``[S#]`` citation tags.

    Returns ``(answer, used_tags)``.
    """
    q_tokens = token_set(question)
    sentences: list[dict] = []
    for ev in evidences:
        for idx, sent in enumerate(sentence_split(ev.get("text", ""))):
            if len(sent) < 30 or len(sent) > 400:
                continue
            toks = tokenize(sent)
            if len(toks) < 3:
                continue
            sentences.append({
                "tag": ev["tag"],
                "text": sent,
                "tokens": toks,
                "order": ev.get("order", 0),
                "score": float(ev.get("score", 0.0)),
            })
    if not sentences:
        return "", []

    df: Counter = Counter()
    for s in sentences:
        df.update(set(s["tokens"]))
    n = len(sentences)
    for s in sentences:
        overlap = q_tokens.intersection(s["tokens"])
        idf_sum = sum(log(1 + n / (df[t] + 0.5)) for t in overlap)
        s["relevance"] = (
            idf_sum / (len(s["tokens"]) ** 0.5 + 1)
            + 0.35 * s["score"]
            + (0.2 if overlap else 0.0)
        )

    ranked = sorted(sentences, key=lambda s: s["relevance"], reverse=True)

    def similar(a: dict, b: dict) -> float:
        ta, tb = set(a["tokens"]), set(b["tokens"])
        return len(ta & tb) / max(1, min(len(ta), len(tb)))

    chosen: list[dict] = []
    # pass 1 — best sentence per evidence source (breadth across sources)
    for tag in dict.fromkeys(s["tag"] for s in ranked):
        best = next(
            (s for s in ranked if s["tag"] == tag and all(similar(s, c) < 0.7 for c in chosen)),
            None,
        )
        if best:
            chosen.append(best)
        if len(chosen) >= max_sentences:
            break
    # pass 2 — fill remaining slots by global salience
    for s in ranked:
        if len(chosen) >= max_sentences:
            break
        if s in chosen:
            continue
        if all(similar(s, c) < 0.7 for c in chosen):
            chosen.append(s)

    chosen.sort(key=lambda s: (s["order"], s["text"]))
    answer = " ".join(f"{s['text']} [{s['tag']}]" for s in chosen)
    tags: list[str] = []
    for s in chosen:
        if s["tag"] not in tags:
            tags.append(s["tag"])
    return answer, tags


def reflex_expand(question: str, seed_texts: list[str], top_n: int = 5) -> list[str]:
    """Pseudo-relevance feedback (Rocchio-style).

    Frequent terms in the top-scoring seed chunks — that are *not* already in
    the query — become 'expected-reward' signals appended to the query,
    sharpening the next propagation pass.
    """
    q_tokens = token_set(question)
    freq: Counter = Counter()
    for text in seed_texts:
        seen: set[str] = set()
        for tok in tokenize(text):
            if tok in q_tokens or tok in seen:
                continue
            seen.add(tok)
            freq[tok] += 1
    if not freq:
        return []
    recurring = [t for t, c in freq.most_common() if c >= 2]
    return recurring[:top_n] or [t for t, _ in freq.most_common(top_n)]
