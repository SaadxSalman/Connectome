"""Shared lexical utilities: tokenisation, sentence splitting, idf, slugify.

Used by the hashing embedder, the local micro-critic circuits and the
Reflex-Arc synthesiser so that every layer agrees on what a "word" is.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log

STOPWORDS = frozenset(
    """a an the and or but if then else of for to in on at by with from as is
    are was were be been being it its this that these those there here i you
    he she they we us our your their his her hers not no nor so too very can
    will just should now do does did doing have has had having than s t don t
    re ve ll m d o y ain aren couldn didn doesn hadn hasn haven isn ma mightn
    mustn needn shan shouldn wasn weren won wouldn also into about over under
    between during without within upon whether while because however therefore
    thus hence among across per via vs etc eg ie both each few more most other
    some such only own same when where why how all any both what which who whom
    let get got go goes going make made use used using one two three first
    second new like well may might must shall would could""".split()
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:,\d{3})*(?:\.\d+)?")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords and 1-char tokens removed."""
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(tok)
    return out


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def sentence_split(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def df_counts(token_lists: list[list[str]]) -> Counter:
    df: Counter = Counter()
    for toks in token_lists:
        df.update(set(toks))
    return df


def idf(df: Counter, n_docs: int, token: str) -> float:
    """Smoothed idf; rare terms dominate, unseen terms default to log(n+1)."""
    return log(1 + n_docs / (df.get(token, 0) + 0.5))


def lexical_overlap(query_tokens: set[str], text: str) -> float:
    """Sqrt-normalised token overlap between the query and a text blob (0..1)."""
    if not query_tokens:
        return 0.0
    inter = query_tokens & token_set(text)
    if not inter:
        return 0.0
    return min(1.0, len(inter) / (len(query_tokens) ** 0.5 + 1))


def slugify(text: str, max_len: int = 60) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "x"


def truncate(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
