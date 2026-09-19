"""Neurodynamics: spreading activation, membrane potentials, lateral inhibition.

This module implements the biophysical metaphors that give SynapseCraft its
noise immunity:

• Spreading activation — graded depolarisation flows along semantic synapses
  (RELATES_TO / MENTIONS / REFERENCES), recovering related context the plain
  vector search missed.

• Membrane potential — each candidate chunk accumulates excitatory
  postsynaptic potentials (semantic + lexical + entity congruence) and
  inhibitory postsynaptic potentials (lateral inhibition + numeric
  contradictions).  It reaches the generation layer only if  V ≥ θ.

• Lateral inhibition — an already-firing chunk suppresses overlapping
  competitors (λ × cosine similarity), the mechanism that keeps context
  small, diverse and cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# synaptic weights (excitatory)
W_SEMANTIC = 0.55   # vector similarity to the propagated query
W_LEXICAL = 0.25    # idf-normalised token overlap
W_ENTITY = 0.20     # named-concept congruence
# inhibitory weights
W_CONFLICT = 0.30   # numeric-contradiction micro-critic
MIN_GAIN = 0.02     # sub-threshold signals die out silently

_NUM_RE = re.compile(r"([A-Za-z][A-Za-z0-9-]{2,})[^.\n]{0,60}?(\d[\d,]*(?:\.\d+)?)")


@dataclass
class GateVerdict:
    """Result of the local-circuit vote on a single candidate chunk."""
    potential: float
    fired: bool
    components: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.55
    reason: str = ""


def membrane_potential(
    sim: float,
    lexical: float,
    entity_score: float,
    lateral: float,
    conflict: float,
    lateral_lambda: float,
    threshold: float,
) -> GateVerdict:
    excit = W_SEMANTIC * max(0.0, sim) + W_LEXICAL * max(0.0, lexical) + W_ENTITY * max(0.0, entity_score)
    inhib = lateral_lambda * max(0.0, lateral) + W_CONFLICT * max(0.0, conflict)
    v = excit - inhib
    components = {
        "semantic": round(W_SEMANTIC * max(0.0, sim), 3),
        "lexical": round(W_LEXICAL * max(0.0, lexical), 3),
        "entity": round(W_ENTITY * max(0.0, entity_score), 3),
        "lateral_inhibition": -round(lateral_lambda * max(0.0, lateral), 3),
        "conflict": -round(W_CONFLICT * max(0.0, conflict), 3),
    }
    fired = v >= threshold
    reason = ""
    if not fired:
        if conflict > 0 and W_CONFLICT * conflict >= max(
            lateral_lambda * lateral, 0.0
        ) and W_CONFLICT * conflict >= v + threshold:
            reason = "suppressed: numeric contradiction with fired evidence"
        elif lateral > 0:
            reason = "suppressed: lateral inhibition (redundant context)"
        else:
            reason = "suppressed: sub-threshold potential"
    return GateVerdict(potential=round(v, 4), fired=fired, components=components,
                       threshold=threshold, reason=reason)


def spreading_activation(
    graph,
    seeds: dict[str, float],
    decay: float,
    max_hops: int = 2,
    min_gain: float = MIN_GAIN,
    skip_kinds: set[str] | None = None,
) -> tuple[dict[str, float], list[tuple[str, str, float, str]]]:
    """Propagate graded depolarisation across the connectome.

    Classic Collins–Loftus spreading activation, run over the structural
    graph: energy leaves every seed chunk along MENTIONS / RELATES_TO /
    REFERENCES synapses, decaying by ``decay`` per hop. Structural synapses
    (CONTAINS / BELONGS / agent wiring) do not carry retrieval signal.

    Returns ``(potentials, events)`` where events are
    ``(source, target, gain, kind)`` tuples for the cockpit visualiser.
    """
    skip = skip_kinds or {"BELONGS", "CONTAINS", "SENSES", "GATES", "DRIVES", "REFLECTS"}
    potentials: dict[str, float] = {}
    events: list[tuple[str, str, float, str]] = []
    frontier = dict(seeds)
    for hop in range(max_hops):
        nxt: dict[str, float] = {}
        for node, energy in frontier.items():
            for nb, edata in graph.neighbors(node):
                kind = edata.get("kind", "")
                if kind in skip:
                    continue
                gain = energy * float(edata.get("weight", 0.5)) * (decay ** hop)
                if gain < min_gain:
                    continue
                nxt[nb] = nxt.get(nb, 0.0) + gain
        for target, gain in nxt.items():
            potentials[target] = potentials.get(target, 0.0) + gain
            events.append((node, target, gain, "propagation"))
        frontier = nxt
    return potentials, events


def numeric_conflicts(text: str, fired_texts: list[str]) -> float:
    """Consistency micro-critic: does this chunk assert a different value for
    a key another chunk already claimed? Returns 1.0 on the first conflict."""
    if not fired_texts:
        return 0.0

    def claims_of(t: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in _NUM_RE.finditer(t):
            key = m.group(1).lower()
            try:
                out[key] = float(m.group(2).replace(",", ""))
            except ValueError:
                continue
        return out

    mine = claims_of(text)
    if not mine:
        return 0.0
    for other in fired_texts[-4:]:
        for key, v in claims_of(other).items():
            if key in mine and mine[key] != 0 and v != 0:
                if abs(mine[key] - v) / max(abs(mine[key]), abs(v)) > 0.15:
                    return 1.0
    return 0.0
