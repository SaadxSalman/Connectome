"""Neurodynamics tests: membrane potentials, spreading activation, conflicts."""

from app.connectome.activation import (
    membrane_potential, numeric_conflicts, spreading_activation,
)


class FakeGraph:
    def __init__(self, adjacency: dict[str, list[tuple[str, float]]]):
        self.adj = adjacency

    def neighbors(self, nid):
        return [(nb, {"kind": "RELATES_TO", "weight": w}) for nb, w in self.adj.get(nid, [])]


def test_membrane_potential_fires_on_strong_signal():
    v = membrane_potential(sim=0.9, lexical=0.6, entity_score=0.5,
                           lateral=0.0, conflict=0.0,
                           lateral_lambda=0.35, threshold=0.55)
    assert v.fired
    assert v.potential >= 0.55


def test_lateral_inhibition_suppresses_redundancy():
    base = dict(lexical=0.6, entity_score=0.5, conflict=0.0,
                lateral_lambda=0.35, threshold=0.55)
    fresh = membrane_potential(sim=0.8, lateral=0.0, **base)
    redundant = membrane_potential(sim=0.8, lateral=0.8, **base)
    assert fresh.fired
    assert not redundant.fired
    assert "lateral inhibition" in redundant.reason


def test_numeric_conflict_flagged():
    fired = ["The patch clamp recorded 120 spikes per second."]
    assert numeric_conflicts("The patch clamp recorded 80 spikes per second.", fired) == 1.0
    assert numeric_conflicts("The patch clamp recorded 120 spikes per second.", fired) == 0.0
    assert numeric_conflicts("No numbers here.", fired) == 0.0


def test_spreading_activation_decays_per_hop():
    g = FakeGraph({"a": [("b", 0.9)], "b": [("c", 0.9)]})
    potentials, events = spreading_activation(g, {"a": 1.0}, decay=0.5, max_hops=2)
    assert potentials["b"] > potentials["c"] > 0
    assert any(e[0] == "a" and e[1] == "b" for e in events)


def test_spreading_activation_dies_out_below_min_gain():
    g = FakeGraph({"a": [("b", 0.9)], "b": [("c", 0.9)]})
    potentials, _ = spreading_activation(g, {"a": 0.02}, decay=0.5, max_hops=3)
    assert "b" not in potentials
