"""Optional Neo4j mirror of the connectome graph.

NetworkX remains the query engine; Neo4j (Desktop / Aura) receives a
persistent, explorable write-only twin of every node and synapse. All writes
are best-effort: mirror failures never propagate into the nervous system.
"""

from __future__ import annotations

from ..config import Settings


def _safe(v):
    return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)


class Neo4jMirror:
    def __init__(self, settings: Settings) -> None:
        self.uri = settings.neo4j_uri
        self.auth = (settings.neo4j_username, settings.neo4j_password)
        self._driver = None
        self._warned = False

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # optional dependency

            self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self._driver.verify_connectivity()
        except Exception as exc:
            self._driver = None
            if not self._warned:
                print(f"[synapsecraft] Neo4j mirror unavailable: {exc}")
                self._warned = True
            raise
        return self._driver

    def _write(self, cypher: str, **params) -> None:
        try:
            with self._get_driver().session() as session:
                session.run(cypher, **params)
        except Exception:
            pass

    def upsert_node(self, nid: str, data: dict) -> None:
        self._write(
            "MERGE (n:Synapse {id:$id}) SET n += $props",
            id=nid,
            props={k: _safe(v) for k, v in data.items()},
        )

    def upsert_edge(self, a: str, b: str, kind: str, weight: float) -> None:
        self._write(
            "MATCH (a:Synapse {id:$a}), (b:Synapse {id:$b}) "
            "MERGE (a)-[r:SYNAPSE]->(b) SET r.kind=$kind, r.weight=$weight",
            a=a,
            b=b,
            kind=kind,
            weight=weight,
        )

    def remove_document(self, doc_id: str) -> None:
        self._write("MATCH (n:Synapse {doc_id:$doc}) DETACH DELETE n", doc=doc_id)
        self._write("MATCH (n:Synapse {id:$doc}) DETACH DELETE n", doc=f"doc:{doc_id}")

    def clear(self) -> None:
        self._write("MATCH (n:Synapse) DETACH DELETE n")

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
