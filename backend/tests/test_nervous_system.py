"""End-to-end nervous-system tests (fully offline Reflex-Arc mode)."""

import asyncio

from tests.conftest import SAMPLE_MD


def test_ingest_then_query_e2e(harness):
    async def flow():
        ingest = await harness.ingest_md()
        counts = harness.graph.counts()
        result = await harness.query(
            "How do T4 neurons detect motion in the optic lobe?"
        )
        return ingest, counts, result

    ingest, counts, result = asyncio.run(flow())
    assert ingest["chunks"] > 0
    assert counts["chunks"] == ingest["chunks"]
    assert counts["documents"] == 1

    assert result["error"] == ""
    assert result["fired"], "no chunk crossed the firing threshold"
    assert len(result["fired"]) <= harness.settings.top_k
    assert result["answer"], "motor output was empty"
    assert result["citations"], "citations were not assembled"
    assert result["grounded"] is True
    assert result["critic_note"]
    assert result["latency_ms"] >= 0
    assert result["suppressed"] >= 0
    assert result["potentials"], "spreading-activation potentials missing"
    # every fired chunk records a membrane-potential trace
    assert result["verdicts"] and result["verdicts"][0]["components"]


def test_answer_cites_only_real_sources(harness):
    async def flow():
        await harness.ingest_md()
        return await harness.query("What do Kenyon cells do for odor coding?")

    result = asyncio.run(flow())
    assert result["tags_used"], "reflex answer carried no [S#] tags"
    max_tag = len(result["fired"])
    for tag in result["tags_used"]:
        assert 1 <= int(tag[1:]) <= max_tag
    cited_ids = {c["tag"] for c in result["citations"]}
    for tag in result["tags_used"]:
        assert tag in cited_ids


def test_unrelated_query_yields_no_evidence(harness):
    async def flow():
        await harness.ingest_md()
        return await harness.query("quantum chess openings in zero gravity")

    result = asyncio.run(flow())
    assert result["answer"] == ""
    assert result["error"], "expected a no-evidence error"
    assert "no signal crossed the firing threshold" in result["error"]


def test_events_streamed_for_the_cockpit(harness):
    async def flow():
        await harness.ingest_md()
        sub_id, q = harness.bus.subscribe()
        await harness.query("optic lobe motion detection")
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        harness.bus.unsubscribe(sub_id)
        return events

    events = asyncio.run(flow())
    kinds = [e["type"] for e in events]
    assert "run_start" in kinds
    assert "run_end" in kinds
    assert "gate" in kinds
    # polarity colouring is always one of the three legal values
    assert all(e.get("polarity") in ("excitatory", "inhibitory", "signal")
               for e in events)
    run_end = next(e for e in events if e["type"] == "run_end")
    assert "message" in run_end


def test_duplicate_ingest_is_noop(harness):
    async def flow():
        first = await harness.ingest_md()
        second = await harness.ingest_md()
        return first, second

    first, second = asyncio.run(flow())
    assert first["chunks"] == second["chunks"]
    assert any("duplicate" in w for w in second["warnings"])
