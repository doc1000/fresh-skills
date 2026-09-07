"""Tool wrappers around compiled subgraphs, plus store thread safety."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from pathlib import Path

from playbook.topics import BertopicConfig

PLACEHOLDER_DATA_DIR = Path(__file__).resolve().parents[1] / "scratch_data"

DEMO_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=2,
    min_cluster_size=2,
    min_samples=1,
    min_topic_n=2,
    representative_n=3,
    n_neighbors=3,
    n_components=2,
)


@pytest.fixture
def runtime(tmp_path):
    from playbook import configure_runtime

    return configure_runtime(
        data_dir=PLACEHOLDER_DATA_DIR,
        store_path=tmp_path / "run_store.sqlite",
        method="jaccard",
        topic_config=DEMO_TOPIC_CONFIG,
    )


def test_get_staging_miss_returns_none(runtime):
    from playbook import runtime as rt

    assert rt.store.get_staging("missing-run", "missing-stage") is None


def test_store_set_staging_from_other_threads(runtime):
    from playbook import runtime as rt

    def write(i: int) -> str:
        stage = f"thread-{i}"
        rt.store.set_staging("thread-run", stage, {"i": i})
        return rt.store.get_staging("thread-run", stage)["i"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write, i) for i in range(16)]
        values = [future.result() for future in as_completed(futures)]
    assert sorted(values) == list(range(16))


def test_cohort_and_classify_tools_share_run(runtime):
    from playbook import classify_intent, classify_subflow, cohort
    from playbook import runtime as rt

    out = cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "cohort_query": {},
        }
    )
    assert out["run_id"]
    assert out["cohort_summary"]["n"] > 0
    assert rt.current_run_id == out["run_id"]

    intent = classify_intent.invoke({})
    assert intent["intent_summary"]["processed"] >= 1
    subflow = classify_subflow.invoke({})
    assert "subflow_summary" in subflow


def test_tools_survive_worker_thread_invoke(runtime):
    from playbook import classify_intent, cohort

    def run_tools() -> dict:
        cohort.invoke(
            {
                "start_date": "2026-09-01",
                "end_date": "2026-09-07",
                "method": "jaccard",
            }
        )
        return classify_intent.invoke({})

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(run_tools).result()
    assert result["intent_summary"]["processed"] >= 1


def test_recommend_then_persist_recc(runtime, stub_topic_fit):
    from playbook import (
        classify_intent,
        classify_subflow,
        cohort,
        discover_intent,
        discover_subflow,
        persist_recc,
        recommend_pathway,
    )
    from playbook import runtime as rt

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "week-tools",
        }
    )
    classify_intent.invoke({})
    classify_subflow.invoke({})
    discovered = discover_intent.invoke({})
    discover_subflow.invoke({})
    intent_changes = discovered["discovery_summary"]["intents"]["changes"]
    assert intent_changes
    row = intent_changes[0]
    assert "examples" in row
    assert "metrics" in row
    assert "review_note" in row
    assert "supporting_task_ids" in row
    assert len(row["supporting_task_ids"]) <= 10

    draft = recommend_pathway.invoke({"target_subflow": "reset_2fa"})
    assert draft["target_subflow"] == "reset_2fa"
    assert rt.store.list_recommendations("week-tools") == []
    change = draft["recommendation_summary"]["changes"][0]
    assert "examples" in change
    assert "metrics" in change
    assert "supporting_task_ids" in change
    assert "review_note" not in change
    assert len(change["supporting_task_ids"]) <= 10

    saved = persist_recc.invoke({"target_subflow": "reset_2fa"})
    assert saved["rec_id"]
    recs = rt.store.list_recommendations("week-tools")
    assert recs
    assert recs[0]["subflow_id"] == "reset_2fa"
    assert "changes" in saved["recommendation_summary"]
    saved_change = saved["recommendation_summary"]["changes"][0]
    assert "examples" in saved_change
    assert "supporting_task_ids" in saved_change


def test_persist_recc_without_draft_returns_error(runtime):
    from playbook import cohort, persist_recc
    from playbook import runtime as rt

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "no-draft",
        }
    )
    out = persist_recc.invoke({"target_subflow": "reset_2fa"})
    assert out["ok"] is False
    assert "recommend_pathway" in out["error"]
    assert out["target_subflow"] == "reset_2fa"
    assert rt.store.list_recommendations("no-draft") == []


def test_retrieve_guidance_tool_catalog_and_query(runtime):
    from playbook.agent import retrieve_guidance

    catalog = retrieve_guidance.invoke({})
    assert catalog["kb_version"]
    assert catalog["intents"]
    assert "id" in catalog["intents"][0]
    assert "subflows" in catalog["intents"][0]
    assert "has_pathway" in catalog["intents"][0]["subflows"][0]
    assert "guidance" not in catalog["intents"][0]

    with_guide = retrieve_guidance.invoke({"include_guidance": True})
    assert "guidance" in with_guide["intents"][0]
    assert "hidden_flow" not in json.dumps(with_guide)

    ranked = retrieve_guidance.invoke({"query": "password reset two-factor"})
    assert ranked["hits"]
    assert "score" in ranked["hits"][0]


def test_cohort_peek_does_not_replace_working_set(runtime):
    from playbook import classify_intent, cohort
    from playbook import runtime as rt

    persisted = cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "peek-run",
        }
    )
    classify_intent.invoke({})
    working = list(rt.store.cohort_ids("peek-run"))
    assert working

    peek = cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "sample_n": 2,
            "cohort_query": {"intent": next(iter(rt.store.latest_intents("peek-run").values()))["intent_id"]},
        }
    )
    assert peek["peek"] is True
    assert "run_id" not in peek
    assert rt.current_run_id == persisted["run_id"]
    assert rt.store.cohort_ids("peek-run") == working
    assert peek["samples"]
    assert "hidden_flow" not in peek["samples"][0]
    assert "opening" in peek["samples"][0]

    task_id = peek["samples"][0]["task_id"]
    detail = cohort.invoke({"task_ids": [task_id]})
    assert "run_id" not in detail
    assert detail["tasks"][0]["task_id"] == task_id
    assert "turns" in detail["tasks"][0]
    assert "hidden_flow" not in detail["tasks"][0]
    assert "hidden_subflow" not in detail["tasks"][0]
    assert rt.store.cohort_ids("peek-run") == working


def test_cohort_filter_persist_and_richer_summaries(runtime, stub_topic_fit):
    from playbook import classify_intent, cohort, discover_intent
    from playbook import runtime as rt

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "filter-run",
        }
    )
    intent = classify_intent.invoke({})
    assert "unknown_ids" in intent["intent_summary"]
    labeled = [
        (tid, row["intent_id"])
        for tid, row in rt.store.latest_intents("filter-run").items()
        if row["intent_id"] != "unknown"
    ]
    assert labeled
    target = labeled[0][1]
    filtered = cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "filter-run",
            "cohort_query": {"intent": target},
        }
    )
    assert filtered["run_id"] == "filter-run"
    assert filtered["cohort_summary"]["filters"]["intent"] == target
    assert all(
        rt.store.latest_intents("filter-run")[tid]["intent_id"] == target
        for tid in rt.store.cohort_ids("filter-run")
    )

    discovered = discover_intent.invoke({})
    changes = discovered["discovery_summary"]["intents"]["changes"]
    assert "changes" in discovered["discovery_summary"]["intents"]
    if changes:
        row = changes[0]
        assert "examples" in row
        assert "metrics" in row
        assert "review_note" in row
        assert "supporting_task_ids" in row
        assert len(row["supporting_task_ids"]) <= 10


def test_empty_path_subflow_is_classifiable(runtime, stub_topic_fit):
    from playbook import classify_intent, classify_subflow, cohort
    from playbook.agent import retrieve_guidance
    from playbook import runtime as rt

    rt.playbook.add_subflow(
        "account_access",
        "account_locked",
        description="account locked lockout",
    )
    catalog = retrieve_guidance.invoke({})
    by_id = {
        row["id"]: row
        for intent in catalog["intents"]
        for row in intent["subflows"]
    }
    assert by_id["account_locked"]["has_pathway"] is False
    assert by_id["recover_username"]["has_pathway"] is True

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "bertopic",
            "run_id": "empty-path",
        }
    )
    classify_intent.invoke({})
    out = classify_subflow.invoke({})
    assert out["subflow_summary"]["processed"] >= 1
    methods = {row["method"] for row in rt.store.latest_subflows("empty-path").values()}
    assert "bertopic_prototype_v1" in methods


def test_discover_subflow_persists_emerging_without_pathway(runtime, stub_topic_fit):
    from playbook import classify_intent, classify_subflow, cohort, discover_subflow
    from playbook import runtime as rt

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "bertopic",
            "run_id": "emerging-run",
        }
    )
    classify_intent.invoke({})
    classify_subflow.invoke({})
    discovered = discover_subflow.invoke({})
    summary = discovered["discovery_summary"]["subflows"]
    assert summary["candidate_count"] >= 0
    if summary["changes"]:
        row = summary["changes"][0]
        assert "examples" in row
        assert "metrics" in row
        assert "review_note" in row
        assert "supporting_task_ids" in row
    emerging = [
        row
        for row in rt.store.list_proposals("emerging-run")
        if row["proposal_type"] == "emerging"
    ]
    accepted = [row for row in emerging if row["review_decision"] == "accept"]
    if accepted:
        candidate = accepted[0]["candidate"]
        parent = accepted[0]["parent_intent"]
        assert candidate in rt.playbook.subflows_for(parent)
        assert not rt.playbook.has_pathway(parent, candidate)
