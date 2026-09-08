"""Cleanup-boundary tests for the extracted LangGraph agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from playbook.topics import BertopicConfig

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_DATA_DIR = Path(__file__).resolve().parent / "fixtures"
DEMO_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=2,
    min_cluster_size=2,
    min_samples=1,
    min_topic_n=2,
    representative_n=3,
    n_neighbors=3,
    n_components=2,
)


def test_nodes_receive_and_return_typed_state(runtime):
    from playbook import MetaAgentState, cohort_query, empty_state

    state = empty_state(
        run_id="week-state-check",
        start="2026-09-01",
        end="2026-09-07",
        method="jaccard",
    )
    assert set(state) >= {
        "run_id",
        "start",
        "end",
        "method",
        "cohort_query",
        "kb_version",
        "current_stage",
        "intent_summary",
        "pending_proposal_ids",
        "errors",
    }
    out = cohort_query(state)
    assert out["run_id"] == "week-state-check"
    assert out["current_stage"] == "select_time_window"
    assert out["start"] == "2026-09-01"
    assert out["end"] == "2026-09-07"


def test_playbook_loader_and_retriever_are_independent():
    from playbook import load_playbook, retrieve_guidance

    playbook = load_playbook()
    assert playbook.intent_ids() == ["account_access", "order_issue"]
    assert playbook.subflows_for("account_access") == [
        "recover_username",
        "recover_password",
        "reset_2fa",
    ]
    assert playbook.subflows_for("order_issue") == [
        "status_mystery_fee",
        "status_delivery_time",
        "manage_upgrade",
        "manage_cancel",
    ]
    assert "recover_username" in playbook.kb
    assert "reset_2fa" in playbook.kb

    ranked = retrieve_guidance(
        "I forgot my username and cannot log into my account.",
        playbook,
    )
    assert ranked
    assert ranked[0]["intent_id"] == "account_access"
    assert "score" in ranked[0]
    assert "doc" in ranked[0]


def test_langchain_tools_are_callable(runtime):
    from playbook import runtime as rt
    from playbook.tools import (
        draft_guideline_stub,
        draft_kb_stub,
        score_intent_similarity,
        score_subflow_similarity,
    )

    task_id = rt.store.fetchall("SELECT task_id FROM tasks ORDER BY task_id LIMIT 1")[0]["task_id"]
    intent = score_intent_similarity.invoke(
        {"conversation_id": task_id, "intent_id": "account_access"}
    )
    username = score_subflow_similarity.invoke(
        {"conversation_id": task_id, "subflow_id": "recover_username"}
    )
    password = score_subflow_similarity.invoke(
        {"conversation_id": task_id, "subflow_id": "recover_password"}
    )
    assert intent >= 0
    assert username >= 0
    assert password >= 0

    kb_draft = draft_kb_stub.invoke({"subflow_id": "reset_2fa", "actions": ["pull-up-account"]})
    guideline = draft_guideline_stub.invoke(
        {
            "flow_title": "Account Access",
            "subflow_title": "Reset Two-Factor Auth",
            "actions": ["pull-up-account"],
            "instructions": ["Review before publishing."],
        }
    )
    assert kb_draft == {"reset_2fa": ["pull-up-account"]}
    assert "Account Access" in guideline


def test_agent_subgraphs_compile_outside_jupyter():
    """Every subgraph the deep agent calls compiles as an importable module."""
    from playbook import (
        build_cohort_graph,
        build_intent_discovery_draft_graph,
        build_intent_graph,
        build_pathway_draft_graph,
        build_subflow_discovery_draft_graph,
        build_subflow_graph,
    )

    builders = [
        build_cohort_graph,
        build_intent_graph,
        build_subflow_graph,
        build_intent_discovery_draft_graph,
        build_subflow_discovery_draft_graph,
        build_pathway_draft_graph,
    ]
    for builder in builders:
        compiled = builder()
        assert compiled is not None
        assert compiled.get_graph() is not None


def test_agent_exposes_exactly_the_documented_tools():
    from playbook import PLAYBOOK_TOOLS

    assert {tool.name for tool in PLAYBOOK_TOOLS} == {
        "cohort",
        "classify_intent",
        "classify_subflow",
        "discover_intent",
        "discover_subflow",
        "recommend_pathway",
        "persist_recc",
        "retrieve_guidance",
    }


def test_demo_data_has_everything_the_app_loads():
    """Guards the app's data contract: a rename here breaks startup, not a test."""
    import ast

    demo = ROOT / "demo_data"
    app = ast.parse((ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    wanted: list[str] = []
    for node in app.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DATA_FILES":
            wanted = [element.value for element in node.value.elts]
    assert wanted, "streamlit_app.DATA_FILES not found"
    for name in wanted:
        assert (demo / name).exists(), f"demo_data is missing {name}"

    from playbook.kb import DEFAULT_DATA_DIR

    assert DEFAULT_DATA_DIR == demo


def test_demo_data_holds_the_full_task_set():
    import json

    tasks = json.loads((ROOT / "demo_data" / "incoming_conversations.json").read_text(encoding="utf-8"))
    assert len(tasks) == 145
    dates = sorted({row["conversation_date"] for row in tasks})
    assert dates[0] == "2026-08-25"
    assert dates[-1] >= "2026-09-14"
