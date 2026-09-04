"""Cleanup-boundary tests for the extracted LangGraph agent."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PY = ROOT / "scratch_modular_meta_agent.py"


@pytest.fixture
def runtime(tmp_path):
    from playbook import configure_runtime

    return configure_runtime(store_path=tmp_path / "run_store.sqlite")


def test_graph_imports_and_compiles_outside_jupyter():
    from playbook import build_graph, build_meta_graph

    compiled = build_graph(False)
    assert compiled is not None
    assert compiled.get_graph() is not None
    assert build_meta_graph(False).name == "meta_agent"


def test_graph_renders_mermaid():
    from playbook import build_graph, render_mermaid

    source = render_mermaid(build_graph(True), xray=1)
    assert "graph" in source or "flowchart" in source
    assert "establish_cohort" in source
    assert "classify" in source
    assert "discover" in source
    assert "recommend" in source


def test_nodes_receive_and_return_typed_state(runtime):
    from playbook import MetaAgentState, cohort_query, empty_state

    state = empty_state(
        run_id="week-state-check",
        cohort_query={"source": "scratch_data/incoming_conversations.json", "window": "demo-week"},
    )
    assert set(state) >= {
        "run_id",
        "cohort_query",
        "kb_version",
        "current_stage",
        "intent_summary",
        "pending_proposal_ids",
        "errors",
    }
    out = cohort_query(state)
    assert out["run_id"] == "week-state-check"
    assert out["current_stage"] == "cohort.query"
    assert isinstance(out, dict)
    merged: MetaAgentState = {**state, **out}
    assert merged["run_id"] == "week-state-check"


def test_playbook_loader_and_retriever_are_independent():
    from playbook import load_playbook, retrieve_guidance

    playbook = load_playbook()
    assert playbook.intent_ids() == ["account_access"]
    assert playbook.subflows_for("account_access") == ["recover_username", "recover_password"]
    assert "recover_username" in playbook.kb
    assert "reset_2fa" not in playbook.kb

    ranked = retrieve_guidance(
        "I forgot my username and cannot log into my account.",
        playbook,
    )
    assert ranked
    assert ranked[0]["intent_id"] == "account_access"
    assert "score" in ranked[0]
    assert "doc" in ranked[0]


def test_langchain_tools_are_callable(runtime):
    from playbook.tools import (
        draft_guideline_stub,
        draft_kb_stub,
        score_intent_similarity,
        score_subflow_similarity,
    )

    intent = score_intent_similarity.invoke(
        {"conversation_id": "u1", "intent_id": "account_access"}
    )
    username = score_subflow_similarity.invoke(
        {"conversation_id": "u1", "subflow_id": "recover_username"}
    )
    password = score_subflow_similarity.invoke(
        {"conversation_id": "u1", "subflow_id": "recover_password"}
    )
    assert intent > 0
    assert username > password

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


def test_agent_flow_reaches_decision_and_review_states(runtime, stub_topic_fit):
    from playbook import invoke_week, simulate_hitl
    from playbook import runtime as rt

    result = invoke_week(
        "week-2026-09-01",
        {
            "source": "scratch_data/incoming_conversations.json",
            "window": "demo-week",
            "as_of": "2026-09-01",
        },
    )
    assert result["current_stage"] == "summarize"
    assert result["run_id"] == "week-2026-09-01"

    proposals = rt.store.list_proposals("week-2026-09-01")
    assert proposals
    assert all(row["review_decision"] in {"accept", "decline"} for row in proposals)
    assert any(row["candidate"] == "shipping_issue" and row["review_decision"] == "accept" for row in proposals)
    assert any(row["candidate"] == "reset_2fa" and row["review_decision"] == "accept" for row in proposals)
    assert any(row["proposal_type"] == "emerging" and row["review_decision"] == "decline" for row in proposals)

    accepted_new_intent = {"proposal_type": "new_intent", "candidate": "shipping_issue"}
    assert simulate_hitl(accepted_new_intent)[0] == "accept"
    assert simulate_hitl({"proposal_type": "outlier", "candidate": None})[0] == "decline"

    assert "shipping_issue" in rt.playbook.intent_ids()
    assert "reset_2fa" in rt.playbook.subflows_for("account_access")
    assert "missing" in rt.playbook.subflows_for("shipping_issue")
    assert result["recommendation_summary"].get("recommended", 0) >= 1
    assert rt.store.list_recommendations("week-2026-09-01")


def test_notebook_imports_implementation_instead_of_duplicating_it():
    source = NOTEBOOK_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
    }
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    assert any(mod == "playbook" or mod.startswith("playbook.") for mod in imported_modules)
    for name in (
        "PlaybookKB",
        "TaskStore",
        "MetaAgentState",
        "build_meta_graph",
        "simulate_hitl",
        "score_intent_similarity",
        "load_playbook",
    ):
        assert name not in defined, f"notebook still defines {name}"
