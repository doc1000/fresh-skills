"""Integration-boundary tests: DS discovery into the existing LangGraph agent."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import pytest

from playbook.schemas import ConversationRecord, DiscoveredTopic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path):
    from playbook import configure_runtime

    return configure_runtime(store_path=tmp_path / "run_store.sqlite")


def _invoke_demo_week():
    from playbook import invoke_week

    return invoke_week(
        "week-2026-09-01",
        {
            "source": "scratch_data/incoming_conversations.json",
            "window": "demo-week",
            "as_of": "2026-09-01",
        },
    )


def test_ds_discovery_is_invoked_from_the_real_agent_path(runtime, stub_topic_fit, monkeypatch):
    from playbook import actions, topics

    intent_calls: list[list[ConversationRecord]] = []
    subflow_calls: list[list[ConversationRecord]] = []
    action_calls: list[list[ConversationRecord]] = []

    real_intent = topics.discover_intent_topics
    real_subflow = topics.discover_subflow_topics
    real_actions = actions.discover_action_paths

    def spy_intent(conversations, **kwargs):
        intent_calls.append(list(conversations))
        return real_intent(conversations, **kwargs)

    def spy_subflow(conversations, **kwargs):
        subflow_calls.append(list(conversations))
        return real_subflow(conversations, **kwargs)

    def spy_actions(conversations, **kwargs):
        action_calls.append(list(conversations))
        return real_actions(conversations, **kwargs)

    monkeypatch.setattr("playbook.graph.discover_intent_topics", spy_intent)
    monkeypatch.setattr("playbook.graph.discover_subflow_topics", spy_subflow)
    monkeypatch.setattr("playbook.graph.discover_action_paths", spy_actions)

    _invoke_demo_week()

    assert intent_calls, "discover_intent_topics was not called from invoke_week"
    assert subflow_calls, "discover_subflow_topics was not called from invoke_week"
    assert action_calls, "discover_action_paths was not called from invoke_week"


def test_discovery_outputs_enter_stable_typed_graph_state(runtime, stub_topic_fit):
    result = _invoke_demo_week()

    assert result["current_stage"] == "summarize"
    topics = result["discovered_topics"]
    subflows = result["discovered_subflows"]
    paths = result["action_paths"]
    assert topics
    assert isinstance(topics, list)
    for row in topics:
        parsed = DiscoveredTopic.model_validate(row)
        assert set(row) == set(parsed.model_dump())
        assert "member_ids" not in row
        assert "cohesive_enough" not in row
        assert "config" not in row
        assert "assignments" not in row
        assert "embedding_model" not in row
    assert subflows
    for row in subflows:
        DiscoveredTopic.model_validate(row)
    assert paths
    for row in paths:
        assert set(row) >= {"actions", "count", "conversation_ids"}


def test_canonical_agent_kb_retriever_is_used(runtime, stub_topic_fit, monkeypatch):
    from playbook import kb

    retrieve_calls: list[str] = []
    real_retrieve = kb.retrieve_guidance

    def spy_retrieve(query, playbook=None, *, top_k=4):
        retrieve_calls.append(query)
        return real_retrieve(query, playbook, top_k=top_k)

    monkeypatch.setattr("playbook.graph.retrieve_guidance", spy_retrieve)

    result = _invoke_demo_week()

    assert retrieve_calls, "retrieve_guidance was not used on the agent path"
    assert result["retrieved_guidance"]
    for row in result["retrieved_guidance"]:
        assert "hits" in row
        assert all("intent_id" in hit and "doc" in hit for hit in row["hits"])

    graph_source = (ROOT / "src" / "playbook" / "graph.py").read_text(encoding="utf-8")
    kb_source = (ROOT / "src" / "playbook" / "kb.py").read_text(encoding="utf-8")
    assert "KB_JSON" not in graph_source
    assert "GUIDELINES_JSON" not in graph_source
    assert "data/raw/kb.json" not in kb_source
    assert "scratch_data" in kb_source
    assert "load_playbook" in kb_source
    assert "retrieve_guidance" in kb_source


def test_graph_still_compiles_and_renders():
    from playbook import build_graph, render_mermaid

    compiled = build_graph(False)
    assert compiled.name == "meta_agent"
    source = render_mermaid(build_graph(True), xray=1)
    assert "establish_cohort" in source
    assert "classify" in source
    assert "discover" in source
    assert "recommend" in source


def test_existing_decision_and_hitl_flow_still_works(runtime, stub_topic_fit):
    from playbook import simulate_hitl
    from playbook import runtime as rt

    result = _invoke_demo_week()
    assert result["current_stage"] == "summarize"
    proposals = rt.store.list_proposals("week-2026-09-01")
    assert proposals
    assert all(row["review_decision"] in {"accept", "decline"} for row in proposals)
    assert any(row["candidate"] == "shipping_issue" and row["review_decision"] == "accept" for row in proposals)
    assert any(row["candidate"] == "reset_2fa" and row["review_decision"] == "accept" for row in proposals)
    assert any(row["proposal_type"] == "emerging" and row["review_decision"] == "decline" for row in proposals)
    assert simulate_hitl({"proposal_type": "new_intent", "candidate": "shipping_issue"})[0] == "accept"
    assert "shipping_issue" in rt.playbook.intent_ids()
    assert "reset_2fa" in rt.playbook.subflows_for("account_access")
    assert "missing" in rt.playbook.subflows_for("shipping_issue")
    assert result["recommendation_summary"].get("recommended", 0) >= 1
    assert rt.store.list_recommendations("week-2026-09-01")


def test_runtime_discovery_does_not_receive_held_out_abcd_labels(runtime, stub_topic_fit, monkeypatch):
    from playbook import topics

    seen: list[ConversationRecord] = []
    real_intent = topics.discover_intent_topics
    real_subflow = topics.discover_subflow_topics

    def capture(conversations, **kwargs):
        seen.extend(conversations)
        return real_intent(conversations, **kwargs)

    def capture_subflow(conversations, **kwargs):
        seen.extend(conversations)
        return real_subflow(conversations, **kwargs)

    monkeypatch.setattr("playbook.graph.discover_intent_topics", capture)
    monkeypatch.setattr("playbook.graph.discover_subflow_topics", capture_subflow)

    _invoke_demo_week()
    assert seen
    for record in seen:
        payload = record.model_dump()
        assert set(payload) == {"conversation_id", "turns", "actions"}
        assert "flow" not in payload
        assert "subflow" not in payload
        assert "hidden_flow" not in payload
        assert "hidden_subflow" not in payload
        assert not hasattr(record, "flow")
        assert not hasattr(record, "hidden_subflow")


def test_no_duplicate_ds_or_agent_implementations_remain():
    graph_path = ROOT / "src" / "playbook" / "graph.py"
    tools_path = ROOT / "src" / "playbook" / "tools.py"
    graph_tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    tools_tree = ast.parse(tools_path.read_text(encoding="utf-8"))

    graph_imports: set[str] = set()
    for node in ast.walk(graph_tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                graph_imports.add(alias.name)
        elif isinstance(node, ast.Import):
            graph_imports.update(alias.name for alias in node.names)

    assert "cluster_conversation_ids" not in graph_imports
    assert "discover_intent_topics" in graph_imports
    assert "discover_subflow_topics" in graph_imports
    assert "discover_action_paths" in graph_imports
    assert "retrieve_guidance" in graph_imports

    tool_names = {
        node.name
        for node in ast.walk(tools_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "cluster_conversation_ids" not in tool_names

    from playbook.graph import (
        _run_topic_discovery,
        intent_discovery_discover,
        pathway_analyze,
        subflow_discovery_discover,
    )

    helper_src = inspect.getsource(_run_topic_discovery)
    intent_src = inspect.getsource(intent_discovery_discover)
    subflow_src = inspect.getsource(subflow_discovery_discover)
    pathway_src = inspect.getsource(pathway_analyze)
    assert "discover_intent_topics" in helper_src
    assert "discover_subflow_topics" in helper_src
    assert "_run_topic_discovery" in intent_src
    assert "_run_topic_discovery" in subflow_src
    assert "discover_action_paths" in pathway_src
    assert "cluster_conversation_ids" not in helper_src
    assert "cluster_conversation_ids" not in intent_src
    assert "cluster_conversation_ids" not in subflow_src


def test_task_adapter_strips_store_labels():
    from playbook.adapters import tasks_to_conversations

    conversations = tasks_to_conversations(
        [
            {
                "task_id": "s1",
                "text": "package missing",
                "actions": ["pull-up-account"],
                "success": 1,
                "hidden_flow": "shipping_issue",
                "hidden_subflow": "missing",
                "turns_json": '[{"speaker": "customer", "text": "package missing"}]',
            }
        ]
    )
    assert len(conversations) == 1
    assert conversations[0].conversation_id == "s1"
    assert conversations[0].actions == ["pull-up-account"]
    assert set(conversations[0].model_dump()) == {"conversation_id", "turns", "actions"}
