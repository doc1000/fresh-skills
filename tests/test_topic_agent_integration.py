"""Integration-boundary tests: DS discovery into the existing LangGraph agent."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import pytest

from playbook.schemas import ConversationRecord, DiscoveredTopic
from playbook.topics import BertopicConfig

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = ROOT / "demo_data"
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
    import conftest as test_conf
    from playbook import configure_runtime

    return configure_runtime(
        data_dir=DEMO_DATA_DIR,
        store_path=tmp_path / "run_store.sqlite",
        vectors_path=tmp_path / "embeddings.duckdb",
        method="jaccard",
        topic_config=DEMO_TOPIC_CONFIG,
        embed_fn_override=test_conf.fake_embed_texts,
        load_env=False,
    )


@pytest.fixture
def bertopic_runtime(tmp_path):
    import conftest as test_conf
    from playbook import configure_runtime

    return configure_runtime(
        data_dir=DEMO_DATA_DIR,
        store_path=tmp_path / "run_store.sqlite",
        vectors_path=tmp_path / "embeddings.duckdb",
        method="bertopic",
        topic_config=DEMO_TOPIC_CONFIG,
        embed_fn_override=test_conf.fake_embed_texts,
        load_env=False,
    )


def _accept_intent_drafts(run_id: str) -> None:
    """Accept every drafted intent, the way a named second `discover_intent` call does.

    Naming is a tool-level concern, so this step goes through the tool even
    though the surrounding steps invoke the subgraphs directly.
    """
    from playbook import discover_intent
    from playbook import runtime as rt

    rt.current_run_id = run_id
    drafts = [
        {"proposal_id": row["proposal_id"], "name": row["candidate"]}
        for row in rt.store.list_proposals(run_id, "new_intent")
        if row["candidate"]
    ]
    if drafts:
        discover_intent.invoke({"names": drafts, "run_id": run_id})


def _run_demo_week(run_id: str = "week-2026-09-01", start: str = "2026-09-01", end: str = "2026-09-07"):
    """Run the subgraphs the deep agent's tools call, keeping the full state.

    The tools return slimmed summaries by design. These assertions are about the
    typed state the subgraph nodes exchange, so the subgraphs are invoked
    directly here — same nodes, same order a full run uses: classify against the
    KB, discover what is left over, accept it, re-classify against the enlarged
    KB, then do the same one level down for subflows.
    """
    from playbook import (
        build_cohort_graph,
        build_intent_discovery_draft_graph,
        build_intent_graph,
        build_subflow_discovery_draft_graph,
        build_subflow_graph,
        empty_state,
        invoke_named,
    )
    from playbook import runtime as rt

    state = empty_state(
        run_id=run_id,
        start=start,
        end=end,
        method=rt.method,
        kb_version=rt.playbook.version,
    )

    def step(compiled, agent):
        nonlocal state
        state = {**state, **invoke_named(compiled, state, agent=agent)}

    step(build_cohort_graph(), "cohort")
    step(build_intent_graph(), "intent")
    step(build_intent_discovery_draft_graph(), "intent_discovery")
    _accept_intent_drafts(run_id)
    # The classifier re-queries the store after a KB mutation; the tasks that
    # match a newly accepted intent are what subflow discovery works on.
    state["kb_version"] = rt.playbook.version
    step(build_intent_graph(), "intent")
    step(build_subflow_graph(), "subflow")
    step(build_subflow_discovery_draft_graph(), "subflow_discovery")
    return state


def test_ds_discovery_is_invoked_from_the_real_agent_path(bertopic_runtime, stub_topic_fit, monkeypatch):
    from playbook import topics

    intent_calls: list[list[ConversationRecord]] = []
    subflow_calls: list[list[ConversationRecord]] = []

    real_intent = topics.discover_intent_topics
    real_subflow = topics.discover_subflow_topics

    def spy_intent(conversations, **kwargs):
        intent_calls.append(list(conversations))
        return real_intent(conversations, **kwargs)

    def spy_subflow(conversations, **kwargs):
        subflow_calls.append(list(conversations))
        return real_subflow(conversations, **kwargs)

    monkeypatch.setattr("playbook.graph.discover_intent_topics", spy_intent)
    monkeypatch.setattr("playbook.graph.discover_subflow_topics", spy_subflow)

    from playbook import runtime as rt

    # Seed-window chats classify under centroids. Discovery wiring is checked
    # on the leftover path the stub clusterer already understands.
    rt.seed_examples = []
    _run_demo_week()

    assert intent_calls, "discover_intent_topics was not called on the tool path"
    assert subflow_calls, "discover_subflow_topics was not called on the tool path"


def test_discovery_outputs_enter_stable_typed_graph_state(bertopic_runtime, stub_topic_fit):
    from playbook import runtime as rt

    rt.seed_examples = []
    result = _run_demo_week()

    assert result["current_stage"] == "summarize_subflow_discovery"
    topics = result["discovered_topics"]
    subflows = result["discovered_subflows"]
    assert topics
    assert isinstance(topics, list)
    for row in topics:
        parsed = DiscoveredTopic.model_validate(row)
        assert set(row) == set(parsed.model_dump())
        assert row["member_ids"]
        assert "cohesive_enough" in row
        assert "config" not in row
        assert "assignments" not in row
        assert "embedding_model" not in row
    assert isinstance(subflows, list)
    for row in subflows:
        DiscoveredTopic.model_validate(row)


def test_canonical_agent_kb_retriever_is_used(bertopic_runtime, stub_topic_fit, monkeypatch):
    from playbook import kb

    retrieve_calls: list[str] = []
    real_retrieve = kb.retrieve_guidance

    def spy_retrieve(query, playbook=None, *, top_k=4):
        retrieve_calls.append(query)
        return real_retrieve(query, playbook, top_k=top_k)

    monkeypatch.setattr("playbook.graph.retrieve_guidance", spy_retrieve)

    from playbook import runtime as rt

    rt.seed_examples = []
    result = _run_demo_week()

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
    assert "demo_data" in kb_source
    assert "load_playbook" in kb_source
    assert "retrieve_guidance" in kb_source

def test_subgraph_nodes_keep_their_real_names():
    """Phase nodes are named for what they do, so a LangSmith trace reads."""
    from playbook import build_intent_graph

    intent = build_intent_graph().get_graph().draw_mermaid()
    assert "select_unresolved_intents" in intent
    assert "score_intents" in intent
    assert "persist_intent_labels" in intent
    assert "summarize_intent_assignments" in intent
    generic = [
        line.strip()
        for line in intent.splitlines()
        if line.strip() in {"query", "process", "persist", "summarize"}
    ]
    assert not generic, generic

def test_discovery_drafts_then_persists_on_a_named_second_call(runtime, stub_topic_fit):
    """Discovery proposes; a second call with names is what writes the KB."""
    from playbook import cohort, discover_intent
    from playbook import runtime as rt

    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "week-2026-09-01",
        }
    )
    draft = discover_intent.invoke({})
    proposals = rt.store.list_proposals("week-2026-09-01")
    assert proposals, "first call should persist proposals"
    assert all(row["review_decision"] is None for row in proposals), "drafts are not auto-approved"
    assert draft["discovery_summary"]["intents"]["candidate_count"] >= 1

    candidates = [row for row in proposals if row["proposal_type"] == "new_intent"]
    assert candidates
    before = set(rt.playbook.intent_ids())
    named = discover_intent.invoke(
        {"names": [{"proposal_id": candidates[0]["proposal_id"], "name": "shipping_issue"}]}
    )
    assert named["ok"] is True
    assert "shipping_issue" in rt.playbook.intent_ids()
    assert set(rt.playbook.intent_ids()) - before == {"shipping_issue"}
    accepted = [
        row
        for row in rt.store.list_proposals("week-2026-09-01")
        if row["review_decision"] == "accept"
    ]
    assert accepted


def test_runtime_discovery_does_not_receive_held_out_abcd_labels(bertopic_runtime, stub_topic_fit, monkeypatch):
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

    _run_demo_week()
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
    assert "jaccard_topic_result" in helper_src
    assert "runtime.method" in helper_src
    assert "_run_topic_discovery" in intent_src
    assert "_run_topic_discovery" in subflow_src
    assert "discover_action_paths" in pathway_src
    assert "discover_common_workflow" in pathway_src
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

def test_jaccard_method_does_not_call_bertopic(tmp_path, monkeypatch):
    import conftest as test_conf
    from playbook import classify_intent, cohort, configure_runtime, discover_intent
    from playbook.topics import TopicDiscoveryResult

    configure_runtime(
        data_dir=DEMO_DATA_DIR,
        store_path=tmp_path / "run_store.sqlite",
        vectors_path=tmp_path / "embeddings.duckdb",
        method="jaccard",
        embed_fn_override=test_conf.fake_embed_texts,
        load_env=False,
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("BERTopic path should not run when METHOD=jaccard")

    monkeypatch.setattr("playbook.graph.discover_intent_topics", boom)
    monkeypatch.setattr("playbook.graph.discover_subflow_topics", boom)
    cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "week-jaccard",
        }
    )
    result = classify_intent.invoke({})
    assert result["current_stage"] == "summarize_intent_assignments"
    assert isinstance(result["intent_summary"], dict)
    assert "classified" in result["intent_summary"]
    discover_intent.invoke({})
    # TopicDiscoveryResult import keeps the type visible if the jaccard path returns it.
    assert TopicDiscoveryResult is not None

def test_public_entrypoint_is_the_cohort_tool(runtime, stub_topic_fit):
    """`cohort` is the entry point: it fixes the working set every later tool reads."""
    from playbook import cohort
    from playbook import runtime as rt

    result = cohort.invoke(
        {
            "start_date": "2026-09-01",
            "end_date": "2026-09-07",
            "method": "jaccard",
            "run_id": "week-2026-09-01",
        }
    )
    assert result["run_id"] == "week-2026-09-01"
    assert result["cohort_summary"]["start"] == "2026-09-01"
    assert result["cohort_summary"]["end"] == "2026-09-07"
    assert result["cohort_summary"]["n"] == len(rt.store.cohort_ids("week-2026-09-01"))
    assert result["cohort_summary"]["n"] > 0
    assert rt.current_run_id == "week-2026-09-01"

def test_start_end_selects_the_cohort_window(tmp_path, stub_topic_fit):
    import conftest as test_conf
    from playbook import cohort, configure_runtime, load_conversations
    from playbook import runtime as rt

    conversations = load_conversations(DEMO_DATA_DIR)
    for i, row in enumerate(conversations):
        row["conversation_date"] = "2026-08-01" if i == 0 else "2026-09-01"
    configure_runtime(
        data_dir=DEMO_DATA_DIR,
        store_path=tmp_path / "run_store.sqlite",
        vectors_path=tmp_path / "embeddings.duckdb",
        method="jaccard",
        topic_config=DEMO_TOPIC_CONFIG,
        conversations=conversations,
        embed_fn_override=test_conf.fake_embed_texts,
        load_env=False,
    )
    result = cohort.invoke(
        {
            "start_date": "2026-08-01",
            "end_date": "2026-08-01",
            "method": "jaccard",
            "run_id": "window-aug",
        }
    )
    assert result["cohort_summary"]["n"] == 1
    assert result["cohort_summary"]["min_conversation_date"] == "2026-08-01"
    assert result["cohort_summary"]["max_conversation_date"] == "2026-08-01"
    assert rt.store.cohort_ids("window-aug") == [conversations[0]["convo_id"]]
