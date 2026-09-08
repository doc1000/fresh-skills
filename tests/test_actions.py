"""Extraction-boundary tests for action-path discovery."""

from __future__ import annotations

import json

from playbook.actions import ActionPathResult, CommonWorkflow, discover_action_paths, discover_common_workflow
from playbook.schemas import ConversationRecord, Turn


def _record(conversation_id: str, actions: list[str]) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=conversation_id,
        turns=[Turn(speaker="customer", text="hello")],
        actions=actions,
    )


def test_action_path_discovery_is_importable_and_independent() -> None:
    conversations = [
        _record("1", ["pull-up-account", "verify-identity"]),
        _record("2", ["pull-up-account", "verify-identity"]),
        _record("3", ["notify-team"]),
    ]
    result = discover_action_paths(conversations)
    assert isinstance(result, ActionPathResult)
    assert result.n_conversations == 3
    assert result.n_unique_paths == 2
    assert result.paths[0].actions == ["pull-up-account", "verify-identity"]
    assert result.paths[0].count == 2
    assert result.paths[0].conversation_ids == ["1", "2"]
    assert result.button_counts == {"pull-up-account": 2, "verify-identity": 2, "notify-team": 1}


def test_action_path_outputs_are_deterministic_and_serializable() -> None:
    conversations = [
        _record("b", ["notify-team"]),
        _record("a", ["pull-up-account"]),
        _record("c", ["pull-up-account"]),
    ]
    first = discover_action_paths(conversations)
    second = discover_action_paths(conversations)
    assert first.model_dump() == second.model_dump()
    dumped = first.model_dump()
    json.dumps(dumped)
    assert dumped["paths"][0]["actions"] == ["pull-up-account"]
    assert dumped["paths"][0]["conversation_ids"] == ["a", "c"]


def _edge_pairs(workflow: CommonWorkflow) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in workflow.edges}


def test_common_workflow_clear_linear_path() -> None:
    traces = [
        ["lookup", "verify", "reset"],
        ["lookup", "verify", "reset"],
        ["lookup", "verify", "reset"],
    ]
    result = discover_common_workflow(traces)
    assert result.tools == ["lookup", "verify", "reset"]
    assert result.tool_support == {"lookup": 1.0, "reset": 1.0, "verify": 1.0}
    assert _edge_pairs(result) == {
        ("lookup", "verify"),
        ("lookup", "reset"),
        ("verify", "reset"),
    }
    by_pair = {(row.a, row.b): row for row in result.precedence}
    assert by_pair[("lookup", "verify")].a_before_b == 1.0
    assert by_pair[("lookup", "verify")].b_before_a == 0.0


def test_common_workflow_orders_an_evenly_split_middle_by_threshold() -> None:
    """An even split clears the 0.35 default in both directions, so it is ordered.

    `min_order_support` is the knob: at the default the pair gets an arbitrary
    but deterministic direction, and raising the bar past 0.5 drops the edge.
    The reported precedence still shows the split honestly either way.
    """
    traces = [
        ["start", "alpha", "bravo", "end"],
        ["start", "bravo", "alpha", "end"],
        ["start", "alpha", "bravo", "end"],
        ["start", "bravo", "alpha", "end"],
    ]
    result = discover_common_workflow(traces)
    assert set(result.tools) == {"start", "alpha", "bravo", "end"}
    assert result.tools[0] == "start"
    assert result.tools[-1] == "end"
    edges = _edge_pairs(result)
    assert ("alpha", "bravo") in edges
    assert ("bravo", "alpha") not in edges
    assert ("start", "end") in edges

    strict = discover_common_workflow(traces, min_order_support=0.5)
    strict_edges = _edge_pairs(strict)
    assert ("alpha", "bravo") not in strict_edges
    assert ("bravo", "alpha") not in strict_edges

    by_pair = {(row.a, row.b): row for row in result.precedence}
    middle = by_pair[("alpha", "bravo")]
    assert middle.a_before_b == 0.5
    assert middle.b_before_a == 0.5
    assert middle.n_both == 4


def test_common_workflow_filters_low_support_tools() -> None:
    traces = [
        ["lookup", "verify"],
        ["lookup", "verify"],
        ["lookup", "verify"],
        ["lookup", "verify"],
        ["lookup", "verify", "notify-team"],
    ]
    result = discover_common_workflow(traces)
    assert result.tools == ["lookup", "verify"]
    assert "notify-team" not in result.tools
    assert result.tool_support["notify-team"] == 0.2
    assert all("notify-team" not in (edge.source, edge.target) for edge in result.edges)


def test_common_workflow_sparse_noisy_traces_still_useful() -> None:
    traces = [
        ["pull-up-account", "verify-identity", "issue-refund"],
        ["pull-up-account", "issue-refund"],
        ["pull-up-account", "note-added", "issue-refund"],
        ["pull-up-account", "verify-identity", "issue-refund", "survey"],
        ["transfer", "pull-up-account", "issue-refund"],
    ]
    result = discover_common_workflow(traces)
    # verify-identity is in 2 of 5 traces, which clears the 0.30 default.
    assert result.tools == ["pull-up-account", "verify-identity", "issue-refund"]
    # Raise the bar to a majority and the sparse middle step drops out.
    strict = discover_common_workflow(traces, min_tool_support=0.5)
    assert strict.tools == ["pull-up-account", "issue-refund"]
    assert result.tool_support["pull-up-account"] == 1.0
    assert result.tool_support["issue-refund"] == 1.0
    assert result.tool_support["note-added"] == 0.2
    assert result.tool_support["survey"] == 0.2
    assert result.tool_support["transfer"] == 0.2
    assert _edge_pairs(strict) == {("pull-up-account", "issue-refund")}
    # At the default bar the retained middle step is ordered between them.
    assert _edge_pairs(result) == {
        ("pull-up-account", "verify-identity"),
        ("pull-up-account", "issue-refund"),
        ("verify-identity", "issue-refund"),
    }
    first = discover_common_workflow(traces)
    second = discover_common_workflow(traces)
    assert first.model_dump() == second.model_dump()
    json.dumps(first.model_dump())
