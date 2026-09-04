"""Extraction-boundary tests for action-path discovery."""

from __future__ import annotations

import json

from playbook.actions import ActionPathResult, discover_action_paths
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
