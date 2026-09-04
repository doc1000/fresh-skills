"""Extraction-boundary tests for BERTopic discovery helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from playbook.schemas import ConversationRecord, Turn
from playbook.topics import (
    BertopicConfig,
    FittedTopics,
    PrototypeMatch,
    TopicDiscoveryResult,
    apply_fit_rule,
    centroid_matrix,
    classify_against,
    discover_intent_topics,
    discover_subflow_topics,
    resolve_label,
)


def _record(conversation_id: str, text: str) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=conversation_id,
        turns=[Turn(speaker="customer", text=text)],
        actions=[],
    )


def _conversations(n: int = 6) -> list[ConversationRecord]:
    return [_record(str(i), f"customer text {i}") for i in range(n)]


def _fake_fit(documents: list[str], config: BertopicConfig, **_kwargs: Any) -> FittedTopics:
    n = len(documents)
    mid = n // 2
    topic_ids = [0] * mid + [1] * (n - mid)
    return FittedTopics(
        topic_ids=topic_ids,
        descriptors={0: "password, reset", 1: "package, missing"},
    )


@pytest.fixture
def patched_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("playbook.topics.fit_topic_model", _fake_fit)


def test_discovery_functions_are_importable_and_callable(patched_fit) -> None:
    conversations = _conversations()
    intents = discover_intent_topics(conversations)
    subflows = discover_subflow_topics(conversations, intent="order_issue")
    assert isinstance(intents, TopicDiscoveryResult)
    assert isinstance(subflows, TopicDiscoveryResult)
    assert len(intents.assignments) == len(conversations)
    assert len(subflows.assignments) == len(conversations)


def test_intent_discovery_is_deterministic_given_fit(patched_fit) -> None:
    conversations = _conversations()
    first = discover_intent_topics(conversations)
    second = discover_intent_topics(conversations)
    assert first.model_dump() == second.model_dump()
    assert first.assignments == [0, 0, 0, 1, 1, 1]


def test_discovery_outputs_are_compact_and_serializable(patched_fit) -> None:
    result = discover_intent_topics(_conversations())
    dumped = result.model_dump()
    assert "model" not in dumped
    payload = json.dumps(dumped)
    assert json.loads(payload)["assignments"] == result.assignments
    for topic in dumped["topics"]:
        assert set(topic) <= {
            "topic_id",
            "size",
            "descriptor",
            "member_ids",
            "representative_ids",
            "cohesive_enough",
            "parent_intent",
        }


def test_topic_information_and_memberships_are_available(patched_fit) -> None:
    conversations = _conversations()
    result = discover_intent_topics(conversations, config=BertopicConfig(min_topic_n=3))
    by_id = {topic.topic_id: topic for topic in result.topics}
    assert by_id[0].descriptor == "password, reset"
    assert by_id[0].member_ids == ["0", "1", "2"]
    assert by_id[1].member_ids == ["3", "4", "5"]
    assert by_id[0].representative_ids == ["0", "1", "2"]
    assert by_id[0].cohesive_enough is True
    assert by_id[0].parent_intent is None
    assert set(result.assignments) == {0, 1}


def test_subflow_discovery_stamps_parent_intent(patched_fit) -> None:
    result = discover_subflow_topics(_conversations(), intent="order_issue")
    assert {topic.parent_intent for topic in result.topics} == {"order_issue"}
    assert all(topic.parent_intent == "order_issue" for topic in result.topics)


def test_discover_skips_when_below_min_to_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*_args: Any, **_kwargs: Any) -> FittedTopics:
        raise AssertionError("fit should not run when n < min_to_cluster")

    monkeypatch.setattr("playbook.topics.fit_topic_model", fail_if_called)
    result = discover_intent_topics(
        _conversations(3),
        config=BertopicConfig(min_to_cluster=5),
    )
    assert result.skipped is True
    assert result.assignments == [-1, -1, -1]
    assert result.topics == []


def test_classify_against_is_deterministic() -> None:
    embeddings = np.array([[1.0, 0.0], [0.2, 0.8], [0.9, 0.1]], dtype=float)
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    first = classify_against(embeddings, prototypes, ["account_access", "order_issue"])
    second = classify_against(embeddings, prototypes, ["account_access", "order_issue"])
    assert first == second
    assert first[0] == PrototypeMatch(pred="account_access", sim=1.0, margin=1.0, fits=True)
    assert first[1].pred == "order_issue"
    assert first[1].fits is True
    payload = json.dumps([row.model_dump() for row in first])
    assert "account_access" in payload


def test_classify_fit_rule_and_resolve_label_match_notebook() -> None:
    assert apply_fit_rule(True, False, "either") is True
    assert apply_fit_rule(True, False, "both") is False
    assert apply_fit_rule(True, False, "centroid") is True
    assert apply_fit_rule(True, False, "guideline") is False
    assert resolve_label(True, "a", 0.8, True, "b", 0.9) == "b"
    assert resolve_label(True, "a", 0.8, False, "b", 0.9) == "a"
    assert resolve_label(False, "a", 0.8, False, "b", 0.9) is None


def test_centroid_matrix_is_mean_of_labeled_rows() -> None:
    embeddings = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=float)
    matrix = centroid_matrix(embeddings, ["a", "a", "b"], ["a", "b"])
    np.testing.assert_allclose(matrix, [[2.0, 0.0], [0.0, 2.0]])
