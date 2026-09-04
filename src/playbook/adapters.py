"""Adapt DS discovery outputs into the existing agent contracts.

This is the only integration boundary. Graph nodes should not import BERTopic
objects or ABCD label fields.

Known mismatches absorbed here (kept visible, not smoothed over):

* TaskStore rows still carry `hidden_flow` / `hidden_subflow` for eval. Runtime
  `ConversationRecord` must not.
* DS `TopicInfo` has memberships and `cohesive_enough`. Agent graph state uses
  plan-frozen `DiscoveredTopic` (no memberships, no model config).
* DS default `min_topic_n` / `min_to_cluster` is 5. The agent demo's unresolved
  leftover sets are often size 3–4, so the runtime path uses a smaller config.
* DS `ActionPathResult` counts every conversation. The pathway node still
  evaluates successful traces only, then reads the DS path counts.
* Canonical retrieval stays `playbook.kb.retrieve_guidance` over `scratch_data/`
  seed files. DS `config.KB_JSON` (`data/raw/kb.json`) is not a runtime source.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from playbook.actions import ActionPathResult
from playbook.schemas import ConversationRecord, DiscoveredTopic, Turn
from playbook.topics import BertopicConfig, TopicDiscoveryResult, TopicInfo

RUNTIME_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=2,
    min_cluster_size=2,
    min_samples=1,
    min_topic_n=2,
    representative_n=3,
    n_neighbors=3,
    n_components=2,
)


def tasks_to_conversations(tasks: Sequence[dict[str, Any]]) -> list[ConversationRecord]:
    """Project store rows to runtime records. Drops hidden ABCD labels."""
    records: list[ConversationRecord] = []
    for task in tasks:
        raw_turns = task.get("turns_json")
        if isinstance(raw_turns, str):
            raw_turns = json.loads(raw_turns)
        turns = [Turn.model_validate(turn) for turn in raw_turns or []]
        if not turns and task.get("text"):
            turns = [Turn(speaker="customer", text=str(task["text"]))]
        records.append(
            ConversationRecord(
                conversation_id=str(task["task_id"]),
                turns=turns,
                actions=list(task.get("actions") or []),
            )
        )
    return records


def topic_to_discovered(topic: TopicInfo) -> DiscoveredTopic:
    return DiscoveredTopic(
        topic_id=topic.topic_id,
        size=topic.size,
        descriptor=topic.descriptor,
        representative_conversation_ids=list(topic.representative_ids),
    )


def topic_result_to_clusters(result: TopicDiscoveryResult) -> list[dict[str, Any]]:
    """Cluster dicts the existing validate/HITL nodes already understand."""
    clusters: list[dict[str, Any]] = []
    for topic in result.topics:
        clusters.append(
            {
                "task_ids": list(topic.member_ids),
                "intent_id": topic.parent_intent,
                "cohesive_enough": topic.cohesive_enough,
                "discovered_topic": topic_to_discovered(topic).model_dump(),
            }
        )
    return clusters


def discovered_topics_from_clusters(clusters: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable graph-state rows. Noise topic `-1` stays out of typed state."""
    topics: list[dict[str, Any]] = []
    for cluster in clusters:
        topic = dict(cluster.get("discovered_topic") or {})
        if topic.get("topic_id", -1) == -1:
            continue
        topics.append(topic)
    return topics


def action_paths_to_state(result: ActionPathResult) -> list[dict[str, Any]]:
    return [path.model_dump() for path in result.paths]
