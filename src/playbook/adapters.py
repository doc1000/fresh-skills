"""Adapt DS discovery outputs into the existing agent contracts.

Graph nodes should not import BERTopic objects or ABCD label fields.

Runtime conversations drop hidden flow/subflow. Topic rows keep memberships
and cohesive_enough so later nodes can override, name, and describe a topic.
Retrieval stays `playbook.kb.retrieve_guidance` over the Agent KB files.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from playbook.actions import ActionPathResult
from playbook.data import conversation_document
from playbook.kb import PlaybookKB
from playbook.schemas import ConversationRecord, DiscoveredTopic, Turn
from playbook.scoring import CLUSTER_THRESHOLD, greedy_jaccard_clusters
from playbook.topics import (
    BertopicConfig,
    TopicDiscoveryResult,
    TopicInfo,
    apply_fit_rule,
    centroid_matrix,
    classify_against,
    embed_texts,
    resolve_label,
)

FitRule = Literal["centroid", "guideline", "either", "both"]

# Starting copy of the DS working-sheet cluster knobs. The notebook should
# overwrite `runtime.topic_config` before invoke.
RUNTIME_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=5,
    min_cluster_size=5,
    min_samples=1,
    min_topic_n=5,
    representative_n=3,
    n_neighbors=12,
    n_components=5,
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
        member_ids=list(topic.member_ids),
        cohesive_enough=topic.cohesive_enough,
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


def _example_document(item: dict[str, Any]) -> str:
    turns = item.get("original") or []
    texts: list[str] = []
    for turn in turns:
        speaker = turn["speaker"] if isinstance(turn, dict) else turn[0]
        text = turn["text"] if isinstance(turn, dict) else turn[1]
        if speaker in {"customer", "agent"} and text:
            texts.append(str(text))
    return "\n".join(texts)


def jaccard_topic_result(
    conversations: Sequence[ConversationRecord],
    *,
    config: BertopicConfig,
    parent_intent: str | None = None,
    threshold: float = CLUSTER_THRESHOLD,
) -> TopicDiscoveryResult:
    records = list(conversations)
    if len(records) < config.min_to_cluster:
        return TopicDiscoveryResult(
            topics=[],
            assignments=[-1] * len(records),
            skipped=True,
            config=config,
        )
    ids = [record.conversation_id for record in records]
    texts = {record.conversation_id: conversation_document(record) for record in records}
    groups = greedy_jaccard_clusters(ids, texts, threshold)
    by_id: dict[str, int] = {}
    topics: list[TopicInfo] = []
    for topic_id, member_ids in enumerate(groups):
        size = len(member_ids)
        words = texts[member_ids[0]].split()[:8]
        topics.append(
            TopicInfo(
                topic_id=topic_id,
                size=size,
                descriptor=", ".join(words),
                member_ids=member_ids,
                representative_ids=member_ids[: config.representative_n],
                cohesive_enough=size >= config.min_topic_n,
                parent_intent=parent_intent,
            )
        )
        for member_id in member_ids:
            by_id[member_id] = topic_id
    return TopicDiscoveryResult(
        topics=topics,
        assignments=[by_id[record.conversation_id] for record in records],
        skipped=False,
        config=config,
    )


def _prototype_rows(
    conversations: Sequence[ConversationRecord],
    playbook: PlaybookKB,
    seed_examples: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    kind: Literal["intent", "subflow"],
    parent_intent: str | None = None,
    min_sim: float,
    min_margin: float,
    fit_rule: FitRule,
) -> list[dict[str, Any]]:
    records = list(conversations)
    if not records or not names:
        return []
    documents = [conversation_document(record) for record in records]
    embeddings = embed_texts(documents)

    centroid_names = list(names)
    seed_docs: list[str] = []
    seed_labels: list[str] = []
    for item in seed_examples:
        scenario = item.get("scenario") or {}
        label = scenario.get("flow") if kind == "intent" else scenario.get("subflow")
        if kind == "subflow" and parent_intent and scenario.get("flow") != parent_intent:
            continue
        if label not in centroid_names:
            continue
        seed_docs.append(_example_document(item))
        seed_labels.append(str(label))

    centroid_matches = None
    if seed_docs:
        seeded_names = [name for name in centroid_names if name in set(seed_labels)]
        if seeded_names:
            prototypes = centroid_matrix(embed_texts(seed_docs), seed_labels, seeded_names)
            centroid_matches = classify_against(
                embeddings, prototypes, seeded_names, min_sim=min_sim, min_margin=min_margin
            )

    if kind == "intent":
        guide_docs = [playbook.guideline_intent_text(name) for name in centroid_names]
    else:
        intent_id = parent_intent or ""
        guide_docs = [playbook.subflow_doc(intent_id, name) for name in centroid_names]
    guideline_matches = classify_against(
        embeddings,
        embed_texts(guide_docs),
        centroid_names,
        min_sim=min_sim,
        min_margin=min_margin,
    )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        centroid = centroid_matches[index] if centroid_matches is not None else None
        guideline = guideline_matches[index]
        fits_c = bool(centroid.fits) if centroid is not None else False
        pred_c = centroid.pred if centroid is not None else ""
        sim_c = centroid.sim if centroid is not None else 0.0
        assigned = apply_fit_rule(fits_c, bool(guideline.fits), fit_rule)
        label = resolve_label(
            fits_c,
            pred_c,
            sim_c,
            bool(guideline.fits),
            guideline.pred,
            guideline.sim,
            fit_rule,
        )
        confidence = 0.0
        if label == pred_c:
            confidence = sim_c
        elif label == guideline.pred:
            confidence = guideline.sim
        rows.append(
            {
                "conversation_id": record.conversation_id,
                "assigned": assigned and label is not None,
                "label": label,
                "confidence": round(float(confidence), 3),
                "centroid_pred": pred_c,
                "centroid_sim": round(float(sim_c), 3),
                "guideline_pred": guideline.pred,
                "guideline_sim": round(float(guideline.sim), 3),
            }
        )
    return rows


def classify_intents_bertopic(
    conversations: Sequence[ConversationRecord],
    playbook: PlaybookKB,
    seed_examples: Sequence[dict[str, Any]],
    *,
    min_sim: float,
    min_margin: float,
    fit_rule: FitRule,
) -> list[dict[str, Any]]:
    names = playbook.intent_ids()
    if not names:
        return [
            {"task_id": record.conversation_id, "intent_id": "unknown", "confidence": 0.0, "scores": []}
            for record in conversations
        ]
    rows = _prototype_rows(
        conversations,
        playbook,
        seed_examples,
        names,
        kind="intent",
        min_sim=min_sim,
        min_margin=min_margin,
        fit_rule=fit_rule,
    )
    results = []
    for row in rows:
        results.append(
            {
                "task_id": row["conversation_id"],
                "intent_id": row["label"] if row["assigned"] else "unknown",
                "confidence": row["confidence"] if row["assigned"] else 0.0,
                "scores": [
                    ("centroid", row["centroid_pred"], row["centroid_sim"]),
                    ("guideline", row["guideline_pred"], row["guideline_sim"]),
                ],
            }
        )
    return results


def classify_subflows_bertopic(
    conversations: Sequence[ConversationRecord],
    playbook: PlaybookKB,
    seed_examples: Sequence[dict[str, Any]],
    intent_by_task: dict[str, str],
    *,
    min_sim: float,
    min_margin: float,
    fit_rule: FitRule,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ConversationRecord]] = {}
    for record in conversations:
        grouped.setdefault(intent_by_task[record.conversation_id], []).append(record)
    results: list[dict[str, Any]] = []
    for intent_id, group in grouped.items():
        names = list(playbook.subflows_for(intent_id))
        if not names:
            for record in group:
                results.append(
                    {
                        "task_id": record.conversation_id,
                        "intent_id": intent_id,
                        "subflow_id": "unknown",
                        "confidence": 0.0,
                        "scores": [],
                    }
                )
            continue
        rows = _prototype_rows(
            group,
            playbook,
            seed_examples,
            names,
            kind="subflow",
            parent_intent=intent_id,
            min_sim=min_sim,
            min_margin=min_margin,
            fit_rule=fit_rule,
        )
        for row in rows:
            results.append(
                {
                    "task_id": row["conversation_id"],
                    "intent_id": intent_id,
                    "subflow_id": row["label"] if row["assigned"] else "unknown",
                    "confidence": row["confidence"] if row["assigned"] else 0.0,
                    "scores": [
                        ("centroid", row["centroid_pred"], row["centroid_sim"]),
                        ("guideline", row["guideline_pred"], row["guideline_sim"]),
                    ],
                }
            )
    return results
