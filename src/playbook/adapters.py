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
    centroid_matrix,
    classify_against,
    embed_texts,
    resolve_label,
)
from playbook.vectors import KB_KIND_INTENT_GUIDE, KB_KIND_SUBFLOW, kb_doc_id_subflow

FitRule = Literal["centroid", "guideline", "either", "both"]

# Starting copy of the DS working-sheet cluster knobs. The notebook should
# overwrite `runtime.topic_config` before invoke.
RUNTIME_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=5,
    min_cluster_size=3,
    min_samples=1,
    min_topic_n=3,
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


def _task_embeddings(records: Sequence[ConversationRecord]) -> np.ndarray:
    from playbook import runtime as rt

    task_ids = [record.conversation_id for record in records]
    if rt.vectors is not None:
        ordered, matrix = rt.vectors.task_matrix(task_ids)
        if len(ordered) == len(task_ids):
            return matrix
    documents = [conversation_document(record) for record in records]
    fn = rt.embed_fn or embed_texts
    return fn(documents)


def _guideline_embeddings(
    playbook: PlaybookKB,
    names: Sequence[str],
    *,
    kind: Literal["intent", "subflow"],
    parent_intent: str | None,
) -> tuple[list[str], np.ndarray]:
    from playbook import runtime as rt

    label_names = list(names)
    if kind == "intent":
        guide_kind = KB_KIND_INTENT_GUIDE
        doc_ids = label_names
    else:
        guide_kind = KB_KIND_SUBFLOW
        intent_id = parent_intent or ""
        doc_ids = [kb_doc_id_subflow(intent_id, name) for name in label_names]

    if rt.vectors is not None:
        stored_ids, matrix, _ = rt.vectors.kb_matrix(guide_kind, doc_ids)
        if len(stored_ids) == len(doc_ids):
            return label_names, matrix

    if kind == "intent":
        guide_docs = [playbook.guideline_intent_text(name) for name in label_names]
    else:
        intent_id = parent_intent or ""
        guide_docs = [playbook.subflow_doc(intent_id, name) for name in label_names]
    fn = rt.embed_fn or embed_texts
    return label_names, fn(guide_docs)


def _seed_label(
    item: dict[str, Any],
    *,
    kind: Literal["intent", "subflow"],
    parent_intent: str | None,
) -> str | None:
    scenario = item.get("scenario") or {}
    if kind == "intent":
        return str(scenario["flow"]) if scenario.get("flow") else None
    if parent_intent and scenario.get("flow") != parent_intent:
        return None
    return str(scenario["subflow"]) if scenario.get("subflow") else None


def _seed_embeddings(items: Sequence[dict[str, Any]]) -> np.ndarray:
    from playbook import runtime as rt

    task_ids = [str(item["convo_id"]) for item in items]
    if rt.vectors is not None:
        ordered, matrix = rt.vectors.task_matrix(task_ids)
        if list(ordered) == task_ids:
            return matrix
    documents = [_example_document(item) for item in items]
    fn = rt.embed_fn or embed_texts
    return fn(documents)


def _centroid_prototypes(
    seed_examples: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    kind: Literal["intent", "subflow"],
    parent_intent: str | None,
) -> np.ndarray | None:
    labeled: list[dict[str, Any]] = []
    labels: list[str] = []
    wanted = set(names)
    for item in seed_examples:
        label = _seed_label(item, kind=kind, parent_intent=parent_intent)
        if label in wanted:
            labeled.append(item)
            labels.append(label)
    if not labeled or any(name not in labels for name in names):
        return None
    return centroid_matrix(_seed_embeddings(labeled), labels, names)


def _winning_confidence(centroid, guideline, label: str | None) -> float:
    if label is None:
        return 0.0
    if centroid.fits and guideline.fits:
        return round(float(centroid.sim if centroid.sim >= guideline.sim else guideline.sim), 3)
    if centroid.fits:
        return round(float(centroid.sim), 3)
    return round(float(guideline.sim), 3)


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
    embeddings = _task_embeddings(records)
    guide_names, guide_matrix = _guideline_embeddings(
        playbook, names, kind=kind, parent_intent=parent_intent
    )
    guideline_matches = classify_against(
        embeddings,
        guide_matrix,
        guide_names,
        min_sim=min_sim,
        min_margin=min_margin,
    )
    centroids = _centroid_prototypes(
        seed_examples, names, kind=kind, parent_intent=parent_intent
    )
    if centroids is None:
        centroid_matches = [None] * len(records)
    else:
        centroid_matches = classify_against(
            embeddings,
            centroids,
            names,
            min_sim=min_sim,
            min_margin=min_margin,
        )

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        guideline = guideline_matches[index]
        centroid = centroid_matches[index]
        if centroid is None:
            label = guideline.pred if guideline.fits and fit_rule in {"guideline", "either"} else None
            rows.append(
                {
                    "conversation_id": record.conversation_id,
                    "assigned": bool(label),
                    "label": label,
                    "confidence": round(float(guideline.sim), 3) if label else 0.0,
                    "centroid_pred": "",
                    "centroid_sim": 0.0,
                    "guideline_pred": guideline.pred,
                    "guideline_sim": round(float(guideline.sim), 3),
                }
            )
            continue
        label = resolve_label(
            centroid.fits,
            centroid.pred,
            centroid.sim,
            guideline.fits,
            guideline.pred,
            guideline.sim,
            fit_rule,
        )
        rows.append(
            {
                "conversation_id": record.conversation_id,
                "assigned": bool(label),
                "label": label,
                "confidence": _winning_confidence(centroid, guideline, label),
                "centroid_pred": centroid.pred,
                "centroid_sim": round(float(centroid.sim), 3),
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
