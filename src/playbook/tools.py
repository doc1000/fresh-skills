"""Explicit LangChain tools for scoring, clustering, and stub drafts."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from playbook import runtime
from playbook.scoring import CLUSTER_THRESHOLD, action_sequence_score, jaccard


@tool
def score_intent_similarity(conversation_id: str, intent_id: str) -> float:
    """Stub semantic similarity of a full interaction against an existing intent."""
    task = runtime.store.get_tasks([conversation_id])[0]
    return round(jaccard(task["text"], runtime.playbook.intent_doc(intent_id)), 3)


@tool
def score_subflow_similarity(conversation_id: str, subflow_id: str) -> float:
    """Stub subflow match: action-sequence LCS vs kb.json buttons."""
    task = runtime.store.get_tasks([conversation_id])[0]
    return round(action_sequence_score(task["actions"], runtime.playbook.kb.get(subflow_id, [])), 3)


@tool
def cluster_conversation_ids(conversation_ids: list[str]) -> list[list[str]]:
    """Greedy Jaccard clustering over full interaction text. Stands in for BERTopic."""
    remaining = list(conversation_ids)
    tasks = {t["task_id"]: t for t in runtime.store.get_tasks(conversation_ids)}
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_text = tasks[seed]["text"]
        group = [seed]
        kept: list[str] = []
        for other in remaining:
            if jaccard(seed_text, tasks[other]["text"]) >= CLUSTER_THRESHOLD:
                group.append(other)
            else:
                kept.append(other)
        remaining = kept
        clusters.append(group)
    return clusters


@tool
def draft_kb_stub(subflow_id: str, actions: list[str]) -> dict[str, list[str]]:
    """Draft an ABCD-shaped kb.json entry. Pass-through: no model."""
    return {subflow_id: actions}


@tool
def draft_guideline_stub(
    flow_title: str,
    subflow_title: str,
    actions: list[str],
    instructions: list[str],
) -> dict[str, Any]:
    """Draft an ABCD-shaped guidelines.json subflow block. Pass-through: no model."""
    return {
        flow_title: {
            "subflows": {
                subflow_title: {
                    "actions": [
                        {
                            "type": "interaction",
                            "button": action.replace("-", " ").title(),
                            "text": f"Observed repeated action [{action}]",
                            "subtext": [],
                        }
                        for action in actions
                    ],
                    "instructions": instructions,
                }
            }
        }
    }
