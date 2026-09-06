"""Explicit LangChain tools for scoring and stub drafts."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from playbook import runtime
from playbook.scoring import jaccard


@tool
def score_intent_similarity(conversation_id: str, intent_id: str) -> float:
    """Stub semantic similarity of a full interaction against an existing intent."""
    task = runtime.store.get_tasks([conversation_id])[0]
    return round(jaccard(task["text"], runtime.playbook.intent_doc(intent_id)), 3)


@tool
def score_subflow_similarity(conversation_id: str, subflow_id: str, intent_id: str = "") -> float:
    """Stub semantic similarity of a conversation against an existing subflow."""
    task = runtime.store.get_tasks([conversation_id])[0]
    parent = intent_id
    if not parent:
        for flow_id in runtime.playbook.intent_ids():
            if subflow_id in runtime.playbook.subflows_for(flow_id):
                parent = flow_id
                break
    return round(jaccard(task["text"], runtime.playbook.subflow_doc(parent, subflow_id)), 3)


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
