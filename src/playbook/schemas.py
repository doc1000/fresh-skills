"""Shared typed contracts between the store, the topic models, and the graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: str
    text: str


class ConversationRecord(BaseModel):
    """Runtime conversation. No ABCD flow/subflow identity."""

    conversation_id: str
    turns: list[Turn]
    actions: list[str]


class DiscoveredTopic(BaseModel):
    """Run-local topic. IDs are not stable across runs."""

    topic_id: int
    size: int
    descriptor: str
    representative_conversation_ids: list[str]
    member_ids: list[str] = Field(default_factory=list)
    cohesive_enough: bool = False


def conversation_document(record: ConversationRecord) -> str:
    """NL document for topic models: customer/agent text only, no action names."""
    return "\n".join(
        turn.text
        for turn in record.turns
        if turn.speaker in {"customer", "agent"} and turn.text
    )
