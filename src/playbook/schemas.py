"""Shared typed contracts. Freeze after Phase 1 merge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: str
    text: str


class ConversationRecord(BaseModel):
    """Runtime conversation. No ABCD flow/subflow identity."""

    conversation_id: str
    turns: list[Turn]
    actions: list[str]


class WeeklyBatch(BaseModel):
    week_id: str
    conversation_ids: list[str]
    conversations: list[ConversationRecord] = Field(default_factory=list)


class DiscoveredTopic(BaseModel):
    """Run-local topic. IDs are not stable across weeks."""

    topic_id: int
    size: int
    descriptor: str
    representative_conversation_ids: list[str]


class RetrievedKB(BaseModel):
    id: str
    title: str
    guideline: str
    score: float | None = None


class TopicCoverage(BaseModel):
    topic_id: int
    retrieved_kb: list[RetrievedKB]
    judgment: Literal["covered", "uncovered", "insufficient"]
    rationale: str


class KBEntry(BaseModel):
    id: str
    title: str
    guideline: str


class ReviewProposal(BaseModel):
    action: Literal["add"] = "add"
    label: str
    synthesized_guideline: str
    evidence_summary: str
    representative_conversation_ids: list[str]
    nearest_kb_ids: list[str]
    rationale: str


class RunResult(BaseModel):
    week_id: str
    thread_id: str
    decision: Literal["propose_add", "no_action"]
    discovered_topics: list[DiscoveredTopic]
    coverage_judgments: list[TopicCoverage]
    proposal: ReviewProposal | None = None
    findings: list[str] = Field(default_factory=list)
    interrupted: bool = False


class ReviewDecision(BaseModel):
    status: Literal["approve", "edit_approve", "reject"]
    edited_guideline: str | None = None


class FixtureLabel(BaseModel):
    """Offline / eval only. Never imported by BERTopic or runtime prompts."""

    conversation_id: str
    week_id: str
    flow: str
    subflow: str
    role: Literal["A", "B", "C", "D"]


class HeldOutGuideline(BaseModel):
    """Eval truth. Not runtime RAG and not BERTopic input."""

    hidden_subflow: str
    canonical_abcd_text: str
