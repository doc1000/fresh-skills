"""Action-path discovery. Independent of topic clustering and notebook orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from playbook.schemas import ConversationRecord


class ActionPath(BaseModel):
    actions: list[str]
    count: int
    conversation_ids: list[str]


class ActionPathResult(BaseModel):
    n_conversations: int
    n_unique_paths: int
    paths: list[ActionPath]
    button_counts: dict[str, int]


def discover_action_paths(conversations: Sequence[ConversationRecord]) -> ActionPathResult:
    """Count exact action sequences and per-button presence. No BERTopic."""
    records = list(conversations)
    members: dict[tuple[str, ...], list[str]] = defaultdict(list)
    buttons: Counter[str] = Counter()
    for record in records:
        path = tuple(record.actions)
        members[path].append(record.conversation_id)
        buttons.update(set(path))
    paths = [
        ActionPath(actions=list(actions), count=len(ids), conversation_ids=ids)
        for actions, ids in sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    return ActionPathResult(
        n_conversations=len(records),
        n_unique_paths=len(paths),
        paths=paths,
        button_counts=dict(buttons.most_common()),
    )
