"""Shared fixtures for agent + DS integration tests."""

from __future__ import annotations

from typing import Any

import pytest

from playbook.topics import BertopicConfig, FittedTopics


def content_based_fit(documents: list[str], config: BertopicConfig, **_kwargs: Any) -> FittedTopics:
    """Deterministic stand-in for BERTopic. Uses conversation text, not ABCD labels."""
    topic_ids: list[int] = []
    descriptors: dict[int, str] = {}
    for document in documents:
        text = document.lower()
        if any(token in text for token in ("package", "shipment", "delivered", "porch", "carrier")):
            topic_ids.append(0)
            descriptors[0] = "package, missing, delivered"
        elif any(token in text for token in ("two-factor", "authenticator")):
            topic_ids.append(0)
            descriptors[0] = "two-factor, authenticator, reset"
        elif any(token in text for token in ("locked", "lockout")):
            topic_ids.append(1)
            descriptors[1] = "account, locked, lockout"
        else:
            topic_ids.append(-1)
            descriptors.setdefault(-1, "")
    return FittedTopics(topic_ids=topic_ids, descriptors=descriptors)


@pytest.fixture
def stub_topic_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("playbook.topics.fit_topic_model", content_based_fit)
