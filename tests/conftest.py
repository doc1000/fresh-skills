"""Shared fixtures for agent + DS integration tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from pathlib import Path

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


def fake_embed_texts(texts, *, embedding_model="all-MiniLM-L6-v2"):
    """Deterministic vectors so bertopic classify tests do not download a model."""
    rows = []
    for text in texts:
        vec = np.zeros(16, dtype=float)
        lowered = str(text).lower()
        if "username" in lowered:
            vec[0] = 1.0
        elif "password" in lowered:
            vec[1] = 1.0
        elif "two-factor" in lowered or "authenticator" in lowered:
            vec[2] = 1.0
        elif "package" in lowered or "shipment" in lowered or "delivered" in lowered:
            vec[3] = 1.0
        elif "locked" in lowered or "lockout" in lowered:
            vec[4] = 1.0
        else:
            vec[abs(hash(lowered)) % 16] = 1.0
        rows.append(vec)
    return np.asarray(rows)


@pytest.fixture
def stub_topic_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("playbook.topics.fit_topic_model", content_based_fit)
    monkeypatch.setattr("playbook.topics.embed_texts", fake_embed_texts)
    monkeypatch.setattr("playbook.adapters.embed_texts", fake_embed_texts)


DEMO_TOPIC_CONFIG = BertopicConfig(
    min_to_cluster=2,
    min_cluster_size=2,
    min_samples=1,
    min_topic_n=2,
    representative_n=3,
    n_neighbors=3,
    n_components=2,
)


@pytest.fixture
def runtime(tmp_path):
    from playbook import configure_runtime

    return configure_runtime(
        data_dir=Path(__file__).resolve().parents[1] / "scratch_data" / "eda",
        store_path=tmp_path / "run_store.sqlite",
        vectors_path=tmp_path / "embeddings.duckdb",
        method="jaccard",
        topic_config=DEMO_TOPIC_CONFIG,
        embed_fn_override=fake_embed_texts,
        load_env=False,
    )
