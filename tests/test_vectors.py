"""Unit tests for DuckDB embedding store."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from playbook.kb import load_playbook
from playbook.vectors import (
    KB_KIND_INTENT,
    VectorStore,
    kb_rows_for_playbook,
    sync_playbook_vectors,
    sync_task_vectors,
    upsert_playbook_change,
)
import conftest as test_conf

fake_embed_texts = test_conf.fake_embed_texts

DATA_DIR = Path(__file__).resolve().parents[1] / "scratch_data" / "eda"


def test_vector_store_kb_query_and_task_matrix(tmp_path):
    playbook = load_playbook(DATA_DIR)
    store = VectorStore(tmp_path / "embeddings.duckdb")
    sync_playbook_vectors(store, playbook, fake_embed_texts)

    rows = kb_rows_for_playbook(playbook)
    intent_row = next(r for r in rows if r[0] == "account_access" and r[1] == KB_KIND_INTENT)
    query_vec = fake_embed_texts([intent_row[2]])[0]
    hits = store.query_kb(KB_KIND_INTENT, query_vec, top_k=2)
    assert hits
    assert hits[0][0] == "account_access"
    assert hits[0][1] > 0.9


def test_upsert_playbook_change_refreshes_intent(tmp_path):
    playbook = load_playbook(DATA_DIR)
    store = VectorStore(tmp_path / "embeddings.duckdb")
    sync_playbook_vectors(store, playbook, fake_embed_texts)
    playbook.add_intent("shipping_issue", "Shipping Issue", "lost packages")
    upsert_playbook_change(store, playbook, fake_embed_texts, intent_ids=["shipping_issue"])
    ids, matrix, _ = store.kb_matrix(KB_KIND_INTENT, ["shipping_issue"])
    assert ids == ["shipping_issue"]
    assert matrix.shape[0] == 1


def test_fit_topic_model_accepts_precomputed_embeddings(monkeypatch):
    captured: dict[str, np.ndarray | None] = {}

    def spy_fit(documents, config, *, embedding_model=None, embeddings=None):
        captured["embeddings"] = embeddings
        from playbook.topics import FittedTopics

        return FittedTopics(topic_ids=[0] * len(documents), descriptors={0: "x"})

    monkeypatch.setattr("playbook.topics.fit_topic_model", spy_fit)
    from playbook.topics import BertopicConfig, discover_topics
    from playbook.schemas import ConversationRecord, Turn

    record = ConversationRecord(
        conversation_id="t1",
        turns=[Turn(speaker="customer", text="package missing")],
        actions=[],
    )
    config = BertopicConfig(
        min_to_cluster=1,
        min_cluster_size=1,
        min_samples=1,
        min_topic_n=1,
        representative_n=1,
        n_neighbors=2,
        n_components=2,
    )
    from playbook import runtime as rt
    from playbook.vectors import VectorStore, sync_task_vectors

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "embeddings.duckdb")
        rt.vectors = store
        rt.embed_fn = fake_embed_texts
        sync_task_vectors(
            store,
            [
                {
                    "task_id": "t1",
                    "turns_json": [{"speaker": "customer", "text": "package missing"}],
                    "text": "package missing",
                    "actions": [],
                }
            ],
            fake_embed_texts,
        )
        discover_topics([record], config=config)
    assert captured["embeddings"] is not None
    assert captured["embeddings"].shape[0] == 1
