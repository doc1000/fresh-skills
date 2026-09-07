"""DuckDB-backed KB and task embeddings for cosine retrieve / classify."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from playbook.data import conversation_document
from playbook.schemas import ConversationRecord, Turn

if TYPE_CHECKING:
    from playbook.kb import PlaybookKB

KB_KIND_INTENT = "intent"
KB_KIND_INTENT_GUIDE = "intent_guide"
KB_KIND_SUBFLOW = "subflow"


def _embedding_to_list(vec: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(vec).ravel()]


def _list_to_embedding(raw: list[float] | str) -> np.ndarray:
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.asarray(raw, dtype=float)


class VectorStore:
    """Thread-local DuckDB file for frozen + incremental embeddings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = duckdb.connect(str(self.path))
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._connect().execute(
            """
            CREATE TABLE IF NOT EXISTS kb_docs (
                doc_id VARCHAR NOT NULL,
                kind VARCHAR NOT NULL,
                text VARCHAR NOT NULL,
                embedding DOUBLE[] NOT NULL,
                PRIMARY KEY (doc_id, kind)
            );
            CREATE TABLE IF NOT EXISTS task_docs (
                task_id VARCHAR PRIMARY KEY,
                text VARCHAR NOT NULL,
                embedding DOUBLE[] NOT NULL
            );
            """
        )

    def upsert_kb(self, rows: Sequence[tuple[str, str, str, np.ndarray]]) -> None:
        """Each row: (doc_id, kind, text, embedding)."""
        if not rows:
            return
        conn = self._connect()
        for doc_id, kind, text, embedding in rows:
            conn.execute(
                """
                INSERT INTO kb_docs (doc_id, kind, text, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (doc_id, kind) DO UPDATE SET
                    text = excluded.text,
                    embedding = excluded.embedding
                """,
                [doc_id, kind, text, _embedding_to_list(embedding)],
            )

    def upsert_tasks(self, rows: Sequence[tuple[str, str, np.ndarray]]) -> None:
        """Each row: (task_id, text, embedding)."""
        if not rows:
            return
        conn = self._connect()
        for task_id, text, embedding in rows:
            conn.execute(
                """
                INSERT INTO task_docs (task_id, text, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT (task_id) DO UPDATE SET
                    text = excluded.text,
                    embedding = excluded.embedding
                """,
                [task_id, text, _embedding_to_list(embedding)],
            )

    def kb_matrix(
        self,
        kind: str,
        doc_ids: Sequence[str] | None = None,
    ) -> tuple[list[str], np.ndarray, list[str]]:
        conn = self._connect()
        if doc_ids:
            placeholders = ",".join("?" * len(doc_ids))
            rows = conn.execute(
                f"""
                SELECT doc_id, text, embedding FROM kb_docs
                WHERE kind = ? AND doc_id IN ({placeholders})
                ORDER BY doc_id
                """,
                [kind, *doc_ids],
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT doc_id, text, embedding FROM kb_docs
                WHERE kind = ?
                ORDER BY doc_id
                """,
                [kind],
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 0)), []
        ids = [str(r[0]) for r in rows]
        texts = [str(r[1]) for r in rows]
        matrix = np.vstack([_list_to_embedding(r[2]) for r in rows])
        return ids, matrix, texts

    def task_matrix(self, task_ids: Sequence[str]) -> tuple[list[str], np.ndarray]:
        if not task_ids:
            return [], np.zeros((0, 0))
        conn = self._connect()
        placeholders = ",".join("?" * len(task_ids))
        rows = conn.execute(
            f"""
            SELECT task_id, embedding FROM task_docs
            WHERE task_id IN ({placeholders})
            ORDER BY task_id
            """,
            list(task_ids),
        ).fetchall()
        by_id = {str(r[0]): _list_to_embedding(r[1]) for r in rows}
        ordered = [tid for tid in task_ids if tid in by_id]
        if not ordered:
            return [], np.zeros((0, 0))
        return ordered, np.vstack([by_id[tid] for tid in ordered])

    def query_kb(
        self,
        kind: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 4,
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        ids, matrix, texts = self.kb_matrix(kind, doc_ids)
        if not ids:
            return []
        q = np.asarray(query_vector, dtype=float).reshape(1, -1)
        sims = cosine_similarity(q, matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [(ids[i], round(float(sims[i]), 3), texts[i]) for i in order]

    def has_kb(self, doc_id: str, kind: str) -> bool:
        row = self._connect().execute(
            "SELECT 1 FROM kb_docs WHERE doc_id = ? AND kind = ? LIMIT 1",
            [doc_id, kind],
        ).fetchone()
        return row is not None

    def has_task(self, task_id: str) -> bool:
        row = self._connect().execute(
            "SELECT 1 FROM task_docs WHERE task_id = ? LIMIT 1",
            [task_id],
        ).fetchone()
        return row is not None


def kb_doc_id_subflow(intent_id: str, subflow_id: str) -> str:
    return f"{intent_id}:{subflow_id}"


def kb_rows_for_playbook(
    playbook: PlaybookKB,
    *,
    intent_ids: Sequence[str] | None = None,
    subflow_pairs: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Return (doc_id, kind, text) rows to embed."""
    rows: list[tuple[str, str, str]] = []
    intents = list(intent_ids) if intent_ids is not None else playbook.intent_ids()
    for intent_id in intents:
        rows.append((intent_id, KB_KIND_INTENT, playbook.intent_doc(intent_id)))
        rows.append((intent_id, KB_KIND_INTENT_GUIDE, playbook.guideline_intent_text(intent_id)))
    if subflow_pairs is not None:
        pairs = list(subflow_pairs)
    elif intent_ids is not None:
        pairs = [
            (intent_id, subflow_id)
            for intent_id in intents
            for subflow_id in playbook.subflows_for(intent_id)
        ]
    else:
        pairs = [
            (intent_id, subflow_id)
            for intent_id in playbook.intent_ids()
            for subflow_id in playbook.subflows_for(intent_id)
        ]
    for intent_id, subflow_id in pairs:
        doc_id = kb_doc_id_subflow(intent_id, subflow_id)
        rows.append((doc_id, KB_KIND_SUBFLOW, playbook.subflow_doc(intent_id, subflow_id)))
    return rows


def task_text_from_store_row(task: dict[str, Any]) -> str:
    raw_turns = task.get("turns_json")
    if isinstance(raw_turns, str):
        raw_turns = json.loads(raw_turns)
    turns = [Turn.model_validate(turn) for turn in raw_turns or []]
    if not turns and task.get("text"):
        turns = [Turn(speaker="customer", text=str(task["text"]))]
    record = ConversationRecord(
        conversation_id=str(task["task_id"]),
        turns=turns,
        actions=list(task.get("actions") or []),
    )
    return conversation_document(record)


def sync_playbook_vectors(
    store: VectorStore,
    playbook: PlaybookKB,
    embed_fn: Callable[..., np.ndarray],
) -> None:
    """Upsert any missing KB rows from the live playbook."""
    pending: list[tuple[str, str, str]] = []
    for doc_id, kind, text in kb_rows_for_playbook(playbook):
        if not store.has_kb(doc_id, kind):
            pending.append((doc_id, kind, text))
    if not pending:
        return
    texts = [row[2] for row in pending]
    vectors = embed_fn(texts)
    upsert = [
        (doc_id, kind, text, vectors[i])
        for i, (doc_id, kind, text) in enumerate(pending)
    ]
    store.upsert_kb(upsert)


def sync_task_vectors(
    store: VectorStore,
    tasks: Sequence[dict[str, Any]],
    embed_fn: Callable[..., np.ndarray],
) -> None:
    pending: list[tuple[str, str]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if store.has_task(task_id):
            continue
        pending.append((task_id, task_text_from_store_row(task)))
    if not pending:
        return
    vectors = embed_fn([text for _, text in pending])
    store.upsert_tasks(
        [(task_id, text, vectors[i]) for i, (task_id, text) in enumerate(pending)]
    )


def upsert_playbook_change(
    store: VectorStore,
    playbook: PlaybookKB,
    embed_fn: Callable[..., np.ndarray],
    *,
    intent_ids: Sequence[str] | None = None,
    subflow_pairs: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Embed and upsert KB docs after persist (always refresh changed ids)."""
    rows = kb_rows_for_playbook(
        playbook,
        intent_ids=intent_ids,
        subflow_pairs=subflow_pairs,
    )
    if not rows:
        return
    texts = [row[2] for row in rows]
    vectors = embed_fn(texts)
    store.upsert_kb(
        [
            (doc_id, kind, text, vectors[i])
            for i, (doc_id, kind, text) in enumerate(rows)
        ]
    )
