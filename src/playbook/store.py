"""SQLite working store for cohort, labels, proposals, and recommendations."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


class TaskStore:
    """Process-local SQLite store.

    LangGraph tools can run on worker threads, so each thread gets its own
    connection to the same file. WAL keeps concurrent reads from colliding.
    """

    def __init__(self, path: Path):
        path = Path(path)
        _remove_sqlite_files(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                actions_json TEXT NOT NULL,
                success INTEGER NOT NULL,
                hidden_flow TEXT,
                hidden_subflow TEXT,
                turns_json TEXT NOT NULL,
                conversation_date TEXT
            );
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                cohort_query TEXT NOT NULL,
                as_of TEXT NOT NULL,
                kb_version INTEGER NOT NULL
            );
            CREATE TABLE cohort (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (run_id, task_id)
            );
            CREATE TABLE worksets (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                task_id TEXT NOT NULL
            );
            CREATE TABLE staging (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (run_id, stage)
            );
            CREATE TABLE intent_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                method TEXT NOT NULL,
                kb_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE subflow_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                subflow_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                method TEXT NOT NULL,
                kb_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                proposal_type TEXT NOT NULL,
                candidate TEXT,
                parent_intent TEXT,
                supporting_task_ids TEXT NOT NULL,
                examples TEXT NOT NULL,
                metrics TEXT NOT NULL,
                review_decision TEXT,
                review_note TEXT,
                resulting_kb_version INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE recommendations (
                rec_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                subflow_id TEXT NOT NULL,
                supporting_task_ids TEXT NOT NULL,
                kb_draft TEXT NOT NULL,
                guideline_draft TEXT NOT NULL,
                evaluation TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE kb_log (
                kb_version INTEGER PRIMARY KEY,
                change_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def load_tasks(self, convos: list[dict[str, Any]]) -> None:
        rows = []
        for convo in convos:
            actions = [t["text"] for t in convo["original"] if t["speaker"] == "action"]
            text = " ".join(t["text"] for t in convo["original"])
            rows.append(
                (
                    convo["convo_id"],
                    text,
                    json.dumps(actions),
                    int(bool(convo.get("success"))),
                    convo.get("scenario", {}).get("flow"),
                    convo.get("scenario", {}).get("subflow"),
                    json.dumps(convo["original"]),
                    convo.get("conversation_date"),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO tasks
            (task_id, text, actions_json, success, hidden_flow, hidden_subflow, turns_json, conversation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def create_run(self, run_id: str, cohort_query: dict[str, Any], kb_version: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, cohort_query, as_of, kb_version) VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(cohort_query), now_iso(), kb_version),
        )
        self.conn.commit()

    def set_run_kb_version(self, run_id: str, kb_version: int) -> None:
        self.conn.execute("UPDATE runs SET kb_version = ? WHERE run_id = ?", (kb_version, run_id))
        self.conn.commit()

    def add_cohort(self, run_id: str, task_ids: list[str]) -> None:
        self.conn.execute("DELETE FROM cohort WHERE run_id = ?", (run_id,))
        self.conn.executemany(
            "INSERT INTO cohort (run_id, task_id) VALUES (?, ?)",
            [(run_id, tid) for tid in task_ids],
        )
        self.conn.commit()

    def cohort_ids(self, run_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT task_id FROM cohort WHERE run_id = ? ORDER BY task_id", (run_id,)
        ).fetchall()
        return [r["task_id"] for r in rows]

    def get_tasks(self, task_ids: list[str]) -> list[dict[str, Any]]:
        if not task_ids:
            return []
        placeholders = ",".join("?" * len(task_ids))
        rows = self.conn.execute(
            f"SELECT * FROM tasks WHERE task_id IN ({placeholders})",
            task_ids,
        ).fetchall()
        by_id = {r["task_id"]: dict(r) | {"actions": json.loads(r["actions_json"])} for r in rows}
        return [by_id[tid] for tid in task_ids if tid in by_id]

    def set_workset(self, run_id: str, stage: str, task_ids: list[str]) -> None:
        self.conn.execute("DELETE FROM worksets WHERE run_id = ? AND stage = ?", (run_id, stage))
        self.conn.executemany(
            "INSERT INTO worksets (run_id, stage, task_id) VALUES (?, ?, ?)",
            [(run_id, stage, tid) for tid in task_ids],
        )
        self.conn.commit()

    def get_workset(self, run_id: str, stage: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT task_id FROM worksets WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchall()
        return [r["task_id"] for r in rows]

    def set_staging(self, run_id: str, stage: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO staging (run_id, stage, payload) VALUES (?, ?, ?)",
            (run_id, stage, json.dumps(payload)),
        )
        self.conn.commit()

    def get_staging(self, run_id: str, stage: str) -> Any:
        row = self.conn.execute(
            "SELECT payload FROM staging WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def persist_intent_labels(self, rows: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO intent_labels
            (task_id, run_id, intent_id, confidence, method, kb_version, created_at)
            VALUES (:task_id, :run_id, :intent_id, :confidence, :method, :kb_version, :created_at)
            """,
            rows,
        )
        self.conn.commit()

    def persist_subflow_labels(self, rows: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """
            INSERT INTO subflow_labels
            (task_id, run_id, intent_id, subflow_id, confidence, method, kb_version, created_at)
            VALUES (:task_id, :run_id, :intent_id, :subflow_id, :confidence, :method, :kb_version, :created_at)
            """,
            rows,
        )
        self.conn.commit()

    def latest_intents(self, run_id: str) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT l.* FROM intent_labels l
            JOIN (
                SELECT task_id, MAX(id) AS max_id
                FROM intent_labels WHERE run_id = ? GROUP BY task_id
            ) m ON l.id = m.max_id
            """,
            (run_id,),
        ).fetchall()
        return {r["task_id"]: r for r in rows}

    def latest_subflows(self, run_id: str) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT l.* FROM subflow_labels l
            JOIN (
                SELECT task_id, MAX(id) AS max_id
                FROM subflow_labels WHERE run_id = ? GROUP BY task_id
            ) m ON l.id = m.max_id
            """,
            (run_id,),
        ).fetchall()
        return {r["task_id"]: r for r in rows}

    def unresolved_intent_ids(self, run_id: str) -> list[str]:
        ids = []
        latest = self.latest_intents(run_id)
        for tid in self.cohort_ids(run_id):
            row = latest.get(tid)
            if row is None or row["intent_id"] == "unknown":
                ids.append(tid)
        return ids

    def unresolved_subflow_ids(self, run_id: str, intent_id: str | None = None) -> list[str]:
        ids = []
        intents = self.latest_intents(run_id)
        subflows = self.latest_subflows(run_id)
        for tid in self.cohort_ids(run_id):
            intent_row = intents.get(tid)
            if intent_row is None or intent_row["intent_id"] == "unknown":
                continue
            if intent_id and intent_row["intent_id"] != intent_id:
                continue
            sub_row = subflows.get(tid)
            if sub_row is None or sub_row["subflow_id"] == "unknown":
                ids.append(tid)
        return ids

    def persist_proposal(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO proposals (
                proposal_id, run_id, proposal_type, candidate, parent_intent,
                supporting_task_ids, examples, metrics, review_decision, review_note,
                resulting_kb_version, created_at
            ) VALUES (
                :proposal_id, :run_id, :proposal_type, :candidate, :parent_intent,
                :supporting_task_ids, :examples, :metrics, :review_decision, :review_note,
                :resulting_kb_version, :created_at
            )
            """,
            row,
        )
        self.conn.commit()

    def list_proposals(self, run_id: str, proposal_type: str | None = None) -> list[dict[str, Any]]:
        if proposal_type:
            rows = self.conn.execute(
                "SELECT * FROM proposals WHERE run_id = ? AND proposal_type = ? ORDER BY created_at",
                (run_id, proposal_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM proposals WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def decide_proposal(self, proposal_id: str, decision: str, note: str) -> None:
        self.conn.execute(
            "UPDATE proposals SET review_decision = ?, review_note = ? WHERE proposal_id = ?",
            (decision, note, proposal_id),
        )
        self.conn.commit()

    def mark_proposal_kb_version(self, proposal_id: str, kb_version: int) -> None:
        self.conn.execute(
            "UPDATE proposals SET resulting_kb_version = ? WHERE proposal_id = ?",
            (kb_version, proposal_id),
        )
        self.conn.commit()

    def persist_recommendation(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO recommendations (
                rec_id, run_id, intent_id, subflow_id, supporting_task_ids,
                kb_draft, guideline_draft, evaluation, created_at
            ) VALUES (
                :rec_id, :run_id, :intent_id, :subflow_id, :supporting_task_ids,
                :kb_draft, :guideline_draft, :evaluation, :created_at
            )
            """,
            row,
        )
        self.conn.commit()

    def list_recommendations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM recommendations WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def log_kb_change(self, kb_version: int, change_type: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO kb_log (kb_version, change_type, payload, created_at) VALUES (?, ?, ?, ?)",
            (kb_version, change_type, json.dumps(payload), now_iso()),
        )
        self.conn.commit()

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]
