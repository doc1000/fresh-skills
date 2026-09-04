"""DuckDB-backed demo loaders. Runtime paths omit ABCD flow/subflow."""

from __future__ import annotations

import json

import duckdb

from playbook.config import DEMO_MANIFEST_JSON, FIXTURE_LABELS_PARQUET, WEEKS_PARQUET
from playbook.schemas import ConversationRecord, FixtureLabel, Turn, WeeklyBatch


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _parquet_path(path) -> str:
    return path.resolve().as_posix()


def parse_raw_conversation(raw: dict) -> ConversationRecord:
    turns = [Turn(speaker=speaker, text=text) for speaker, text in raw["original"]]
    actions: list[str] = []
    for item in raw.get("delexed") or []:
        if item.get("speaker") != "action":
            continue
        targets = item.get("targets") or []
        if len(targets) > 2 and targets[2]:
            actions.append(str(targets[2]))
    return ConversationRecord(
        conversation_id=str(raw["convo_id"]),
        turns=turns,
        actions=actions,
    )


def conversation_document(record: ConversationRecord) -> str:
    """NL document for topic models: original customer/agent text only, no action names."""
    return "\n".join(
        turn.text for turn in record.turns if turn.speaker in {"customer", "agent"} and turn.text
    )


def load_week(week_id: str) -> WeeklyBatch:
    if not WEEKS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {WEEKS_PARQUET}. Run: python scripts/prepare_demo.py"
        )
    con = _connect()
    rows = con.execute(
        """
        SELECT conversation_id, turns_json, actions_json
        FROM read_parquet(?)
        WHERE week_id = ?
        ORDER BY conversation_id
        """,
        [_parquet_path(WEEKS_PARQUET), week_id],
    ).fetchall()
    conversations = [
        ConversationRecord(
            conversation_id=conversation_id,
            turns=[Turn.model_validate(turn) for turn in json.loads(turns_json)],
            actions=json.loads(actions_json),
        )
        for conversation_id, turns_json, actions_json in rows
    ]
    return WeeklyBatch(
        week_id=week_id,
        conversation_ids=[record.conversation_id for record in conversations],
        conversations=conversations,
    )


def load_fixture_labels() -> list[FixtureLabel]:
    """Offline labels for fixture assertions and eval. Do not pass into BERTopic."""
    if not FIXTURE_LABELS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {FIXTURE_LABELS_PARQUET}. Run: python scripts/prepare_demo.py"
        )
    con = _connect()
    rows = con.execute(
        """
        SELECT conversation_id, week_id, flow, subflow, role
        FROM read_parquet(?)
        ORDER BY week_id, conversation_id
        """,
        [_parquet_path(FIXTURE_LABELS_PARQUET)],
    ).fetchall()
    return [
        FixtureLabel(
            conversation_id=conversation_id,
            week_id=week_id,
            flow=flow,
            subflow=subflow,
            role=role,
        )
        for conversation_id, week_id, flow, subflow, role in rows
    ]


def load_manifest() -> dict:
    return json.loads(DEMO_MANIFEST_JSON.read_text(encoding="utf-8"))
