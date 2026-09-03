"""Fixture and leakage contracts for the Phase 1 demo subset."""

from __future__ import annotations

from collections import defaultdict

import duckdb
import pyarrow.parquet as pq

from playbook.config import FIXTURE_LABELS_PARQUET, WEEKS_PARQUET
from playbook.data import load_fixture_labels, load_manifest, load_week
from playbook.fixtures import WEEK_MEMBERSHIP


def test_only_one_top_level_flow() -> None:
    flows = {label.flow for label in load_fixture_labels()}
    assert flows == {"shipping_issue"}


def test_no_more_than_four_demo_subflows() -> None:
    subflows = {label.subflow for label in load_fixture_labels()}
    assert subflows == {"manage", "cost", "missing", "status"}
    assert len(subflows) <= 4


def test_week2_introduces_exactly_one_new_subflow() -> None:
    by_week: dict[str, set[str]] = defaultdict(set)
    for label in load_fixture_labels():
        by_week[label.week_id].add(label.subflow)
    introduced = by_week["week_2"] - by_week["week_1"]
    assert introduced == {"status"}


def test_week3_introduces_no_new_subflow() -> None:
    by_week: dict[str, set[str]] = defaultdict(set)
    for label in load_fixture_labels():
        by_week[label.week_id].add(label.subflow)
    introduced = by_week["week_3"] - by_week["week_1"] - by_week["week_2"]
    assert introduced == set()


def test_demo_fixture_is_deterministic() -> None:
    first = load_week("week_1")
    second = load_week("week_1")
    assert first.conversation_ids == second.conversation_ids
    expected = sorted(
        str(convo_id)
        for role_ids in WEEK_MEMBERSHIP["week_1"].values()
        for convo_id in role_ids
    )
    assert first.conversation_ids == expected
    manifest = load_manifest()
    assert manifest["week_membership"]["week_1"] == WEEK_MEMBERSHIP["week_1"]


def test_runtime_batch_omits_subflow_labels() -> None:
    batch = load_week("week_1")
    assert batch.conversations
    for record in batch.conversations:
        assert set(record.model_dump()) == {"conversation_id", "turns", "actions"}
        assert "flow" not in record.model_dump()
        assert "subflow" not in record.model_dump()

    runtime_cols = set(pq.read_schema(WEEKS_PARQUET).names)
    assert runtime_cols == {"conversation_id", "week_id", "turns_json", "actions_json"}
    assert "flow" not in runtime_cols
    assert "subflow" not in runtime_cols
    assert "role" not in runtime_cols

    leaked = duckdb.execute(
        """
        SELECT column_name
        FROM (DESCRIBE SELECT * FROM read_parquet(?))
        WHERE lower(column_name) IN ('flow', 'subflow', 'role')
        """,
        [WEEKS_PARQUET.resolve().as_posix()],
    ).fetchall()
    assert leaked == []

    label_cols = set(pq.read_schema(FIXTURE_LABELS_PARQUET).names)
    assert {"flow", "subflow", "role"} <= label_cols


def test_week1_has_recurring_held_out_subflow() -> None:
    labels = [label for label in load_fixture_labels() if label.week_id == "week_1"]
    missing = [label for label in labels if label.subflow == "missing"]
    assert len(missing) >= 2
    assert {label.role for label in missing} == {"C"}
