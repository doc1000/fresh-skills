"""Build deterministic demo Parquet + hidden labels from pinned ABCD IDs."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playbook.config import (  # noqa: E402
    ABCD_JSON,
    DATA_DEMO,
    DATA_REFERENCE,
    DEMO_MANIFEST_JSON,
    FIXTURE_LABELS_PARQUET,
    WEEKS_PARQUET,
)
from playbook.data import parse_raw_conversation  # noqa: E402
from playbook.fixtures import (  # noqa: E402
    DEMO_FLOW,
    ROLE_TO_SUBFLOW,
    WEEK_MEMBERSHIP,
    all_pinned_ids,
)


def _index_abcd() -> dict[int, dict]:
    if not ABCD_JSON.exists():
        raise FileNotFoundError(f"Missing {ABCD_JSON}. Run: python scripts/download_abcd.py")
    raw = json.loads(ABCD_JSON.read_text(encoding="utf-8"))
    by_id: dict[int, dict] = {}
    for split in raw.values():
        for convo in split:
            by_id[int(convo["convo_id"])] = convo
    return by_id


def main() -> None:
    by_id = _index_abcd()
    DATA_DEMO.mkdir(parents=True, exist_ok=True)
    DATA_REFERENCE.mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict] = []
    label_rows: list[dict] = []
    seen: set[int] = set()

    for week_id, roles in WEEK_MEMBERSHIP.items():
        for role, convo_ids in roles.items():
            expected_subflow = ROLE_TO_SUBFLOW[role]
            for convo_id in convo_ids:
                if convo_id in seen:
                    raise ValueError(f"duplicate pinned id {convo_id}")
                seen.add(convo_id)
                if convo_id not in by_id:
                    raise KeyError(f"pinned convo_id {convo_id} not in ABCD")
                raw = by_id[convo_id]
                flow = raw["scenario"]["flow"]
                subflow = raw["scenario"]["subflow"]
                if flow != DEMO_FLOW:
                    raise ValueError(f"{convo_id} flow={flow}, expected {DEMO_FLOW}")
                if subflow != expected_subflow:
                    raise ValueError(
                        f"{convo_id} subflow={subflow}, expected {expected_subflow} for role {role}"
                    )
                record = parse_raw_conversation(raw)
                runtime_rows.append(
                    {
                        "conversation_id": record.conversation_id,
                        "week_id": week_id,
                        "turns_json": json.dumps([turn.model_dump() for turn in record.turns]),
                        "actions_json": json.dumps(record.actions),
                    }
                )
                label_rows.append(
                    {
                        "conversation_id": record.conversation_id,
                        "week_id": week_id,
                        "flow": flow,
                        "subflow": subflow,
                        "role": role,
                    }
                )

    pq.write_table(pa.Table.from_pylist(runtime_rows), WEEKS_PARQUET)
    pq.write_table(pa.Table.from_pylist(label_rows), FIXTURE_LABELS_PARQUET)

    week_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in label_rows:
        week_counts[row["week_id"]][row["role"]] = week_counts[row["week_id"]].get(row["role"], 0) + 1

    manifest = {
        "flow": DEMO_FLOW,
        "role_to_subflow": ROLE_TO_SUBFLOW,
        "week_membership": WEEK_MEMBERSHIP,
        "week_counts": {week: dict(counts) for week, counts in week_counts.items()},
        "n_conversations": len(all_pinned_ids()),
        "runtime_parquet": str(WEEKS_PARQUET.relative_to(ROOT)).replace("\\", "/"),
        "labels_parquet": str(FIXTURE_LABELS_PARQUET.relative_to(ROOT)).replace("\\", "/"),
        "notes": (
            "Runtime parquet has no flow/subflow. Labels are offline-only. "
            "BERTopic topic IDs are not stored here and are not stable across weeks."
        ),
    }
    DEMO_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {WEEKS_PARQUET}")
    print(f"wrote {FIXTURE_LABELS_PARQUET}")
    print(f"wrote {DEMO_MANIFEST_JSON}")
    print("week counts", dict(week_counts))


if __name__ == "__main__":
    main()
