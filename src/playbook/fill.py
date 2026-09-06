"""Lookup ABCD rows and dump Agent KB + incoming tasks. No agent invoke."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from playbook.config import ABCD_JSON, GUIDELINES_JSON
from playbook.kb import DEFAULT_DATA_DIR

EDA_DATA_DIR = DEFAULT_DATA_DIR


def _require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/download_abcd.py")
    return path


def load_abcd(abcd_json: Path | None = None) -> list[dict[str, Any]]:
    path = _require(Path(abcd_json) if abcd_json is not None else ABCD_JSON)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [item for split in raw.values() for item in split]


def load_guidelines(path: Path | None = None) -> dict[str, Any]:
    return json.loads(_require(Path(path) if path is not None else GUIDELINES_JSON).read_text(encoding="utf-8"))


def first_customer(item: dict[str, Any]) -> str:
    for speaker, text in _pairs(item.get("original") or []):
        if speaker == "customer":
            return str(text).strip()
    return ""


def _pairs(original: Sequence[Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for turn in original:
        if isinstance(turn, dict):
            pairs.append((str(turn["speaker"]), str(turn["text"])))
        else:
            pairs.append((str(turn[0]), str(turn[1])))
    return pairs


def to_store_conversation(item: dict[str, Any], *, role: str) -> dict[str, Any]:
    original = [{"speaker": speaker, "text": text} for speaker, text in _pairs(item.get("original") or [])]
    actions = [text for speaker, text in _pairs(item.get("original") or []) if speaker == "action"]
    scenario = dict(item.get("scenario") or {})
    return {
        "convo_id": str(item["convo_id"]),
        "scenario": scenario,
        "original": original,
        "success": bool(item.get("success", True)),
        "role": role,
        "actions": actions,
    }


def button_actions(block: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for action in block.get("actions") or []:
        button = str(action.get("button") or "").strip()
        if not button or button.upper() == "N/A":
            continue
        actions.append(button.lower().replace(" ", "-"))
    return actions


def assign_conversation_dates(
    rows: list[dict[str, Any]],
    *,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """ABCD chats have no timestamps. Stamp synthetic ISO dates across [start, end]."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d < start_d:
        raise ValueError(f"fill_date_start {start} is after fill_date_end {end}")
    span = (end_d - start_d).days
    n = len(rows)
    for i, row in enumerate(rows):
        offset = 0 if n <= 1 else round(i * span / (n - 1))
        row["conversation_date"] = (start_d + timedelta(days=offset)).isoformat()
    return rows


def catalog_abcd(
    *,
    min_opening: int,
    abcd_json: Path | None = None,
) -> list[dict[str, Any]]:
    """Every flow/subflow in ABCD, with counts before/after the opening filter."""
    counts: Counter[tuple[str, str]] = Counter()
    kept: Counter[tuple[str, str]] = Counter()
    for item in load_abcd(abcd_json):
        flow = str((item.get("scenario") or {}).get("flow") or "")
        subflow = str((item.get("scenario") or {}).get("subflow") or "")
        key = (flow, subflow)
        counts[key] += 1
        if len(first_customer(item)) >= min_opening:
            kept[key] += 1
    rows = [
        {
            "flow": flow,
            "subflow": subflow,
            "n": counts[(flow, subflow)],
            "n_after_opening_filter": kept[(flow, subflow)],
        }
        for flow, subflow in sorted(counts)
    ]
    return rows


def pick(
    items: Sequence[dict[str, Any]],
    flow: str,
    subflow: str,
    n: int,
    *,
    min_opening: int,
) -> list[dict[str, Any]]:
    matched = [
        item
        for item in items
        if (item.get("scenario") or {}).get("flow") == flow
        and (item.get("scenario") or {}).get("subflow") == subflow
        and len(first_customer(item)) >= min_opening
    ]
    matched = sorted(matched, key=lambda item: int(item["convo_id"]))
    if len(matched) < n:
        raise ValueError(f"{flow}/{subflow}: {len(matched)} after filter, need {n}")
    return matched[:n]


def current_kb_view(data_dir: Path | None = None) -> dict[str, Any]:
    root = Path(data_dir) if data_dir is not None else EDA_DATA_DIR
    ontology_path = root / "seed_ontology.json"
    if not ontology_path.exists():
        return {"data_dir": str(root), "exists": False, "intents": [], "subflows": {}}
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    return {
        "data_dir": str(root),
        "exists": True,
        "intents": list(ontology.get("intents", {}).get("flows") or []),
        "subflows": dict(ontology.get("intents", {}).get("subflows") or {}),
        "flow_titles": dict(ontology.get("flow_titles") or {}),
        "subflow_titles": dict(ontology.get("subflow_titles") or {}),
    }


def fill_kb_and_tasks(
    *,
    seed_subflows: dict[str, list[str]],
    noise_pairs: Sequence[tuple[str, str]],
    probe_subflow: tuple[str, str],
    probe_intent: tuple[str, str],
    per_seed: int,
    per_probe: int,
    min_opening: int,
    intent_titles: dict[str, str],
    subflow_titles: dict[str, str],
    fill_date_start: str,
    fill_date_end: str,
    data_dir: Path | None = None,
    abcd_json: Path | None = None,
    guidelines_json: Path | None = None,
) -> dict[str, Any]:
    """Write Agent KB files and incoming tasks from the current lever values."""
    root = Path(data_dir) if data_dir is not None else EDA_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    items = load_abcd(abcd_json)
    guidelines = load_guidelines(guidelines_json)

    seed_items: list[dict[str, Any]] = []
    for flow, subflows in seed_subflows.items():
        for subflow in subflows:
            seed_items.extend(pick(items, flow, subflow, per_seed, min_opening=min_opening))
    noise_items = [
        pick(items, flow, subflow, 1, min_opening=min_opening)[0] for flow, subflow in noise_pairs
    ]
    probe_subflow_items = pick(items, *probe_subflow, per_probe, min_opening=min_opening)
    probe_intent_items = pick(items, *probe_intent, per_probe, min_opening=min_opening)

    week_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for role, group in (
        ("seed", seed_items),
        ("noise", noise_items),
        ("probe_subflow", probe_subflow_items),
        ("probe_intent", probe_intent_items),
    ):
        for item in group:
            row = to_store_conversation(item, role=role)
            week_rows.append(row)
            inventory.append(
                {
                    "conversation_id": row["convo_id"],
                    "role": role,
                    "true_flow": row["scenario"].get("flow"),
                    "true_subflow": row["scenario"].get("subflow"),
                    "opening": first_customer(item)[:80],
                }
            )
    assign_conversation_dates(week_rows, start=fill_date_start, end=fill_date_end)
    dates_by_id = {row["convo_id"]: row["conversation_date"] for row in week_rows}
    for row in inventory:
        row["conversation_date"] = dates_by_id[row["conversation_id"]]

    ontology = {
        "intents": {
            "flows": list(seed_subflows),
            "subflows": {flow: list(subflows) for flow, subflows in seed_subflows.items()},
        },
        "flow_titles": {flow: intent_titles[flow] for flow in seed_subflows},
        "subflow_titles": {
            subflow: subflow_titles[subflow]
            for subflows in seed_subflows.values()
            for subflow in subflows
        },
        "notes": "Filled from ABCD for the current notebook levers. Not a hidden default.",
    }
    seed_guidelines: dict[str, Any] = {}
    seed_kb: dict[str, list[str]] = {}
    missing_titles: list[str] = []
    for flow, subflows in seed_subflows.items():
        title = intent_titles[flow]
        source = guidelines.get(title)
        if source is None:
            missing_titles.append(title)
            continue
        block = {"description": source.get("description") or "", "subflows": {}}
        for subflow in subflows:
            sub_title = subflow_titles[subflow]
            sub_block = (source.get("subflows") or {}).get(sub_title)
            if sub_block is None:
                missing_titles.append(f"{title}/{sub_title}")
                continue
            block["subflows"][sub_title] = sub_block
            seed_kb[subflow] = button_actions(sub_block)
        seed_guidelines[title] = block
    if missing_titles:
        raise KeyError(f"guidelines missing titles: {missing_titles}")

    incoming = [
        {
            "convo_id": row["convo_id"],
            "scenario": row["scenario"],
            "original": row["original"],
            "success": row["success"],
            "conversation_date": row["conversation_date"],
        }
        for row in week_rows
    ]
    seed_examples = [
        {
            "convo_id": row["convo_id"],
            "scenario": row["scenario"],
            "original": row["original"],
            "success": row["success"],
        }
        for row in week_rows
        if row["role"] == "seed"
    ]

    (root / "seed_ontology.json").write_text(json.dumps(ontology, indent=2) + "\n", encoding="utf-8")
    (root / "seed_kb.json").write_text(json.dumps(seed_kb, indent=2) + "\n", encoding="utf-8")
    (root / "seed_guidelines.json").write_text(json.dumps(seed_guidelines, indent=2) + "\n", encoding="utf-8")
    (root / "incoming_conversations.json").write_text(json.dumps(incoming, indent=2) + "\n", encoding="utf-8")
    (root / "seed_examples.json").write_text(json.dumps(seed_examples, indent=2) + "\n", encoding="utf-8")
    (root / "fill_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    counts = Counter(row["role"] for row in inventory)
    dates = [row["conversation_date"] for row in week_rows]
    return {
        "data_dir": str(root),
        "n_tasks": len(incoming),
        "n_seed_examples": len(seed_examples),
        "fill_date_start": fill_date_start,
        "fill_date_end": fill_date_end,
        "min_conversation_date": min(dates) if dates else None,
        "max_conversation_date": max(dates) if dates else None,
        "role_counts": dict(counts),
        "intents_in_kb": ontology["intents"]["flows"],
        "subflows_in_kb": ontology["intents"]["subflows"],
        "files": [
            "seed_ontology.json",
            "seed_kb.json",
            "seed_guidelines.json",
            "incoming_conversations.json",
            "seed_examples.json",
            "fill_inventory.json",
        ],
        "inventory": inventory,
    }
