"""Fill writes Agent KB + tasks from explicit seed levers."""

from __future__ import annotations

import json
from pathlib import Path

from playbook.fill import catalog_abcd, fill_kb_and_tasks, to_store_conversation


def _convo(convo_id: int, flow: str, subflow: str, opening: str) -> dict:
    return {
        "convo_id": convo_id,
        "scenario": {"flow": flow, "subflow": subflow},
        "original": [
            ["customer", opening],
            ["agent", "I can help with that."],
            ["action", "pull-up-account"],
        ],
        "success": True,
    }


def _abcd_bundle(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    convo_id = 1
    specs = [
        ("account_access", "recover_username", "I forgot my username and cannot log into my account today."),
        ("account_access", "recover_password", "I need to reset my password I know my username already."),
        ("order_issue", "status_payment_method", "Why was I charged a mystery payment method fee on this order."),
        ("troubleshoot_site", "slow_speed", "The website is loading extremely slowly and I cannot check out."),
        ("shipping_issue", "missing", "My package is missing from the porch after it was marked delivered."),
    ]
    for flow, subflow, opening in specs:
        for _ in range(3):
            rows.append(_convo(convo_id, flow, subflow, opening))
            convo_id += 1
    abcd_path = tmp_path / "abcd.json"
    abcd_path.write_text(json.dumps({"train": rows}), encoding="utf-8")
    guidelines = {
        "Account Access": {
            "description": "login problems",
            "subflows": {
                "Recover Username": {
                    "actions": [{"button": "Pull up Account", "text": "load the user", "subtext": []}],
                    "instructions": ["Get the username."],
                },
                "Recover Password": {
                    "actions": [
                        {"button": "Pull up Account", "text": "load", "subtext": []},
                        {"button": "Make Password", "text": "reset", "subtext": []},
                    ],
                    "instructions": ["Make a new password."],
                },
            },
        }
    }
    guidelines_path = tmp_path / "guidelines.json"
    guidelines_path.write_text(json.dumps(guidelines), encoding="utf-8")
    return abcd_path, guidelines_path


def test_fill_writes_kb_and_tasks_without_invoking_agent(tmp_path: Path) -> None:
    abcd_path, guidelines_path = _abcd_bundle(tmp_path)
    out_dir = tmp_path / "eda"
    result = fill_kb_and_tasks(
        seed_subflows={"account_access": ["recover_username", "recover_password"]},
        noise_pairs=[("shipping_issue", "missing")],
        probe_subflow=("order_issue", "status_payment_method"),
        probe_intent=("troubleshoot_site", "slow_speed"),
        per_seed=2,
        per_probe=2,
        min_opening=25,
        intent_titles={"account_access": "Account Access"},
        subflow_titles={
            "recover_username": "Recover Username",
            "recover_password": "Recover Password",
        },
        data_dir=out_dir,
        abcd_json=abcd_path,
        guidelines_json=guidelines_path,
        fill_date_start="2026-08-25",
        fill_date_end="2026-09-14",
    )
    ontology = json.loads((out_dir / "seed_ontology.json").read_text(encoding="utf-8"))
    incoming = json.loads((out_dir / "incoming_conversations.json").read_text(encoding="utf-8"))
    examples = json.loads((out_dir / "seed_examples.json").read_text(encoding="utf-8"))
    kb = json.loads((out_dir / "seed_kb.json").read_text(encoding="utf-8"))
    assert ontology["intents"]["flows"] == ["account_access"]
    assert ontology["intents"]["subflows"]["account_access"] == ["recover_username", "recover_password"]
    assert "reset_2fa" not in ontology["intents"]["subflows"]["account_access"]
    assert result["n_tasks"] == 2 * 2 + 1 + 2 + 2
    assert result["n_seed_examples"] == 4
    assert len(incoming) == result["n_tasks"]
    assert len(examples) == 4
    assert incoming[0]["original"][0] == {"speaker": "customer", "text": incoming[0]["original"][0]["text"]}
    assert "scenario" not in json.dumps(kb)
    assert kb["recover_username"] == ["pull-up-account"]
    assert all("conversation_date" in row for row in incoming)
    assert incoming[0]["conversation_date"] == "2026-08-25"
    assert incoming[-1]["conversation_date"] == "2026-09-14"
    assert result["min_conversation_date"] == "2026-08-25"
    catalog = catalog_abcd(min_opening=25, abcd_json=abcd_path)
    assert any(row["flow"] == "account_access" and row["n"] == 3 for row in catalog)


def test_store_conversation_drops_tuple_turns() -> None:
    row = to_store_conversation(
        _convo(9, "account_access", "recover_username", "I forgot my username and cannot log in."),
        role="seed",
    )
    assert row["original"][0]["speaker"] == "customer"
    assert row["role"] == "seed"
