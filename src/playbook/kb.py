"""Canonical playbook / KB load and retrieval."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from playbook.scoring import jaccard

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "scratch_data" / "eda"

DEFAULT_FLOW_TITLES = {"account_access": "Account Access"}
DEFAULT_SUBFLOW_TITLES = {
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
    "reset_2fa": "Reset Two-Factor Auth",
    "missing": "Missing Item",
}


class PlaybookKB:
    def __init__(
        self,
        ontology: dict,
        kb: dict,
        guidelines: dict,
        version: int = 1,
        flow_titles: dict[str, str] | None = None,
        subflow_titles: dict[str, str] | None = None,
    ):
        self.ontology = deepcopy(ontology)
        self.kb = deepcopy(kb)
        self.guidelines = deepcopy(guidelines)
        self.version = version
        stored_flow_titles = ontology.get("flow_titles") if isinstance(ontology, dict) else None
        stored_subflow_titles = ontology.get("subflow_titles") if isinstance(ontology, dict) else None
        self.flow_titles = dict(flow_titles or stored_flow_titles or DEFAULT_FLOW_TITLES)
        self.subflow_titles = dict(subflow_titles or stored_subflow_titles or DEFAULT_SUBFLOW_TITLES)
        self.title_to_flow_id = {title: flow_id for flow_id, title in self.flow_titles.items()}

    def intent_ids(self) -> list[str]:
        return list(self.ontology["intents"]["flows"])

    def subflows_for(self, intent_id: str) -> list[str]:
        return list(self.ontology["intents"]["subflows"].get(intent_id, []))

    def flow_title(self, intent_id: str) -> str:
        return self.flow_titles.get(intent_id, intent_id.replace("_", " ").title())

    def subflow_title(self, subflow_id: str) -> str:
        return self.subflow_titles.get(subflow_id, subflow_id.replace("_", " ").title())

    def intent_doc(self, intent_id: str) -> str:
        title = self.flow_title(intent_id)
        body = self.guidelines.get(title, {})
        subflow_titles = list(body.get("subflows", {}) or [])
        subflow_ids = self.subflows_for(intent_id)
        return " ".join(
            [title, intent_id, body.get("description", ""), *subflow_titles, *subflow_ids]
        )

    def guideline_intent_text(self, intent_id: str) -> str:
        title = self.flow_title(intent_id)
        desc = (self.guidelines.get(title) or {}).get("description") or ""
        return f"{title}. {desc}".strip()

    def _subflow_node(self, intent_id: str, subflow_id: str) -> dict[str, Any]:
        title = self.flow_title(intent_id)
        sub_title = self.subflow_title(subflow_id)
        return ((self.guidelines.get(title) or {}).get("subflows") or {}).get(sub_title) or {}

    def subflow_doc(self, intent_id: str, subflow_id: str) -> str:
        """Issue-type document for classify. Title and description only — no actions."""
        title = self.flow_title(intent_id)
        sub_title = self.subflow_title(subflow_id)
        node = self._subflow_node(intent_id, subflow_id)
        description = node.get("description") or ""
        instructions = " ".join(node.get("instructions") or [])
        return " ".join(
            part for part in [title, sub_title, subflow_id, description, instructions] if part
        ).strip()

    def guideline_subflow_text(self, intent_id: str, subflow_id: str) -> str:
        title = self.flow_title(intent_id)
        sub_title = self.subflow_title(subflow_id)
        node = self._subflow_node(intent_id, subflow_id)
        instructions = " ".join(node.get("instructions") or [])
        bits: list[str] = []
        for action in node.get("actions") or []:
            bits.append(action.get("text") or "")
            bits.extend(action.get("subtext") or [])
        return f"{title} / {sub_title}. {instructions} {' '.join(bits)}".strip()

    def has_pathway(self, intent_id: str, subflow_id: str) -> bool:
        if self.kb.get(subflow_id):
            return True
        node = self._subflow_node(intent_id, subflow_id)
        return bool(node.get("actions"))

    def add_intent(self, intent_id: str, title: str, description: str) -> int:
        if intent_id not in self.ontology["intents"]["flows"]:
            self.ontology["intents"]["flows"].append(intent_id)
        self.ontology["intents"]["subflows"].setdefault(intent_id, [])
        self.flow_titles[intent_id] = title
        self.title_to_flow_id[title] = intent_id
        self.guidelines.setdefault(title, {"description": description, "subflows": {}})
        self.guidelines[title]["description"] = description
        self.version += 1
        return self.version

    def add_subflow(
        self,
        intent_id: str,
        subflow_id: str,
        actions: list[str] | None = None,
        description: str = "",
    ) -> int:
        existing = self.ontology["intents"]["subflows"].setdefault(intent_id, [])
        if subflow_id not in existing:
            existing.append(subflow_id)
        if actions is not None:
            self.kb[subflow_id] = list(actions)
        else:
            self.kb.setdefault(subflow_id, [])
        if description:
            title = self.flow_title(intent_id)
            self.guidelines.setdefault(title, {"description": "", "subflows": {}})
            self.guidelines[title].setdefault("subflows", {})
            sub_title = self.subflow_titles.setdefault(
                subflow_id, subflow_id.replace("_", " ").title()
            )
            block = self.guidelines[title]["subflows"].setdefault(sub_title, {})
            block["description"] = description
            block.setdefault("instructions", [])
            block.setdefault("actions", [])
        self.version += 1
        return self.version

    def attach_guideline(self, intent_id: str, subflow_id: str, draft: dict[str, Any]) -> int:
        title = self.flow_title(intent_id)
        self.guidelines.setdefault(title, {"description": "", "subflows": {}})
        self.guidelines[title].setdefault("subflows", {})
        sub_title = self.subflow_titles.get(subflow_id, subflow_id.replace("_", " ").title())
        incoming = draft.get(title, {}).get("subflows", {})
        block = incoming.get(sub_title) or next(iter(incoming.values()), {})
        self.guidelines[title]["subflows"][sub_title] = block
        self.version += 1
        return self.version


def load_playbook(data_dir: Path | None = None) -> PlaybookKB:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    ontology = json.loads((root / "seed_ontology.json").read_text(encoding="utf-8"))
    kb = json.loads((root / "seed_kb.json").read_text(encoding="utf-8"))
    guidelines = json.loads((root / "seed_guidelines.json").read_text(encoding="utf-8"))
    return PlaybookKB(
        ontology,
        kb,
        guidelines,
        flow_titles=ontology.get("flow_titles"),
        subflow_titles=ontology.get("subflow_titles"),
    )


def load_conversations(data_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return json.loads((root / "incoming_conversations.json").read_text(encoding="utf-8"))


def _live_playbook(playbook: PlaybookKB | None) -> PlaybookKB:
    if playbook is not None:
        return playbook
    from playbook import runtime

    kb = runtime.playbook
    if kb is None:
        raise RuntimeError("Call configure_runtime() before reading the playbook.")
    return kb


def kb_catalog(playbook: PlaybookKB | None = None, *, include_guidance: bool = False) -> dict[str, Any]:
    """Live taxonomy. Does not read seed files."""
    kb = _live_playbook(playbook)
    intents = []
    for intent_id in kb.intent_ids():
        title = kb.flow_title(intent_id)
        body = kb.guidelines.get(title) or {}
        subflows = []
        for subflow_id in kb.subflows_for(intent_id):
            row: dict[str, Any] = {
                "id": subflow_id,
                "title": kb.subflow_title(subflow_id),
            }
            if include_guidance:
                row["has_pathway"] = kb.has_pathway(intent_id, subflow_id)
                row["guidance"] = kb.guideline_subflow_text(intent_id, subflow_id)
                row["actions"] = list(kb.kb.get(subflow_id) or [])
            subflows.append(row)
        intent_row: dict[str, Any] = {
            "id": intent_id,
            "title": title,
            "description": body.get("description") or "",
            "subflows": subflows,
        }
        if include_guidance:
            intent_row["guidance"] = kb.guideline_intent_text(intent_id)
        intents.append(intent_row)
    return {"kb_version": kb.version, "intents": intents}


def retrieve_guidance(
    query: str,
    playbook: PlaybookKB | None = None,
    *,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Rank playbook intents by cosine similarity to query (vector store when configured)."""
    kb = _live_playbook(playbook)
    from playbook import runtime as rt

    if rt.vectors is not None and rt.embed_fn is not None and query.strip():
        from playbook.vectors import KB_KIND_INTENT

        query_vec = rt.embed_fn([query])[0]
        hits: list[dict[str, Any]] = []
        for intent_id, score, doc in rt.vectors.query_kb(
            KB_KIND_INTENT,
            query_vec,
            top_k=top_k,
            doc_ids=kb.intent_ids(),
        ):
            hits.append(
                {
                    "intent_id": intent_id,
                    "score": score,
                    "doc": doc,
                    "subflows": kb.subflows_for(intent_id),
                }
            )
        return hits

    ranked = []
    for intent_id in kb.intent_ids():
        doc = kb.intent_doc(intent_id)
        ranked.append(
            {
                "intent_id": intent_id,
                "score": round(jaccard(query, doc), 3),
                "doc": doc,
                "subflows": kb.subflows_for(intent_id),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]
