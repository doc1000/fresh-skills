# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modular meta-agent: control state vs working data
#
# Worktree-local notebook. Behavioral baseline is `scratch_langgraph_flow_working`:
# Jaccard intent match, action-sequence LCS subflow match, greedy clustering, stub
# pathway drafts. This notebook **does not** change that matching math.
#
# What changes is the **graph architecture**:
#
# ```text
# LangGraph state  = orchestration / control state
# Task store       = working data (query → process → persist → summarize)
# KB               = durable learned structure
# LangSmith        = traces / metadata
# ```
#
# LangGraph state contains information required to decide what happens next.
# Detailed task records are queried when needed, persisted externally, and then released.
#
# Conceptual split:
#
# ```text
# CLASSIFY   Match tasks against existing KB structure.
# DISCOVER   Identify and validate missing intents or subflows.
# RECOMMEND  Generate pathway / guideline knowledge for validated subflows.
# ```
#
# Classify and discover stay structured data-science / LLM-assisted workflows.
# Recommend is the agentic seam (still a stub draft here, same as the baseline).

# %% [markdown]
# ## X-ray 0 — system context
#
# ```mermaid
# flowchart LR
#     TS[Task Source] <--> MA[Meta Agent]
#     MA <--> KB[KB]
#     MA --> LS[LangSmith traces]
# ```

# %% [markdown]
# ## X-ray 1 — top-level agent workflow
#
# ```mermaid
# flowchart TD
#     C[Establish Cohort] --> CL[Classify]
#     CL --> D[Discover]
#     D --> R[Recommend]
#     R --> S[Summarize]
# ```

# %% [markdown]
# ## X-ray 2 — shared execution pattern
#
# Every subgraph follows the same loop. Task objects live inside the subgraph only.
#
# ```mermaid
# flowchart LR
#     Q[Query store] --> P[Process]
#     P --> W[Persist]
#     W --> U[Summarize into graph state]
# ```
#
# After a KB mutation, the next classifier **re-queries the store**. It does not
# reuse an in-memory leftover list of unresolved tasks.

# %%
from __future__ import annotations

import json
import os
import operator
import re
import sqlite3
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from IPython.display import Markdown, display, JSON
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langsmith import Client

ROOT = Path(".").resolve()
DATA = ROOT / "scratch_data"
STORE_PATH = DATA / "run_store.sqlite"
ENV_PATH = ROOT / ".env"

INTENT_THRESHOLD = 0.07
SUBFLOW_THRESHOLD = 0.75
CLUSTER_THRESHOLD = 0.20
MIN_CLUSTER_SIZE = 2
MIN_SUCCESS_FOR_PATTERN = 2
PATTERN_SUPPORT = 0.66
LOW_INTENT_BAND = 0.09
LOW_SUBFLOW_BAND = 0.85


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(ENV_PATH)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "fresh-skills")

print("LANGSMITH_TRACING:", os.environ.get("LANGSMITH_TRACING"))
print("LANGSMITH_PROJECT:", os.environ.get("LANGSMITH_PROJECT"))
print("LANGSMITH_API_KEY set:", bool(os.environ.get("LANGSMITH_API_KEY")))

# %% [markdown]
# ## Seed playbook
#
# Same incomplete ABCD-shaped seed as the working notebook:
# **Account Access** with `recover_username` and `recover_password` only.
# `reset_2fa` and all of shipping are omitted on purpose.

# %%
ontology = json.loads((DATA / "seed_ontology.json").read_text(encoding="utf-8"))
kb_actions = json.loads((DATA / "seed_kb.json").read_text(encoding="utf-8"))
guidelines = json.loads((DATA / "seed_guidelines.json").read_text(encoding="utf-8"))
conversations = json.loads((DATA / "incoming_conversations.json").read_text(encoding="utf-8"))

print("ontology flows:", ontology["intents"]["flows"])
print("ontology subflows:", ontology["intents"]["subflows"])
print("kb keys:", list(kb_actions))
print("guideline flows:", list(guidelines))
print("incoming ids:", [c["convo_id"] for c in conversations])


# %% [markdown]
# Hidden `scenario.flow` / `scenario.subflow` are **offline labels** for evaluation
# in this notebook. Classifiers and discovery never read them.

# %% [markdown]
# ## Playbook KB (durable structure)
#
# Graph state holds `kb_version` only. Nodes load the current structure from this
# object, and approved HITL decisions mutate it.


# %%
FLOW_TITLES = {"account_access": "Account Access"}
SUBFLOW_TITLES = {
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
    "reset_2fa": "Reset Two-Factor Auth",
    "missing": "Missing Item",
}
TITLE_TO_FLOW_ID = {title: flow_id for flow_id, title in FLOW_TITLES.items()}


class PlaybookKB:
    def __init__(self, ontology: dict, kb: dict, guidelines: dict, version: int = 1):
        self.ontology = deepcopy(ontology)
        self.kb = deepcopy(kb)
        self.guidelines = deepcopy(guidelines)
        self.version = version

    def intent_ids(self) -> list[str]:
        return list(self.ontology["intents"]["flows"])

    def subflows_for(self, intent_id: str) -> list[str]:
        return list(self.ontology["intents"]["subflows"].get(intent_id, []))

    def flow_title(self, intent_id: str) -> str:
        return FLOW_TITLES.get(intent_id, intent_id.replace("_", " ").title())

    def intent_doc(self, intent_id: str) -> str:
        title = self.flow_title(intent_id)
        body = self.guidelines.get(title, {})
        subflow_titles = list(body.get("subflows", {}) or [])
        subflow_ids = self.subflows_for(intent_id)
        return " ".join(
            [title, intent_id, body.get("description", ""), *subflow_titles, *subflow_ids]
        )

    def add_intent(self, intent_id: str, title: str, description: str) -> int:
        if intent_id not in self.ontology["intents"]["flows"]:
            self.ontology["intents"]["flows"].append(intent_id)
        self.ontology["intents"]["subflows"].setdefault(intent_id, [])
        FLOW_TITLES[intent_id] = title
        TITLE_TO_FLOW_ID[title] = intent_id
        self.guidelines.setdefault(title, {"description": description, "subflows": {}})
        self.guidelines[title]["description"] = description
        self.version += 1
        return self.version

    def add_subflow(self, intent_id: str, subflow_id: str, actions: list[str]) -> int:
        existing = self.ontology["intents"]["subflows"].setdefault(intent_id, [])
        if subflow_id not in existing:
            existing.append(subflow_id)
        self.kb[subflow_id] = list(actions)
        self.version += 1
        return self.version

    def attach_guideline(self, intent_id: str, subflow_id: str, draft: dict[str, Any]) -> int:
        title = self.flow_title(intent_id)
        self.guidelines.setdefault(title, {"description": "", "subflows": {}})
        self.guidelines[title].setdefault("subflows", {})
        sub_title = SUBFLOW_TITLES.get(subflow_id, subflow_id.replace("_", " ").title())
        incoming = draft.get(title, {}).get("subflows", {})
        block = incoming.get(sub_title) or next(iter(incoming.values()), {})
        self.guidelines[title]["subflows"][sub_title] = block
        self.version += 1
        return self.version


playbook = PlaybookKB(ontology, kb_actions, guidelines)
print("kb_version", playbook.version)
print("intent docs:", {i: playbook.intent_doc(i) for i in playbook.intent_ids()})

# %%
JSON(playbook)

# %% [markdown]
# ## Task store (working data)
#
# SQLite is the source of truth for cohort membership, labels, proposals, and
# recommendations. Graph nodes query it, persist into it, and drop the detailed
# rows before returning.
#
# Future production source: a date-range query against external task/call data.
# This demo loads the sample JSON once, then treats the store as the cohort backend.


# %%
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, path: Path):
        if path.exists():
            path.unlink()
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

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
                turns_json TEXT NOT NULL
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
                )
            )
        self.conn.executemany(
            """
            INSERT INTO tasks
            (task_id, text, actions_json, success, hidden_flow, hidden_subflow, turns_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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
        return json.loads(row["payload"]) if row else []

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


store = TaskStore(STORE_PATH)
store.load_tasks(conversations)
print("loaded tasks:", store.fetchall("SELECT task_id, hidden_flow, hidden_subflow, success FROM tasks"))

# %% [markdown]
# ## Graph state
#
# Compact control state. No task corpus, no conversation bodies, no shrinking
# ID lists of the whole batch.


# %%
class MetaAgentState(TypedDict, total=False):
    run_id: str
    cohort_query: dict[str, Any]
    kb_version: int
    current_stage: str
    target_subflow: str
    intent_summary: dict[str, Any]
    subflow_summary: dict[str, Any]
    discovery_summary: dict[str, Any]
    recommendation_summary: dict[str, Any]
    pending_proposal_ids: list[str]
    approved_change_ids: list[str]
    errors: Annotated[list[dict[str, Any]], operator.add]
    # I think that the summaries could be passed as messages with differnt labels or roles.  more mutable.

def empty_state(**kwargs: Any) -> MetaAgentState:
    state: MetaAgentState = {
        "run_id": "",
        "cohort_query": {},
        "kb_version": playbook.version,
        "current_stage": "",
        "target_subflow": "",
        "intent_summary": {},
        "subflow_summary": {},
        "discovery_summary": {},
        "recommendation_summary": {},
        "pending_proposal_ids": [],
        "approved_change_ids": [],
        "errors": [],
    }
    state.update(kwargs)
    return state


def show(obj: Any, title: str = "") -> None:
    if title:
        display(Markdown(f"**{title}**"))
    display(Markdown(f"```json\n{json.dumps(obj, indent=2, default=str)}\n```"))


def show_state(state: MetaAgentState, keys: list[str] | None = None, title: str = "state") -> None:
    picked = keys or [
        "run_id",
        "kb_version",
        "current_stage",
        "intent_summary",
        "subflow_summary",
        "discovery_summary",
        "recommendation_summary",
        "pending_proposal_ids",
        "approved_change_ids",
    ]
    show({k: state.get(k) for k in picked}, title)


def show_rows(rows: list[dict[str, Any]], title: str, keep: list[str] | None = None) -> None:
    trimmed = []
    for row in rows:
        item = {k: row[k] for k in keep if k in row} if keep else dict(row)
        trimmed.append(item)
    show(trimmed, title)


def run_config(state: MetaAgentState, *, agent: str) -> dict[str, Any]:
    run_id = state.get("run_id") or "unassigned"
    return {
        "run_name": f"{agent}:{run_id}",
        "tags": ["demo", "modular-meta-agent", agent, f"run:{run_id}"],
        "metadata": {
            "agent": agent,
            "run_id": run_id,
            "kb_version": state.get("kb_version", playbook.version),
        },
    }


def add_phase_node(graph: StateGraph, name: str, fn, *, agent: str, phase: str) -> None:
    graph.add_node(name, fn, metadata={"agent": agent, "phase": phase})


def show_mermaid(compiled, *, xray: bool | int = False, title: str = "") -> None:
    source = compiled.get_graph(xray=xray).draw_mermaid()
    if title:
        display(Markdown(f"### {title}"))
    display(Markdown(f"```mermaid\n{source}\n```"))

# %% [markdown]
# ## Scoring helpers (unchanged from the working notebook)
#
# Intent / clustering: Jaccard on flattened turn text vs a slim playbook document.
# Subflow: LCS ratio of observed action buttons vs `kb.json`.
#
# A later classification step could retrieve similar already-labeled tasks and
# majority/LLM-vote instead of this centroid-like document match. Not built here;
# the current stub is enough to exercise the branches.


# %%
def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokenize(a), tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lcs_len(left: list[str], right: list[str]) -> int:
    n, m = len(left), len(right)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if left[i - 1] == right[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[n][m]


def action_sequence_score(observed: list[str], expected: list[str]) -> float:
    if not observed or not expected:
        return 0.0
    return lcs_len(observed, expected) / max(len(observed), len(expected))


@tool
def score_intent_similarity(conversation_id: str, intent_id: str) -> float:
    """Stub semantic similarity of a full interaction against an existing intent."""
    task = store.get_tasks([conversation_id])[0]
    return round(jaccard(task["text"], playbook.intent_doc(intent_id)), 3)


@tool
def score_subflow_similarity(conversation_id: str, subflow_id: str) -> float:
    """Stub subflow match: action-sequence LCS vs kb.json buttons."""
    task = store.get_tasks([conversation_id])[0]
    return round(action_sequence_score(task["actions"], playbook.kb.get(subflow_id, [])), 3)


@tool
def cluster_conversation_ids(conversation_ids: list[str]) -> list[list[str]]:
    """Greedy Jaccard clustering over full interaction text. Stands in for BERTopic."""
    remaining = list(conversation_ids)
    tasks = {t["task_id"]: t for t in store.get_tasks(conversation_ids)}
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_text = tasks[seed]["text"]
        group = [seed]
        kept: list[str] = []
        for other in remaining:
            if jaccard(seed_text, tasks[other]["text"]) >= CLUSTER_THRESHOLD:
                group.append(other)
            else:
                kept.append(other)
        remaining = kept
        clusters.append(group)
    return clusters


@tool
def draft_kb_stub(subflow_id: str, actions: list[str]) -> dict[str, list[str]]:
    """Draft an ABCD-shaped kb.json entry. Pass-through: no model."""
    return {subflow_id: actions}


@tool
def draft_guideline_stub(
    flow_title: str,
    subflow_title: str,
    actions: list[str],
    instructions: list[str],
) -> dict[str, Any]:
    """Draft an ABCD-shaped guidelines.json subflow block. Pass-through: no model."""
    return {
        flow_title: {
            "subflows": {
                subflow_title: {
                    "actions": [
                        {
                            "type": "interaction",
                            "button": action.replace("-", " ").title(),
                            "text": f"Observed repeated action [{action}]",
                            "subtext": [],
                        }
                        for action in actions
                    ],
                    "instructions": instructions,
                }
            }
        }
    }


print("sanity scores (intent=account_access)")
for cid in ["u1", "p1", "t1", "l1", "s1", "x1"]:
    intent = score_intent_similarity.invoke({"conversation_id": cid, "intent_id": "account_access"})
    username = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_username"})
    password = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_password"})
    print(f"  {cid:3}  intent={intent:.3f}  username={username:.3f}  password={password:.3f}")

# %% [markdown]
# ## HITL placeholder
#
# Not a LangGraph `interrupt()`. This function stands in for accept / modify /
# decline so the demo can persist KB changes. Replace later with native HITL.


# %%
SIMULATED_REVIEW = {
    "shipping_issue": ("accept", "Coherent new intent; enough shipping evidence."),
    "reset_2fa": ("accept", "Repeatable 2FA reset pattern."),
    "missing": ("accept", "Repeatable missing-package resolution."),
}


def simulate_hitl(proposal: dict[str, Any]) -> tuple[str, str]:
    candidate = proposal.get("candidate") or ""
    ptype = proposal.get("proposal_type")
    if ptype in {"outlier", "emerging"}:
        return "decline", "Placeholder HITL: monitor / no playbook change."
    if candidate in SIMULATED_REVIEW:
        return SIMULATED_REVIEW[candidate]
    return "decline", "Placeholder HITL: not in the demo accept list."


ACTION_FINGERPRINTS = {
    ("pull-up-account", "enter-details", "send-link"): "reset_2fa",
    (
        "pull-up-account",
        "validate-purchase",
        "record-reason",
        "update-order",
        "make-purchase",
    ): "missing",
}


def propose_intent_label(task_ids: list[str]) -> str:
    blob = " ".join(t["text"] for t in store.get_tasks(task_ids))
    if "package" in blob or "shipment" in blob or "delivered" in blob:
        return "shipping_issue"
    return "new_intent"


def propose_subflow_label(task_ids: list[str], fallback: str) -> tuple[str, list[str]]:
    successful = [t for t in store.get_tasks(task_ids) if t["success"]]
    if not successful:
        return fallback, []
    sequences = [tuple(t["actions"]) for t in successful]
    actions = list(Counter(sequences).most_common(1)[0][0])
    label = ACTION_FINGERPRINTS.get(tuple(actions), "_".join(actions[-2:]) if actions else fallback)
    return label, actions


def mean_pairwise_jaccard(task_ids: list[str]) -> float:
    texts = [t["text"] for t in store.get_tasks(task_ids)]
    pairs = [jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# %% [markdown]
# ## Establish cohort
#
# Creates `run_id`, records the query parameters, and writes `run_id | task_id`.
# Downstream nodes reload the same cohort with `run_id` — they do not carry IDs.


# %%
def cohort_query(state: MetaAgentState) -> dict[str, Any]:
    query = dict(
        state.get("cohort_query")
        or {
            "source": "scratch_data/incoming_conversations.json",
            "window": "demo-week",
            "as_of": "2026-09-01",
        }
    )
    run_id = state.get("run_id") or f"week-{query.get('as_of', 'demo')}"
    ids = [row["task_id"] for row in store.fetchall("SELECT task_id FROM tasks ORDER BY task_id")]
    store.set_workset(run_id, "cohort", ids)
    store.set_staging(run_id, "cohort_query", query)
    return {"run_id": run_id, "cohort_query": query, "current_stage": "cohort.query"}


def cohort_process(state: MetaAgentState) -> dict[str, Any]:
    return {"current_stage": "cohort.process"}


def cohort_persist(state: MetaAgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    query = store.get_staging(run_id, "cohort_query")
    ids = store.get_workset(run_id, "cohort")
    store.create_run(run_id, query, playbook.version)
    store.add_cohort(run_id, ids)
    return {"current_stage": "cohort.persist", "kb_version": playbook.version}


def cohort_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = store.cohort_ids(state["run_id"])
    summary = {"total": len(ids), "source": (state.get("cohort_query") or {}).get("source")}
    store.set_staging(state["run_id"], "cohort_summary", summary)
    return {"current_stage": "cohort.summarize"}


def build_cohort_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", cohort_query, agent="meta", phase="query")
    add_phase_node(graph, "process", cohort_process, agent="meta", phase="process")
    add_phase_node(graph, "persist", cohort_persist, agent="meta", phase="persist")
    add_phase_node(graph, "summarize", cohort_summarize, agent="meta", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "process")
    graph.add_edge("process", "persist")
    graph.add_edge("persist", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="establish_cohort")


cohort_graph = build_cohort_graph()

# %% [markdown]
# ## Classify intents
#
# `query cohort tasks requiring intent classification → compare against existing
# intents → persist intent labels → summarize coverage.`
#
# Re-running after a KB change picks up remaining `unknown` rows from the store.


# %%
def classify_intents_process(task_ids: list[str]) -> list[dict[str, Any]]:
    results = []
    intent_ids = playbook.intent_ids()
    for tid in task_ids:
        if not intent_ids:
            results.append({"task_id": tid, "intent_id": "unknown", "confidence": 0.0})
            continue
        scored = [
            (intent_id, score_intent_similarity.invoke({"conversation_id": tid, "intent_id": intent_id}))
            for intent_id in intent_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_id, best = scored[0]
        results.append(
            {
                "task_id": tid,
                "intent_id": best_id if best >= INTENT_THRESHOLD else "unknown",
                "confidence": best,
                "scores": scored,
            }
        )
    return results


def intent_query(state: MetaAgentState) -> dict[str, Any]:
    ids = store.unresolved_intent_ids(state["run_id"])
    store.set_workset(state["run_id"], "intent", ids)
    return {"current_stage": "intent.query"}


def intent_process(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_workset(state["run_id"], "intent")
    store.set_staging(state["run_id"], "intent", classify_intents_process(ids))
    return {"current_stage": "intent.process"}


def intent_persist(state: MetaAgentState) -> dict[str, Any]:
    staged = store.get_staging(state["run_id"], "intent")
    rows = [
        {
            "task_id": row["task_id"],
            "run_id": state["run_id"],
            "intent_id": row["intent_id"],
            "confidence": row["confidence"],
            "method": "jaccard_intent_doc_v1",
            "kb_version": playbook.version,
            "created_at": now_iso(),
        }
        for row in staged
    ]
    if rows:
        store.persist_intent_labels(rows)
    return {"current_stage": "intent.persist", "kb_version": playbook.version}


def _intent_summary(run_id: str, processed: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [r for r in processed if r["intent_id"] != "unknown"]
    unknown = [r for r in processed if r["intent_id"] == "unknown"]
    low = [r for r in classified if r["confidence"] < LOW_INTENT_BAND]
    latest = store.latest_intents(run_id)
    by_intent: dict[str, int] = {}
    for row in latest.values():
        by_intent[row["intent_id"]] = by_intent.get(row["intent_id"], 0) + 1
    return {
        "total": len(store.cohort_ids(run_id)),
        "processed": len(processed),
        "classified": len(classified),
        "unknown": len(unknown),
        "low_confidence": len(low),
        "by_intent": by_intent,
        "still_unresolved": len(store.unresolved_intent_ids(run_id)),
    }


def intent_summarize(state: MetaAgentState) -> dict[str, Any]:
    processed = store.get_staging(state["run_id"], "intent")
    summary = _intent_summary(state["run_id"], processed)
    return {
        "current_stage": "intent.summarize",
        "intent_summary": summary,
        "kb_version": playbook.version,
    }


def build_intent_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", intent_query, agent="intent", phase="query")
    add_phase_node(graph, "process", intent_process, agent="intent", phase="process")
    add_phase_node(graph, "persist", intent_persist, agent="intent", phase="persist")
    add_phase_node(graph, "summarize", intent_summarize, agent="intent", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "process")
    graph.add_edge("process", "persist")
    graph.add_edge("persist", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="classify_intents")


intent_graph = build_intent_graph()

# %% [markdown]
# ## Classify subflows
#
# Same shape as intents, scoped to tasks that already have an intent.
# Valid subflows are read from the current KB for that intent.


# %%
def classify_subflows_process(task_ids: list[str], run_id: str) -> list[dict[str, Any]]:
    latest_intents = store.latest_intents(run_id)
    results = []
    for tid in task_ids:
        intent_id = latest_intents[tid]["intent_id"]
        candidates = [sid for sid in playbook.subflows_for(intent_id) if sid in playbook.kb]
        if not candidates:
            results.append(
                {
                    "task_id": tid,
                    "intent_id": intent_id,
                    "subflow_id": "unknown",
                    "confidence": 0.0,
                    "scores": [],
                }
            )
            continue
        scored = [
            (sid, score_subflow_similarity.invoke({"conversation_id": tid, "subflow_id": sid}))
            for sid in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_id, best = scored[0]
        results.append(
            {
                "task_id": tid,
                "intent_id": intent_id,
                "subflow_id": best_id if best >= SUBFLOW_THRESHOLD else "unknown",
                "confidence": best,
                "scores": scored,
            }
        )
    return results


def subflow_query(state: MetaAgentState) -> dict[str, Any]:
    ids = store.unresolved_subflow_ids(state["run_id"])
    store.set_workset(state["run_id"], "subflow", ids)
    return {"current_stage": "subflow.query"}


def subflow_process(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_workset(state["run_id"], "subflow")
    store.set_staging(state["run_id"], "subflow", classify_subflows_process(ids, state["run_id"]))
    return {"current_stage": "subflow.process"}


def subflow_persist(state: MetaAgentState) -> dict[str, Any]:
    staged = store.get_staging(state["run_id"], "subflow")
    rows = [
        {
            "task_id": row["task_id"],
            "run_id": state["run_id"],
            "intent_id": row["intent_id"],
            "subflow_id": row["subflow_id"],
            "confidence": row["confidence"],
            "method": "lcs_kb_actions_v1",
            "kb_version": playbook.version,
            "created_at": now_iso(),
        }
        for row in staged
    ]
    if rows:
        store.persist_subflow_labels(rows)
    return {"current_stage": "subflow.persist", "kb_version": playbook.version}


def _subflow_summary(run_id: str, processed: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [r for r in processed if r["subflow_id"] != "unknown"]
    unknown = [r for r in processed if r["subflow_id"] == "unknown"]
    low = [r for r in classified if r["confidence"] < LOW_SUBFLOW_BAND]
    by_intent: dict[str, dict[str, int]] = {}
    latest = store.latest_subflows(run_id)
    intents = store.latest_intents(run_id)
    for tid, row in latest.items():
        intent_row = intents.get(tid)
        intent_id = row["intent_id"] or (intent_row["intent_id"] if intent_row else "?")
        by_intent.setdefault(intent_id, {})
        by_intent[intent_id][row["subflow_id"]] = by_intent[intent_id].get(row["subflow_id"], 0) + 1
    return {
        "processed": len(processed),
        "classified": len(classified),
        "unknown": len(unknown),
        "low_confidence": len(low),
        "by_intent": by_intent,
        "still_unresolved": len(store.unresolved_subflow_ids(run_id)),
    }


def subflow_summarize(state: MetaAgentState) -> dict[str, Any]:
    processed = store.get_staging(state["run_id"], "subflow")
    return {
        "current_stage": "subflow.summarize",
        "subflow_summary": _subflow_summary(state["run_id"], processed),
        "kb_version": playbook.version,
    }


def build_subflow_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", subflow_query, agent="subflow", phase="query")
    add_phase_node(graph, "process", subflow_process, agent="subflow", phase="process")
    add_phase_node(graph, "persist", subflow_persist, agent="subflow", phase="persist")
    add_phase_node(graph, "summarize", subflow_summarize, agent="subflow", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "process")
    graph.add_edge("process", "persist")
    graph.add_edge("persist", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="classify_subflows")


subflow_graph = build_subflow_graph()

# %% [markdown]
# ## Discovery: intents
#
# Operates across unresolved-intent tasks. Distinct from subflow discovery, which
# is scoped to an already-known intent.
#
# Approved proposals write labels for their supporting task IDs into the store.
# Reclassify then **re-queries** remaining `unknown` rows against the new KB; it
# does not reuse the in-memory cluster list.


# %%
def _examples(task_ids: list[str]) -> list[dict[str, str]]:
    out = []
    for task in store.get_tasks(task_ids)[:3]:
        out.append({"task_id": task["task_id"], "text": task["text"][:180]})
    return out


def intent_discovery_query(state: MetaAgentState) -> dict[str, Any]:
    ids = store.unresolved_intent_ids(state["run_id"])
    store.set_workset(state["run_id"], "intent_discovery", ids)
    return {"current_stage": "intent_discovery.query"}


def intent_discovery_discover(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_workset(state["run_id"], "intent_discovery")
    clusters = cluster_conversation_ids.invoke({"conversation_ids": ids}) if ids else []
    store.set_staging(state["run_id"], "intent_discovery_clusters", clusters)
    return {"current_stage": "intent_discovery.discover"}


def intent_discovery_validate(state: MetaAgentState) -> dict[str, Any]:
    clusters = store.get_staging(state["run_id"], "intent_discovery_clusters")
    candidates = []
    for cluster in clusters:
        mean = mean_pairwise_jaccard(cluster)
        coherent = len(cluster) >= MIN_CLUSTER_SIZE and mean >= CLUSTER_THRESHOLD
        if coherent:
            label = propose_intent_label(cluster)
            candidates.append(
                {
                    "proposal_type": "new_intent",
                    "candidate": label,
                    "parent_intent": None,
                    "supporting_task_ids": cluster,
                    "metrics": {"size": len(cluster), "mean_jaccard": round(mean, 3), "coherent": True},
                }
            )
        else:
            candidates.append(
                {
                    "proposal_type": "outlier",
                    "candidate": None,
                    "parent_intent": None,
                    "supporting_task_ids": cluster,
                    "metrics": {"size": len(cluster), "mean_jaccard": round(mean, 3), "coherent": False},
                }
            )
    store.set_staging(state["run_id"], "intent_discovery", candidates)
    return {"current_stage": "intent_discovery.validate"}


def intent_discovery_persist_proposal(state: MetaAgentState) -> dict[str, Any]:
    pending_ids = []
    for cand in store.get_staging(state["run_id"], "intent_discovery"):
        proposal_id = new_id("prop")
        pending_ids.append(proposal_id)
        store.persist_proposal(
            {
                "proposal_id": proposal_id,
                "run_id": state["run_id"],
                "proposal_type": cand["proposal_type"],
                "candidate": cand["candidate"],
                "parent_intent": cand["parent_intent"],
                "supporting_task_ids": json.dumps(cand["supporting_task_ids"]),
                "examples": json.dumps(_examples(cand["supporting_task_ids"])),
                "metrics": json.dumps(cand["metrics"]),
                "review_decision": None,
                "review_note": None,
                "resulting_kb_version": None,
                "created_at": now_iso(),
            }
        )
    store.set_staging(state["run_id"], "intent_discovery_ids", pending_ids)
    return {"current_stage": "intent_discovery.persist_proposal", "pending_proposal_ids": pending_ids}


def intent_discovery_hitl(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_staging(state["run_id"], "intent_discovery_ids")
    for proposal in store.list_proposals(state["run_id"]):
        if proposal["proposal_id"] not in ids:
            continue
        decision, note = simulate_hitl(proposal)
        store.decide_proposal(proposal["proposal_id"], decision, note)
    return {"current_stage": "intent_discovery.hitl"}


def intent_discovery_persist_kb(state: MetaAgentState) -> dict[str, Any]:
    approved = []
    ids = set(store.get_staging(state["run_id"], "intent_discovery_ids") or [])
    for proposal in store.list_proposals(state["run_id"], "new_intent"):
        if proposal["proposal_id"] not in ids or proposal["review_decision"] != "accept":
            continue
        description = (
            "package shipment delivered missing items carrier porch tracking "
            "order purchase replacement"
            if proposal["candidate"] == "shipping_issue"
            else "newly discovered operational intent"
        )
        version = playbook.add_intent(
            proposal["candidate"],
            proposal["candidate"].replace("_", " ").title(),
            description,
        )
        evidence_ids = json.loads(proposal["supporting_task_ids"])
        metrics = json.loads(proposal["metrics"])
        store.persist_intent_labels(
            [
                {
                    "task_id": tid,
                    "run_id": state["run_id"],
                    "intent_id": proposal["candidate"],
                    "confidence": metrics.get("mean_jaccard") or 1.0,
                    "method": "discovery_evidence_v1",
                    "kb_version": version,
                    "created_at": now_iso(),
                }
                for tid in evidence_ids
            ]
        )
        store.log_kb_change(version, "add_intent", {"intent": proposal["candidate"], "tasks": evidence_ids})
        store.mark_proposal_kb_version(proposal["proposal_id"], version)
        store.set_run_kb_version(state["run_id"], version)
        approved.append(proposal["proposal_id"])
    return {
        "current_stage": "intent_discovery.persist_kb",
        "kb_version": playbook.version,
        "approved_change_ids": approved,
    }


def intent_discovery_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = set(store.get_staging(state["run_id"], "intent_discovery_ids") or [])
    rows = [p for p in store.list_proposals(state["run_id"]) if p["proposal_id"] in ids]
    summary = {
        "unresolved_tasks": len(store.get_workset(state["run_id"], "intent_discovery")),
        "candidate_count": sum(1 for p in rows if p["proposal_type"] == "new_intent"),
        "outlier_count": sum(1 for p in rows if p["proposal_type"] == "outlier"),
        "approved_count": sum(1 for p in rows if p["review_decision"] == "accept"),
        "rejected_count": sum(1 for p in rows if p["review_decision"] == "decline"),
    }
    merged = dict(state.get("discovery_summary") or {})
    merged["intents"] = summary
    return {"current_stage": "intent_discovery.summarize", "discovery_summary": merged}


def build_intent_discovery_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", intent_discovery_query, agent="intent_discovery", phase="query")
    add_phase_node(graph, "discover", intent_discovery_discover, agent="intent_discovery", phase="process")
    add_phase_node(graph, "validate", intent_discovery_validate, agent="intent_discovery", phase="process")
    add_phase_node(
        graph, "persist_proposal", intent_discovery_persist_proposal, agent="intent_discovery", phase="persist"
    )
    add_phase_node(graph, "hitl", intent_discovery_hitl, agent="intent_discovery", phase="process")
    add_phase_node(graph, "persist_kb", intent_discovery_persist_kb, agent="intent_discovery", phase="persist")
    add_phase_node(graph, "summarize", intent_discovery_summarize, agent="intent_discovery", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "discover")
    graph.add_edge("discover", "validate")
    graph.add_edge("validate", "persist_proposal")
    graph.add_edge("persist_proposal", "hitl")
    graph.add_edge("hitl", "persist_kb")
    graph.add_edge("persist_kb", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="discover_intents")


intent_discovery_graph = build_intent_discovery_graph()

# %% [markdown]
# ## Discovery: subflows
#
# Same broad procedure, but the query is "tasks with an intent and unresolved
# subflow", clustered **within** that intent.


# %%
def subflow_discovery_query(state: MetaAgentState) -> dict[str, Any]:
    ids = store.unresolved_subflow_ids(state["run_id"])
    store.set_workset(state["run_id"], "subflow_discovery", ids)
    return {"current_stage": "subflow_discovery.query"}


def subflow_discovery_discover(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_workset(state["run_id"], "subflow_discovery")
    intents = store.latest_intents(state["run_id"])
    grouped: dict[str, list[str]] = {}
    for tid in ids:
        grouped.setdefault(intents[tid]["intent_id"], []).append(tid)
    clusters = []
    for intent_id, group in grouped.items():
        for cluster in cluster_conversation_ids.invoke({"conversation_ids": group}):
            clusters.append({"intent_id": intent_id, "task_ids": cluster})
    store.set_staging(state["run_id"], "subflow_discovery_clusters", clusters)
    return {"current_stage": "subflow_discovery.discover"}


def subflow_discovery_validate(state: MetaAgentState) -> dict[str, Any]:
    candidates = []
    for cluster in store.get_staging(state["run_id"], "subflow_discovery_clusters"):
        task_ids = cluster["task_ids"]
        intent_id = cluster["intent_id"]
        mean = mean_pairwise_jaccard(task_ids)
        coherent = len(task_ids) >= MIN_CLUSTER_SIZE and mean >= CLUSTER_THRESHOLD
        successful = [t for t in store.get_tasks(task_ids) if t["success"]]
        if not coherent:
            ptype = "outlier"
            label, actions = None, []
        elif len(successful) < MIN_SUCCESS_FOR_PATTERN:
            ptype = "emerging"
            label, actions = None, []
        else:
            sequences = [tuple(t["actions"]) for t in successful]
            best, count = Counter(sequences).most_common(1)[0]
            if best and count / len(successful) >= PATTERN_SUPPORT:
                ptype = "new_subflow"
                label, actions = propose_subflow_label(task_ids, "new_subflow")
            else:
                ptype = "emerging"
                label, actions = None, []
        candidates.append(
            {
                "proposal_type": ptype,
                "candidate": label,
                "parent_intent": intent_id,
                "supporting_task_ids": task_ids,
                "metrics": {
                    "size": len(task_ids),
                    "mean_jaccard": round(mean, 3),
                    "successful": len(successful),
                    "actions": actions,
                    "coherent": coherent,
                },
            }
        )
    store.set_staging(state["run_id"], "subflow_discovery", candidates)
    return {"current_stage": "subflow_discovery.validate"}


def subflow_discovery_persist_proposal(state: MetaAgentState) -> dict[str, Any]:
    pending_ids = []
    for cand in store.get_staging(state["run_id"], "subflow_discovery"):
        proposal_id = new_id("prop")
        pending_ids.append(proposal_id)
        store.persist_proposal(
            {
                "proposal_id": proposal_id,
                "run_id": state["run_id"],
                "proposal_type": cand["proposal_type"],
                "candidate": cand["candidate"],
                "parent_intent": cand["parent_intent"],
                "supporting_task_ids": json.dumps(cand["supporting_task_ids"]),
                "examples": json.dumps(_examples(cand["supporting_task_ids"])),
                "metrics": json.dumps(cand["metrics"]),
                "review_decision": None,
                "review_note": None,
                "resulting_kb_version": None,
                "created_at": now_iso(),
            }
        )
    store.set_staging(state["run_id"], "subflow_discovery_ids", pending_ids)
    return {"current_stage": "subflow_discovery.persist_proposal", "pending_proposal_ids": pending_ids}


def subflow_discovery_hitl(state: MetaAgentState) -> dict[str, Any]:
    ids = store.get_staging(state["run_id"], "subflow_discovery_ids")
    for proposal in store.list_proposals(state["run_id"]):
        if proposal["proposal_id"] not in ids:
            continue
        decision, note = simulate_hitl(proposal)
        store.decide_proposal(proposal["proposal_id"], decision, note)
    return {"current_stage": "subflow_discovery.hitl"}


def subflow_discovery_persist_kb(state: MetaAgentState) -> dict[str, Any]:
    approved = list(state.get("approved_change_ids") or [])
    ids = set(store.get_staging(state["run_id"], "subflow_discovery_ids") or [])
    for proposal in store.list_proposals(state["run_id"], "new_subflow"):
        if proposal["proposal_id"] not in ids or proposal["review_decision"] != "accept":
            continue
        metrics = json.loads(proposal["metrics"])
        version = playbook.add_subflow(
            proposal["parent_intent"],
            proposal["candidate"],
            metrics.get("actions") or [],
        )
        evidence_ids = json.loads(proposal["supporting_task_ids"])
        store.persist_subflow_labels(
            [
                {
                    "task_id": tid,
                    "run_id": state["run_id"],
                    "intent_id": proposal["parent_intent"],
                    "subflow_id": proposal["candidate"],
                    "confidence": 1.0,
                    "method": "discovery_evidence_v1",
                    "kb_version": version,
                    "created_at": now_iso(),
                }
                for tid in evidence_ids
            ]
        )
        store.log_kb_change(
            version,
            "add_subflow",
            {
                "intent": proposal["parent_intent"],
                "subflow": proposal["candidate"],
                "tasks": evidence_ids,
            },
        )
        store.mark_proposal_kb_version(proposal["proposal_id"], version)
        store.set_run_kb_version(state["run_id"], version)
        approved.append(proposal["proposal_id"])
    return {
        "current_stage": "subflow_discovery.persist_kb",
        "kb_version": playbook.version,
        "approved_change_ids": approved,
    }


def subflow_discovery_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = set(store.get_staging(state["run_id"], "subflow_discovery_ids") or [])
    rows = [p for p in store.list_proposals(state["run_id"]) if p["proposal_id"] in ids]
    summary = {
        "unresolved_tasks": len(store.get_workset(state["run_id"], "subflow_discovery")),
        "candidate_count": sum(1 for p in rows if p["proposal_type"] == "new_subflow"),
        "emerging_count": sum(1 for p in rows if p["proposal_type"] == "emerging"),
        "approved_count": sum(1 for p in rows if p["review_decision"] == "accept"),
        "rejected_count": sum(1 for p in rows if p["review_decision"] == "decline"),
    }
    merged = dict(state.get("discovery_summary") or {})
    merged["subflows"] = summary
    return {"current_stage": "subflow_discovery.summarize", "discovery_summary": merged}


def build_subflow_discovery_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", subflow_discovery_query, agent="subflow_discovery", phase="query")
    add_phase_node(graph, "discover", subflow_discovery_discover, agent="subflow_discovery", phase="process")
    add_phase_node(graph, "validate", subflow_discovery_validate, agent="subflow_discovery", phase="process")
    add_phase_node(
        graph, "persist_proposal", subflow_discovery_persist_proposal, agent="subflow_discovery", phase="persist"
    )
    add_phase_node(graph, "hitl", subflow_discovery_hitl, agent="subflow_discovery", phase="process")
    add_phase_node(graph, "persist_kb", subflow_discovery_persist_kb, agent="subflow_discovery", phase="persist")
    add_phase_node(graph, "summarize", subflow_discovery_summarize, agent="subflow_discovery", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "discover")
    graph.add_edge("discover", "validate")
    graph.add_edge("validate", "persist_proposal")
    graph.add_edge("persist_proposal", "hitl")
    graph.add_edge("hitl", "persist_kb")
    graph.add_edge("persist_kb", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="discover_subflows")


subflow_discovery_graph = build_subflow_discovery_graph()

# %% [markdown]
# ## Recommend pathways
#
# This is the reused "evaluate candidate subflow" idea from the working notebook,
# now limited to **validated** subflows. Clustering already happened in discovery.
# The subgraph queries the store for tasks associated with that subflow, analyzes
# successful action patterns, drafts KB + guideline, evaluates support, persists.


# %%
def pathway_query(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    latest = store.latest_subflows(state["run_id"])
    ids = [tid for tid, row in latest.items() if row["subflow_id"] == subflow_id]
    if not ids:
        for proposal in store.list_proposals(state["run_id"], "new_subflow"):
            if proposal["candidate"] == subflow_id and proposal["review_decision"] == "accept":
                ids = json.loads(proposal["supporting_task_ids"])
                break
    store.set_workset(state["run_id"], f"pathway:{subflow_id}", ids)
    return {"current_stage": "pathway.query"}


def pathway_analyze(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    ids = store.get_workset(state["run_id"], f"pathway:{subflow_id}")
    tasks = store.get_tasks(ids)
    successful = [t for t in tasks if t["success"]]
    sequences = [tuple(t["actions"]) for t in successful]
    best, count = Counter(sequences).most_common(1)[0] if sequences else ((), 0)
    payload = {
        "task_ids": ids,
        "successful": len(successful),
        "actions": list(best),
        "support": (count / len(successful)) if successful else 0.0,
    }
    store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    return {"current_stage": "pathway.analyze"}


def pathway_recommend(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = dict(store.get_staging(state["run_id"], f"pathway:{subflow_id}"))
    latest = store.latest_subflows(state["run_id"])
    sample_id = (payload.get("task_ids") or [None])[0]
    sample_row = latest.get(sample_id) if sample_id else None
    intent_id = sample_row["intent_id"] if sample_row else None
    if not intent_id:
        for proposal in store.list_proposals(state["run_id"], "new_subflow"):
            if proposal["candidate"] == subflow_id:
                intent_id = proposal["parent_intent"]
                break
    flow_title = playbook.flow_title(intent_id or "new_intent")
    actions = payload.get("actions") or []
    payload["intent_id"] = intent_id
    payload["kb_draft"] = draft_kb_stub.invoke({"subflow_id": subflow_id, "actions": actions})
    payload["guideline_draft"] = draft_guideline_stub.invoke(
        {
            "flow_title": flow_title,
            "subflow_title": SUBFLOW_TITLES.get(subflow_id, subflow_id.replace("_", " ").title()),
            "actions": actions,
            "instructions": [
                "Inferred from repeated successful traces in this batch.",
                "Needs human review before it becomes live playbook text.",
            ],
        }
    )
    store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    return {"current_stage": "pathway.recommend"}


def pathway_evaluate(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = dict(store.get_staging(state["run_id"], f"pathway:{subflow_id}"))
    ok = (
        payload.get("successful", 0) >= MIN_SUCCESS_FOR_PATTERN
        and payload.get("support", 0) >= PATTERN_SUPPORT
        and bool(payload.get("actions"))
    )
    payload["evaluation"] = {
        "supported": ok,
        "successful": payload.get("successful"),
        "support": round(payload.get("support", 0.0), 3),
    }
    store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    return {"current_stage": "pathway.evaluate"}


def pathway_persist(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = store.get_staging(state["run_id"], f"pathway:{subflow_id}")
    rec_id = new_id("rec")
    store.persist_recommendation(
        {
            "rec_id": rec_id,
            "run_id": state["run_id"],
            "intent_id": payload.get("intent_id") or "",
            "subflow_id": subflow_id,
            "supporting_task_ids": json.dumps(payload.get("task_ids") or []),
            "kb_draft": json.dumps(payload.get("kb_draft") or {}),
            "guideline_draft": json.dumps(payload.get("guideline_draft") or {}),
            "evaluation": json.dumps(payload.get("evaluation") or {}),
            "created_at": now_iso(),
        }
    )
    if payload.get("evaluation", {}).get("supported") and payload.get("intent_id"):
        version = playbook.attach_guideline(
            payload["intent_id"], subflow_id, payload.get("guideline_draft") or {}
        )
        store.log_kb_change(version, "attach_guideline", {"subflow": subflow_id})
        store.set_run_kb_version(state["run_id"], version)
    store.set_staging(state["run_id"], f"pathway:{subflow_id}:rec_id", rec_id)
    return {"current_stage": "pathway.persist", "kb_version": playbook.version}


def pathway_summarize(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = store.get_staging(state["run_id"], f"pathway:{subflow_id}")
    item = {
        "subflow_id": subflow_id,
        "intent_id": payload.get("intent_id"),
        "supported": payload.get("evaluation", {}).get("supported"),
        "n_tasks": len(payload.get("task_ids") or []),
        "support": payload.get("evaluation", {}).get("support"),
    }
    merged = dict(state.get("recommendation_summary") or {})
    items = list(merged.get("items") or [])
    items.append(item)
    merged["items"] = items
    merged["recommended"] = sum(1 for i in items if i.get("supported"))
    return {
        "current_stage": "pathway.summarize",
        "recommendation_summary": merged,
        "kb_version": playbook.version,
    }


def build_pathway_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "query", pathway_query, agent="pathway", phase="query")
    add_phase_node(graph, "analyze", pathway_analyze, agent="pathway", phase="process")
    add_phase_node(graph, "recommend", pathway_recommend, agent="pathway", phase="process")
    add_phase_node(graph, "evaluate", pathway_evaluate, agent="pathway", phase="process")
    add_phase_node(graph, "persist", pathway_persist, agent="pathway", phase="persist")
    add_phase_node(graph, "summarize", pathway_summarize, agent="pathway", phase="summarize")
    graph.add_edge(START, "query")
    graph.add_edge("query", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "evaluate")
    graph.add_edge("evaluate", "persist")
    graph.add_edge("persist", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="recommend_pathway")


pathway_graph = build_pathway_graph()

# %% [markdown]
# Isolated `invoke` wrappers: the working notebook found that adding a compiled
# subgraph as a parent node can re-fire `START` when state schemas are shared.
# Display graphs use the compiled subgraphs; the run graph calls `invoke`.


# %%
def invoke_named(compiled, state: MetaAgentState, *, agent: str) -> MetaAgentState:
    return compiled.invoke(state, config=run_config(state, agent=agent))


def node_establish_cohort(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(cohort_graph, state, agent="meta")


def node_classify_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(intent_graph, state, agent="intent")


def node_classify_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(subflow_graph, state, agent="subflow")


def node_discover_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(intent_discovery_graph, state, agent="intent_discovery")


def node_reclassify_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(intent_graph, state, agent="intent")


def node_discover_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(subflow_discovery_graph, state, agent="subflow_discovery")


def node_reclassify_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(subflow_graph, state, agent="subflow")


def node_recommend(state: MetaAgentState) -> dict[str, Any]:
    out = dict(state)
    for proposal in store.list_proposals(state["run_id"], "new_subflow"):
        if proposal["review_decision"] != "accept":
            continue
        out = invoke_named(
            pathway_graph,
            {**out, "target_subflow": proposal["candidate"]},
            agent="pathway",
        )
    return {
        "current_stage": "recommend",
        "recommendation_summary": out.get("recommendation_summary") or {"items": [], "recommended": 0},
        "kb_version": playbook.version,
    }


def node_summarize_run(state: MetaAgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    summary = {
        "run_id": run_id,
        "kb_version": playbook.version,
        "cohort_size": len(store.cohort_ids(run_id)),
        "intent_summary": state.get("intent_summary"),
        "subflow_summary": state.get("subflow_summary"),
        "discovery_summary": state.get("discovery_summary"),
        "recommendation_summary": state.get("recommendation_summary"),
        "intents_in_kb": playbook.intent_ids(),
        "subflows_in_kb": playbook.ontology["intents"]["subflows"],
        "unresolved_intents": store.unresolved_intent_ids(run_id),
        "unresolved_subflows": store.unresolved_subflow_ids(run_id),
    }
    store.set_staging(run_id, "run_summary", summary)
    return {"current_stage": "summarize", "run_id": run_id, "kb_version": playbook.version}


def build_classify_graph(use_compiled: bool):
    graph = StateGraph(MetaAgentState)
    if use_compiled:
        graph.add_node("classify_intents", intent_graph, metadata={"agent": "intent", "phase": "process"})
        graph.add_node("classify_subflows", subflow_graph, metadata={"agent": "subflow", "phase": "process"})
    else:
        graph.add_node("classify_intents", node_classify_intents, metadata={"agent": "intent", "phase": "process"})
        graph.add_node("classify_subflows", node_classify_subflows, metadata={"agent": "subflow", "phase": "process"})
    graph.add_edge(START, "classify_intents")
    graph.add_edge("classify_intents", "classify_subflows")
    graph.add_edge("classify_subflows", END)
    return graph.compile(name="classify")


def build_discover_graph(use_compiled: bool):
    graph = StateGraph(MetaAgentState)
    if use_compiled:
        graph.add_node("discover_intents", intent_discovery_graph, metadata={"agent": "intent_discovery", "phase": "process"})
        graph.add_node("reclassify_intents", intent_graph, metadata={"agent": "intent", "phase": "process"})
        graph.add_node("discover_subflows", subflow_discovery_graph, metadata={"agent": "subflow_discovery", "phase": "process"})
        graph.add_node("reclassify_subflows", subflow_graph, metadata={"agent": "subflow", "phase": "process"})
    else:
        graph.add_node("discover_intents", node_discover_intents, metadata={"agent": "intent_discovery", "phase": "process"})
        graph.add_node("reclassify_intents", node_reclassify_intents, metadata={"agent": "intent", "phase": "process"})
        graph.add_node("discover_subflows", node_discover_subflows, metadata={"agent": "subflow_discovery", "phase": "process"})
        graph.add_node("reclassify_subflows", node_reclassify_subflows, metadata={"agent": "subflow", "phase": "process"})
    graph.add_edge(START, "discover_intents")
    graph.add_edge("discover_intents", "reclassify_intents")
    graph.add_edge("reclassify_intents", "discover_subflows")
    graph.add_edge("discover_subflows", "reclassify_subflows")
    graph.add_edge("reclassify_subflows", END)
    return graph.compile(name="discover")


def build_meta_graph(use_compiled: bool):
    graph = StateGraph(MetaAgentState)
    graph.add_node("establish_cohort", node_establish_cohort, metadata={"agent": "meta", "phase": "query"})
    if use_compiled:
        graph.add_node("classify", build_classify_graph(True), metadata={"agent": "meta", "phase": "process"})
        graph.add_node("discover", build_discover_graph(True), metadata={"agent": "meta", "phase": "process"})
    else:
        graph.add_node("classify", build_classify_graph(False), metadata={"agent": "meta", "phase": "process"})
        graph.add_node("discover", build_discover_graph(False), metadata={"agent": "meta", "phase": "process"})
    graph.add_node("recommend", node_recommend, metadata={"agent": "pathway", "phase": "process"})
    graph.add_node("summarize", node_summarize_run, metadata={"agent": "meta", "phase": "summarize"})
    graph.add_edge(START, "establish_cohort")
    graph.add_edge("establish_cohort", "classify")
    graph.add_edge("classify", "discover")
    graph.add_edge("discover", "recommend")
    graph.add_edge("recommend", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(name="meta_agent")


classify_display = build_classify_graph(True)
discover_display = build_discover_graph(True)
meta_display = build_meta_graph(True)
meta_graph = build_meta_graph(False)

# %% [markdown]
# ## X-ray 3 — compiled graphs
#
# Top-level first (internals collapsed). Subgraphs below.

# %%
show_mermaid(meta_display, title="Top-level meta-agent")
show_mermaid(classify_display, title="Classify")
show_mermaid(discover_display, title="Discover (includes reclassify after KB mutation)")
show_mermaid(intent_graph, title="classify_intents")
show_mermaid(subflow_graph, title="classify_subflows")
show_mermaid(intent_discovery_graph, title="discover_intents")
show_mermaid(subflow_discovery_graph, title="discover_subflows")
show_mermaid(pathway_graph, title="recommend_pathway")
show_mermaid(meta_display, xray=1, title="Top-level xray=1")

# %% [markdown]
# ## Evaluation helper (offline labels only)
#
# Classifiers never see `hidden_flow` / `hidden_subflow`. This is notebook-only
# coverage / correctness visibility.


# %%
def eval_intents(run_id: str) -> list[dict[str, Any]]:
    latest = store.latest_intents(run_id)
    rows = []
    for task in store.get_tasks(store.cohort_ids(run_id)):
        pred = latest.get(task["task_id"])
        pred_id = pred["intent_id"] if pred else None
        truth = task["hidden_flow"]
        if pred_id is None or pred_id == "unknown":
            outcome = "unresolved"
        elif truth in {None, "outlier"}:
            outcome = "unresolved" if pred_id == "unknown" else "predicted_without_gt"
        elif pred_id == truth:
            outcome = "correct"
        else:
            outcome = "incorrect"
        rows.append(
            {
                "task_id": task["task_id"],
                "predicted": pred_id,
                "hidden": truth,
                "confidence": pred["confidence"] if pred else None,
                "outcome": outcome,
            }
        )
    return rows


def eval_subflows(run_id: str) -> list[dict[str, Any]]:
    latest = store.latest_subflows(run_id)
    rows = []
    for task in store.get_tasks(store.cohort_ids(run_id)):
        pred = latest.get(task["task_id"])
        pred_id = pred["subflow_id"] if pred else None
        truth = task["hidden_subflow"]
        if pred_id is None or pred_id == "unknown":
            outcome = "unresolved"
        elif pred_id == truth:
            outcome = "correct"
        else:
            outcome = "incorrect"
        rows.append(
            {
                "task_id": task["task_id"],
                "predicted": pred_id,
                "hidden": truth,
                "confidence": pred["confidence"] if pred else None,
                "outcome": outcome,
            }
        )
    return rows


def count_outcomes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return counts

# %% [markdown]
# ---
# # Walkthrough
#
# Expected operational outcomes (same as the working notebook):
#
# | ids | hidden | hoped-for path |
# | --- | --- | --- |
# | `u1`, `p1` | known subflows | classify existing, no structural change |
# | `t1–t3` | `reset_2fa` | existing intent, discover + recommend subflow |
# | `l1–l3` | locked / mixed | coherent cluster, no successful pattern, decline |
# | `s1–s3` | shipping missing | discover intent, reclassify, discover subflow, recommend |
# | `x1` | store window | outlier / monitor |

# %% [markdown]
# ### 1. Inspect current KB

# %%
show(
    {
        "version": playbook.version,
        "flows": playbook.intent_ids(),
        "subflows": playbook.ontology["intents"]["subflows"],
        "kb": playbook.kb,
    },
    "seed KB",
)

# %% [markdown]
# ### 2. Establish weekly cohort

# %%
state = empty_state(
    run_id="week-2026-09-01",
    cohort_query={
        "source": "scratch_data/incoming_conversations.json",
        "window": "demo-week",
        "as_of": "2026-09-01",
    },
)
state = invoke_named(cohort_graph, state, agent="meta")
show_state(state, keys=["run_id", "cohort_query", "kb_version", "current_stage"], title="after cohort")
show_rows(
    store.fetchall("SELECT run_id, task_id FROM cohort ORDER BY task_id"),
    "cohort mapping (run_id | task_id)",
)

# %% [markdown]
# ### 3–4. Intent classification + persisted records

# %%
state = invoke_named(intent_graph, state, agent="intent")
show_state(state, keys=["kb_version", "current_stage", "intent_summary"], title="after intent classification")
show_rows(
    store.fetchall(
        """
        SELECT task_id, intent_id, confidence, method, kb_version
        FROM intent_labels WHERE run_id = ? ORDER BY id
        """,
        (state["run_id"],),
    ),
    "persisted intent labels",
)
intent_eval = eval_intents(state["run_id"])
show_rows(intent_eval, "intent eval vs hidden labels")
print("intent outcomes:", count_outcomes(intent_eval))

# %% [markdown]
# ### 5–6. Subflow classification + summary

# %%
state = invoke_named(subflow_graph, state, agent="subflow")
show_state(state, keys=["kb_version", "current_stage", "subflow_summary"], title="after subflow classification")
show_rows(
    store.fetchall(
        """
        SELECT task_id, intent_id, subflow_id, confidence, method, kb_version
        FROM subflow_labels WHERE run_id = ? ORDER BY id
        """,
        (state["run_id"],),
    ),
    "persisted subflow labels",
)
print("subflow outcomes:", count_outcomes(eval_subflows(state["run_id"])))

# %% [markdown]
# ### 7–11. Discover intents, simulated HITL, persist KB, reclassify
#
# Unresolved intents are **re-queried from the store**, not passed as a Python list.

# %%
print("unresolved intents before discovery:", store.unresolved_intent_ids(state["run_id"]))
state = invoke_named(intent_discovery_graph, state, agent="intent_discovery")
show_state(
    state,
    keys=["kb_version", "current_stage", "discovery_summary", "pending_proposal_ids", "approved_change_ids"],
    title="after intent discovery",
)
show_rows(
    [
        {
            "proposal_id": p["proposal_id"],
            "type": p["proposal_type"],
            "candidate": p["candidate"],
            "tasks": p["supporting_task_ids"],
            "metrics": p["metrics"],
            "decision": p["review_decision"],
            "note": p["review_note"],
            "kb_version": p["resulting_kb_version"],
        }
        for p in store.list_proposals(state["run_id"])
    ],
    "intent-stage proposals + HITL",
)
print("KB intents now:", playbook.intent_ids())
print("unresolved intents after KB persist (before reclassify):", store.unresolved_intent_ids(state["run_id"]))
state = invoke_named(intent_graph, state, agent="intent")
show_state(state, keys=["intent_summary"], title="after reclassify intents")
show_rows(eval_intents(state["run_id"]), "intent eval after reclassify")
print("intent outcomes after reclassify:", count_outcomes(eval_intents(state["run_id"])))
print("unresolved intents after reclassify:", store.unresolved_intent_ids(state["run_id"]))

# %% [markdown]
# ### 12–15. Discover subflows by intent, HITL, persist, reclassify

# %%
print("unresolved subflows before discovery:", store.unresolved_subflow_ids(state["run_id"]))
state = invoke_named(subflow_discovery_graph, state, agent="subflow_discovery")
show_state(
    state,
    keys=["kb_version", "current_stage", "discovery_summary", "approved_change_ids"],
    title="after subflow discovery",
)
show_rows(
    [
        {
            "proposal_id": p["proposal_id"],
            "type": p["proposal_type"],
            "intent": p["parent_intent"],
            "candidate": p["candidate"],
            "tasks": p["supporting_task_ids"],
            "decision": p["review_decision"],
            "note": p["review_note"],
            "kb_version": p["resulting_kb_version"],
        }
        for p in store.list_proposals(state["run_id"])
        if p["proposal_type"] in {"new_subflow", "emerging", "outlier"}
    ],
    "subflow-stage proposals + HITL",
)
print("KB subflows now:", playbook.ontology["intents"]["subflows"])
state = invoke_named(subflow_graph, state, agent="subflow")
show_state(state, keys=["subflow_summary"], title="after reclassify subflows")
show_rows(eval_subflows(state["run_id"]), "subflow eval after reclassify")
print("subflow outcomes:", count_outcomes(eval_subflows(state["run_id"])))

# %% [markdown]
# ### 16–17. Pathway recommendation for newly validated subflows

# %%
for proposal in store.list_proposals(state["run_id"], "new_subflow"):
    if proposal["review_decision"] != "accept":
        continue
    print("recommend pathway for", proposal["candidate"], "tasks", proposal["supporting_task_ids"])
    state = invoke_named(
        pathway_graph,
        {**state, "target_subflow": proposal["candidate"]},
        agent="pathway",
    )
    show_state(state, keys=["recommendation_summary", "kb_version"], title=f"after pathway {proposal['candidate']}")

show_rows(
    [
        {
            "rec_id": r["rec_id"],
            "intent_id": r["intent_id"],
            "subflow_id": r["subflow_id"],
            "evaluation": r["evaluation"],
            "kb_draft": r["kb_draft"],
        }
        for r in store.list_recommendations(state["run_id"])
    ],
    "persisted recommendations",
)

# %% [markdown]
# ### 18–19. Final run summary and KB changes

# %%
state = {**state, **node_summarize_run(state)}
show(store.get_staging(state["run_id"], "run_summary"), "final run summary")
show(
    {
        "version": playbook.version,
        "flows": playbook.intent_ids(),
        "subflows": playbook.ontology["intents"]["subflows"],
        "kb": playbook.kb,
        "guideline_flow_keys": {k: list(v.get("subflows", {})) for k, v in playbook.guidelines.items()},
    },
    "final KB",
)
show_rows(store.fetchall("SELECT kb_version, change_type, payload FROM kb_log ORDER BY kb_version"), "KB mutation log")
print("still unresolved intents:", store.unresolved_intent_ids(state["run_id"]))
print("still unresolved subflows:", store.unresolved_subflow_ids(state["run_id"]))

# %% [markdown]
# ### 20. Orchestrated parent invoke
#
# Fresh `run_id` so LangSmith has one parent trace whose children are the
# subgraphs above. KB is already mutated from the walkthrough; that is fine —
# the orchestrated run should mostly classify against the updated structure
# and find little left to discover.

# %%
orchestrated = invoke_named(
    meta_graph,
    empty_state(
        run_id="week-orchestrated",
        cohort_query={
            "source": "scratch_data/incoming_conversations.json",
            "window": "demo-week",
            "as_of": "2026-09-01",
            "note": "second invoke against already-updated KB",
        },
    ),
    agent="meta",
)
show_state(orchestrated, title="orchestrated parent state")
show(store.get_staging("week-orchestrated", "run_summary"), "orchestrated run summary")

# %% [markdown]
# ## LangSmith
#
# Filter a parent run by tag `run:week-2026-09-01` or `run:week-orchestrated`.
# Expand children. Node metadata is `agent` + `phase`, so the intent subgraph
# reads as:
#
# ```text
# intent → query
# intent → process
# intent → persist
# intent → summarize
# ```
#
# `run_id` and `kb_version` are on the invoke config as well.
#
# **Thread / session identity (not implemented here):** a later HITL phase can
# map `run_id` to a LangGraph thread id (`demo-week-1`) and a LangSmith session
# so interrupt/resume lands in the same conversation. This notebook only sets
# clean trace names, tags, and metadata.

# %%
try:
    client = Client()
    project = os.environ.get("LANGSMITH_PROJECT", "fresh-skills")
    runs = list(client.list_runs(project_name=project, is_root=True, limit=20))
    interesting = [
        run
        for run in runs
        if any(
            key in (run.name or "")
            for key in ("meta_agent", "classify", "discover", "recommend", "establish", "pathway", "intent", "subflow")
        )
    ]
    if interesting:
        display(Markdown("**Recent root LangSmith runs** (open a `meta_agent` / `classify_intents` parent and expand)"))
        for run in interesting[:8]:
            print(f"{run.name:40}  {getattr(run, 'url', '')}")
    else:
        print("No matching root runs yet. Re-run with LANGSMITH_TRACING=true and open the project in LangSmith.")
except Exception as exc:
    print("LangSmith lookup skipped:", type(exc).__name__, exc)

# %% [markdown]
# ## Notes for later
#
# - Classification still uses Jaccard / LCS, not retrieval of similar labeled
#   tasks. That swap is local to `classify_intents_process` / `classify_subflows_process`.
# - HITL is `simulate_hitl`. Real work is `interrupt()` + `Command(resume=...)`.
# - Pathway drafts are still pass-through stubs. That is the honest agentic seam.
# - Do not copy this notebook over `demo_story.ipynb`; merge-back is `/apply-worktree`.
