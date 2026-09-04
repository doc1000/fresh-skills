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
# # Integrate topic discovery with the LangGraph agent
#
# Worktree-local EDA. Fill writes files. The agent only loads a time window and
# `invoke`s. Those are separate steps.
#
# ```text
# 1. edit levers
# 2. fill_kb_and_tasks  → writes Agent KB files and incoming conversations
# 3. configure_runtime  → loads those files into the store
# 4. inspect intent-classify subgraph
# 5. agent = build_graph(...); agent.invoke(payload)
# ```
#
# Later windows: change `START` / `END` only. Do not recouple that to
# `SEED_SUBFLOWS` / `PER_SEED`.

# %%
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
from IPython.display import JSON, Markdown, display

import playbook
from playbook import (
    build_graph,
    build_intent_graph,
    configure_runtime,
    render_mermaid,
    retrieve_guidance,
)
from playbook import runtime as rt
from playbook.adapters import tasks_to_conversations
from playbook.config import ABCD_JSON, ROOT
from playbook.fill import EDA_DATA_DIR, catalog_abcd, current_kb_view, fill_kb_and_tasks
from playbook.graph import classify_intents_process
from playbook.topics import BertopicConfig

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)

print("playbook.__file__", playbook.__file__)
print("sys.executable", sys.executable)
print("ROOT", ROOT)
assert Path(playbook.__file__).resolve().is_relative_to(ROOT / "src"), playbook.__file__
assert Path(sys.executable).resolve().is_relative_to(ROOT / ".venv"), sys.executable
print("configure_runtime", inspect.signature(configure_runtime))

# %% [markdown]
# ## Levers
#
# Fill/sampling knobs write files. Agent-window knobs select which already-filled
# chats are in the run. ABCD has no chat timestamps; fill stamps synthetic
# `conversation_date` values across `FILL_DATE_START`…`FILL_DATE_END`.
#
# **Classification**
#
# - `MIN_SIM` — cosine to the best prototype must be at least this. Below that, nearest label is a weak match.
# - `MIN_MARGIN` — best prototype must beat second-best by this. Blocks close calls.
# - `FIT_RULE` — how centroid vs guideline vote: `centroid` / `guideline` / `either` / `both`. Unassigned tickets go to discovery. If both fit but disagree, keep the higher similarity.
#
# **Sampling the fill (not the agent window)**
#
# - `MIN_OPENING` — drop conversations whose first customer message is shorter than this.
# - `PER_SEED` — conversations taken from each **known** playbook subflow (the ones in `SEED_SUBFLOWS`).
# - `PER_PROBE` — conversations for the held-out subflow and the held-out intent.
# - `NOISE_PAIRS` — one conversation each; they should not form a topic.
#
# **Clustering / “is this a real topic?”**
#
# - `MIN_TO_CLUSTER` — do not cluster if the leftover pile is smaller than this.
# - `MIN_CLUSTER_SIZE` — HDBSCAN smallest cluster. Smaller blobs become noise.
# - `MIN_TOPIC_N` — after clustering, `cohesive_enough` requires `size >= MIN_TOPIC_N` and topic != -1.
#
# **Method**
#
# - `"jaccard"` — token overlap classify + greedy Jaccard clusters. Fast walkthrough.
# - `"bertopic"` — centroid/guideline embeddings + BERTopic. Runtime path.

# %%
MIN_SIM = 0.60
MIN_MARGIN = 0.05
FIT_RULE = "either"
MIN_CLUSTER_SIZE = 5
MIN_TOPIC_N = 5
MIN_TO_CLUSTER = 5
PER_SEED = 15
PER_PROBE = 15
MIN_OPENING = 25
METHOD = "jaccard"  # "bertopic" for the live embedding path

# ABCD has no conversation timestamps. Fill stamps these synthetic ISO dates.
FILL_DATE_START = "2026-08-25"
FILL_DATE_END = "2026-09-14"

# Agent payload window. Change only these to run a different cohort.
RUN_ID = "eda-window"
START = "2026-09-01"
END = "2026-09-07"

SEED_SUBFLOWS = {
    "account_access": ["recover_username", "recover_password", "reset_2fa"],
    "order_issue": [
        "status_mystery_fee",
        "status_delivery_time",
        "manage_upgrade",
        "manage_cancel",
    ],
}
INTENT_GUIDELINE_TITLE = {
    "account_access": "Account Access",
    "order_issue": "Order Issue",
}
SUBFLOW_GUIDELINE_TITLE = {
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
    "reset_2fa": "Reset Two-Factor Auth",
    "status_mystery_fee": "Status Mystery Fee",
    "status_delivery_time": "Status Delivery Time",
    "manage_upgrade": "Manage Upgrade",
    "manage_cancel": "Manage Cancel",
}
PROBE_SUBFLOW = ("order_issue", "status_payment_method")
PROBE_INTENT = ("troubleshoot_site", "slow_speed")
NOISE_PAIRS = [
    ("shipping_issue", "missing"),
    ("shipping_issue", "cost"),
    ("product_defect", "return_stain"),
    ("product_defect", "refund_initiate"),
    ("purchase_dispute", "promo_code_invalid"),
    ("purchase_dispute", "out_of_stock_general"),
    ("subscription_inquiry", "manage_pay_bill"),
    ("manage_account", "manage_change_address"),
    ("storewide_query", "pricing_3"),
    ("single_item_query", "boots_how_2"),
]

topic_config = BertopicConfig(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_to_cluster=MIN_TO_CLUSTER,
    min_topic_n=MIN_TOPIC_N,
    n_neighbors=12,
    seed=42,
)

pd.DataFrame(
    [
        {"knob": "METHOD", "value": METHOD, "means": "one switch for classify + discover; also on invoke payload"},
        {"knob": "FILL_DATE_START", "value": FILL_DATE_START, "means": "synthetic conversation_date origin (fill only)"},
        {"knob": "FILL_DATE_END", "value": FILL_DATE_END, "means": "synthetic conversation_date last day (fill only)"},
        {"knob": "START", "value": START, "means": "agent payload: inclusive window start"},
        {"knob": "END", "value": END, "means": "agent payload: inclusive window end"},
        {"knob": "RUN_ID", "value": RUN_ID, "means": "agent payload: this invoke's run id"},
        {"knob": "MIN_SIM", "value": MIN_SIM, "means": "minimum cosine to accept a prototype"},
        {"knob": "MIN_MARGIN", "value": MIN_MARGIN, "means": "best prototype must beat #2 by this"},
        {"knob": "FIT_RULE", "value": FIT_RULE, "means": "centroid vs guideline vote"},
        {"knob": "MIN_OPENING", "value": MIN_OPENING, "means": "drop short first customer turns"},
        {"knob": "PER_SEED", "value": PER_SEED, "means": "n per known subflow written on fill"},
        {"knob": "PER_PROBE", "value": PER_PROBE, "means": "n for held-out subflow and intent"},
        {"knob": "MIN_TO_CLUSTER", "value": MIN_TO_CLUSTER, "means": "skip clustering below this n"},
        {"knob": "MIN_CLUSTER_SIZE", "value": MIN_CLUSTER_SIZE, "means": "HDBSCAN smallest cluster"},
        {"knob": "MIN_TOPIC_N", "value": MIN_TOPIC_N, "means": "cohesive_enough if size >= this"},
    ]
)

# %% [markdown]
# ## What ABCD can give you
#
# Every flow/subflow, not just the seeded ones. `n_after_opening_filter` is how
# many you can actually pick. `in_seed_kb` is whether that subflow is in
# `SEED_SUBFLOWS`. `role` is set only when this sheet selected the pair.

# %%
# ! python ../scripts/download_abcd.py

# %%
if not ABCD_JSON.exists():
    raise FileNotFoundError(f"Missing {ABCD_JSON}. Run: python scripts/download_abcd.py")

seed_pairs = {(flow, subflow) for flow, subflows in SEED_SUBFLOWS.items() for subflow in subflows}
selected_roles = {
    **{pair: "seed" for pair in seed_pairs},
    PROBE_SUBFLOW: "probe_subflow",
    PROBE_INTENT: "probe_intent",
    **{pair: "noise" for pair in NOISE_PAIRS},
}
catalog = pd.DataFrame(catalog_abcd(min_opening=MIN_OPENING))
catalog["in_seed_kb"] = [
    (row.flow, row.subflow) in seed_pairs for row in catalog.itertuples()
]
catalog["role"] = [
    selected_roles.get((row.flow, row.subflow), "") for row in catalog.itertuples()
]
catalog = catalog.rename(columns={"n_after_opening_filter": "n after opening filter"})
display(Markdown("**Every ABCD flow/subflow**"))
display(catalog.sort_values(["flow", "subflow"]))
display(pd.DataFrame(
    [
        {"field": "n_pairs", "value": len(catalog), "means": "distinct flow/subflow rows in ABCD"},
        {"field": "n_in_seed_kb", "value": int(catalog["in_seed_kb"].sum()), "means": "will be written into Agent KB on fill"},
        {"field": "n_with_role", "value": int((catalog["role"] != "").sum()), "means": "seed + probe + noise selected on this sheet"},
    ]
))

# %% [markdown]
# ## Current Agent KB on disk
#
# This is whatever was last filled into `scratch_data/eda/`. It is not the agent yet.

# %%
kb_now = current_kb_view(EDA_DATA_DIR)
display(Markdown(f"KB dir: `{kb_now['data_dir']}` exists={kb_now['exists']}"))
if EDA_DATA_DIR.exists():
    display(pd.DataFrame(
        [
            {"file": path.name, "bytes": path.stat().st_size, "means": "Agent KB / fill artifact on disk"}
            for path in sorted(EDA_DATA_DIR.glob("*.json"))
        ]
    ))
if kb_now["exists"]:
    kb_rows = []
    for intent, subflows in (kb_now.get("subflows") or {}).items():
        for subflow in subflows:
            kb_rows.append({"intent": intent, "subflow": subflow, "in_seed_kb_file": True})
    display(pd.DataFrame(kb_rows))
    display(pd.DataFrame(
        [{"intent": intent, "title": kb_now.get("flow_titles", {}).get(intent, "")} for intent in kb_now["intents"]]
    ))
else:
    display(Markdown("No Agent KB files yet. Run fill."))

# %% [markdown]
# ## Fill (manual)
#
# Run this cell after you change the fill levers. It only looks up ABCD rows and
# dumps files, including synthetic `conversation_date`. It does **not** compile
# or invoke the agent.

# %%
fill = fill_kb_and_tasks(
    seed_subflows=SEED_SUBFLOWS,
    noise_pairs=NOISE_PAIRS,
    probe_subflow=PROBE_SUBFLOW,
    probe_intent=PROBE_INTENT,
    per_seed=PER_SEED,
    per_probe=PER_PROBE,
    min_opening=MIN_OPENING,
    intent_titles=INTENT_GUIDELINE_TITLE,
    subflow_titles=SUBFLOW_GUIDELINE_TITLE,
    fill_date_start=FILL_DATE_START,
    fill_date_end=FILL_DATE_END,
    data_dir=EDA_DATA_DIR,
)
inventory = pd.DataFrame(fill["inventory"])
display(pd.DataFrame(
    [
        {"field": "n_tasks", "value": fill["n_tasks"], "means": "conversations written to incoming_conversations.json"},
        {"field": "n_seed_examples", "value": fill["n_seed_examples"], "means": "centroid prototypes; not graph state"},
        {"field": "intents_in_kb", "value": ", ".join(fill["intents_in_kb"]), "means": "Agent KB flows after this fill"},
        {"field": "fill_date_start", "value": fill["fill_date_start"], "means": "synthetic date stamped on first chat"},
        {"field": "fill_date_end", "value": fill["fill_date_end"], "means": "synthetic date stamped on last chat"},
        {"field": "min_conversation_date", "value": fill["min_conversation_date"], "means": "actual min date written"},
        {"field": "max_conversation_date", "value": fill["max_conversation_date"], "means": "actual max date written"},
    ]
))
display(inventory.groupby(["role", "true_flow", "true_subflow"]).size().reset_index(name="n"))
display(inventory[["conversation_id", "role", "true_flow", "true_subflow", "conversation_date"]].head(12))

# %% [markdown]
# ## Load runtime from those files
#
# Separate from fill. This rebuilds the SQLite store from the files you just wrote.
# It still does not invoke the agent. `method=` must be accepted here; if this
# TypeErrors, the kernel is not this worktree's package (see `playbook.__file__`).

# %%
playbook_kb, store = configure_runtime(
    data_dir=EDA_DATA_DIR,
    store_path=EDA_DATA_DIR / "run_store.sqlite",
    method=METHOD,
    topic_config=topic_config,
    min_sim=MIN_SIM,
    min_margin=MIN_MARGIN,
    fit_rule=FIT_RULE,
)
task_n = store.fetchall("SELECT COUNT(*) AS n FROM tasks")[0]["n"]
adapted = tasks_to_conversations(
    store.get_tasks([row["task_id"] for row in store.fetchall("SELECT task_id FROM tasks LIMIT 1")])
)
window = store.fetchall(
    """
    SELECT
        COUNT(*) AS n_chats,
        MIN(conversation_date) AS min_conversation_date,
        MAX(conversation_date) AS max_conversation_date
    FROM tasks
    WHERE conversation_date >= ? AND conversation_date <= ?
    """,
    (START, END),
)[0]
display(pd.DataFrame(
    [
        {"check": "METHOD", "value": rt.method, "means": "classify + discover both read this"},
        {"check": "tasks in store", "value": task_n, "means": "all filled chats; not yet windowed"},
        {"check": "KB intents", "value": ", ".join(playbook_kb.intent_ids()), "means": "retrieve_guidance corpus"},
        {"check": "runtime record fields", "value": ", ".join(sorted(adapted[0].model_dump())), "means": "no hidden flow/subflow"},
        {"check": "seed_examples", "value": len(rt.seed_examples), "means": "centroid side file, not graph state"},
        {"check": "kb_version", "value": playbook_kb.version, "means": "goes on the invoke payload"},
    ]
))
display(Markdown("**Window the agent will load** (store query, same predicate as the cohort node)"))
display(pd.DataFrame(
    [
        {
            "start": START,
            "end": END,
            "n_chats_in_window": window["n_chats"],
            "min_conversation_date": window["min_conversation_date"],
            "max_conversation_date": window["max_conversation_date"],
            "means": "inclusive ISO dates; fill stamped these, agent only filters",
        }
    ]
))
assert set(adapted[0].model_dump()) == {"conversation_id", "turns", "actions"}

# %% [markdown]
# ## Intent-classify subgraph
#
# Phase metadata is still query/process/persist/summarize. Node **names** are the
# work: `select_unresolved_intents` → `score_intents` → `persist_intent_labels` →
# `summarize_intent_assignments`. Scoring lives in `classify_intents_process`.

# %%
intent_graph = build_intent_graph()
display(Markdown(f"```mermaid\n{render_mermaid(intent_graph, xray=True)}\n```"))
display(pd.DataFrame(
    [
        {"node": "select_unresolved_intents", "does": "workset of chats with no intent or intent=unknown"},
        {"node": "score_intents", "does": "classify_intents_process: jaccard or bertopic prototypes"},
        {"node": "persist_intent_labels", "does": "write intent_id + confidence to the store"},
        {"node": "summarize_intent_assignments", "does": "classified vs unknown counts on state"},
    ]
))
display(Markdown("```python\n" + inspect.getsource(classify_intents_process) + "\n```"))

# %% [markdown]
# ## Invoke the live agent
#
# No stubs. No direct `discover_*` calls. Agent KB is the only retriever.
# `invoke_week` is a test wrapper; this sheet calls `agent.invoke`.

# %%
agent = build_graph()
payload = {
    "run_id": RUN_ID,
    "start": START,
    "end": END,
    "method": METHOD,
    "kb_version": playbook_kb.version,
}
display(pd.DataFrame(
    [
        {"key": "run_id", "value": payload["run_id"], "means": "this run's id; labels and proposals hang off it"},
        {"key": "start", "value": payload["start"], "means": "inclusive ISO date; chats on/after this day"},
        {"key": "end", "value": payload["end"], "means": "inclusive ISO date; chats on/before this day"},
        {"key": "method", "value": payload["method"], "means": "jaccard | bertopic for classify + discover"},
        {"key": "kb_version", "value": payload["kb_version"], "means": "Agent KB version at invoke time"},
    ]
))
result = agent.invoke(payload)

# %% [markdown]
# ### Cohort actually loaded

# %%
display(pd.DataFrame(
    [
        {
            "start": (result.get("cohort_summary") or {}).get("start"),
            "end": (result.get("cohort_summary") or {}).get("end"),
            "n_chats_in_window": (result.get("cohort_summary") or {}).get("n"),
            "min_conversation_date": (result.get("cohort_summary") or {}).get("min_conversation_date"),
            "max_conversation_date": (result.get("cohort_summary") or {}).get("max_conversation_date"),
            "means": "dates the cohort node actually loaded",
        }
    ]
))

intents = store.latest_intents(RUN_ID)
subflows = store.latest_subflows(RUN_ID)
assignment_rows = []
for task_id in store.cohort_ids(RUN_ID):
    intent_row = intents.get(task_id)
    subflow_row = subflows.get(task_id)
    intent_id = intent_row["intent_id"] if intent_row else ""
    subflow_id = subflow_row["subflow_id"] if subflow_row else ""
    unknown = (not intent_id or intent_id == "unknown") or (not subflow_id or subflow_id == "unknown")
    assignment_rows.append(
        {
            "conversation_id": task_id,
            "intent": intent_id,
            "subflow": subflow_id,
            "intent_confidence": None if intent_row is None else intent_row["confidence"],
            "subflow_confidence": None if subflow_row is None else subflow_row["confidence"],
            "status": "unknown" if unknown else "assigned",
        }
    )
assignments = pd.DataFrame(assignment_rows)
display(Markdown("**Assigned vs unknown after invoke**"))
display(assignments.groupby(["status", "intent", "subflow"]).size().reset_index(name="n"))
display(assignments)

# %% [markdown]
# ### Subgraph summaries
#
# These are the user-facing counts each subgraph already puts on state.

# %%
def counts_table(summary: dict, title: str):
    display(Markdown(f"**{title}**"))
    rows = [{"field": key, "value": value} for key, value in (summary or {}).items() if not isinstance(value, (dict, list))]
    display(pd.DataFrame(rows))
    nested = {key: value for key, value in (summary or {}).items() if isinstance(value, (dict, list))}
    if nested:
        display(JSON(nested))


counts_table(result.get("intent_summary"), "intent_summary — classified vs leftover")
counts_table(result.get("subflow_summary"), "subflow_summary — assigned vs unknown")
display(Markdown("**discovery_summary** — candidate / outlier / HITL counts"))
display(JSON(result.get("discovery_summary") or {}))
counts_table(result.get("recommendation_summary"), "recommendation_summary — pathway drafts")

topics = pd.DataFrame(result.get("discovered_topics") or [])
if not topics.empty:
    display(Markdown("**discovered_topics** — memberships + cohesive_enough stay on state so the agent can name/override"))
    display(topics[["topic_id", "size", "cohesive_enough", "descriptor", "member_ids"]])
subflows_df = pd.DataFrame(result.get("discovered_subflows") or [])
if not subflows_df.empty:
    display(Markdown("**discovered_subflows**"))
    display(subflows_df[["topic_id", "size", "cohesive_enough", "descriptor", "member_ids"]])

hits = result.get("retrieved_guidance") or []
if hits:
    display(Markdown("**retrieved_guidance** — `playbook.kb.retrieve_guidance` only"))
    display(pd.DataFrame(
        [
            {
                "topic_id": row.get("topic_id"),
                "query": (row.get("query") or "")[:80],
                "top_intent": (row.get("hits") or [{}])[0].get("intent_id"),
                "top_score": (row.get("hits") or [{}])[0].get("score"),
            }
            for row in hits
        ]
    ))

# %% [markdown]
# Full graph state (expandable). Not a substitute for the tables above.

# %%
display(JSON({key: result.get(key) for key in (
    "run_id",
    "start",
    "end",
    "method",
    "current_stage",
    "kb_version",
    "cohort_summary",
    "intent_summary",
    "subflow_summary",
    "discovery_summary",
    "recommendation_summary",
    "discovered_topics",
    "discovered_subflows",
    "action_paths",
    "retrieved_guidance",
)}))

manual = retrieve_guidance("package missing delivered", playbook_kb)
display(pd.DataFrame(manual))
