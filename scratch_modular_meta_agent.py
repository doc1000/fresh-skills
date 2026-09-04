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
# Worktree-local review notebook. Implementation lives in `src/playbook/`.
# This notebook imports that code, renders the graphs, and walks the current
# classify → discover → recommend flow.
#
# Behavioral baseline is unchanged from the extracted modular agent:
# Jaccard intent match, action-sequence LCS subflow match, greedy clustering,
# stub pathway drafts, and `simulate_hitl`.
#
# ```text
# notebook / UI
#      ↓ imports
# compact LangGraph application code
#      ↓
# stable callable agent boundaries
# ```

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
from pathlib import Path
from typing import Any

from IPython.display import JSON, Markdown, display

from playbook import (
    build_classify_graph,
    build_cohort_graph,
    build_discover_graph,
    build_intent_discovery_graph,
    build_intent_graph,
    build_meta_graph,
    build_pathway_graph,
    build_subflow_discovery_graph,
    build_subflow_graph,
    configure_runtime,
    empty_state,
    invoke_named,
    invoke_week,
    load_playbook,
    node_summarize_run,
    render_mermaid,
    retrieve_guidance,
)
from playbook import runtime as rt
from playbook.tools import score_intent_similarity, score_subflow_similarity

ROOT = Path(".").resolve()
playbook, store = configure_runtime(data_dir=ROOT / "scratch_data")

print("LANGSMITH_TRACING:", os.environ.get("LANGSMITH_TRACING"))
print("LANGSMITH_PROJECT:", os.environ.get("LANGSMITH_PROJECT"))
print("LANGSMITH_API_KEY set:", bool(os.environ.get("LANGSMITH_API_KEY")))
print("kb_version", playbook.version)
print("intent docs:", {i: playbook.intent_doc(i) for i in playbook.intent_ids()})
print("loaded tasks:", store.fetchall("SELECT task_id, hidden_flow, hidden_subflow, success FROM tasks"))


# %%
def show(obj: Any, title: str = "") -> None:
    if title:
        display(Markdown(f"**{title}**"))
    display(Markdown(f"```json\n{json.dumps(obj, indent=2, default=str)}\n```"))


def show_state(state, keys: list[str] | None = None, title: str = "state") -> None:
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


def show_mermaid(compiled, *, xray: bool | int = False, title: str = "") -> None:
    source = render_mermaid(compiled, xray=xray)
    if title:
        display(Markdown(f"### {title}"))
    display(Markdown(f"```mermaid\n{source}\n```"))


# %% [markdown]
# Hidden `scenario.flow` / `scenario.subflow` are **offline labels** for evaluation
# in this notebook. Classifiers and discovery never read them.

# %%
JSON(
    {
        "version": playbook.version,
        "flows": playbook.intent_ids(),
        "subflows": playbook.ontology["intents"]["subflows"],
        "kb": playbook.kb,
    }
)

# %% [markdown]
# ## Canonical loader / retriever / tools
#
# `load_playbook` and `retrieve_guidance` are independent of the graph. The
# LangChain tools still use the bound runtime store.

# %%
seed = load_playbook(ROOT / "scratch_data")
print(retrieve_guidance("I forgot my username and cannot log into my account.", seed))
print("sanity scores (intent=account_access)")
for cid in ["u1", "p1", "t1", "l1", "s1", "x1"]:
    intent = score_intent_similarity.invoke({"conversation_id": cid, "intent_id": "account_access"})
    username = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_username"})
    password = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_password"})
    print(f"  {cid:3}  intent={intent:.3f}  username={username:.3f}  password={password:.3f}")

# %% [markdown]
# ## Compiled graphs
#
# Display graphs use compiled subgraphs; the run graph calls `invoke`.

# %%
cohort_graph = build_cohort_graph()
intent_graph = build_intent_graph()
subflow_graph = build_subflow_graph()
intent_discovery_graph = build_intent_discovery_graph()
subflow_discovery_graph = build_subflow_discovery_graph()
pathway_graph = build_pathway_graph()
classify_display = build_classify_graph(True)
discover_display = build_discover_graph(True)
meta_display = build_meta_graph(True)
meta_graph = build_meta_graph(False)

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
# Expected operational outcomes:
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
# Fresh `run_id` so LangSmith has one parent trace. The walkthrough already
# mutated the bound KB, so this second invoke mostly classifies against the
# updated structure.

# %%
orchestrated = invoke_week(
    "week-orchestrated",
    {
        "source": "scratch_data/incoming_conversations.json",
        "window": "demo-week",
        "as_of": "2026-09-01",
        "note": "second invoke against already-updated KB",
    },
    compiled=meta_graph,
)
show_state(orchestrated, title="orchestrated parent state")
show(store.get_staging("week-orchestrated", "run_summary"), "orchestrated run summary")
print("runtime kb_version", rt.playbook.version)

# %% [markdown]
# ## LangSmith
#
# Filter a parent run by tag `run:week-2026-09-01` or `run:week-orchestrated`.
# Node metadata is `agent` + `phase`.
#
# **Thread / session identity (not implemented here):** a later HITL phase can
# map `run_id` to a LangGraph thread id and resume with `Command(resume=...)`.
# This notebook still uses `simulate_hitl`.

# %%
try:
    from langsmith import Client

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
#   tasks. That swap is local to the classify process functions in `playbook.graph`.
# - HITL is `simulate_hitl`. Real work is `interrupt()` + `Command(resume=...)`.
# - Pathway drafts are still pass-through stubs.
# - Do not copy this notebook over `demo_story.ipynb`; merge-back is `/apply-worktree`.
