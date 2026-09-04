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
# Worktree-local review notebook. It imports `src/playbook` and walks:
#
# ```text
# weekly batch
# → DS topic/intent discovery
# → DS subflow discovery
# → DS action-path discovery
# → normalized agent state
# → current playbook/RAG lookup
# → agent decision / HITL
# ```
#
# Ownership is unchanged:
#
# * **Agent** owns orchestration, `MetaAgentState`, KB/playbook loading, RAG, tools, HITL, persistence.
# * **DS** owns `discover_intent_topics`, `discover_subflow_topics`, `discover_action_paths`.
# * `playbook.adapters` is the only boundary. BERTopic objects do not enter graph state.
#
# This notebook does **not** hide mismatches.

# %%
from __future__ import annotations

import json

from IPython.display import Markdown, display

from playbook import (
    build_graph,
    configure_runtime,
    invoke_week,
    load_playbook,
    render_mermaid,
    retrieve_guidance,
)
from playbook import runtime as rt
from playbook.actions import discover_action_paths
from playbook.adapters import RUNTIME_TOPIC_CONFIG, tasks_to_conversations
from playbook.data import load_week
from playbook.kb import DEFAULT_DATA_DIR
from playbook.schemas import DiscoveredTopic
from playbook.topics import BertopicConfig, discover_intent_topics

playbook, store = configure_runtime(
    data_dir=DEFAULT_DATA_DIR,
    store_path=DEFAULT_DATA_DIR / "integrate_run_store.sqlite",
)


def show(obj, title: str = "") -> None:
    if title:
        display(Markdown(f"**{title}**"))
    display(Markdown(f"```json\n{json.dumps(obj, indent=2, default=str)}\n```"))


# %% [markdown]
# ## Mismatches left visible
#
# | Area | DS side | Agent side | Adapter choice |
# |------|---------|------------|----------------|
# | Weekly batch | `load_week("week_1")` from `data/demo/weeks.parquet` | `configure_runtime` + `scratch_data/incoming_conversations.json` | Demo HITL still uses the agent cohort. DS week is shown separately. |
# | Conversation | `ConversationRecord` (no labels) | `TaskStore` rows still have `hidden_flow` / `hidden_subflow` | `tasks_to_conversations` drops those fields. |
# | Topic output | `TopicInfo` + `TopicDiscoveryResult` (memberships, `cohesive_enough`, `BertopicConfig`) | Plan-frozen `DiscoveredTopic` | Graph state stores `DiscoveredTopic.model_dump()` only. |
# | Cluster size | Defaults `min_to_cluster=5`, `min_topic_n=5` | Demo leftover sets are often 3–4 conversations | Runtime uses `RUNTIME_TOPIC_CONFIG` (min 2). |
# | Retrieval | `config.KB_JSON` → `data/raw/kb.json` (path constant only) | `load_playbook` / `retrieve_guidance` over `scratch_data/seed_*.json` | Agent KB is the only runtime retriever. |
# | Action paths | Counts every conversation | Pathway node evaluates successful traces, then reads DS paths | Intentional: DS function is called; success filter stays on the agent. |
# | Labels | Never produced by BERTopic | `propose_*_label` + `simulate_hitl` still name/accept proposals | Discovery ≠ naming. |

# %%
ds_defaults = BertopicConfig()
show(
    {
        "ds_default_min_to_cluster": ds_defaults.min_to_cluster,
        "ds_default_min_topic_n": ds_defaults.min_topic_n,
        "runtime_min_to_cluster": RUNTIME_TOPIC_CONFIG.min_to_cluster,
        "runtime_min_topic_n": RUNTIME_TOPIC_CONFIG.min_topic_n,
        "agent_kb_dir": str(DEFAULT_DATA_DIR),
        "agent_kb_files": ["seed_ontology.json", "seed_kb.json", "seed_guidelines.json"],
        "ds_raw_kb_constant": "playbook.config.KB_JSON (not used at runtime)",
    },
    "integration choices",
)

# %% [markdown]
# ## 1. Weekly batch
#
# Two batch sources exist after the merge. The agent path uses the scratch cohort
# so the existing HITL demo still runs. The DS parquet week is unlabeled.

# %%
ds_week = load_week("week_1")
agent_tasks = store.fetchall("SELECT task_id, hidden_flow, hidden_subflow, success FROM tasks")
agent_batch = tasks_to_conversations(store.get_tasks([row["task_id"] for row in agent_tasks]))

show(
    {
        "ds_week_id": ds_week.week_id,
        "ds_conversation_ids": ds_week.conversation_ids,
        "ds_record_fields": sorted(ds_week.conversations[0].model_dump()),
        "agent_task_ids": [row["task_id"] for row in agent_tasks],
        "agent_store_still_has_hidden_labels": bool(agent_tasks[0].get("hidden_flow")),
        "adapted_record_fields": sorted(agent_batch[0].model_dump()),
    },
    "batch",
)
assert set(agent_batch[0].model_dump()) == {"conversation_id", "turns", "actions"}

# %% [markdown]
# ## 2–4. DS discovery functions (callable, not a new architecture)
#
# `fit_topic_model` is stubbed here so the notebook stays fast and deterministic.
# The real `discover_*` wrappers still run. Production nodes call the same wrappers.

# %%
from playbook import topics as topic_mod

original_fit = topic_mod.fit_topic_model


def notebook_fit(documents, config, **kwargs):
    topic_ids = []
    descriptors = {}
    for document in documents:
        text = document.lower()
        if any(token in text for token in ("package", "shipment", "delivered", "porch", "carrier")):
            topic_ids.append(0)
            descriptors[0] = "package, missing, delivered"
        elif any(token in text for token in ("two-factor", "authenticator")):
            topic_ids.append(0)
            descriptors[0] = "two-factor, authenticator, reset"
        elif any(token in text for token in ("locked", "lockout")):
            topic_ids.append(1)
            descriptors[1] = "account, locked, lockout"
        else:
            topic_ids.append(-1)
            descriptors.setdefault(-1, "")
    from playbook.topics import FittedTopics

    return FittedTopics(topic_ids=topic_ids, descriptors=descriptors)


topic_mod.fit_topic_model = notebook_fit
print("fit_topic_model stubbed:", topic_mod.fit_topic_model is not original_fit)
print("discover_intent_topics still the package function:", discover_intent_topics)

direct_intents = discover_intent_topics(agent_batch, config=RUNTIME_TOPIC_CONFIG)
direct_paths = discover_action_paths(agent_batch)
show([topic.model_dump() for topic in direct_intents.topics], "direct DS intent topics (includes noise topic -1)")
show(direct_paths.model_dump(), "direct DS action paths")

# %% [markdown]
# ## 5–7. Real agent path: graph state, RAG, decision
#
# `invoke_week` still uses the existing meta-agent:
# `establish_cohort → classify → discover → recommend → summarize`.
# Discover nodes now call the DS functions and `retrieve_guidance`.

# %%
result = invoke_week(
    "week-2026-09-01",
    {
        "source": "scratch_data/incoming_conversations.json",
        "window": "demo-week",
        "as_of": "2026-09-01",
    },
)

show(
    {
        "run_id": result["run_id"],
        "current_stage": result["current_stage"],
        "kb_version": result["kb_version"],
        "discovered_topics": result["discovered_topics"],
        "discovered_subflows": result["discovered_subflows"],
        "action_paths": result["action_paths"],
        "retrieved_guidance": result["retrieved_guidance"],
        "discovery_summary": result["discovery_summary"],
        "recommendation_summary": result["recommendation_summary"],
    },
    "normalized graph state",
)

for row in result["discovered_topics"]:
    DiscoveredTopic.model_validate(row)
for row in result["discovered_subflows"]:
    DiscoveredTopic.model_validate(row)

proposals = [
    {
        "proposal_type": row["proposal_type"],
        "candidate": row["candidate"],
        "review_decision": row["review_decision"],
        "parent_intent": row["parent_intent"],
    }
    for row in rt.store.list_proposals("week-2026-09-01")
]
show(proposals, "HITL decisions")
show(rt.store.list_recommendations("week-2026-09-01"), "pathway recommendations")

# %% [markdown]
# ## Canonical KB / playbook path
#
# One retriever: `playbook.kb.retrieve_guidance` over `scratch_data` seed files.

# %%
seed = load_playbook(DEFAULT_DATA_DIR)
manual_hits = retrieve_guidance("package missing delivered", seed)
show(
    {
        "retriever": "playbook.kb.retrieve_guidance",
        "data_dir": str(DEFAULT_DATA_DIR),
        "seed_intents": seed.intent_ids(),
        "seed_subflows": seed.ontology["intents"]["subflows"],
        "manual_hits": manual_hits,
        "graph_used_same_retriever": bool(result["retrieved_guidance"]),
        "live_intents_after_hitl": rt.playbook.intent_ids(),
        "live_subflows_after_hitl": rt.playbook.ontology["intents"]["subflows"],
    },
    "canonical KB",
)

# %% [markdown]
# ## Mermaid (compiled subgraphs, x-ray)

# %%
source = render_mermaid(build_graph(True), xray=1)
display(Markdown(f"```mermaid\n{source}\n```"))
print("graph name:", build_graph(False).name)
print("mermaid has discover:", "discover" in source)
print("mermaid has recommend:", "recommend" in source)

# %%
topic_mod.fit_topic_model = original_fit
print("restored fit_topic_model")
