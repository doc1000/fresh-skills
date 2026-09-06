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
# # Deep agent: subgraphs as tools
#
# Worktree-local notebook. The compiled graphs stay as they are; this notebook
# wires them as LangChain tools and optionally asks a deep agent to choose
# among them.
#
# Run Jupyter from this worktree root after `uv sync`.
#
# ```text
# user request
#      ↓
# create_deep_agent
#      ↓
# cohort / classify_* / discover_* / recommend_pathway / persist_recc
#      ↓
# compiled subgraphs + TaskStore
# ```

# %%
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

print(load_dotenv(".env", override=True))
key = os.getenv("OPENAI_API_KEY")
print(key[:10] if key else "OPENAI_API_KEY not found")

from playbook import (
    SYSTEM_PROMPT,
    classify_intent,
    classify_subflow,
    cohort,
    configure_runtime,
    create_playbook_agent,
    discover_intent,
    discover_subflow,
    persist_recc,
    recommend_pathway,
)
from playbook import runtime as rt
from playbook.fill import EDA_DATA_DIR

ROOT = Path(".").resolve()
playbook, store = configure_runtime(
    data_dir=EDA_DATA_DIR,
    store_path=EDA_DATA_DIR / "run_store.sqlite",
)



print("tasks", len(store.fetchall("SELECT task_id FROM tasks")))
print("intents", playbook.intent_ids())
print("LANGSMITH_TRACING", os.environ.get("LANGSMITH_TRACING"))

# %% [markdown]
# ## Direct tool calls
#
# These invoke the compiled graphs with no model. Use this cell to confirm
# the SQLite store survives LangGraph worker threads.

# %%
cohort_out = cohort.invoke(
    {
        "start_date": "2026-09-01",
        "end_date": "2026-09-07",
        "method": "jaccard",
        "cohort_query": {},
        "run_id": "notebook-deep-agent",
    }
)
intent_out = classify_intent.invoke({})
subflow_out = classify_subflow.invoke({})
print(cohort_out)
print(intent_out)
print(subflow_out)

# %% [markdown]
# Discovery and pathway tools. `recommend_pathway` drafts only;
# `persist_recc` writes the store / KB.

# %%
print("------------\ndiscover_intent")
print(discover_intent.invoke({}))
print("\n------------\ndiscover_subflow")
print(discover_subflow.invoke({}))
print("\n------------\nrecommend_pathway")
print(recommend_pathway.invoke({"target_subflow": "reset_2fa"}))
print("\n------------\persist_recc_pathway")
print(persist_recc.invoke({"target_subflow": "reset_2fa"}))
print("\n------------\nlist reccs")
print("recs", store.list_recommendations(rt.current_run_id))
print("\n------------\nkb version")
print("kb_version", playbook.version)

# %% [markdown]
# ### current seeded data
# seed: ~Aug 25–Sep 8
#
# noise: Sep 9–10
#
# status_payment_method: Sep 10–12
#
# slow_speed: Sep 12–14

# %% [markdown]
# ## Deep agent (optional)
#
# Needs `OPENAI_API_KEY`. The agent picks tools from the request instead of
# following a fixed classify → discover → recommend graph.

# %%
from langchain.chat_models import init_chat_model

if os.environ.get("OPENAI_API_KEY"):
    model = init_chat_model("openai:gpt-4.1-mini", temperature=0)
    agent = create_playbook_agent(model=model)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Classify the 2026-09-01 to 2026-09-07 cohort with jaccard. "
                        "Stop after classification unless unresolved tasks clearly "
                        "need discovery."
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": "notebook-deep-agent"}},
    )
    print(result["messages"][-1].content)
    result["messages"][-1].content.pretty_print()
    
else:
    print("Set OPENAI_API_KEY to invoke create_playbook_agent.")
    print("System prompt starts with:")
    print(SYSTEM_PROMPT.splitlines()[0])

# %%
thread_id="notebook-deep-agent"
def user_message(agent, content, thread_id):
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        content
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    #print(result["messages"][-1].content)
    result["messages"][-1].pretty_print()


# %%
content = """yes, run discovery over this cohort"""
user_message(agent, content, thread_id)

# %%
content = """what is the name of the new intent?"""
user_message(agent, content, thread_id)

# %%
content = """what are the available intent labels in the kb?"""
user_message(agent, content, thread_id)

# %%
content = """yes, classify unresolved tasks with the new intent"""
user_message(agent, content, thread_id)

# %%
content = """classify empty subflows"""
user_message(agent, content, thread_id)

# %%
content = """yes, for both"""
user_message(agent, content, thread_id)

# %%
content = """yes, review the emerging subflows"""
user_message(agent, content, thread_id)

# %%
