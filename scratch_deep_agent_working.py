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
hf auth whoami

# %%
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")

# %%
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint

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
# close an existing store connection so this block can be rerun
store = getattr(rt, "store", None)
conn = getattr(getattr(store, "_local", None), "conn", None) if store else None
if conn is not None:
    conn.close()
    store._local.conn = None

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
        "method": "bert",
        "cohort_query": {},
        "run_id": "notebook-deep-agent",
    }
)
intent_out = classify_intent.invoke({})
subflow_out = classify_subflow.invoke({})

print(f"cohort output\n")
pprint(cohort_out)
print("\nIntent output")
pprint(intent_out)
print("\nsubflow output")
pprint(subflow_out)

# %% [markdown]
# Discovery and pathway tools. `recommend_pathway` drafts only;
# `persist_recc` writes the store / KB.

# %%
print("------------\ndiscover_intent")
pprint(discover_intent.invoke({}))
print("\n------------\ndiscover_subflow")
pprint(discover_subflow.invoke({}))
print("\n------------\nrecommend_pathway")
pprint(recommend_pathway.invoke({"target_subflow": "reset_2fa"}))
print("\n------------\persist_recc_pathway")
pprint(persist_recc.invoke({"target_subflow": "reset_2fa"}))
print("\n------------\nlist reccs")
pprint(store.list_recommendations(rt.current_run_id))
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
from langchain.chat_models import init_chat_model

if os.environ.get("OPENAI_API_KEY"):
    model = init_chat_model("openai:gpt-4.1-mini", temperature=0)
    agent = create_playbook_agent(model=model)
    # or construct it yourself:
    # agent = create_deep_agent(model=model, tools=[...], system_prompt=SYSTEM_PROMPT)
    content = """
    "Classify the 2026-09-01 to 2026-09-07 cohort with jaccard. 
    Stop after classification unless unresolved tasks clearly need discovery.
    """
    user_message(agent, content, thread_id)    
else:
    print("Set OPENAI_API_KEY to invoke create_playbook_agent.")
    print("System prompt starts with:")
    print(SYSTEM_PROMPT.splitlines()[0])

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
content = """what are the existing subflows in the kb?"""
user_message(agent, content, thread_id)

# %%
content = """go ahead and classify using existing subflows"""
user_message(agent, content, thread_id)

# %%

content = """why were none classified to existing subflows?"""
user_message(agent, content, thread_id)

# %%
content = """yes, discover new subflows"""
user_message(agent, content, thread_id)

# %%
content = """tell me about the emerging subflow proposals"""
user_message(agent, content, thread_id)

# %%
content = """can you get task summaries for a sample from prop_efba1912"""
user_message(agent, content, thread_id)

# %%
