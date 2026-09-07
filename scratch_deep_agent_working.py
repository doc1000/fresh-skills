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
starting_intro = """
Hello.  I am ready to identify customer service call intentions (intent) and sub-intentions (subflow).
I can: 
- Classify tasks according to intents that are in the knowledge base (kb).
- Discover new intent and subflows and can add them to the knowledge base.
- Reccomend pathways or agent workflows that succeeded for subflows and write them as guidelines in the knowledge base.
- explore the current kb
- run all processes to first classify according the existing kb, then discover new intents and subflows, and finally recommend to workflows

Pass date range or I will simply look for unclassified tasks.

"""

# %%
from langchain.chat_models import init_chat_model

if os.environ.get("OPENAI_API_KEY"):
    model = init_chat_model("openai:gpt-4.1-mini", temperature=0)
    agent = create_playbook_agent(model=model)
    # or construct it yourself:
    # agent = create_deep_agent(model=model, tools=[...], system_prompt=SYSTEM_PROMPT)
    content = """
    "Classify the 2026-08-25 to 2026-09-08 cohort with bertopic. 
    Stop after classification unless unresolved tasks clearly need discovery.
    """
    user_message(agent, content, thread_id)    
else:
    print("Set OPENAI_API_KEY to invoke create_playbook_agent.")
    print("System prompt starts with:")
    print(SYSTEM_PROMPT.splitlines()[0])

# %%
content = """let's classify sept 9-12th"""
user_message(agent, content, thread_id)

# %%
content = """run discovery"""
user_message(agent, content, thread_id)

# %%
content = """what new intents were rejected?"""
user_message(agent, content, thread_id)

# %%
content = """what else needs to be done?"""
user_message(agent, content, thread_id)

# %%
content = """classify the subflows"""
user_message(agent, content, thread_id)

# %%
content = """discover new subflows over the the range 9-9-26 to 9-12-26"""
user_message(agent, content, thread_id)

# %%
content = """recommend pathways"""
user_message(agent, content, thread_id)

# %%

content = """what are the recommended paths that were rejected?  i want more details"""
user_message(agent, content, thread_id)

# %%
content = """approve the payment_status_payment_method pathway"""
user_message(agent, content, thread_id)

# %%
content = """what are the current intents and subflows?"""
user_message(agent, content, thread_id)

# %%
content = """show me the pathway for payment status payment method"""
user_message(agent, content, thread_id)

# %% [markdown]
# ## target agent eval prompts
# content = """Classify the 2026-08-25 to 2026-09-08 cohort with bertopic. 
#     Stop after classification unless unresolved tasks clearly need discovery.
#     """
#
# discovery_content = """Run discovery on tasks between 2026-09-08 and 2026-09-12"""
# pathway_content = """Are there new pathways in the 2026-09-08 and 2026-09-12 cohort"""
#
#
#
#
# {"inputs": {"content": "Classify the 2026-09-01 to 2026-09-07 cohort with jaccard. Stop after classification."}, "outputs": {"expected_tools": ["cohort", "classify_intent", "classify_subflow"], "forbidden_tools": ["discover_intent", "discover_subflow", "recommend_pathway", "persist_recc"]}}

# %%
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))

# %%
print("Tools:")
for name in agent.nodes["tools"].bound._tools_by_name:
    print(f"  - {name}")

# %%
from IPython.display import Markdown, display


STANDARD_DEEP_AGENT_TOOLS = {
    "task",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "delete",
    "execute"
}


def show_agent_graph_with_domain_tools(agent):
    mermaid = agent.get_graph().draw_mermaid()

    all_tools = agent.nodes["tools"].bound._tools_by_name.keys()

    domain_tools = sorted(
        name
        for name in all_tools
        if name not in STANDARD_DEEP_AGENT_TOOLS
    )

    tool_label = "tools<br/>" + "<br/>".join(
        f"• {name}" for name in domain_tools
    )

    mermaid = mermaid.replace(
        "tools(tools)",
        f'tools["{tool_label}"]'
    )

    display(Markdown(f"```mermaid\n{mermaid}\n```"))

    #return mermaid


#show_agent_graph_with_domain_tools(agent)

# %%
import re

def tool_mermaid(tool: StructuredTool):
    source = inspect.getsource(tool.func)

    match = re.search(r'\b(build_\w+_graph)\s*\(', source)

    if not match:
        display(Markdown(f"### {tool.name}:\n{tool.description}"))
        return

    graph_builder_name = match.group(1)

    module = inspect.getmodule(tool.func)
    builder = getattr(module, graph_builder_name)

    graph = builder()
    mermaid = graph.get_graph().draw_mermaid()

    mermaid = (
        mermaid
        .replace("graph TD", "graph LR")
        .replace("flowchart TD", "flowchart LR")
    )

    display(Markdown(f"### {tool.name}\n```mermaid\n{mermaid}\n```"))


# %% [markdown]
# ## Customer Intent, Subflow and Workflow Discovery 
# ### Top-level agent workflow
# **What**: The agent operates on top of an existing customer support agent. It processes a multitude help desk tasks, aggregating them classify customer intent, the subflows within those intents and identify successful agent workflows.  
#
# **Who**: Internal help desk agent managers.  It does not interact with customers directly.
#
# **Why**: The customer complaint agent needs to identify new customer intentions, subflow and turn pathways to stay fresh.  Normally, memory only automatically improves using single traces, which can miss overall trends and causes regressions.  This agent uses machine learning topic classification and discovery techniques to provide fresh guidance based on analysis of aggregate tasks traces.  
#
# **How**: The meta agent isolates task trace data in subgraphs, accessed by tools, which then pass summarized data to the parent agent.  The objective is to allow the meta agent to see the big picture while preventing state from being overrun by individual task data.
#
#
# ```mermaid
# flowchart LR
#
#     
#     T[(Task DB/VDB)]
#
#     
#
#     subgraph P[Agent Workflow]
#         direction LR
#
#         C[Establish Cohort]
#
#         subgraph A[Analysis]
#             direction TB
#             CL[Classify]
#             D[Discover]
#             R[Recommend]
#         end
#
#         K[(Knowledge Base)]
#         L[LangSmith]
#
#         C --> CL
#         C --> D
#         C --> R
#
#         CL <--> K
#         D <--> K
#         R <--> K
#     end
#     
#     T --> C
#     
# ```
#
# The cohort is established based on agent requests for particular date ranges, intents and/or subflows with missing labels.  Rather than passing actual traces, **pointers** to the cohort are passed to subsequent nodes.  This also keeps the aggregates clean if new tasks are loaded mid-run.

# %% [markdown]
# ## Deep agent structure: model with tool calls in a loop
# This provides agent decision making with user interaction.  
#
# I had originally planned a strict workflow that reflects a deterministic ML approach:
#
# ```mermaid
# flowchart LR
#     C[Establish Cohort] --> CL[Classify]
#     CL --> D[Discover]
#     D --> R[Recommend]
#     R --> S[Summarize]
# ```
#
# However, it did not allow for as much flexibility around dates, observability and choices around discovering new intents before classifying.
#
# The chosen agent structure looks like this:

# %%
show_agent_graph_with_domain_tools(agent)

# %% [markdown]
# The agent has a limited number tools which call statistical workflows, a tool to retrieve data from the knowledge base (retrieve_guidance) and a separate persistence tool for writing reccomneded pathways to the knowledge base.
#
# There are interrupts baked into any writebacks to the knowledge base, but are set to be authorized automatically for this demo.
#
# Each turn of the main and sub-graphs are sent to langsmith, but the main agent state does not see sub-agent state.

# %% [markdown]
# ## Subgraphs as tools — shared execution pattern
#
# Every subgraph follows the same structural loop. Task objects live inside the subgraph only.
#
# The agent calls tools that 
# 1) query multiple tasks
# 2) process tasks using statistical operatations
# 3) persist the results 
# 4) summarize results for agent
#
# The subgraph nodes are labelled with the subgraph name and phases:
#  
# ```mermaid
# flowchart LR
#     Q[Query] --> P[Process]
#     P --> W[Persist]
#     W --> U[Summarize]
# ```
#
# I decided to create a shared structure for the tools so that when following the traces there is a shared concept base.  I find it gets confusing trying to follow tools that name each step in their own way.  The meta data simplifies this by labelling the family that each step belongs to.  
#
# In addition, the steps can be isolated within langsmith for specific analysis.  They can also be fairly easily found and changed in the code base if a specific phase needs to be universally changed.  
# I found that with the returned tool summaries.  At first they were too sparse (overprotecting the agent state memory) and left the agent ignorant of proposed changes and details.  Increasing the richness of the return summary was simpler with the phase labels.
#
#
# The agent does not directly call or process tasks unless it is specifically asked to assess examples.  It can call cohort and ask for task samples or get details from the knowledge base.  
#
# After a KB mutation, the next classifier **re-queries the store**. It does not
# reuse an in-memory leftover list of unresolved tasks.

# %% [markdown]
# ## Tool descriptions and subgraph flows

# %%
all_tools = agent.nodes["tools"].bound._tools_by_name.items() 
domain_tools = list( tool for name, tool in all_tools if name not in STANDARD_DEEP_AGENT_TOOLS )

for tool in domain_tools:
    tool_mermaid(tool)


# %% [markdown]
# ## Future improvements
#
# Wire in true interupts, allowing direct editing of recommended new knowledge base entries.  
# Knowledge base entries had pydantic shapes and validation.
# Process for re-classification of task intent and subflows using updated knowledge base.  
# Comparison of existing guidelines (workflow pathways) vs new guidelines with more data.  Calling customer facing agent evals vs updated knowledge base before committal to avoid regression.
# Direct agent evaluation of new kb entries based on task cohort samples.
# Integrate performance evals, checking how well classification, discovery and pathways compare to gold standards.  Current evals are build around agent execution, not statistical performance.
#

# %%
def explore_tool(tool: StructuredTool):
    print("name:", tool.name)
    print("description:", tool.description)
    print("args:", tool.args)
    print("args_schema:", tool.args_schema)
    print("func:", tool.func)
    print("coroutine:", tool.coroutine)


# %%
explore_tool(discover_intent)


# %%
