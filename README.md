# fresh-skills (topic + agent integration worktree)

LangGraph playbook-maintenance agent with BERTopic discovery used as callable
capabilities. Neither subsystem was redesigned.

```text
weekly batch
     ↓
DS discover_intent_topics / discover_subflow_topics / discover_action_paths
     ↓
adapters → MetaAgentState
     ↓
load_playbook / retrieve_guidance
     ↓
existing classify → discover → recommend → HITL flow
```

* **Agent owns** orchestration, graph state, KB/playbook loading, RAG, tools, HITL, persistence.
* **DS owns** intent/topic discovery, subflow discovery, and action-path discovery.
* `src/playbook/adapters.py` is the only boundary. BERTopic objects do not enter graph state.

## Setup

```text
uv sync
```

Copy `.env` if you want LangSmith traces (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).

## Primary workflow

```python
from playbook import build_graph, configure_runtime, retrieve_guidance

playbook, store = configure_runtime(method="jaccard")
agent = build_graph()
result = agent.invoke(
    {
        "run_id": "week-2026-09-01",
        "start": "2026-09-01",
        "end": "2026-09-07",
        "method": "jaccard",
        "kb_version": playbook.version,
    }
)
```

Canonical runtime KB is still `scratch_data/seed_*.json` via `load_playbook` /
`retrieve_guidance`. DS `data/raw/kb.json` is not a runtime source.

## Integration notebook

`notebooks/integrate_topic_agent.ipynb` imports `src/playbook`, fills Agent KB
files by hand, then calls `agent.invoke` with an explicit `start`/`end` window.
`invoke_week` remains a thin test wrapper.

## Tests

```text
uv run pytest
```
