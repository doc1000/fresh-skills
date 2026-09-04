# fresh-skills (LangGraph agent worktree)

Playbook-maintenance agent extracted from the modular meta-agent EDA.

```text
notebook / UI
     ↓ imports
src/playbook
     ↓
build_graph / invoke_week / load_playbook / retrieve_guidance
```

This worktree does **not** include BERTopic or the data-science topic-discovery contract.

## Setup

```text
uv sync
```

Copy `.env` if you want LangSmith traces (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).

## Primary workflow

```python
from playbook import configure_runtime, invoke_week, load_playbook, retrieve_guidance

configure_runtime()
result = invoke_week("week-2026-09-01")
```

Canonical seed files live in `scratch_data/`:

- `seed_ontology.json`
- `seed_kb.json`
- `seed_guidelines.json`
- `incoming_conversations.json`

## Review notebook

`scratch_modular_meta_agent.ipynb` imports `src/playbook` and walks classify → discover → recommend.

Older `scratch_langgraph_flow*.ipynb` notebooks are historical EDA.

## Tests

```text
uv run pytest
```
