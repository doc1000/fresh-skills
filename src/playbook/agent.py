"""Deep agent that treats compiled subgraphs as tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from playbook import runtime
from playbook.graph import (
    SAMPLE_N_MAX,
    TASK_IDS_MAX,
    build_cohort_graph,
    build_intent_discovery_graph,
    build_intent_graph,
    build_pathway_draft_graph,
    build_subflow_discovery_graph,
    build_subflow_graph,
    empty_state,
    invoke_named,
    pathway_persist,
    pathway_summarize,
    public_task,
    select_task_ids,
    task_card,
)
from playbook.kb import kb_catalog
from playbook.kb import retrieve_guidance as rank_guidance

SYSTEM_PROMPT = """You manage the knowledge used by a customer-support agent to identify customer intents, subflows, and successful response pathways.

The task store contains customer-support interactions with timestamps, intent labels, subflow labels, and other task metadata. The knowledge base contains the currently recognized intents, subflows, and pathway guidelines.

Your job is to maintain and improve this knowledge using the available tools. Do not assume a fixed workflow. Select and reuse tools based on the user's request and the evidence you find.

Use `retrieve_guidance` to inspect the live knowledge base. An empty query returns the current intent and subflow catalog. Pass a query to rank existing intents. Set include_guidance to read guideline text. This is the live playbook, not seed data.

Use `cohort` to retrieve the tasks needed for an analysis. Cohorts may be selected by date, intent, subflow, labeling status, or other supported criteria. A persist call (filters only, no sample_n or task_ids) replaces the working set used by later tools and returns a run_id. A filtered persist is the new working set — classify and discover will only see that slice, not the previous broader cohort. sample_n or task_ids is a peek: it does not replace the working set and does not return a run_id. Broaden or refine the persisted cohort when the available evidence is insufficient.

Use `classify_intent` when tasks lack an intent and there is reason to believe they can be assigned to an existing intent.

Use `classify_subflow` when tasks have an intent but lack a subflow and there is reason to believe they fit an existing subflow.

Review classification results for semantic consistency with both the tasks and the existing knowledge base.

Use `discover_intent` when tasks do not appear to match the existing intent taxonomy. Treat discovery as evidence for a possible new intent, not automatic proof that one should be created. Ensure proposed names are clear, distinct, and consistent with existing naming conventions.

Use `discover_subflow` when enough related tasks exist within an intent to investigate whether a meaningful new subflow is present. Retrieve additional relevant tasks when necessary to establish sufficient evidence.

When a new or changed subflow appears warranted, use `recommend_pathway` to identify successful response patterns that could improve the knowledge base. Prefer concise, actionable guidance grounded in successful task traces.

Knowledge-base changes require the approval and persistence behavior implemented by the relevant tools. Never bypass those controls.

Stop when the user's request has been satisfied, when no justified change is supported by the available evidence, or when further progress requires human input.
"""

GUIDANCE_CHAR_CAP = 8000


def _require_runtime() -> None:
    if runtime.store is None or runtime.playbook is None:
        raise RuntimeError("Call configure_runtime() before using playbook tools.")


def _remember_run(run_id: str) -> str:
    runtime.current_run_id = run_id
    return run_id


def _run_id(run_id: str = "") -> str:
    rid = run_id or runtime.current_run_id or ""
    if not rid:
        raise ValueError("run_id is required; call cohort first or pass run_id.")
    return rid


def _slim(result: dict[str, Any], *keys: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in result:
            out[key] = result[key]
    return out


def _cap_json(payload: dict[str, Any], cap: int) -> dict[str, Any]:
    payload.setdefault("truncated", False)
    raw = json.dumps(payload)
    if len(raw) <= cap:
        return payload
    payload["truncated"] = True
    for row in payload.get("intents") or []:
        row.pop("guidance", None)
        for sub in row.get("subflows") or []:
            sub.pop("guidance", None)
    for hit in payload.get("hits") or []:
        hit.pop("guidance", None)
        hit.pop("subflow_guidance", None)
    raw = json.dumps(payload)
    if len(raw) <= cap:
        return payload
    if payload.get("intents"):
        payload["intents"] = payload["intents"][:2]
    if payload.get("hits"):
        payload["hits"] = payload["hits"][:2]
    return payload


def _enrich_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kb = runtime.playbook
    enriched = []
    for hit in hits:
        row = dict(hit)
        row["guidance"] = kb.guideline_intent_text(hit["intent_id"])
        row["subflow_guidance"] = [
            {
                "id": sid,
                "title": kb.subflow_title(sid),
                "guidance": kb.guideline_subflow_text(hit["intent_id"], sid),
            }
            for sid in hit.get("subflows") or []
        ]
        enriched.append(row)
    return enriched


@tool
def retrieve_guidance(
    query: str = "",
    include_guidance: bool = False,
    top_k: int = 4,
) -> dict:
    """Read the live knowledge base.

    Empty query: intent and subflow catalog.
    Query text: rank existing intents (Jaccard).
    include_guidance: add guideline text."""
    _require_runtime()
    if query.strip():
        hits = rank_guidance(query, runtime.playbook, top_k=top_k)
        if include_guidance:
            hits = _enrich_hits(hits)
        payload: dict[str, Any] = {
            "kb_version": runtime.playbook.version,
            "hits": hits,
            "truncated": False,
        }
    else:
        payload = kb_catalog(runtime.playbook, include_guidance=include_guidance)
        payload["truncated"] = False
    return _cap_json(payload, GUIDANCE_CHAR_CAP)


@tool
def cohort(
    start_date: str = "",
    end_date: str = "",
    method: str = "jaccard",
    cohort_query: dict[str, Any] | None = None,
    run_id: str = "",
    sample_n: int = 0,
    task_ids: list[str] | None = None,
) -> dict:
    """Retrieve a cohort of customer-support tasks, or peek at a few of them.

    Persist (default): date/filter only. Replaces the working set and
    returns run_id. Later tools use this slice only.

    Peek: pass sample_n or task_ids. Does not replace the working set
    and does not return run_id."""
    _require_runtime()
    query = dict(cohort_query or {})
    query.setdefault("start", start_date)
    query.setdefault("end", end_date)
    query.setdefault("method", method)
    ids_arg = [tid for tid in (task_ids or []) if tid]
    if sample_n or ids_arg:
        return _peek_cohort(
            start=start_date or str(query.get("start") or ""),
            end=end_date or str(query.get("end") or ""),
            query=query,
            run_id=run_id,
            sample_n=sample_n,
            task_ids=ids_arg,
        )
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required to persist a cohort.")
    state = empty_state(
        run_id=run_id,
        start=start_date,
        end=end_date,
        method=method,
        cohort_query=query,
    )
    result = invoke_named(build_cohort_graph(), state, agent="meta")
    _remember_run(result["run_id"])
    return {
        "run_id": result["run_id"],
        "current_stage": result["current_stage"],
        "cohort_summary": result["cohort_summary"],
        "kb_version": result.get("kb_version"),
    }


def _peek_cohort(
    *,
    start: str,
    end: str,
    query: dict[str, Any],
    run_id: str,
    sample_n: int,
    task_ids: list[str],
) -> dict[str, Any]:
    label_run = run_id or runtime.current_run_id or ""
    out: dict[str, Any] = {"peek": True}
    if task_ids:
        tasks = runtime.store.get_tasks(task_ids[:TASK_IDS_MAX])
        out["tasks"] = [public_task(task, label_run) for task in tasks]
        return out
    ids = select_task_ids(start=start, end=end, run_id=label_run, query=query)
    n = min(max(sample_n, 0), SAMPLE_N_MAX)
    sample = runtime.store.get_tasks(ids[:n])
    out["samples"] = [task_card(task, label_run) for task in sample]
    out["n_matching"] = len(ids)
    return out


@tool
def classify_intent(run_id: str = "") -> dict:
    """Assign existing intents to unlabeled cohort tasks.

    Use this when tasks lack an intent and there is reason to believe
    they can be assigned to an existing intent."""
    _require_runtime()
    rid = _run_id(run_id)
    result = invoke_named(build_intent_graph(), empty_state(run_id=rid), agent="intent")
    return _slim(result, "current_stage", "intent_summary", "kb_version")


@tool
def classify_subflow(run_id: str = "") -> dict:
    """Assign existing subflows to tasks that already have an intent.

    Use this when tasks have an intent but lack a subflow and there is
    reason to believe they fit an existing subflow."""
    _require_runtime()
    rid = _run_id(run_id)
    result = invoke_named(build_subflow_graph(), empty_state(run_id=rid), agent="subflow")
    return _slim(result, "current_stage", "subflow_summary", "kb_version")


@tool
def discover_intent(run_id: str = "") -> dict:
    """Look for a possible new intent among unresolved tasks.

    Treat discovery as evidence for a candidate intent, not automatic
    proof that one should be created."""
    _require_runtime()
    rid = _run_id(run_id)
    result = invoke_named(
        build_intent_discovery_graph(), empty_state(run_id=rid), agent="intent_discovery"
    )
    return _slim(
        result,
        "current_stage",
        "discovery_summary",
        "pending_proposal_ids",
        "approved_change_ids",
        "kb_version",
    )


@tool
def discover_subflow(run_id: str = "") -> dict:
    """Look for a possible new subflow among unresolved tasks in an intent.

    Retrieve additional relevant tasks when the current evidence is thin."""
    _require_runtime()
    rid = _run_id(run_id)
    result = invoke_named(
        build_subflow_discovery_graph(), empty_state(run_id=rid), agent="subflow_discovery"
    )
    return _slim(
        result,
        "current_stage",
        "discovery_summary",
        "pending_proposal_ids",
        "approved_change_ids",
        "kb_version",
    )


@tool
def recommend_pathway(target_subflow: str, run_id: str = "") -> dict:
    """Draft a pathway recommendation from successful task traces.

    Does not write the knowledge base. Use persist_recc after review
    when the draft is supported and should be kept."""
    _require_runtime()
    rid = _run_id(run_id)
    result = invoke_named(
        build_pathway_draft_graph(),
        empty_state(run_id=rid, target_subflow=target_subflow),
        agent="pathway",
    )
    payload = runtime.store.get_staging(rid, f"pathway:{target_subflow}")
    evaluation = payload.get("evaluation") if isinstance(payload, dict) else {}
    return {
        "current_stage": result.get("current_stage"),
        "target_subflow": target_subflow,
        "recommendation_summary": result.get("recommendation_summary"),
        "evaluation": evaluation,
        "kb_version": result.get("kb_version"),
    }


@tool
def persist_recc(target_subflow: str, run_id: str = "") -> dict:
    """Persist an evaluated pathway recommendation into the store and knowledge base.

    Call this after recommend_pathway. Unsupported drafts are stored
    but not attached to the live knowledge base."""
    _require_runtime()
    rid = _run_id(run_id)
    state = empty_state(run_id=rid, target_subflow=target_subflow)
    persisted = pathway_persist(state)
    summarized = pathway_summarize(state)
    rec_id = runtime.store.get_staging(rid, f"pathway:{target_subflow}:rec_id")
    return {
        "current_stage": persisted.get("current_stage"),
        "rec_id": rec_id,
        "kb_version": persisted.get("kb_version"),
        "recommendation_summary": summarized.get("recommendation_summary"),
    }


PLAYBOOK_TOOLS = [
    retrieve_guidance,
    cohort,
    classify_intent,
    classify_subflow,
    discover_intent,
    discover_subflow,
    recommend_pathway,
    persist_recc,
]


def create_playbook_agent(model: Any | None = None, **kwargs: Any):
    """Build a deep agent whose tools are the compiled playbook subgraphs."""
    from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import MemorySaver

    if model is None:
        model = init_chat_model("openai:gpt-4.1-mini", temperature=0)
    kwargs.setdefault("checkpointer", MemorySaver())
    kwargs.setdefault("name", "playbook_deep_agent")
    return create_deep_agent(
        model=model,
        tools=PLAYBOOK_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        **kwargs,
    )
