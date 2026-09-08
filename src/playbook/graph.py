"""LangGraph construction, nodes, and invoke helpers."""

from __future__ import annotations

import json
import operator
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from playbook import runtime
from playbook.actions import discover_action_paths, discover_common_workflow
from playbook.adapters import (
    RUNTIME_TOPIC_CONFIG,
    action_paths_to_state,
    classify_intents_bertopic,
    classify_subflows_bertopic,
    discovered_topics_from_clusters,
    jaccard_topic_result,
    tasks_to_conversations,
    topic_result_to_clusters,
)
from playbook.kb import retrieve_guidance
from playbook.scoring import (
    INTENT_THRESHOLD,
    LOW_INTENT_BAND,
    LOW_SUBFLOW_BAND,
    SUBFLOW_THRESHOLD,
    jaccard,
)
from playbook.store import now_iso
from playbook.topics import discover_intent_topics, discover_subflow_topics
from playbook.vectors import upsert_playbook_change
from playbook.tools import (
    draft_guideline_stub,
    draft_kb_stub,
    score_intent_similarity,
    score_subflow_similarity,
)


class MetaAgentState(TypedDict, total=False):
    run_id: str
    start: str
    end: str
    method: str
    cohort_query: dict[str, Any]
    cohort_summary: dict[str, Any]
    kb_version: int
    current_stage: str
    target_subflow: str
    intent_summary: dict[str, Any]
    subflow_summary: dict[str, Any]
    discovery_summary: dict[str, Any]
    recommendation_summary: dict[str, Any]
    pending_proposal_ids: list[str]
    approved_change_ids: list[str]
    discovered_topics: list[dict[str, Any]]
    discovered_subflows: list[dict[str, Any]]
    action_paths: list[dict[str, Any]]
    retrieved_guidance: list[dict[str, Any]]
    errors: Annotated[list[dict[str, Any]], operator.add]


SUMMARY_ID_CAP = 20
SUMMARY_TASK_IDS = 10
SAMPLE_N_MAX = 5
TASK_IDS_MAX = 2

def empty_state(**kwargs: Any) -> MetaAgentState:
    version = runtime.playbook.version if runtime.playbook is not None else 1
    state: MetaAgentState = {
        "run_id": "",
        "start": "",
        "end": "",
        "method": runtime.method,
        "cohort_query": {},
        "cohort_summary": {},
        "kb_version": version,
        "current_stage": "",
        "target_subflow": "",
        "intent_summary": {},
        "subflow_summary": {},
        "discovery_summary": {},
        "recommendation_summary": {},
        "pending_proposal_ids": [],
        "approved_change_ids": [],
        "discovered_topics": [],
        "discovered_subflows": [],
        "action_paths": [],
        "retrieved_guidance": [],
        "errors": [],
    }
    state.update(kwargs)
    return state


def _conversations_for(task_ids: list[str]):
    return tasks_to_conversations(runtime.store.get_tasks(task_ids))


def _retrieve_for_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topic in topics:
        query = topic.get("descriptor") or ""
        if not query:
            query = " ".join(topic.get("representative_conversation_ids") or [])
        rows.append(
            {
                "topic_id": topic["topic_id"],
                "query": query,
                "hits": retrieve_guidance(query),
            }
        )
    return rows


def _topic_config():
    return runtime.topic_config or RUNTIME_TOPIC_CONFIG


def _run_topic_discovery(
    task_ids: list[str],
    *,
    intent: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversations = _conversations_for(task_ids)
    config = _topic_config()
    if runtime.method == "bertopic":
        if intent is None:
            result = discover_intent_topics(conversations, config=config)
        else:
            result = discover_subflow_topics(conversations, intent=intent, config=config)
    else:
        result = jaccard_topic_result(conversations, config=config, parent_intent=intent)
    clusters = topic_result_to_clusters(result)
    if intent is not None:
        for cluster in clusters:
            cluster["intent_id"] = intent
    return clusters, discovered_topics_from_clusters(clusters)


def _merge_retrieved_guidance(
    state: MetaAgentState,
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    retrieved = list(state.get("retrieved_guidance") or [])
    retrieved.extend(_retrieve_for_topics(topics))
    return retrieved


def _sync_kb_vectors_after_persist(
    *,
    intent_ids: list[str] | None = None,
    subflow_pairs: list[tuple[str, str]] | None = None,
) -> None:
    if runtime.vectors is None or runtime.embed_fn is None or runtime.playbook is None:
        return
    upsert_playbook_change(
        runtime.vectors,
        runtime.playbook,
        runtime.embed_fn,
        intent_ids=intent_ids,
        subflow_pairs=subflow_pairs,
    )


def run_config(state: MetaAgentState, *, agent: str) -> dict[str, Any]:
    run_id = state.get("run_id") or "unassigned"
    version = runtime.playbook.version if runtime.playbook is not None else state.get("kb_version", 1)
    return {
        "run_name": f"{agent}:{run_id}",
        "tags": ["demo", "modular-meta-agent", agent, f"run:{run_id}"],
        "metadata": {
            "agent": agent,
            "run_id": run_id,
            "kb_version": version,
        },
    }


def add_phase_node(graph: StateGraph, name: str, fn, *, agent: str, phase: str) -> None:
    graph.add_node(name, fn, metadata={"agent": agent, "phase": phase})


def render_mermaid(compiled, *, xray: bool | int = False) -> str:
    return compiled.get_graph(xray=xray).draw_mermaid()


def simulate_hitl(proposal: dict[str, Any]) -> tuple[str, str]:
    ptype = proposal.get("proposal_type")
    if ptype == "outlier":
        return "decline", "Stub HITL: monitor / no playbook change."
    candidate = proposal.get("candidate") or "proposal"
    return "accept", f"Stub HITL: accept {candidate}."


def propose_intent_label(task_ids: list[str], descriptor: str = "") -> str:
    blob = " ".join(t["text"] for t in runtime.store.get_tasks(task_ids))
    if "package" in blob or "shipment" in blob or "delivered" in blob:
        return "shipping_issue"
    source = descriptor.strip()
    if not source:
        source = " ".join(t["text"] for t in runtime.store.get_tasks(task_ids)[:3])
    return _slug_label(source, "new_intent")


def _slug_label(text: str, fallback: str) -> str:
    parts: list[str] = []
    for raw in text.replace(",", " ").replace("-", " ").split():
        token = "".join(ch for ch in raw.lower() if ch.isalnum())
        if len(token) > 2:
            parts.append(token)
        if len(parts) == 4:
            break
    return "_".join(parts) if parts else fallback


def propose_subflow_label(
    task_ids: list[str],
    fallback: str,
    descriptor: str = "",
) -> tuple[str, list[str]]:
    source = descriptor.strip()
    if not source:
        source = " ".join(t["text"] for t in runtime.store.get_tasks(task_ids)[:3])
    return _slug_label(source, fallback), []


def mean_pairwise_jaccard(task_ids: list[str]) -> float:
    texts = [t["text"] for t in runtime.store.get_tasks(task_ids)]
    pairs = [jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def cohort_query(state: MetaAgentState) -> dict[str, Any]:
    query = dict(state.get("cohort_query") or {})
    start = state.get("start") or query.get("start")
    end = state.get("end") or query.get("end")
    method = state.get("method") or query.get("method") or runtime.method
    if not start or not end:
        raise ValueError("agent payload requires start and end (ISO dates YYYY-MM-DD)")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    runtime.method = method
    query["start"] = start
    query["end"] = end
    query["method"] = method
    run_id = state.get("run_id") or f"run-{start}-to-{end}"
    version = runtime.playbook.version if runtime.playbook is not None else state.get("kb_version", 1)
    runtime.store.set_staging(run_id, "cohort_query", query)
    return {
        "run_id": run_id,
        "start": start,
        "end": end,
        "method": method,
        "cohort_query": query,
        "kb_version": version,
        "current_stage": "select_time_window",
    }


def _cohort_filters(query: dict[str, Any] | None) -> dict[str, Any]:
    query = query or {}
    filters: dict[str, Any] = {}
    intent = query.get("intent") or query.get("intent_id")
    subflow = query.get("subflow") or query.get("subflow_id")
    unlabeled = query.get("unlabeled")
    if intent:
        filters["intent"] = intent
    if subflow:
        filters["subflow"] = subflow
    if unlabeled not in (None, "", False):
        filters["unlabeled"] = unlabeled
    return filters


def apply_cohort_filters(ids: list[str], run_id: str, query: dict[str, Any] | None) -> list[str]:
    filters = _cohort_filters(query)
    if not filters:
        return ids
    intents = runtime.store.latest_intents(run_id) if run_id else {}
    subflows = runtime.store.latest_subflows(run_id) if run_id else {}
    kept: list[str] = []
    for tid in ids:
        intent_row = intents.get(tid)
        sub_row = subflows.get(tid)
        intent_id = intent_row["intent_id"] if intent_row else None
        subflow_id = sub_row["subflow_id"] if sub_row else None
        if filters.get("intent") and intent_id != filters["intent"]:
            continue
        if filters.get("subflow") and subflow_id != filters["subflow"]:
            continue
        unlabeled = filters.get("unlabeled")
        if unlabeled in {True, "intent"}:
            if intent_id and intent_id != "unknown":
                continue
        elif unlabeled == "subflow":
            if not intent_id or intent_id == "unknown":
                continue
            if subflow_id and subflow_id != "unknown":
                continue
        kept.append(tid)
    return kept


def select_task_ids(
    *,
    start: str = "",
    end: str = "",
    run_id: str = "",
    query: dict[str, Any] | None = None,
) -> list[str]:
    if start and end:
        ids = [
            row["task_id"]
            for row in runtime.store.fetchall(
                """
                SELECT task_id FROM tasks
                WHERE conversation_date >= ? AND conversation_date <= ?
                ORDER BY task_id
                """,
                (start, end),
            )
        ]
    elif run_id:
        ids = runtime.store.cohort_ids(run_id)
    else:
        ids = [row["task_id"] for row in runtime.store.fetchall("SELECT task_id FROM tasks ORDER BY task_id")]
    return apply_cohort_filters(ids, run_id, query)


def _opening(task: dict[str, Any]) -> str:
    turns = task.get("turns_json")
    if isinstance(turns, str):
        turns = json.loads(turns)
    for turn in turns or []:
        speaker = turn["speaker"] if isinstance(turn, dict) else turn[0]
        text = turn["text"] if isinstance(turn, dict) else turn[1]
        if speaker == "customer" and text:
            return str(text).strip()[:200]
    return str(task.get("text") or "")[:200]


def _task_labels(task_id: str, run_id: str) -> tuple[str | None, str | None]:
    if not run_id:
        return None, None
    intent_row = runtime.store.latest_intents(run_id).get(task_id)
    sub_row = runtime.store.latest_subflows(run_id).get(task_id)
    intent_id = intent_row["intent_id"] if intent_row else None
    subflow_id = sub_row["subflow_id"] if sub_row else None
    return intent_id, subflow_id


def task_card(task: dict[str, Any], run_id: str = "") -> dict[str, Any]:
    intent_id, subflow_id = _task_labels(task["task_id"], run_id)
    return {
        "task_id": task["task_id"],
        "conversation_date": task.get("conversation_date"),
        "success": bool(task.get("success")),
        "opening": _opening(task),
        "actions": list(task.get("actions") or []),
        "intent_id": intent_id,
        "subflow_id": subflow_id,
    }


def public_task(task: dict[str, Any], run_id: str = "") -> dict[str, Any]:
    turns = task.get("turns_json")
    if isinstance(turns, str):
        turns = json.loads(turns)
    intent_id, subflow_id = _task_labels(task["task_id"], run_id)
    return {
        "task_id": task["task_id"],
        "conversation_date": task.get("conversation_date"),
        "success": bool(task.get("success")),
        "actions": list(task.get("actions") or []),
        "turns": turns or [],
        "intent_id": intent_id,
        "subflow_id": subflow_id,
    }


def _as_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return list(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value)


def proposal_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes = []
    for row in rows:
        supporting = _as_list(row.get("supporting_task_ids"))
        changes.append(
            {
                "proposal_id": row.get("proposal_id"),
                "type": row.get("proposal_type"),
                "candidate": row.get("candidate"),
                "parent": row.get("parent_intent"),
                "decision": row.get("review_decision"),
                "n_tasks": len(supporting),
                "kb_version": row.get("resulting_kb_version"),
                "examples": _as_list(row.get("examples")),
                "metrics": _as_dict(row.get("metrics")),
                "review_note": row.get("review_note"),
                "supporting_task_ids": supporting[:SUMMARY_TASK_IDS],
            }
        )
    return changes


def cohort_process(state: MetaAgentState) -> dict[str, Any]:
    ids = select_task_ids(
        start=state["start"],
        end=state["end"],
        run_id=state["run_id"],
        query=state.get("cohort_query"),
    )
    runtime.store.set_workset(state["run_id"], "cohort", ids)
    return {"current_stage": "load_window_conversations"}


def cohort_persist(state: MetaAgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    query = runtime.store.get_staging(run_id, "cohort_query") or {}
    ids = runtime.store.get_workset(run_id, "cohort")
    runtime.store.create_run(run_id, query, runtime.playbook.version)
    runtime.store.add_cohort(run_id, ids)
    return {"current_stage": "persist_cohort", "kb_version": runtime.playbook.version}


def cohort_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.cohort_ids(state["run_id"])
    dates = [
        row["conversation_date"]
        for row in runtime.store.get_tasks(ids)
        if row.get("conversation_date")
    ]
    summary = {
        "start": state.get("start"),
        "end": state.get("end"),
        "n": len(ids),
        "min_conversation_date": min(dates) if dates else None,
        "max_conversation_date": max(dates) if dates else None,
        "method": state.get("method") or runtime.method,
        "task_ids": ids[:SUMMARY_ID_CAP],
    }
    filters = _cohort_filters(state.get("cohort_query"))
    if filters:
        summary["filters"] = filters
    runtime.store.set_staging(state["run_id"], "cohort_summary", summary)
    return {"current_stage": "summarize_cohort", "cohort_summary": summary}


def build_cohort_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "select_time_window", cohort_query, agent="meta", phase="query")
    add_phase_node(graph, "load_window_conversations", cohort_process, agent="meta", phase="process")
    add_phase_node(graph, "persist_cohort", cohort_persist, agent="meta", phase="persist")
    add_phase_node(graph, "summarize_cohort", cohort_summarize, agent="meta", phase="summarize")
    graph.add_edge(START, "select_time_window")
    graph.add_edge("select_time_window", "load_window_conversations")
    graph.add_edge("load_window_conversations", "persist_cohort")
    graph.add_edge("persist_cohort", "summarize_cohort")
    graph.add_edge("summarize_cohort", END)
    return graph.compile(name="establish_cohort")


def classify_intents_process(task_ids: list[str]) -> list[dict[str, Any]]:
    if runtime.method == "bertopic":
        return classify_intents_bertopic(
            _conversations_for(task_ids),
            runtime.playbook,
            runtime.seed_examples,
            min_sim=runtime.min_sim,
            min_margin=runtime.min_margin,
            fit_rule=runtime.fit_rule,
        )
    results = []
    intent_ids = runtime.playbook.intent_ids()
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
    ids = runtime.store.unresolved_intent_ids(state["run_id"])
    runtime.store.set_workset(state["run_id"], "intent", ids)
    return {"current_stage": "select_unresolved_intents"}


def intent_process(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_workset(state["run_id"], "intent")
    runtime.store.set_staging(state["run_id"], "intent", classify_intents_process(ids))
    return {"current_stage": "score_intents"}


def intent_persist(state: MetaAgentState) -> dict[str, Any]:
    staged = runtime.store.get_staging(state["run_id"], "intent") or []
    rows = [
        {
            "task_id": row["task_id"],
            "run_id": state["run_id"],
            "intent_id": row["intent_id"],
            "confidence": row["confidence"],
            "method": "bertopic_prototype_v1" if runtime.method == "bertopic" else "jaccard_intent_doc_v1",
            "kb_version": runtime.playbook.version,
            "created_at": now_iso(),
        }
        for row in staged
    ]
    if rows:
        runtime.store.persist_intent_labels(rows)
    return {"current_stage": "persist_intent_labels", "kb_version": runtime.playbook.version}


def _intent_summary(run_id: str, processed: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [r for r in processed if r["intent_id"] != "unknown"]
    unknown = [r for r in processed if r["intent_id"] == "unknown"]
    low = [r for r in classified if r["confidence"] < LOW_INTENT_BAND]
    latest = runtime.store.latest_intents(run_id)
    by_intent: dict[str, int] = {}
    for row in latest.values():
        by_intent[row["intent_id"]] = by_intent.get(row["intent_id"], 0) + 1
    return {
        "total": len(runtime.store.cohort_ids(run_id)),
        "processed": len(processed),
        "classified": len(classified),
        "unknown": len(unknown),
        "low_confidence": len(low),
        "by_intent": by_intent,
        "still_unresolved": len(runtime.store.unresolved_intent_ids(run_id)),
        "unknown_ids": [r["task_id"] for r in unknown][:SUMMARY_ID_CAP],
        "low_confidence_ids": [r["task_id"] for r in low][:SUMMARY_ID_CAP],
    }


def intent_summarize(state: MetaAgentState) -> dict[str, Any]:
    processed = runtime.store.get_staging(state["run_id"], "intent") or []
    summary = _intent_summary(state["run_id"], processed)
    return {
        "current_stage": "summarize_intent_assignments",
        "intent_summary": summary,
        "kb_version": runtime.playbook.version,
    }


def build_intent_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "select_unresolved_intents", intent_query, agent="intent", phase="query")
    add_phase_node(graph, "score_intents", intent_process, agent="intent", phase="process")
    add_phase_node(graph, "persist_intent_labels", intent_persist, agent="intent", phase="persist")
    add_phase_node(graph, "summarize_intent_assignments", intent_summarize, agent="intent", phase="summarize")
    graph.add_edge(START, "select_unresolved_intents")
    graph.add_edge("select_unresolved_intents", "score_intents")
    graph.add_edge("score_intents", "persist_intent_labels")
    graph.add_edge("persist_intent_labels", "summarize_intent_assignments")
    graph.add_edge("summarize_intent_assignments", END)
    return graph.compile(name="classify_intents")


def classify_subflows_process(task_ids: list[str], run_id: str) -> list[dict[str, Any]]:
    latest_intents = runtime.store.latest_intents(run_id)
    if runtime.method == "bertopic":
        intent_by_task = {tid: latest_intents[tid]["intent_id"] for tid in task_ids}
        return classify_subflows_bertopic(
            _conversations_for(task_ids),
            runtime.playbook,
            runtime.seed_examples,
            intent_by_task,
            min_sim=runtime.min_sim,
            min_margin=runtime.min_margin,
            fit_rule=runtime.fit_rule,
        )
    results = []
    for tid in task_ids:
        intent_id = latest_intents[tid]["intent_id"]
        candidates = list(runtime.playbook.subflows_for(intent_id))
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
            (
                sid,
                score_subflow_similarity.invoke(
                    {"conversation_id": tid, "subflow_id": sid, "intent_id": intent_id}
                ),
            )
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
    ids = runtime.store.unresolved_subflow_ids(state["run_id"])
    runtime.store.set_workset(state["run_id"], "subflow", ids)
    return {"current_stage": "select_unresolved_subflows"}


def subflow_process(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_workset(state["run_id"], "subflow")
    runtime.store.set_staging(state["run_id"], "subflow", classify_subflows_process(ids, state["run_id"]))
    return {"current_stage": "score_subflows"}


def subflow_persist(state: MetaAgentState) -> dict[str, Any]:
    staged = runtime.store.get_staging(state["run_id"], "subflow") or []
    rows = [
        {
            "task_id": row["task_id"],
            "run_id": state["run_id"],
            "intent_id": row["intent_id"],
            "subflow_id": row["subflow_id"],
            "confidence": row["confidence"],
            "method": "bertopic_prototype_v1" if runtime.method == "bertopic" else "jaccard_subflow_doc_v1",
            "kb_version": runtime.playbook.version,
            "created_at": now_iso(),
        }
        for row in staged
    ]
    if rows:
        runtime.store.persist_subflow_labels(rows)
    return {"current_stage": "persist_subflow_labels", "kb_version": runtime.playbook.version}


def _subflow_summary(run_id: str, processed: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [r for r in processed if r["subflow_id"] != "unknown"]
    unknown = [r for r in processed if r["subflow_id"] == "unknown"]
    low = [r for r in classified if r["confidence"] < LOW_SUBFLOW_BAND]
    by_intent: dict[str, dict[str, int]] = {}
    latest = runtime.store.latest_subflows(run_id)
    intents = runtime.store.latest_intents(run_id)
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
        "still_unresolved": len(runtime.store.unresolved_subflow_ids(run_id)),
        "unknown_ids": [r["task_id"] for r in unknown][:SUMMARY_ID_CAP],
        "low_confidence_ids": [r["task_id"] for r in low][:SUMMARY_ID_CAP],
    }


def subflow_summarize(state: MetaAgentState) -> dict[str, Any]:
    processed = runtime.store.get_staging(state["run_id"], "subflow") or []
    return {
        "current_stage": "summarize_subflow_assignments",
        "subflow_summary": _subflow_summary(state["run_id"], processed),
        "kb_version": runtime.playbook.version,
    }


def build_subflow_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "select_unresolved_subflows", subflow_query, agent="subflow", phase="query")
    add_phase_node(graph, "score_subflows", subflow_process, agent="subflow", phase="process")
    add_phase_node(graph, "persist_subflow_labels", subflow_persist, agent="subflow", phase="persist")
    add_phase_node(graph, "summarize_subflow_assignments", subflow_summarize, agent="subflow", phase="summarize")
    graph.add_edge(START, "select_unresolved_subflows")
    graph.add_edge("select_unresolved_subflows", "score_subflows")
    graph.add_edge("score_subflows", "persist_subflow_labels")
    graph.add_edge("persist_subflow_labels", "summarize_subflow_assignments")
    graph.add_edge("summarize_subflow_assignments", END)
    return graph.compile(name="classify_subflows")


def _examples(task_ids: list[str]) -> list[dict[str, str]]:
    out = []
    for task in runtime.store.get_tasks(task_ids)[:3]:
        out.append({"task_id": task["task_id"], "text": task["text"][:180]})
    return out


def intent_discovery_query(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.unresolved_intent_ids(state["run_id"])
    runtime.store.set_workset(state["run_id"], "intent_discovery", ids)
    return {"current_stage": "select_unresolved_intents"}


def intent_discovery_discover(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_workset(state["run_id"], "intent_discovery")
    clusters, topics = _run_topic_discovery(ids)
    runtime.store.set_staging(state["run_id"], "intent_discovery_clusters", clusters)
    return {
        "current_stage": "discover_intent_topics",
        "discovered_topics": topics,
        "retrieved_guidance": _merge_retrieved_guidance(state, topics),
    }


def intent_discovery_validate(state: MetaAgentState) -> dict[str, Any]:
    clusters = runtime.store.get_staging(state["run_id"], "intent_discovery_clusters") or []
    candidates = []
    for cluster in clusters:
        task_ids = cluster["task_ids"]
        coherent = bool(cluster.get("cohesive_enough"))
        mean = mean_pairwise_jaccard(task_ids)
        topic = cluster.get("discovered_topic") or {}
        descriptor = topic.get("descriptor") or ""
        if coherent:
            label = propose_intent_label(task_ids, descriptor)
            candidates.append(
                {
                    "proposal_type": "new_intent",
                    "candidate": label,
                    "parent_intent": None,
                    "supporting_task_ids": task_ids,
                    "metrics": {
                        "size": len(task_ids),
                        "mean_jaccard": round(mean, 3),
                        "coherent": True,
                        "descriptor": descriptor,
                        "source": "discover_intent_topics",
                    },
                }
            )
        else:
            candidates.append(
                {
                    "proposal_type": "outlier",
                    "candidate": None,
                    "parent_intent": None,
                    "supporting_task_ids": task_ids,
                    "metrics": {
                        "size": len(task_ids),
                        "mean_jaccard": round(mean, 3),
                        "coherent": False,
                        "source": "discover_intent_topics",
                    },
                }
            )
    runtime.store.set_staging(state["run_id"], "intent_discovery", candidates)
    return {"current_stage": "validate_intent_topics"}


def intent_discovery_persist_proposal(state: MetaAgentState) -> dict[str, Any]:
    pending_ids = []
    for cand in runtime.store.get_staging(state["run_id"], "intent_discovery") or []:
        proposal_id = new_id("prop")
        pending_ids.append(proposal_id)
        runtime.store.persist_proposal(
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
    runtime.store.set_staging(state["run_id"], "intent_discovery_ids", pending_ids)
    return {"current_stage": "persist_intent_proposals", "pending_proposal_ids": pending_ids}


def intent_discovery_hitl(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_staging(state["run_id"], "intent_discovery_ids") or []
    for proposal in runtime.store.list_proposals(state["run_id"]):
        if proposal["proposal_id"] not in ids:
            continue
        decision, note = simulate_hitl(proposal)
        runtime.store.decide_proposal(proposal["proposal_id"], decision, note)
    return {"current_stage": "review_intent_proposals"}


def intent_discovery_persist_kb(state: MetaAgentState) -> dict[str, Any]:
    approved = []
    ids = set(runtime.store.get_staging(state["run_id"], "intent_discovery_ids") or [])
    for proposal in runtime.store.list_proposals(state["run_id"], "new_intent"):
        if proposal["proposal_id"] not in ids or proposal["review_decision"] != "accept":
            continue
        description = (
            "package shipment delivered missing items carrier porch tracking "
            "order purchase replacement"
            if proposal["candidate"] == "shipping_issue"
            else "newly discovered operational intent"
        )
        version = runtime.playbook.add_intent(
            proposal["candidate"],
            proposal["candidate"].replace("_", " ").title(),
            description,
        )
        evidence_ids = json.loads(proposal["supporting_task_ids"])
        metrics = json.loads(proposal["metrics"])
        runtime.store.persist_intent_labels(
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
        runtime.store.log_kb_change(version, "add_intent", {"intent": proposal["candidate"], "tasks": evidence_ids})
        runtime.store.mark_proposal_kb_version(proposal["proposal_id"], version)
        runtime.store.set_run_kb_version(state["run_id"], version)
        _sync_kb_vectors_after_persist(intent_ids=[proposal["candidate"]])
        approved.append(proposal["proposal_id"])
    return {
        "current_stage": "persist_accepted_intents",
        "kb_version": runtime.playbook.version,
        "approved_change_ids": approved,
    }


def intent_discovery_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = set(runtime.store.get_staging(state["run_id"], "intent_discovery_ids") or [])
    rows = [p for p in runtime.store.list_proposals(state["run_id"]) if p["proposal_id"] in ids]
    summary = {
        "unresolved_tasks": len(runtime.store.get_workset(state["run_id"], "intent_discovery")),
        "candidate_count": sum(1 for p in rows if p["proposal_type"] == "new_intent"),
        "outlier_count": sum(1 for p in rows if p["proposal_type"] == "outlier"),
        "approved_count": sum(1 for p in rows if p["review_decision"] == "accept"),
        "rejected_count": sum(1 for p in rows if p["review_decision"] == "decline"),
        "inserted": [
            p["candidate"]
            for p in rows
            if p["review_decision"] == "accept" and p.get("candidate")
        ],
        "changes": proposal_changes(rows),
    }
    merged = dict(state.get("discovery_summary") or {})
    merged["intents"] = summary
    return {"current_stage": "summarize_intent_discovery", "discovery_summary": merged}


def build_intent_discovery_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(
        graph, "select_unresolved_intents", intent_discovery_query, agent="intent_discovery", phase="query"
    )
    add_phase_node(
        graph, "discover_intent_topics", intent_discovery_discover, agent="intent_discovery", phase="process"
    )
    add_phase_node(
        graph, "validate_intent_topics", intent_discovery_validate, agent="intent_discovery", phase="process"
    )
    add_phase_node(
        graph, "persist_intent_proposals", intent_discovery_persist_proposal, agent="intent_discovery", phase="persist"
    )
    add_phase_node(
        graph, "review_intent_proposals", intent_discovery_hitl, agent="intent_discovery", phase="process"
    )
    add_phase_node(
        graph, "persist_accepted_intents", intent_discovery_persist_kb, agent="intent_discovery", phase="persist"
    )
    add_phase_node(
        graph, "summarize_intent_discovery", intent_discovery_summarize, agent="intent_discovery", phase="summarize"
    )
    graph.add_edge(START, "select_unresolved_intents")
    graph.add_edge("select_unresolved_intents", "discover_intent_topics")
    graph.add_edge("discover_intent_topics", "validate_intent_topics")
    graph.add_edge("validate_intent_topics", "persist_intent_proposals")
    graph.add_edge("persist_intent_proposals", "review_intent_proposals")
    graph.add_edge("review_intent_proposals", "persist_accepted_intents")
    graph.add_edge("persist_accepted_intents", "summarize_intent_discovery")
    graph.add_edge("summarize_intent_discovery", END)
    return graph.compile(name="discover_intents")


def build_intent_discovery_draft_graph():
    """Cluster and persist proposals only. KB write is a second tool call with names."""
    graph = StateGraph(MetaAgentState)
    add_phase_node(
        graph, "select_unresolved_intents", intent_discovery_query, agent="intent_discovery", phase="query"
    )
    add_phase_node(
        graph, "discover_intent_topics", intent_discovery_discover, agent="intent_discovery", phase="process"
    )
    add_phase_node(
        graph, "validate_intent_topics", intent_discovery_validate, agent="intent_discovery", phase="process"
    )
    add_phase_node(
        graph, "persist_intent_proposals", intent_discovery_persist_proposal, agent="intent_discovery", phase="persist"
    )
    add_phase_node(
        graph, "summarize_intent_discovery", intent_discovery_summarize, agent="intent_discovery", phase="summarize"
    )
    graph.add_edge(START, "select_unresolved_intents")
    graph.add_edge("select_unresolved_intents", "discover_intent_topics")
    graph.add_edge("discover_intent_topics", "validate_intent_topics")
    graph.add_edge("validate_intent_topics", "persist_intent_proposals")
    graph.add_edge("persist_intent_proposals", "summarize_intent_discovery")
    graph.add_edge("summarize_intent_discovery", END)
    return graph.compile(name="discover_intents_draft")


def subflow_discovery_query(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.unresolved_subflow_ids(state["run_id"])
    runtime.store.set_workset(state["run_id"], "subflow_discovery", ids)
    return {"current_stage": "select_unresolved_subflows"}


def subflow_discovery_discover(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_workset(state["run_id"], "subflow_discovery")
    intents = runtime.store.latest_intents(state["run_id"])
    grouped: dict[str, list[str]] = {}
    for tid in ids:
        grouped.setdefault(intents[tid]["intent_id"], []).append(tid)
    clusters: list[dict[str, Any]] = []
    topics: list[dict[str, Any]] = []
    for intent_id, group in grouped.items():
        group_clusters, group_topics = _run_topic_discovery(group, intent=intent_id)
        clusters.extend(group_clusters)
        topics.extend(group_topics)
    runtime.store.set_staging(state["run_id"], "subflow_discovery_clusters", clusters)
    return {
        "current_stage": "discover_subflow_topics",
        "discovered_subflows": topics,
        "retrieved_guidance": _merge_retrieved_guidance(state, topics),
    }


def subflow_discovery_validate(state: MetaAgentState) -> dict[str, Any]:
    candidates = []
    for cluster in runtime.store.get_staging(state["run_id"], "subflow_discovery_clusters") or []:
        task_ids = cluster["task_ids"]
        intent_id = cluster["intent_id"]
        mean = mean_pairwise_jaccard(task_ids)
        coherent = bool(cluster.get("cohesive_enough"))
        topic = cluster.get("discovered_topic") or {}
        descriptor = topic.get("descriptor") or ""
        successful = [t for t in runtime.store.get_tasks(task_ids) if t["success"]]
        if not coherent:
            ptype = "outlier"
            label, actions = None, []
        else:
            ptype = "emerging"
            label, actions = propose_subflow_label(task_ids, "new_subflow", descriptor)
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
                    "descriptor": descriptor,
                    "coherent": coherent,
                    "source": "discover_subflow_topics",
                },
            }
        )
    runtime.store.set_staging(state["run_id"], "subflow_discovery", candidates)
    return {"current_stage": "validate_subflow_topics"}


def subflow_discovery_persist_proposal(state: MetaAgentState) -> dict[str, Any]:
    pending_ids = []
    for cand in runtime.store.get_staging(state["run_id"], "subflow_discovery") or []:
        proposal_id = new_id("prop")
        pending_ids.append(proposal_id)
        runtime.store.persist_proposal(
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
    runtime.store.set_staging(state["run_id"], "subflow_discovery_ids", pending_ids)
    return {"current_stage": "persist_subflow_proposals", "pending_proposal_ids": pending_ids}


def subflow_discovery_hitl(state: MetaAgentState) -> dict[str, Any]:
    ids = runtime.store.get_staging(state["run_id"], "subflow_discovery_ids") or []
    for proposal in runtime.store.list_proposals(state["run_id"]):
        if proposal["proposal_id"] not in ids:
            continue
        decision, note = simulate_hitl(proposal)
        runtime.store.decide_proposal(proposal["proposal_id"], decision, note)
    return {"current_stage": "review_subflow_proposals"}


def subflow_discovery_persist_kb(state: MetaAgentState) -> dict[str, Any]:
    approved = list(state.get("approved_change_ids") or [])
    ids = set(runtime.store.get_staging(state["run_id"], "subflow_discovery_ids") or [])
    for proposal in runtime.store.list_proposals(state["run_id"]):
        if proposal["proposal_id"] not in ids or proposal["review_decision"] != "accept":
            continue
        if proposal["proposal_type"] not in {"new_subflow", "emerging"}:
            continue
        if not proposal.get("candidate"):
            continue
        metrics = json.loads(proposal["metrics"])
        version = runtime.playbook.add_subflow(
            proposal["parent_intent"],
            proposal["candidate"],
            actions=metrics.get("actions") or None,
            description=metrics.get("descriptor") or "",
        )
        evidence_ids = json.loads(proposal["supporting_task_ids"])
        runtime.store.persist_subflow_labels(
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
        runtime.store.log_kb_change(
            version,
            "add_subflow",
            {
                "intent": proposal["parent_intent"],
                "subflow": proposal["candidate"],
                "tasks": evidence_ids,
            },
        )
        runtime.store.mark_proposal_kb_version(proposal["proposal_id"], version)
        runtime.store.set_run_kb_version(state["run_id"], version)
        _sync_kb_vectors_after_persist(
            intent_ids=[proposal["parent_intent"]],
            subflow_pairs=[(proposal["parent_intent"], proposal["candidate"])],
        )
        approved.append(proposal["proposal_id"])
    return {
        "current_stage": "persist_accepted_subflows",
        "kb_version": runtime.playbook.version,
        "approved_change_ids": approved,
    }


def subflow_discovery_summarize(state: MetaAgentState) -> dict[str, Any]:
    ids = set(runtime.store.get_staging(state["run_id"], "subflow_discovery_ids") or [])
    rows = [p for p in runtime.store.list_proposals(state["run_id"]) if p["proposal_id"] in ids]
    summary = {
        "unresolved_tasks": len(runtime.store.get_workset(state["run_id"], "subflow_discovery")),
        "candidate_count": sum(1 for p in rows if p["proposal_type"] in {"new_subflow", "emerging"}),
        "emerging_count": sum(1 for p in rows if p["proposal_type"] == "emerging"),
        "approved_count": sum(1 for p in rows if p["review_decision"] == "accept"),
        "rejected_count": sum(1 for p in rows if p["review_decision"] == "decline"),
        "inserted": [
            {"id": p["candidate"], "parent": p["parent_intent"]}
            for p in rows
            if p["review_decision"] == "accept" and p.get("candidate")
        ],
        "changes": proposal_changes(rows),
    }
    merged = dict(state.get("discovery_summary") or {})
    merged["subflows"] = summary
    return {"current_stage": "summarize_subflow_discovery", "discovery_summary": merged}


def build_subflow_discovery_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(
        graph, "select_unresolved_subflows", subflow_discovery_query, agent="subflow_discovery", phase="query"
    )
    add_phase_node(
        graph, "discover_subflow_topics", subflow_discovery_discover, agent="subflow_discovery", phase="process"
    )
    add_phase_node(
        graph, "validate_subflow_topics", subflow_discovery_validate, agent="subflow_discovery", phase="process"
    )
    add_phase_node(
        graph,
        "persist_subflow_proposals",
        subflow_discovery_persist_proposal,
        agent="subflow_discovery",
        phase="persist",
    )
    add_phase_node(
        graph, "review_subflow_proposals", subflow_discovery_hitl, agent="subflow_discovery", phase="process"
    )
    add_phase_node(
        graph, "persist_accepted_subflows", subflow_discovery_persist_kb, agent="subflow_discovery", phase="persist"
    )
    add_phase_node(
        graph, "summarize_subflow_discovery", subflow_discovery_summarize, agent="subflow_discovery", phase="summarize"
    )
    graph.add_edge(START, "select_unresolved_subflows")
    graph.add_edge("select_unresolved_subflows", "discover_subflow_topics")
    graph.add_edge("discover_subflow_topics", "validate_subflow_topics")
    graph.add_edge("validate_subflow_topics", "persist_subflow_proposals")
    graph.add_edge("persist_subflow_proposals", "review_subflow_proposals")
    graph.add_edge("review_subflow_proposals", "persist_accepted_subflows")
    graph.add_edge("persist_accepted_subflows", "summarize_subflow_discovery")
    graph.add_edge("summarize_subflow_discovery", END)
    return graph.compile(name="discover_subflows")


def build_subflow_discovery_draft_graph():
    """Cluster and persist proposals only. KB write is a second tool call with names."""
    graph = StateGraph(MetaAgentState)
    add_phase_node(
        graph, "select_unresolved_subflows", subflow_discovery_query, agent="subflow_discovery", phase="query"
    )
    add_phase_node(
        graph, "discover_subflow_topics", subflow_discovery_discover, agent="subflow_discovery", phase="process"
    )
    add_phase_node(
        graph, "validate_subflow_topics", subflow_discovery_validate, agent="subflow_discovery", phase="process"
    )
    add_phase_node(
        graph,
        "persist_subflow_proposals",
        subflow_discovery_persist_proposal,
        agent="subflow_discovery",
        phase="persist",
    )
    add_phase_node(
        graph, "summarize_subflow_discovery", subflow_discovery_summarize, agent="subflow_discovery", phase="summarize"
    )
    graph.add_edge(START, "select_unresolved_subflows")
    graph.add_edge("select_unresolved_subflows", "discover_subflow_topics")
    graph.add_edge("discover_subflow_topics", "validate_subflow_topics")
    graph.add_edge("validate_subflow_topics", "persist_subflow_proposals")
    graph.add_edge("persist_subflow_proposals", "summarize_subflow_discovery")
    graph.add_edge("summarize_subflow_discovery", END)
    return graph.compile(name="discover_subflows_draft")


def pathway_query(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    latest = runtime.store.latest_subflows(state["run_id"])
    ids = [tid for tid, row in latest.items() if row["subflow_id"] == subflow_id]
    if not ids:
        for proposal in runtime.store.list_proposals(state["run_id"], "new_subflow"):
            if proposal["candidate"] == subflow_id and proposal["review_decision"] == "accept":
                ids = json.loads(proposal["supporting_task_ids"])
                break
    runtime.store.set_workset(state["run_id"], f"pathway:{subflow_id}", ids)
    return {"current_stage": "select_pathway_tasks"}


def pathway_analyze(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    ids = runtime.store.get_workset(state["run_id"], f"pathway:{subflow_id}")
    tasks = runtime.store.get_tasks(ids)
    successful = [t for t in tasks if t["success"]]
    traces = [list(task.get("actions") or []) for task in successful]
    workflow = discover_common_workflow(traces)
    path_result = discover_action_paths(tasks_to_conversations(successful))
    best = path_result.paths[0] if path_result.paths else None
    actions = list(workflow.tools)
    if not actions and best:
        actions = list(best.actions)[:3]
    if workflow.tools:
        support = sum(workflow.tool_support[tool] for tool in workflow.tools) / len(workflow.tools)
    elif best and path_result.n_conversations:
        support = best.count / path_result.n_conversations
    else:
        support = 0.0
    payload = {
        "task_ids": ids,
        "successful": path_result.n_conversations,
        "actions": actions,
        "support": support,
        "workflow": workflow.model_dump(),
    }
    runtime.store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    merged_paths = list(state.get("action_paths") or [])
    merged_paths.extend(action_paths_to_state(path_result))
    return {"current_stage": "analyze_action_paths", "action_paths": merged_paths}


def pathway_recommend(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = dict(runtime.store.get_staging(state["run_id"], f"pathway:{subflow_id}") or {})
    latest = runtime.store.latest_subflows(state["run_id"])
    sample_id = (payload.get("task_ids") or [None])[0]
    sample_row = latest.get(sample_id) if sample_id else None
    intent_id = sample_row["intent_id"] if sample_row else None
    if not intent_id:
        for proposal in runtime.store.list_proposals(state["run_id"], "new_subflow"):
            if proposal["candidate"] == subflow_id:
                intent_id = proposal["parent_intent"]
                break
    flow_title = runtime.playbook.flow_title(intent_id or "new_intent")
    actions = payload.get("actions") or []
    payload["intent_id"] = intent_id
    payload["kb_draft"] = draft_kb_stub.invoke({"subflow_id": subflow_id, "actions": actions})
    payload["guideline_draft"] = draft_guideline_stub.invoke(
        {
            "flow_title": flow_title,
            "subflow_title": runtime.playbook.subflow_titles.get(
                subflow_id, subflow_id.replace("_", " ").title()
            ),
            "actions": actions,
            "instructions": [
                "Inferred from repeated successful traces in this batch.",
                "Needs human review before it becomes live playbook text.",
            ],
        }
    )
    runtime.store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    return {"current_stage": "draft_pathway"}


def pathway_evaluate(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = dict(runtime.store.get_staging(state["run_id"], f"pathway:{subflow_id}") or {})
    workflow = payload.get("workflow") or {}
    actions = payload.get("actions") or []
    payload["evaluation"] = {
        "supported": bool(actions),
        "successful": payload.get("successful"),
        "support": round(payload.get("support", 0.0), 3),
        "mode": "common_workflow",
        "workflow": workflow,
    }
    runtime.store.set_staging(state["run_id"], f"pathway:{subflow_id}", payload)
    return {"current_stage": "evaluate_pathway"}


def pathway_persist(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = runtime.store.get_staging(state["run_id"], f"pathway:{subflow_id}") or {}
    rec_id = new_id("rec")
    runtime.store.persist_recommendation(
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
        version = runtime.playbook.attach_guideline(
            payload["intent_id"], subflow_id, payload.get("guideline_draft") or {}
        )
        runtime.store.log_kb_change(version, "attach_guideline", {"subflow": subflow_id})
        runtime.store.set_run_kb_version(state["run_id"], version)
        _sync_kb_vectors_after_persist(
            subflow_pairs=[(payload["intent_id"], subflow_id)],
        )
    runtime.store.set_staging(state["run_id"], f"pathway:{subflow_id}:rec_id", rec_id)
    return {"current_stage": "persist_pathway", "kb_version": runtime.playbook.version}


def pathway_summarize(state: MetaAgentState) -> dict[str, Any]:
    subflow_id = state["target_subflow"]
    payload = runtime.store.get_staging(state["run_id"], f"pathway:{subflow_id}") or {}
    rec_id = runtime.store.get_staging(state["run_id"], f"pathway:{subflow_id}:rec_id")
    task_ids = list(payload.get("task_ids") or [])
    evaluation = payload.get("evaluation") or {}
    item = {
        "subflow_id": subflow_id,
        "intent_id": payload.get("intent_id"),
        "supported": evaluation.get("supported"),
        "n_tasks": len(task_ids),
        "support": evaluation.get("support"),
        "draft_headline": " → ".join(payload.get("actions") or []) or None,
        "evaluation": evaluation,
        "examples": _examples(task_ids),
        "metrics": evaluation,
        "supporting_task_ids": task_ids[:SUMMARY_TASK_IDS],
    }
    if rec_id:
        item["rec_id"] = rec_id
    merged = dict(state.get("recommendation_summary") or {})
    items = list(merged.get("items") or [])
    items.append(item)
    merged["items"] = items
    merged["recommended"] = sum(1 for i in items if i.get("supported"))
    merged["changes"] = [
        {
            "type": "pathway_draft",
            "candidate": subflow_id,
            "parent": payload.get("intent_id"),
            "decision": "supported" if item.get("supported") else "unsupported",
            "n_tasks": item["n_tasks"],
            "kb_version": runtime.playbook.version if item.get("supported") and rec_id else None,
            "rec_id": rec_id or None,
            "examples": item["examples"],
            "metrics": evaluation,
            "supporting_task_ids": item["supporting_task_ids"],
        }
    ]
    return {
        "current_stage": "summarize_pathway",
        "recommendation_summary": merged,
        "kb_version": runtime.playbook.version,
    }


def build_pathway_graph():
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "select_pathway_tasks", pathway_query, agent="pathway", phase="query")
    add_phase_node(graph, "analyze_action_paths", pathway_analyze, agent="pathway", phase="process")
    add_phase_node(graph, "draft_pathway", pathway_recommend, agent="pathway", phase="process")
    add_phase_node(graph, "evaluate_pathway", pathway_evaluate, agent="pathway", phase="process")
    add_phase_node(graph, "persist_pathway", pathway_persist, agent="pathway", phase="persist")
    add_phase_node(graph, "summarize_pathway", pathway_summarize, agent="pathway", phase="summarize")
    graph.add_edge(START, "select_pathway_tasks")
    graph.add_edge("select_pathway_tasks", "analyze_action_paths")
    graph.add_edge("analyze_action_paths", "draft_pathway")
    graph.add_edge("draft_pathway", "evaluate_pathway")
    graph.add_edge("evaluate_pathway", "persist_pathway")
    graph.add_edge("persist_pathway", "summarize_pathway")
    graph.add_edge("summarize_pathway", END)
    return graph.compile(name="recommend_pathway")


def build_pathway_draft_graph():
    """Recommend a pathway without writing the recommendation or KB."""
    graph = StateGraph(MetaAgentState)
    add_phase_node(graph, "select_pathway_tasks", pathway_query, agent="pathway", phase="query")
    add_phase_node(graph, "analyze_action_paths", pathway_analyze, agent="pathway", phase="process")
    add_phase_node(graph, "draft_pathway", pathway_recommend, agent="pathway", phase="process")
    add_phase_node(graph, "evaluate_pathway", pathway_evaluate, agent="pathway", phase="process")
    add_phase_node(graph, "summarize_pathway", pathway_summarize, agent="pathway", phase="summarize")
    graph.add_edge(START, "select_pathway_tasks")
    graph.add_edge("select_pathway_tasks", "analyze_action_paths")
    graph.add_edge("analyze_action_paths", "draft_pathway")
    graph.add_edge("draft_pathway", "evaluate_pathway")
    graph.add_edge("evaluate_pathway", "summarize_pathway")
    graph.add_edge("summarize_pathway", END)
    return graph.compile(name="recommend_pathway_draft")


def invoke_named(compiled, state: MetaAgentState, *, agent: str) -> MetaAgentState:
    return compiled.invoke(state, config=run_config(state, agent=agent))


def node_establish_cohort(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_cohort_graph(), state, agent="meta")


def node_classify_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_intent_graph(), state, agent="intent")


def node_classify_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_subflow_graph(), state, agent="subflow")


def node_discover_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_intent_discovery_graph(), state, agent="intent_discovery")


def node_reclassify_intents(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_intent_graph(), state, agent="intent")


def node_discover_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_subflow_discovery_graph(), state, agent="subflow_discovery")


def node_reclassify_subflows(state: MetaAgentState) -> dict[str, Any]:
    return invoke_named(build_subflow_graph(), state, agent="subflow")


def node_recommend(state: MetaAgentState) -> dict[str, Any]:
    out = dict(state)
    for proposal in runtime.store.list_proposals(state["run_id"], "new_subflow"):
        if proposal["review_decision"] != "accept":
            continue
        out = invoke_named(
            build_pathway_graph(),
            {**out, "target_subflow": proposal["candidate"]},
            agent="pathway",
        )
    return {
        "current_stage": "recommend",
        "recommendation_summary": out.get("recommendation_summary") or {"items": [], "recommended": 0},
        "kb_version": runtime.playbook.version,
        "action_paths": out.get("action_paths") or [],
    }


def node_summarize_run(state: MetaAgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    summary = {
        "run_id": run_id,
        "kb_version": runtime.playbook.version,
        "cohort_size": len(runtime.store.cohort_ids(run_id)),
        "intent_summary": state.get("intent_summary"),
        "subflow_summary": state.get("subflow_summary"),
        "discovery_summary": state.get("discovery_summary"),
        "recommendation_summary": state.get("recommendation_summary"),
        "intents_in_kb": runtime.playbook.intent_ids(),
        "subflows_in_kb": runtime.playbook.ontology["intents"]["subflows"],
        "unresolved_intents": runtime.store.unresolved_intent_ids(run_id),
        "unresolved_subflows": runtime.store.unresolved_subflow_ids(run_id),
    }
    runtime.store.set_staging(run_id, "run_summary", summary)
    return {"current_stage": "summarize", "run_id": run_id, "kb_version": runtime.playbook.version}


def build_classify_graph(use_compiled: bool):
    graph = StateGraph(MetaAgentState)
    if use_compiled:
        graph.add_node("classify_intents", build_intent_graph(), metadata={"agent": "intent", "phase": "process"})
        graph.add_node("classify_subflows", build_subflow_graph(), metadata={"agent": "subflow", "phase": "process"})
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
        graph.add_node(
            "discover_intents", build_intent_discovery_graph(), metadata={"agent": "intent_discovery", "phase": "process"}
        )
        graph.add_node("reclassify_intents", build_intent_graph(), metadata={"agent": "intent", "phase": "process"})
        graph.add_node(
            "discover_subflows",
            build_subflow_discovery_graph(),
            metadata={"agent": "subflow_discovery", "phase": "process"},
        )
        graph.add_node("reclassify_subflows", build_subflow_graph(), metadata={"agent": "subflow", "phase": "process"})
    else:
        graph.add_node(
            "discover_intents", node_discover_intents, metadata={"agent": "intent_discovery", "phase": "process"}
        )
        graph.add_node("reclassify_intents", node_reclassify_intents, metadata={"agent": "intent", "phase": "process"})
        graph.add_node(
            "discover_subflows", node_discover_subflows, metadata={"agent": "subflow_discovery", "phase": "process"}
        )
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


def build_graph(use_compiled: bool = False):
    return build_meta_graph(use_compiled)


def invoke_week(run_id: str, cohort_query: dict[str, Any] | None = None, *, compiled=None) -> MetaAgentState:
    """Thin test wrapper. The public entrypoint is `build_graph(...); agent.invoke(payload)`."""
    graph = compiled or build_graph()
    query = dict(cohort_query or {})
    payload = empty_state(
        run_id=run_id,
        start=query.get("start") or "",
        end=query.get("end") or "",
        method=query.get("method") or runtime.method,
        kb_version=runtime.playbook.version if runtime.playbook is not None else 1,
        cohort_query=query,
    )
    return graph.invoke(payload, config=run_config(payload, agent="meta"))
