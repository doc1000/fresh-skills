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
# # Scratch: playbook-maintenance LangGraph (flow EDA)
#
# Historical EDA notebook. The supported implementation is `src/playbook/`
# (reviewed by `scratch_modular_meta_agent`). This file keeps the earlier
# parent + subflow-subgraph sketch for provenance.
#
# Worktree-local notebook. Goal is to **see the control flow**, not to implement real
# retrieval, BERTopic, or LLM judges yet. Nodes and tools are stubs / pass-throughs.
#
# Two questions the graph should answer:
#
# 1. **Where does this belong?** — existing intent vs new intent vs outlier
# 2. **Is this a real new operational pattern, and can we encode guidance?** —
#    reusable subflow evaluation, invoked from either path
#
# The reusable lower half is a compiled LangGraph **subgraph**. The parent graph
# **loops** while a candidate queue remains, so the diagram has a cycle.

# %% [markdown]
# ## Target shape
#
# ```mermaid
# flowchart TD
#     A[Incoming customer requests] --> B[Top-level intent alignment]
#     B --> C{Matches existing intent?}
#     C -->|Yes| D[Assign to existing intent]
#     D --> E[Compare against existing subflows]
#     E --> F{Matches existing subflow?}
#     F -->|Yes| G[Existing flow / no structural change]
#     F -->|No| H[Candidate unmatched request set]
#     C -->|No| I[Collect requests outside existing intents]
#     I --> J[Cluster / topic discovery]
#     J --> K{Coherent new intent?}
#     K -->|No| L[Keep as outliers / monitor]
#     K -->|Yes| M[Propose candidate new intent]
#     M --> N[Identify candidate subflow within new intent]
#     H --> S1
#     N --> S1
#     subgraph SUBFLOW["Reusable: Evaluate Candidate Subflow"]
#         S1[Cluster / group candidate requests]
#         S1 --> S2{Coherent single subflow?}
#         S2 -->|No| S3[Noise / mixed cases]
#         S2 -->|Yes| S4[Analyze successful turn traces]
#         S4 --> S5{Clear repeatable resolution pattern?}
#         S5 -->|Yes| S6[Recommend new subflow]
#         S6 --> S7[Draft KB entry]
#         S6 --> S8[Draft guideline update]
#         S7 --> S9[Human review]
#         S8 --> S9
#         S5 -->|No| S10[Emerging issue]
#         S10 --> S11[Notify manager]
#     end
# ```

# %%
from __future__ import annotations

import json
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from IPython.display import Markdown, display
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

ROOT = Path(".").resolve()
DATA = ROOT / "scratch_data"

INTENT_THRESHOLD = 0.07
SUBFLOW_THRESHOLD = 0.75
CLUSTER_THRESHOLD = 0.20
MIN_CLUSTER_SIZE = 2
MIN_SUCCESS_FOR_PATTERN = 2
PATTERN_SUPPORT = 0.66

# %% [markdown]
# ## Seed playbook (ABCD-shaped, incomplete on purpose)
#
# Structures match the public ABCD files:
#
# - `ontology.json` — `intents.flows` / `intents.subflows`
# - `kb.json` — `{subflow_id: [action-button, ...]}`
# - `guidelines.json` — `{Flow Title: {description, subflows: {Name: {actions, instructions}}}}`
#
# Seed coverage: **Account Access** with `recover_username` and `recover_password` only.
# Real ABCD also has `reset_2fa`; we leave it out so the existing-intent / new-subflow
# path has a clean gap. Shipping is absent entirely.

# %%
ontology = json.loads((DATA / "seed_ontology.json").read_text(encoding="utf-8"))
kb = json.loads((DATA / "seed_kb.json").read_text(encoding="utf-8"))
guidelines = json.loads((DATA / "seed_guidelines.json").read_text(encoding="utf-8"))
conversations = json.loads((DATA / "incoming_conversations.json").read_text(encoding="utf-8"))
CONVO_BY_ID = {c["convo_id"]: c for c in conversations}

print("ontology flows:", ontology["intents"]["flows"])
print("ontology subflows:", ontology["intents"]["subflows"])
print("kb keys:", list(kb))
print("guideline flows:", list(guidelines))
print("incoming ids:", [c["convo_id"] for c in conversations])


# %%
def show_turns(convo_id: str) -> None:
    convo = CONVO_BY_ID[convo_id]
    print(f"=== {convo_id}  hidden={convo['scenario']}  success={convo['success']} ===")
    for turn in convo["original"]:
        print(f"  [{turn['speaker']}] {turn['text']}")


show_turns("t1")
print()
show_turns("s1")

# %% [markdown]
# Hidden `scenario.flow` / `scenario.subflow` are **offline labels** for checking the
# stub. Alignment and clustering below use only flattened turn text + action names.

# %% [markdown]
# ## Stub similarity and playbook documents
#
# Stand-ins, not product math:
#
# - **Intent / clustering:** Jaccard on the full interaction (all turns) vs a
#   *slim* playbook document (flow title, description, subflow names). Using the
#   full guideline prose leaked generic "account / email / name" into shipping.
# - **Subflow:** longest-common-subsequence ratio of observed action buttons vs
#   `kb.json`. That matches the real ABCD KB shape.


# %%
def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokenize(a), tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lcs_len(left: list[str], right: list[str]) -> int:
    n, m = len(left), len(right)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if left[i - 1] == right[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[n][m]


def action_sequence_score(observed: list[str], expected: list[str]) -> float:
    if not observed or not expected:
        return 0.0
    return lcs_len(observed, expected) / max(len(observed), len(expected))


def flatten_interaction(convo: dict[str, Any]) -> str:
    return " ".join(turn["text"] for turn in convo["original"])


def action_sequence(convo: dict[str, Any]) -> list[str]:
    return [turn["text"] for turn in convo["original"] if turn["speaker"] == "action"]


TITLE_TO_FLOW_ID = {"Account Access": "account_access"}
SUBFLOW_TITLE_TO_ID = {
    "Recover Username": "recover_username",
    "Recover Password": "recover_password",
}

INTENT_DOCS: dict[str, str] = {}
for flow_title, flow_body in guidelines.items():
    flow_id = TITLE_TO_FLOW_ID[flow_title]
    INTENT_DOCS[flow_id] = " ".join(
        [flow_title, flow_id, flow_body["description"], *flow_body["subflows"], *SUBFLOW_TITLE_TO_ID.values()]
    )

print("intent docs:", INTENT_DOCS)
print("kb action lists:", kb)

# %% [markdown]
# ## Pass-through tools
#
# These are the seams where a later phase can drop in embeddings, BERTopic, or an
# LLM judge. Nodes call tools instead of inlining the stub math.


# %%
@tool
def score_intent_similarity(conversation_id: str, intent_id: str) -> float:
    """Stub semantic similarity of a full interaction against an existing intent."""
    return round(jaccard(flatten_interaction(CONVO_BY_ID[conversation_id]), INTENT_DOCS[intent_id]), 3)


@tool
def score_subflow_similarity(conversation_id: str, subflow_id: str) -> float:
    """Stub subflow match: action-sequence LCS vs kb.json buttons."""
    return round(action_sequence_score(action_sequence(CONVO_BY_ID[conversation_id]), kb[subflow_id]), 3)


@tool
def cluster_conversation_ids(conversation_ids: list[str]) -> list[list[str]]:
    """Greedy Jaccard clustering over full interaction text. Stands in for BERTopic."""
    remaining = list(conversation_ids)
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_text = flatten_interaction(CONVO_BY_ID[seed])
        group = [seed]
        kept: list[str] = []
        for other in remaining:
            other_text = flatten_interaction(CONVO_BY_ID[other])
            if jaccard(seed_text, other_text) >= CLUSTER_THRESHOLD:
                group.append(other)
            else:
                kept.append(other)
        remaining = kept
        clusters.append(group)
    return clusters


@tool
def extract_action_sequence(conversation_id: str) -> list[str]:
    """Return the action-button sequence from a conversation."""
    return action_sequence(CONVO_BY_ID[conversation_id])


@tool
def draft_kb_stub(subflow_id: str, actions: list[str]) -> dict[str, list[str]]:
    """Draft an ABCD-shaped kb.json entry. Pass-through: no model."""
    return {subflow_id: actions}


@tool
def draft_guideline_stub(
    flow_title: str,
    subflow_title: str,
    actions: list[str],
    instructions: list[str],
) -> dict[str, Any]:
    """Draft an ABCD-shaped guidelines.json subflow block. Pass-through: no model."""
    return {
        flow_title: {
            "subflows": {
                subflow_title: {
                    "actions": [
                        {
                            "type": "interaction",
                            "button": action.replace("-", " ").title(),
                            "text": f"Observed repeated action [{action}]",
                            "subtext": [],
                        }
                        for action in actions
                    ],
                    "instructions": instructions,
                }
            }
        }
    }


@tool
def notify_manager(message: str) -> str:
    """Stub manager notification. No side effects."""
    return f"[notify_manager] {message}"


# %%
print("sanity scores (intent=account_access)")
for cid in ["u1", "p1", "t1", "l1", "s1", "x1"]:
    intent = score_intent_similarity.invoke({"conversation_id": cid, "intent_id": "account_access"})
    username = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_username"})
    password = score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": "recover_password"})
    print(f"  {cid:3}  intent={intent:.3f}  username={username:.3f}  password={password:.3f}")

# %% [markdown]
# Intent Jaccard on a slim playbook doc is thin (password/lockout sit near 0.07–0.08;
# shipping stays ~0.03). If a known account conversation falls into `unmatched`,
# lower `INTENT_THRESHOLD` slightly before trusting the walk.

# %% [markdown]
# ## Graph state
#
# IDs and decisions only. Conversation bodies stay in `CONVO_BY_ID`.
# `findings` / `notifications` / `trace` append via reducers so the subgraph can
# loop without clobbering earlier work.


# %%
class PlaybookState(TypedDict):
    incoming_ids: list[str]
    intent_assignment: dict[str, str]
    assigned_existing_ids: list[str]
    unmatched_existing_ids: list[str]
    outside_ids: list[str]
    pending: dict[str, Any]
    candidate_queue: list[dict[str, Any]]
    current_candidate: dict[str, Any] | None
    group_queue: list[list[str]]
    current_group: list[str]
    findings: Annotated[list[dict[str, Any]], operator.add]
    notifications: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]


def touched(name: str) -> dict[str, list[str]]:
    return {"trace": [name]}


# %% [markdown]
# ## Reusable subgraph — evaluate a candidate set
#
# Invoked for:
#
# - unmatched requests that already sit under an existing intent
# - requests that formed a proposed **new** intent
#
# Inside the subgraph we cluster again, then loop over groups. That is the
# recursive piece: one candidate set can yield several subflow decisions.


# %%
def cluster_candidates(state: PlaybookState) -> dict[str, Any]:
    candidate = state["current_candidate"] or {}
    clusters = cluster_conversation_ids.invoke({"conversation_ids": candidate.get("conversation_ids", [])})
    return {**touched("cluster_candidates"), "group_queue": clusters, "current_group": []}


def take_next_group(state: PlaybookState) -> dict[str, Any]:
    queue = list(state.get("group_queue") or [])
    if not queue:
        return {**touched("take_next_group"), "current_group": [], "group_queue": []}
    current, *rest = queue
    return {**touched("take_next_group"), "current_group": current, "group_queue": rest}


def route_has_group(state: PlaybookState) -> Literal["judge_coherence", "subgraph_done"]:
    return "judge_coherence" if state.get("current_group") else "subgraph_done"


def judge_coherence(state: PlaybookState) -> dict[str, Any]:
    return touched("judge_coherence")


def route_coherence(state: PlaybookState) -> Literal["analyze_success_traces", "mark_noise"]:
    group = state.get("current_group") or []
    if len(group) < MIN_CLUSTER_SIZE:
        return "mark_noise"
    texts = [flatten_interaction(CONVO_BY_ID[cid]) for cid in group]
    pairwise = [jaccard(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    mean = sum(pairwise) / len(pairwise) if pairwise else 0.0
    return "analyze_success_traces" if mean >= CLUSTER_THRESHOLD else "mark_noise"


def mark_noise(state: PlaybookState) -> dict[str, Any]:
    group = state.get("current_group") or []
    candidate = state.get("current_candidate") or {}
    msg = notify_manager.invoke(
        {"message": f"Mixed/small cluster under {candidate.get('intent')} ids={group}"}
    )
    return {
        **touched("mark_noise"),
        "notifications": [msg],
        "findings": [
            {
                "kind": "noise_or_outlier",
                "intent": candidate.get("intent"),
                "source": candidate.get("source"),
                "conversation_ids": group,
            }
        ],
    }


def analyze_success_traces(state: PlaybookState) -> dict[str, Any]:
    return touched("analyze_success_traces")


def route_pattern(state: PlaybookState) -> Literal["recommend_subflow", "recommend_emerging"]:
    group = [CONVO_BY_ID[cid] for cid in state.get("current_group") or []]
    successful = [c for c in group if c.get("success")]
    if len(successful) < MIN_SUCCESS_FOR_PATTERN:
        return "recommend_emerging"
    sequences = [tuple(extract_action_sequence.invoke({"conversation_id": c["convo_id"]})) for c in successful]
    best, count = Counter(sequences).most_common(1)[0]
    return "recommend_subflow" if best and count / len(successful) >= PATTERN_SUPPORT else "recommend_emerging"


def _common_actions(state: PlaybookState) -> list[str]:
    successful = [
        CONVO_BY_ID[cid] for cid in state.get("current_group") or [] if CONVO_BY_ID[cid].get("success")
    ]
    if not successful:
        return []
    sequences = [tuple(action_sequence(c)) for c in successful]
    return list(Counter(sequences).most_common(1)[0][0])


ACTION_FINGERPRINTS = {
    ("pull-up-account", "enter-details", "send-link"): "reset_2fa",
    (
        "pull-up-account",
        "validate-purchase",
        "record-reason",
        "update-order",
        "make-purchase",
    ): "missing",
}
SUBFLOW_TITLES = {
    "reset_2fa": "Reset Two-Factor Auth",
    "missing": "Missing Item",
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
}


def _slug_from_actions(actions: list[str], fallback: str) -> str:
    if not actions:
        return fallback
    return ACTION_FINGERPRINTS.get(tuple(actions)) or "_".join(actions[-2:])


def recommend_subflow(state: PlaybookState) -> dict[str, Any]:
    candidate = state.get("current_candidate") or {}
    actions = _common_actions(state)
    label = _slug_from_actions(actions, candidate.get("proposed_subflow") or "new_subflow")
    pending = {
        "kind": "recommend_subflow",
        "intent": candidate.get("intent"),
        "source": candidate.get("source"),
        "label": label,
        "actions": actions,
        "conversation_ids": list(state.get("current_group") or []),
    }
    return {**touched("recommend_subflow"), "pending": pending}


def draft_kb_entry(state: PlaybookState) -> dict[str, Any]:
    pending = dict(state.get("pending") or {})
    pending["kb_draft"] = draft_kb_stub.invoke(
        {"subflow_id": pending["label"], "actions": pending["actions"]}
    )
    return {**touched("draft_kb_entry"), "pending": pending}


def draft_guideline_update(state: PlaybookState) -> dict[str, Any]:
    pending = dict(state.get("pending") or {})
    flow_title = str(pending.get("intent") or "New Intent").replace("_", " ").title()
    pending["guideline_draft"] = draft_guideline_stub.invoke(
        {
            "flow_title": flow_title,
            "subflow_title": SUBFLOW_TITLES.get(
                pending["label"], str(pending["label"]).replace("_", " ").title()
            ),
            "actions": pending["actions"],
            "instructions": [
                "Inferred from repeated successful traces in this batch.",
                "Needs human review before it becomes live playbook text.",
            ],
        }
    )
    return {**touched("draft_guideline_update"), "pending": pending}


def human_review(state: PlaybookState) -> dict[str, Any]:
    pending = dict(state.get("pending") or {})
    pending["review"] = {"status": "pending_review", "stub": True}
    return {**touched("human_review"), "findings": [pending], "pending": {}}


def recommend_emerging(state: PlaybookState) -> dict[str, Any]:
    candidate = state.get("current_candidate") or {}
    pending = {
        "kind": "emerging_no_guidance",
        "intent": candidate.get("intent"),
        "source": candidate.get("source"),
        "conversation_ids": list(state.get("current_group") or []),
    }
    return {**touched("recommend_emerging"), "pending": pending}


def notify_manager_node(state: PlaybookState) -> dict[str, Any]:
    pending = state.get("pending") or {}
    msg = notify_manager.invoke(
        {
            "message": (
                f"Coherent cluster under {pending.get('intent')} but no repeatable successful "
                f"pattern. ids={pending.get('conversation_ids')}"
            )
        }
    )
    return {**touched("notify_manager"), "findings": [pending], "notifications": [msg], "pending": {}}


def subgraph_done(state: PlaybookState) -> dict[str, Any]:
    return touched("subgraph_done")


def build_subflow_graph():
    graph = StateGraph(PlaybookState)
    graph.add_node("cluster_candidates", cluster_candidates)
    graph.add_node("take_next_group", take_next_group)
    graph.add_node("judge_coherence", judge_coherence)
    graph.add_node("mark_noise", mark_noise)
    graph.add_node("analyze_success_traces", analyze_success_traces)
    graph.add_node("recommend_subflow", recommend_subflow)
    graph.add_node("draft_kb_entry", draft_kb_entry)
    graph.add_node("draft_guideline_update", draft_guideline_update)
    graph.add_node("human_review", human_review)
    graph.add_node("recommend_emerging", recommend_emerging)
    graph.add_node("notify_manager", notify_manager_node)
    graph.add_node("subgraph_done", subgraph_done)

    graph.add_edge(START, "cluster_candidates")
    graph.add_edge("cluster_candidates", "take_next_group")
    graph.add_conditional_edges(
        "take_next_group",
        route_has_group,
        {"judge_coherence": "judge_coherence", "subgraph_done": "subgraph_done"},
    )
    graph.add_conditional_edges(
        "judge_coherence",
        route_coherence,
        {"analyze_success_traces": "analyze_success_traces", "mark_noise": "mark_noise"},
    )
    graph.add_edge("mark_noise", "take_next_group")
    graph.add_conditional_edges(
        "analyze_success_traces",
        route_pattern,
        {"recommend_subflow": "recommend_subflow", "recommend_emerging": "recommend_emerging"},
    )
    graph.add_edge("recommend_subflow", "draft_kb_entry")
    graph.add_edge("draft_kb_entry", "draft_guideline_update")
    graph.add_edge("draft_guideline_update", "human_review")
    graph.add_edge("human_review", "take_next_group")
    graph.add_edge("recommend_emerging", "notify_manager")
    graph.add_edge("notify_manager", "take_next_group")
    graph.add_edge("subgraph_done", END)
    return graph.compile(name="evaluate_candidate_subflow")


subflow_graph = build_subflow_graph()


def show_mermaid(compiled, *, xray: bool | int = False, title: str = "") -> None:
    source = compiled.get_graph(xray=xray).draw_mermaid()
    if title:
        display(Markdown(f"### {title}"))
    display(Markdown(f"```mermaid\n{source}\n```"))
    print(source)


show_mermaid(subflow_graph, title="Reusable subgraph (evaluate candidate subflow)")

# %% [markdown]
# The cycle `take_next_group → … → take_next_group` is the subgraph recursion:
# cluster once, then evaluate each group until the queue is empty.

# %% [markdown]
# ## Parent graph — where does this belong?


# %%
def ingest_requests(state: PlaybookState) -> dict[str, Any]:
    return {**touched("ingest_requests"), "incoming_ids": [c["convo_id"] for c in conversations]}


def align_intents(state: PlaybookState) -> dict[str, Any]:
    assignment: dict[str, str] = {}
    for cid in state["incoming_ids"]:
        scored = [
            (intent_id, score_intent_similarity.invoke({"conversation_id": cid, "intent_id": intent_id}))
            for intent_id in INTENT_DOCS
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_id, best = scored[0]
        assignment[cid] = best_id if best >= INTENT_THRESHOLD else "unmatched"
    return {**touched("align_intents"), "intent_assignment": assignment}


def partition_by_intent(state: PlaybookState) -> dict[str, Any]:
    assigned = [cid for cid, intent in state["intent_assignment"].items() if intent != "unmatched"]
    outside = [cid for cid, intent in state["intent_assignment"].items() if intent == "unmatched"]
    return {
        **touched("partition_by_intent"),
        "assigned_existing_ids": assigned,
        "outside_ids": outside,
    }


def compare_subflows(state: PlaybookState) -> dict[str, Any]:
    covered: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for cid in state.get("assigned_existing_ids") or []:
        scored = [
            (subflow_id, score_subflow_similarity.invoke({"conversation_id": cid, "subflow_id": subflow_id}))
            for subflow_id in kb
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_id, best = scored[0]
        if best >= SUBFLOW_THRESHOLD:
            covered.append(
                {
                    "kind": "existing_flow_no_change",
                    "conversation_id": cid,
                    "intent": state["intent_assignment"][cid],
                    "subflow": best_id,
                    "score": best,
                }
            )
        else:
            unmatched.append(cid)
    queue = list(state.get("candidate_queue") or [])
    if unmatched:
        queue.append(
            {
                "source": "existing_intent_unmatched_subflow",
                "intent": state["intent_assignment"][unmatched[0]],
                "conversation_ids": unmatched,
            }
        )
    return {
        **touched("compare_subflows"),
        "unmatched_existing_ids": unmatched,
        "candidate_queue": queue,
        "findings": covered,
    }


def discover_new_intents(state: PlaybookState) -> dict[str, Any]:
    outside = list(state.get("outside_ids") or [])
    clusters = cluster_conversation_ids.invoke({"conversation_ids": outside}) if outside else []
    queue = list(state.get("candidate_queue") or [])
    findings: list[dict[str, Any]] = []
    notifications: list[str] = []
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_SIZE:
            findings.append({"kind": "outlier_monitor", "conversation_ids": cluster})
            notifications.append(
                notify_manager.invoke({"message": f"Keep monitoring outlier cluster {cluster}"})
            )
            continue
        label = _propose_intent_label(cluster)
        findings.append(
            {
                "kind": "propose_new_intent",
                "intent": label,
                "conversation_ids": cluster,
            }
        )
        queue.append(
            {
                "source": "new_intent",
                "intent": label,
                "conversation_ids": cluster,
            }
        )
    return {
        **touched("discover_new_intents"),
        "candidate_queue": queue,
        "findings": findings,
        "notifications": notifications,
    }


def _propose_intent_label(cluster: list[str]) -> str:
    blob = " ".join(flatten_interaction(CONVO_BY_ID[cid]) for cid in cluster)
    if "package" in blob or "shipment" in blob or "delivered" in blob:
        return "shipping_issue"
    return "new_intent"


def take_next_candidate(state: PlaybookState) -> dict[str, Any]:
    queue = list(state.get("candidate_queue") or [])
    if not queue:
        return {**touched("take_next_candidate"), "current_candidate": None, "candidate_queue": []}
    current, *rest = queue
    return {**touched("take_next_candidate"), "current_candidate": current, "candidate_queue": rest}


def route_has_candidate(state: PlaybookState) -> Literal["evaluate_candidate_subflow", "done"]:
    return "evaluate_candidate_subflow" if state.get("current_candidate") else "done"


def done(state: PlaybookState) -> dict[str, Any]:
    return touched("done")


def build_parent_graph(subflow):
    graph = StateGraph(PlaybookState)
    graph.add_node("ingest_requests", ingest_requests)
    graph.add_node("align_intents", align_intents)
    graph.add_node("partition_by_intent", partition_by_intent)
    graph.add_node("compare_subflows", compare_subflows)
    graph.add_node("discover_new_intents", discover_new_intents)
    graph.add_node("take_next_candidate", take_next_candidate)
    graph.add_node("evaluate_candidate_subflow", subflow)
    graph.add_node("done", done)

    graph.add_edge(START, "ingest_requests")
    graph.add_edge("ingest_requests", "align_intents")
    graph.add_edge("align_intents", "partition_by_intent")
    graph.add_edge("partition_by_intent", "compare_subflows")
    graph.add_edge("compare_subflows", "discover_new_intents")
    graph.add_edge("discover_new_intents", "take_next_candidate")
    graph.add_conditional_edges(
        "take_next_candidate",
        route_has_candidate,
        {"evaluate_candidate_subflow": "evaluate_candidate_subflow", "done": "done"},
    )
    graph.add_edge("evaluate_candidate_subflow", "take_next_candidate")
    graph.add_edge("done", END)
    return graph.compile(name="playbook_maintenance")


def evaluate_candidate_subflow(state: PlaybookState) -> dict[str, Any]:
    """Invoke the compiled subgraph in isolation.

    Adding the compiled subgraph as a parent node looks right in mermaid, but
    with a shared state schema LangGraph can re-fire the parent's START when
    the subgraph starts. That replayed ingest/compare and duplicated findings.
    Isolated invoke keeps the recursive *shape* and a clean single walk.
    """
    before_findings = len(state.get("findings") or [])
    before_notes = len(state.get("notifications") or [])
    before_trace = len(state.get("trace") or [])
    inner = subflow_graph.invoke(state)
    return {
        **touched("evaluate_candidate_subflow"),
        "findings": inner["findings"][before_findings:],
        "notifications": inner["notifications"][before_notes:],
        "trace": inner["trace"][before_trace:],
        "group_queue": inner.get("group_queue") or [],
        "current_group": inner.get("current_group") or [],
        "pending": inner.get("pending") or {},
    }


parent_display = build_parent_graph(subflow_graph)
parent_graph = build_parent_graph(evaluate_candidate_subflow)
show_mermaid(parent_display, title="Parent graph (subgraph collapsed)")

# %% [markdown]
# ## Expanded diagram (recursion visible)
#
# `xray=1` inlines the reusable subgraph. You should see:
#
# - top half: ingest → align → partition → compare / discover
# - a **cycle** `take_next_candidate ⇄ evaluate_candidate_subflow`
# - inside the subgraph, another **cycle** over `take_next_group`

# %%
show_mermaid(parent_display, xray=1, title="Parent + subgraph (xray=1)")

# %% [markdown]
# ## Run the stubbed batch
#
# Expected operational outcomes (not asserted as unit tests — this is EDA):
#
# | ids | hidden label | hoped-for path |
# | --- | --- | --- |
# | `u1`, `p1` | known subflows | existing flow, no structural change |
# | `t1-t3` | `reset_2fa` | existing intent, new coherent subflow, draft KB + guideline |
# | `l1-l3` | locked / mixed | coherent-ish cluster, no successful pattern, notify manager |
# | `s1-s3` | shipping missing | new intent, then same subflow loop, draft KB + guideline |
# | `x1` | store window | outlier / monitor |

# %%
initial: PlaybookState = {
    "incoming_ids": [],
    "intent_assignment": {},
    "assigned_existing_ids": [],
    "unmatched_existing_ids": [],
    "outside_ids": [],
    "pending": {},
    "candidate_queue": [],
    "current_candidate": None,
    "group_queue": [],
    "current_group": [],
    "findings": [],
    "notifications": [],
    "trace": [],
}

result = parent_graph.invoke(initial)

print("intent_assignment:")
for cid, intent in result["intent_assignment"].items():
    hidden = CONVO_BY_ID[cid]["scenario"]
    print(f"  {cid:3} → {intent:22}  hidden={hidden}")

print("\ntrace:")
print("  " + " → ".join(result["trace"]))

print("\nnotifications:")
for note in result["notifications"]:
    print(" ", note)

print("\nfindings:")
print(json.dumps(result["findings"], indent=2))

# %% [markdown]
# ## What the walk is teaching
#
# **Kept from the mermaid**
#
# - Top half is placement. Bottom half is "is this a real, encodable pattern?"
# - One compiled subgraph, invoked from both the existing-intent gap and the
#   new-intent proposal. In LangGraph that is a node, not a copied chain.
# - Recursion is a queue loop, not a mystery recursive LLM. Easy to trace.
# - The **drawn** parent uses the compiled subgraph as a node (`parent_display`).
#   The **run** parent calls `subflow_graph.invoke` inside a wrapper so a shared
#   state schema does not re-fire the parent's `START`. Same topology either way.
#
# **Simplifications I would keep questioning**
#
# 1. **Align-then-cluster vs cluster-then-align.** We aligned each conversation
#    to a known intent first. A later BERTopic-first design (batch topics, then
#    retrieve KB) is closer to the written Phase 3 plan. This notebook is the
#    "where does it belong" framing; those two may collapse into one coverage
#    judge.
# 2. **Draft KB and guideline are sequential.** The mermaid fans them out in
#    parallel. Sequential avoids a LangGraph double-join. Parallel is cosmetic.
# 3. **Human review is a pass node.** Native `interrupt()` belongs in a later
#    phase, not this flow sketch.
# 4. **Jaccard is not semantics.** It is good enough to force the branches to
#    fire on constructed stubs. Real matching should be retrieval + an agent
#    coverage judgment, not a fixed threshold.
# 5. **Success is a fixture field.** Runtime would have to infer resolution
#    quality from the trace. That is a real product question.
# 6. **New-intent label is a keyword hack.** Fine for seeing the edge; not a
#    naming policy.
#
# **Graph I would actually implement next**
#
# If this still feels right after looking at the xray diagram, the smallest
# honest production-shaped graph is probably:
#
# `discover_topics → evaluate_coverage → (synthesize | notify | end) → human_review → update_kb`
#
# with `evaluate_candidate_subflow` living *inside* coverage/synthesize rather
# than as a second pipeline. The value of *this* sketch is making the
# existing-intent gap and the new-intent proposal share that inner loop.

# %% [markdown]
# Merge-back is `/apply-worktree`. This notebook is worktree-local EDA
# (`scratch_langgraph_flow`); do not copy it over `demo_story.ipynb`.
