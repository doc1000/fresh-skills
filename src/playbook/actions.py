"""Action-path discovery. Independent of topic clustering and notebook orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from playbook.schemas import ConversationRecord

DEFAULT_MIN_TOOL_SUPPORT = 0.30
DEFAULT_MIN_ORDER_SUPPORT = 0.35


class ActionPath(BaseModel):
    actions: list[str]
    count: int
    conversation_ids: list[str]


class ActionPathResult(BaseModel):
    n_conversations: int
    n_unique_paths: int
    paths: list[ActionPath]
    button_counts: dict[str, int]


class WorkflowEdge(BaseModel):
    source: str
    target: str
    precedence: float


class PairPrecedence(BaseModel):
    a: str
    b: str
    a_before_b: float
    b_before_a: float
    n_both: int


class CommonWorkflow(BaseModel):
    n_traces: int
    tools: list[str]
    tool_support: dict[str, float]
    edges: list[WorkflowEdge]
    precedence: list[PairPrecedence]
    min_tool_support: float
    min_order_support: float


def discover_action_paths(conversations: Sequence[ConversationRecord]) -> ActionPathResult:
    """Count exact action sequences and per-button presence. No BERTopic."""
    records = list(conversations)
    members: dict[tuple[str, ...], list[str]] = defaultdict(list)
    buttons: Counter[str] = Counter()
    for record in records:
        path = tuple(record.actions)
        members[path].append(record.conversation_id)
        buttons.update(set(path))
    paths = [
        ActionPath(actions=list(actions), count=len(ids), conversation_ids=ids)
        for actions, ids in sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    return ActionPathResult(
        n_conversations=len(records),
        n_unique_paths=len(paths),
        paths=paths,
        button_counts=dict(buttons.most_common()),
    )


def _first_index(trace: Sequence[str], tool: str) -> int | None:
    for index, name in enumerate(trace):
        if name == tool:
            return index
    return None


def _order_tools(
    tools: Sequence[str],
    edges: Sequence[WorkflowEdge],
    tool_support: dict[str, float],
) -> list[str]:
    nodes = list(tools)
    incoming = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        if edge.source not in incoming or edge.target not in incoming:
            continue
        outgoing[edge.source].append(edge.target)
        incoming[edge.target] += 1

    def sort_key(node: str) -> tuple[float, str]:
        return (-tool_support.get(node, 0.0), node)

    ready = sorted((node for node in nodes if incoming[node] == 0), key=sort_key)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for nxt in outgoing[node]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=sort_key)
    leftover = sorted((node for node in nodes if node not in ordered), key=sort_key)
    return ordered + leftover


def discover_common_workflow(
    traces: Sequence[Sequence[str]],
    *,
    min_tool_support: float = DEFAULT_MIN_TOOL_SUPPORT,
    min_order_support: float = DEFAULT_MIN_ORDER_SUPPORT,
) -> CommonWorkflow:
    """Infer a compact common workflow from tool-call traces.

    DFG-inspired heuristic, tuned by two thresholds rather than by a notion of
    statistical significance:

    * `min_tool_support` (0.30) keeps a tool that appears in at least that
      fraction of traces.
    * `min_order_support` (0.35) adds a directed edge when that fraction of the
      traces containing both tools saw one before the other. At 0.35 both
      directions of an evenly split pair clear the bar, and the first in sorted
      order wins — the pair is ordered arbitrarily but deterministically.

    Raise `min_order_support` above 0.5 to keep evenly split pairs unordered.
    """
    records = [list(trace) for trace in traces]
    n_traces = len(records)
    if n_traces == 0:
        return CommonWorkflow(
            n_traces=0,
            tools=[],
            tool_support={},
            edges=[],
            precedence=[],
            min_tool_support=min_tool_support,
            min_order_support=min_order_support,
        )

    present = Counter()
    for trace in records:
        present.update(set(trace))
    tool_support = {
        tool: round(present[tool] / n_traces, 3)
        for tool in sorted(present)
    }
    retained = [tool for tool, support in tool_support.items() if support >= min_tool_support]

    pairs: list[PairPrecedence] = []
    edges: list[WorkflowEdge] = []
    for i, a in enumerate(retained):
        for b in retained[i + 1 :]:
            a_first = 0
            b_first = 0
            n_both = 0
            for trace in records:
                ia = _first_index(trace, a)
                ib = _first_index(trace, b)
                if ia is None or ib is None:
                    continue
                n_both += 1
                if ia < ib:
                    a_first += 1
                elif ib < ia:
                    b_first += 1
            if n_both == 0:
                continue
            a_before_b = round(a_first / n_both, 3)
            b_before_a = round(b_first / n_both, 3)
            pairs.append(
                PairPrecedence(
                    a=a,
                    b=b,
                    a_before_b=a_before_b,
                    b_before_a=b_before_a,
                    n_both=n_both,
                )
            )
            if a_before_b > min_order_support:
                edges.append(WorkflowEdge(source=a, target=b, precedence=a_before_b))
            elif b_before_a > min_order_support:
                edges.append(WorkflowEdge(source=b, target=a, precedence=b_before_a))

    edges.sort(key=lambda edge: (edge.source, edge.target))
    pairs.sort(key=lambda row: (row.a, row.b))
    return CommonWorkflow(
        n_traces=n_traces,
        tools=_order_tools(retained, edges, tool_support),
        tool_support=tool_support,
        edges=edges,
        precedence=pairs,
        min_tool_support=min_tool_support,
        min_order_support=min_order_support,
    )
