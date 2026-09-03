"""Pinned demo subset. Offline fixture construction only — not runtime answers."""

from __future__ import annotations

DEMO_FLOW = "shipping_issue"

# A/B live in the seed KB later. C is the Week 1 gap. D arrives in Week 2.
ROLE_TO_SUBFLOW = {
    "A": "manage",
    "B": "cost",
    "C": "missing",
    "D": "status",
}

SUBFLOW_TO_ROLE = {subflow: role for role, subflow in ROLE_TO_SUBFLOW.items()}

# Distinctive opening turns, unique IDs, C over-represented in Week 1.
WEEK_MEMBERSHIP: dict[str, dict[str, list[int]]] = {
    "week_1": {
        "A": [264, 729, 830, 936, 1226],
        "B": [194, 542, 653, 717, 888],
        "C": [169, 450, 481, 587, 695, 1821, 1877, 1944],
        "D": [],
    },
    "week_2": {
        "A": [1558, 1914, 2033],
        "B": [901, 1012, 1043],
        "C": [2343, 2396, 2513],
        "D": [228, 322, 434, 1128, 1439, 1582, 2199, 2372],
    },
    "week_3": {
        "A": [1031, 1466, 1909, 1983],
        "B": [1162, 1175, 1503, 1869],
        "C": [2584, 2670, 2810, 2820],
        "D": [1419, 1722, 2062, 2415],
    },
}


def all_pinned_ids() -> list[int]:
    ids: list[int] = []
    for week in WEEK_MEMBERSHIP.values():
        for role_ids in week.values():
            ids.extend(role_ids)
    return ids
