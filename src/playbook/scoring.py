"""Stub similarity used by classification, clustering, and retrieval."""

from __future__ import annotations

import re

INTENT_THRESHOLD = 0.07
SUBFLOW_THRESHOLD = 0.07
CLUSTER_THRESHOLD = 0.20
MIN_CLUSTER_SIZE = 2
MIN_SUCCESS_FOR_PATTERN = 2
PATTERN_SUPPORT = 0.66
LOW_INTENT_BAND = 0.09
LOW_SUBFLOW_BAND = 0.09


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


def greedy_jaccard_clusters(
    ids: list[str],
    texts: dict[str, str],
    threshold: float = CLUSTER_THRESHOLD,
) -> list[list[str]]:
    """Greedy Jaccard clustering. Fast stand-in for BERTopic."""
    remaining = list(ids)
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_text = texts.get(seed, "")
        group = [seed]
        kept: list[str] = []
        for other in remaining:
            if jaccard(seed_text, texts.get(other, "")) >= threshold:
                group.append(other)
            else:
                kept.append(other)
        remaining = kept
        clusters.append(group)
    return clusters
