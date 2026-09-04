"""BERTopic discovery helpers. Notebook-facing; no LangGraph state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from playbook.data import conversation_document
from playbook.schemas import ConversationRecord

FitRule = Literal["centroid", "guideline", "either", "both"]

EXTRA_STOP = frozenset(
    {
        "agent",
        "customer",
        "help",
        "thank",
        "thanks",
        "ok",
        "okay",
        "please",
        "hi",
        "hello",
        "yes",
        "no",
        "today",
        "need",
        "want",
        "let",
        "one",
        "moment",
        "great",
        "good",
        "day",
        "welcome",
        "acme",
        "acmebrands",
        "id",
        "account",
        "order",
        "email",
        "username",
        "name",
        "full",
    }
)


class BertopicConfig(BaseModel):
    """Explicit BERTopic settings used by the Phase 1 notebooks."""

    embedding_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 5
    min_samples: int = 1
    n_neighbors: int = 12
    n_components: int = 5
    min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    extra_stopwords: list[str] = Field(default_factory=lambda: sorted(EXTRA_STOP))
    seed: int = 42
    min_to_cluster: int = 5
    min_topic_n: int = 5
    representative_n: int = 3


class FittedTopics(BaseModel):
    """Compact BERTopic output. No model object."""

    topic_ids: list[int]
    descriptors: dict[int, str]


class TopicInfo(BaseModel):
    topic_id: int
    size: int
    descriptor: str
    member_ids: list[str]
    representative_ids: list[str]
    cohesive_enough: bool
    parent_intent: str | None = None


class TopicDiscoveryResult(BaseModel):
    topics: list[TopicInfo]
    assignments: list[int]
    skipped: bool = False
    config: BertopicConfig


class PrototypeMatch(BaseModel):
    pred: str
    sim: float
    margin: float
    fits: bool


@lru_cache(maxsize=4)
def _sentence_transformer(name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def get_embedder(embedding_model: str | Any = "all-MiniLM-L6-v2") -> Any:
    if isinstance(embedding_model, str):
        return _sentence_transformer(embedding_model)
    return embedding_model


def embed_texts(
    texts: Sequence[str],
    *,
    embedding_model: str | Any = "all-MiniLM-L6-v2",
) -> np.ndarray:
    embedder = get_embedder(embedding_model)
    return np.asarray(embedder.encode(list(texts), show_progress_bar=False))


def classify_against(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    names: Sequence[str],
    *,
    min_sim: float = 0.60,
    min_margin: float = 0.10,
) -> list[PrototypeMatch]:
    """Nearest prototype. `fits` = sim >= min_sim and margin >= min_margin."""
    if prototypes.ndim != 2 or prototypes.shape[0] != len(names):
        raise ValueError("proto rows must match names")
    sims = cosine_similarity(embeddings, prototypes)
    rows: list[PrototypeMatch] = []
    for row in sims:
        order = np.argsort(row)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        sim = float(row[best])
        margin = float(row[best] - row[second]) if len(names) > 1 else sim
        rows.append(
            PrototypeMatch(
                pred=names[best],
                sim=round(sim, 3),
                margin=round(margin, 3),
                fits=bool(sim >= min_sim and margin >= min_margin),
            )
        )
    return rows


def apply_fit_rule(fits_centroid: bool, fits_guideline: bool, rule: FitRule = "either") -> bool:
    if rule == "centroid":
        return fits_centroid
    if rule == "guideline":
        return fits_guideline
    if rule == "both":
        return fits_centroid and fits_guideline
    if rule == "either":
        return fits_centroid or fits_guideline
    raise ValueError(f"unknown FIT_RULE {rule}")


def resolve_label(
    fits_c: bool,
    pred_c: str,
    sim_c: float,
    fits_g: bool,
    pred_g: str,
    sim_g: float,
    rule: FitRule = "either",
) -> str | None:
    """Assigned label used by the next stage. None = unmatched under the fit rule."""
    if not apply_fit_rule(fits_c, fits_g, rule):
        return None
    if fits_c and fits_g:
        return pred_c if sim_c >= sim_g else pred_g
    if fits_c:
        return pred_c
    if fits_g:
        return pred_g
    return None


def centroid_matrix(
    embeddings: np.ndarray,
    labels: Sequence[str],
    names: Sequence[str],
) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    missing = [name for name in names if not groups[name]]
    if missing:
        raise ValueError(f"no seed items for {missing}")
    return np.vstack([embeddings[groups[name]].mean(axis=0) for name in names])


def _stop_words(config: BertopicConfig) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return list(frozenset(ENGLISH_STOP_WORDS).union(config.extra_stopwords))


def fit_topic_model(
    documents: Sequence[str],
    config: BertopicConfig | None = None,
    *,
    embedding_model: str | Any | None = None,
) -> FittedTopics:
    """Fit BERTopic and return topic ids + descriptors only."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    config = config or BertopicConfig()
    docs = list(documents)
    n = len(docs)
    embedder = get_embedder(embedding_model or config.embedding_model)
    model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(
            n_neighbors=max(2, min(config.n_neighbors, n - 2)),
            n_components=min(config.n_components, max(2, n - 2)),
            min_dist=config.min_dist,
            metric=config.umap_metric,
            random_state=config.seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min(config.min_cluster_size, max(2, n)),
            min_samples=config.min_samples,
            metric=config.hdbscan_metric,
            cluster_selection_method=config.cluster_selection_method,
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words=_stop_words(config),
            ngram_range=config.ngram_range,
            min_df=config.min_df,
        ),
        calculate_probabilities=False,
        verbose=False,
    )
    topic_ids, _ = model.fit_transform(docs)
    topic_ids = [int(topic_id) for topic_id in topic_ids]
    descriptors: dict[int, str] = {}
    for topic_id in set(topic_ids):
        if topic_id == -1:
            descriptors[topic_id] = ""
            continue
        words = [word for word, _ in (model.get_topic(topic_id) or [])[:8]]
        descriptors[topic_id] = ", ".join(words)
    return FittedTopics(topic_ids=topic_ids, descriptors=descriptors)


def _summarize_topics(
    conversation_ids: Sequence[str],
    fitted: FittedTopics,
    config: BertopicConfig,
    *,
    parent_intent: str | None = None,
) -> TopicDiscoveryResult:
    by_topic: dict[int, list[str]] = defaultdict(list)
    for conversation_id, topic_id in zip(conversation_ids, fitted.topic_ids, strict=True):
        by_topic[int(topic_id)].append(str(conversation_id))
    topics = []
    for topic_id in sorted(by_topic):
        member_ids = by_topic[topic_id]
        size = len(member_ids)
        topics.append(
            TopicInfo(
                topic_id=topic_id,
                size=size,
                descriptor=fitted.descriptors.get(topic_id, ""),
                member_ids=member_ids,
                representative_ids=member_ids[: config.representative_n],
                cohesive_enough=topic_id != -1 and size >= config.min_topic_n,
                parent_intent=parent_intent,
            )
        )
    return TopicDiscoveryResult(
        topics=topics,
        assignments=[int(topic_id) for topic_id in fitted.topic_ids],
        skipped=False,
        config=config,
    )


def discover_topics(
    conversations: Sequence[ConversationRecord],
    *,
    config: BertopicConfig | None = None,
    parent_intent: str | None = None,
    embedding_model: str | Any | None = None,
) -> TopicDiscoveryResult:
    """Cluster conversations with BERTopic. Shared entry point for intent/subflow discovery."""
    config = config or BertopicConfig()
    records = list(conversations)
    if len(records) < config.min_to_cluster:
        return TopicDiscoveryResult(
            topics=[],
            assignments=[-1] * len(records),
            skipped=True,
            config=config,
        )
    fitted = fit_topic_model(
        [conversation_document(record) for record in records],
        config,
        embedding_model=embedding_model,
    )
    return _summarize_topics(
        [record.conversation_id for record in records],
        fitted,
        config,
        parent_intent=parent_intent,
    )


def discover_intent_topics(
    conversations: Sequence[ConversationRecord],
    *,
    config: BertopicConfig | None = None,
    embedding_model: str | Any | None = None,
) -> TopicDiscoveryResult:
    """Candidate new intents: cluster conversations that did not fit known intents."""
    return discover_topics(conversations, config=config, embedding_model=embedding_model)


def discover_subflow_topics(
    conversations: Sequence[ConversationRecord],
    *,
    intent: str | None = None,
    config: BertopicConfig | None = None,
    embedding_model: str | Any | None = None,
) -> TopicDiscoveryResult:
    """Candidate new subflows: cluster leftovers inside one intent (known or newly found)."""
    return discover_topics(
        conversations,
        config=config,
        parent_intent=intent,
        embedding_model=embedding_model,
    )
