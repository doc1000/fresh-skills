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
# # Phase 1 — ABCD EDA + BERTopic feasibility + fixtures
#
# Review gate for the demo subset. This notebook **shows the failures**, not only the winner.
#
# Offline labels (`flow` / `subflow`) are used here to *choose* the experiment and to *sanity-check* BERTopic. They are **not** runtime inputs. Runtime batches loaded at the end have no subflow fields.

# %%
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
from dotenv import load_dotenv
from IPython.display import display, JSON

def _find_root() -> Path:
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "playbook" / "config.py").exists():
            return candidate
    raise RuntimeError("Could not find worktree root containing src/playbook")


ROOT = _find_root()
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from playbook.config import ABCD_JSON, GUIDELINES_JSON, KB_JSON, ONTOLOGY_JSON
from playbook.data import conversation_document, load_fixture_labels, load_week
from playbook.fixtures import DEMO_FLOW, ROLE_TO_SUBFLOW, WEEK_MEMBERSHIP

load_dotenv(ROOT / ".env")
con = duckdb.connect(database=":memory:")
print("ROOT", ROOT)
print("ABCD exists", ABCD_JSON.exists(), ABCD_JSON.stat().st_size if ABCD_JSON.exists() else 0)

# %% [markdown]
# ## 1. Load ABCD into DuckDB
#
# Conversations stay as JSON on disk; DuckDB is the query path for counts. Nested dialogue is parsed in Python only when we need excerpts or BERTopic documents.

# %%
raw = json.loads(ABCD_JSON.read_text(encoding="utf-8"))
index_rows = []
for split, convos in raw.items():
    for item in convos:
        index_rows.append(
            {
                "split": split,
                "conversation_id": str(item["convo_id"]),
                "flow": item["scenario"]["flow"],
                "subflow": item["scenario"]["subflow"],
            }
        )
con.register("abcd_index", pa.Table.from_pylist(index_rows))
display(con.execute("SELECT split, count(*) AS n FROM abcd_index GROUP BY 1 ORDER BY 1").df())
print("total", con.execute("SELECT count(*) FROM abcd_index").fetchone()[0])

ontology = json.loads(ONTOLOGY_JSON.read_text(encoding="utf-8"))
guidelines = json.loads(GUIDELINES_JSON.read_text(encoding="utf-8"))
kb = json.loads(KB_JSON.read_text(encoding="utf-8"))
print("ontology flows", ontology["intents"]["flows"])
print("n subflows listed", sum(len(v) for v in ontology["intents"]["subflows"].values()))

# %%
display(JSON(guidelines, expanded=False))

# %% [markdown]
# ## 2. Flow landscape
#
# Plan shortlist: `account_access` (3 subflows), `shipping_issue` (exactly 4), `product_defect` (6; only if we pick four). Avoid FAQ-like `storewide_query` / `single_item_query` unless nothing else works.

# %%
display(
    con.execute(
        """
        SELECT flow, count(*) AS n_convos, count(DISTINCT subflow) AS n_subflows
        FROM abcd_index
        GROUP BY 1
        ORDER BY n_convos DESC
        """
    ).df()
)

# %% [markdown]
# ## 3. Shortlist subflow counts + canonical actions
#
# `kb.json` is the compact action-button list per subflow. Distinct sequences matter more than raw counts.

# %%
SHORTLIST = ["account_access", "shipping_issue", "product_defect"]
display(
    con.execute(
        """
        SELECT flow, subflow, count(*) AS n
        FROM abcd_index
        WHERE flow IN ('account_access', 'shipping_issue', 'product_defect')
        GROUP BY 1, 2
        ORDER BY flow, n DESC
        """
    ).df()
)
for flow in SHORTLIST:
    print(f"\n== {flow} kb actions ==")
    for subflow in ontology["intents"]["subflows"][flow]:
        print(f"  {subflow:20s} {kb[subflow]}")

# %% [markdown]
# Observed action sequences are noisier than the canonical KB lists (agents skip steps). Still, modes match the playbook.

# %%
all_convos = [item for split in raw.values() for item in split]
by_id = {int(item["convo_id"]): item for item in all_convos}


def action_names(item: dict) -> tuple[str, ...]:
    names = []
    for turn in item.get("delexed") or []:
        if turn.get("speaker") == "action":
            targets = turn.get("targets") or []
            if len(targets) > 2 and targets[2]:
                names.append(str(targets[2]))
    return tuple(names)


def first_customer(item: dict) -> str:
    for speaker, text in item["original"]:
        if speaker == "customer":
            return text.strip()
    return ""


for flow in SHORTLIST:
    print(f"\n==== {flow} top observed action sequences ====")
    grouped: dict[str, list] = defaultdict(list)
    for item in all_convos:
        if item["scenario"]["flow"] == flow:
            grouped[item["scenario"]["subflow"]].append(item)
    for subflow, items in grouped.items():
        counts = Counter(action_names(c) for c in items)
        top_seq, top_n = counts.most_common(1)[0]
        print(f"  {subflow:20s} unique_seqs={len(counts):3d}  mode_n={top_n:3d}  {top_seq}")

# %% [markdown]
# ## 4. Excerpts (why language matters)
#
# BERTopic will see original customer/agent text, **not** action-button names (those would leak subflow identity).

# %%
EXCERPT_IDS = {
    "account_access / recover_password": 2652,
    "account_access / reset_2fa": 7225,
    "shipping_issue / missing": 169,
    "shipping_issue / cost": 194,
    "shipping_issue / manage": 264,
    "shipping_issue / status": 228,
    "product_defect / return_stain": None,
}


def show_excerpt(item: dict, n_turns: int = 6) -> None:
    print(f"id={item['convo_id']} {item['scenario']['flow']}/{item['scenario']['subflow']}")
    print("actions", list(action_names(item)))
    for speaker, text in item["original"][:n_turns]:
        print(f"  {speaker:10s} {text[:140]}")
    print()


for item in all_convos:
    if item["scenario"]["flow"] == "product_defect" and item["scenario"]["subflow"] == "return_stain":
        EXCERPT_IDS["product_defect / return_stain"] = int(item["convo_id"])
        break

for label, convo_id in EXCERPT_IDS.items():
    print(f"-- {label} --")
    show_excerpt(by_id[convo_id])

# %% [markdown]
# ## 5. Rejected subsets
#
# **`storewide_query` / `single_item_query`:** FAQ search-button workflows. Ontology lists 4 storewide subflows; the data actually stores `pricing_1`…`policy_4`. Language is catalog Q&A, not an operational playbook the agent would maintain week to week.
#
# **`account_access`:** Best language separation (username vs password vs 2FA) and short, readable guidelines. **Rejected as the demo flow** because it has only three subflows, so Week 2 cannot introduce a fourth held-out workflow from the same top-level flow.
#
# **`product_defect`:** Six subflows, but `return_stain` / `return_color` / `return_size` share the same canonical actions and nearly identical guideline text. Picking four would still leave two return variants that are not operationally distinct.
#
# **`shipping_issue`:** Exactly four subflows, four different procedures (change shipment, shipping-fee complaint, never-arrived reship, email/status discrepancy), and enough conversations each (~240+). Language overlaps more than account_access — that is the honest BERTopic test.

# %%
display(
    con.execute(
        """
        SELECT flow, count(DISTINCT subflow) AS n_subflows, count(*) AS n
        FROM abcd_index
        WHERE flow IN ('storewide_query', 'single_item_query', 'account_access',
                       'product_defect', 'shipping_issue')
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
)
print("storewide_query actual subflows", 
      con.execute("SELECT DISTINCT subflow FROM abcd_index WHERE flow='storewide_query' ORDER BY 1").fetchall()[:8], "...")

# %% [markdown]
# ## 6. BERTopic feasibility
#
# Settings kept small and documented (not a topic-model research pass):
#
# - documents = original customer+agent text (no action names)
# - embedding = `all-MiniLM-L6-v2`
# - UMAP `random_state=42`, `min_dist=0.0`, `metric=cosine`
# - HDBSCAN `min_samples=1`, `cluster_selection_method=eom`
# - CountVectorizer with extra stopwords (`agent`, `customer`, `help`, `account`, `order`, `id`, …) so descriptors are not dominated by chat boilerplate
#
# We do **not** require `topic_id == subflow`. Success = a human can read a topic descriptor + representatives and recognize a recurring operational pattern.

# %%
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

EXTRA_STOP = {
    "agent", "customer", "help", "thank", "thanks", "ok", "okay", "please",
    "hi", "hello", "yes", "no", "today", "need", "want", "let", "one",
    "moment", "great", "good", "day", "welcome", "acme", "acmebrands",
    "id", "account", "order", "email", "username", "name", "full",
}
STOP = list(frozenset(ENGLISH_STOP_WORDS).union(EXTRA_STOP))


def conversation_text(item: dict) -> str:
    return "\n".join(
        text for speaker, text in item["original"] if speaker in {"customer", "agent"} and text
    )


def balanced_sample(flow: str, per_subflow: int) -> list[dict]:
    grouped: dict[str, list] = defaultdict(list)
    for item in all_convos:
        if item["scenario"]["flow"] == flow:
            grouped[item["scenario"]["subflow"]].append(item)
    picked = []
    for subflow, items in sorted(grouped.items()):
        items = sorted(items, key=lambda c: int(c["convo_id"]))
        picked.extend(items[:per_subflow])
    return picked


def fit_bertopic(docs: list[str], *, min_cluster_size: int, n_neighbors: int, seed: int = 42):
    n = len(docs)
    umap_model = UMAP(
        n_neighbors=max(2, min(n_neighbors, n - 2)),
        n_components=min(5, max(2, n - 2)),
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=1),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs)
    return model, topics


def topic_crosstab(items: list[dict], topics: list[int], model: BERTopic) -> None:
    labels = [item["scenario"]["subflow"] for item in items]
    print("topic sizes", Counter(topics).most_common())
    by_topic: dict[int, Counter] = defaultdict(Counter)
    for topic_id, label in zip(topics, labels):
        by_topic[topic_id][label] += 1
    for topic_id in sorted(by_topic):
        words = [w for w, _ in (model.get_topic(topic_id) or [])[:6]] if topic_id != -1 else []
        print(f"  topic {topic_id:3d} n={sum(by_topic[topic_id].values()):3d}  {words}  {by_topic[topic_id].most_common()}")

# %% [markdown]
# ### 6a. `account_access` — cleaner clusters, cannot supply a fourth subflow

# %%
access = balanced_sample("account_access", 40)
access_model, access_topics = fit_bertopic(
    [conversation_text(item) for item in access], min_cluster_size=8, n_neighbors=12
)
print("account_access n=120")
topic_crosstab(access, access_topics, access_model)

# %% [markdown]
# Password / username / 2FA fall into mostly-pure topics. Usable, but Week 2 has no in-flow `D`.

# %% [markdown]
# ### 6b. `product_defect` — returns collapse

# %%
defect = balanced_sample("product_defect", 30)
defect_model, defect_topics = fit_bertopic(
    [conversation_text(item) for item in defect], min_cluster_size=8, n_neighbors=12
)
print("product_defect n=180")
topic_crosstab(defect, defect_topics, defect_model)

# %% [markdown]
# The three return reasons merge. Refund initiate/update also mix. Not four demo-distinct workflows.

# %% [markdown]
# ### 6c. `shipping_issue` — merges and splits, but descriptors are operational

# %%
shipping = balanced_sample("shipping_issue", 40)
shipping_model, shipping_topics = fit_bertopic(
    [conversation_text(item) for item in shipping], min_cluster_size=8, n_neighbors=12
)
print("shipping_issue n=160")
topic_crosstab(shipping, shipping_topics, shipping_model)

# %% [markdown]
# Typical pattern (exact topic IDs will jitter slightly with UMAP): a **cost/refund/cancel** topic, a **missing/waiting** topic, a **change-address** topic, and status split between “check shipment” and “confirm address”. That is good enough for a coverage-judgment demo. We will not chase purity.

# %% [markdown]
# ### 6d. Week-sized batch (the actual runtime scale)
#
# A 13-conversation batch with `min_cluster_size=2` **fragmented**. ~18–20 conversations with `min_cluster_size=4` is the smallest setting that still produced recurring topics rather than pairs.

# %%
from playbook.data import parse_raw_conversation

week1_raw = []
for role, ids in WEEK_MEMBERSHIP["week_1"].items():
    week1_raw.extend(by_id[i] for i in ids)
week1_docs = [conversation_document(parse_raw_conversation(item)) for item in week1_raw]
week1_model, week1_topics = fit_bertopic(week1_docs, min_cluster_size=4, n_neighbors=6)
print("pinned week_1 n=", len(week1_raw))
topic_crosstab(week1_raw, week1_topics, week1_model)
print("\nrepresentatives / descriptors")
for topic_id in sorted(set(week1_topics)):
    if topic_id == -1:
        continue
    words = [w for w, _ in week1_model.get_topic(topic_id)[:8]]
    members = [item for item, t in zip(week1_raw, week1_topics) if t == topic_id]
    print(f"\ntopic {topic_id} {words}")
    for item in members[:3]:
        print(f"  [{item['scenario']['subflow']}] {first_customer(item)[:120]}")

# %% [markdown]
# Week 1 is expected to surface at least a **missing-package** topic (held-out `C`) and a **shipping-cost** topic (seed `B`). `manage` (`A`) may split into the others — acceptable noise. Downstream coverage judgment uses retrieval, not topic-id equality.

# %% [markdown]
# ## 7. Chosen experiment
#
# | role | subflow | week 1 | week 2 | week 3 | seed KB (later) |
# | --- | --- | --- | --- | --- | --- |
# | A | `manage` | yes | yes | yes | covered |
# | B | `cost` | yes | yes | yes | covered |
# | C | `missing` | recurring | still present | present | **held out** |
# | D | `status` | absent | **introduced** | present | **held out** |
#
# Flow: **`shipping_issue`**. BERTopic settings above are the Phase 1 default (HDBSCAN, not KMeans). Custom clustering is **not** justified — HDBSCAN already yields readable recurring candidates on this subset.

# %%
print("flow", DEMO_FLOW)
print("roles", ROLE_TO_SUBFLOW)
for week_id, roles in WEEK_MEMBERSHIP.items():
    counts = {role: len(ids) for role, ids in roles.items() if ids}
    print(week_id, counts, "n=", sum(counts.values()))

# %% [markdown]
# ## 8. Hidden-label week membership (offline)
#
# These labels must not appear on `load_week()`.

# %%
if not (ROOT / "data" / "demo" / "weeks.parquet").exists():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_demo.py")])

labels = load_fixture_labels()
rows = [
    {
        "week_id": label.week_id,
        "role": label.role,
        "subflow": label.subflow,
        "conversation_id": label.conversation_id,
    }
    for label in labels
]
con.register("fixture_labels", pa.Table.from_pylist(rows))
display(
    con.execute(
        """
        SELECT week_id, role, subflow, count(*) AS n
        FROM fixture_labels
        GROUP BY 1, 2, 3
        ORDER BY week_id, role
        """
    ).df()
)
display(
    con.execute(
        """
        SELECT week_id, list(conversation_id ORDER BY conversation_id) AS conversation_ids
        FROM fixture_labels
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
)

# %% [markdown]
# ## 9. Runtime contract: unlabeled weekly batches

# %%
for week_id in ("week_1", "week_2", "week_3"):
    batch = load_week(week_id)
    sample = batch.conversations[0]
    print(
        week_id,
        "n=",
        len(batch.conversation_ids),
        "fields=",
        set(sample.model_dump()),
        "n_actions=",
        len(sample.actions),
        "doc_chars=",
        len(conversation_document(sample)),
    )
    assert "subflow" not in sample.model_dump()
    assert "flow" not in sample.model_dump()

print("first customer turn, week_1 sample:")
print(next(t.text for t in load_week("week_1").conversations[0].turns if t.speaker == "customer"))

# %% [markdown]
# ## 10. Tests + LangSmith env (no eval dataset yet)

# %%
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_demo_fixtures.py"],
    cwd=ROOT,
    check=False,
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
print("pytest exit", result.returncode)
assert result.returncode == 0

print("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING"))
print("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT"))
print("LANGSMITH_API_KEY set", bool(os.getenv("LANGSMITH_API_KEY")))
print("OPENAI_API_KEY set", bool(os.getenv("OPENAI_API_KEY")))
try:
    from langsmith import Client

    if os.getenv("LANGSMITH_API_KEY"):
        Client()
        print("LangSmith Client() constructed")
    else:
        print("LangSmith key absent — skip Client ping")
except ImportError:
    print("langsmith not installed in Phase 1 — env placeholders only")

# %% [markdown]
# ## 11. Phase 1 decision (for merge review)
#
# - **Flow:** `shipping_issue`
# - **A/B/C/D:** `manage` / `cost` / `missing` / `status`
# - **BERTopic:** off-the-shelf, usable on ~20-conversation weeks with the stopword + `min_cluster_size=4` settings above. Expect merges (status↔missing) and splits (manage). Do not unit-test topic IDs.
# - **Refactor:** `NO REFACTOR`
# - **Not done here:** seed KB, RAG, graph, HITL
