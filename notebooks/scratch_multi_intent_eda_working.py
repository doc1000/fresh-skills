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
# # Scratch — classify, then discover
#
# One pipeline. Offline `flow` / `subflow` labels are on the sample for inspection; they are not inputs to classification or clustering.
#
# ```text
# 1. classify each conversation against existing INTENTS
#    (conversation-centroid  and  intent guideline text)
# 2. BERTopic on conversations that do not fit
#    → cohesive topic with enough members = candidate new intent
# 3. within each assigned known intent, classify against existing SUBFLOWS
#    (same two matchers)
# 4. BERTopic on unclassified leftovers inside that intent
#    → cohesive topic = candidate new subflow
# 5. same clustering on each candidate new intent
#    (no existing subflows there, so everyone is unclassified)
# ```
#
# Thresholds below are round numbers, not tuned. Change them and re-run.

# %%
from dotenv import load_dotenv

load_dotenv(".env")

# %%
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

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

from playbook.actions import discover_action_paths
from playbook.config import ABCD_JSON, GUIDELINES_JSON, KB_JSON
from playbook.data import conversation_document, parse_raw_conversation
from playbook.topics import (
    EXTRA_STOP,
    BertopicConfig,
    apply_fit_rule,
    centroid_matrix,
    classify_against,
    discover_intent_topics,
    discover_subflow_topics,
    embed_texts,
    resolve_label,
)

# --- knobs (untuned) ---
MIN_SIM = 0.60
MIN_MARGIN = 0.05
# A conversation is assigned if it fits this matcher set:
#   "centroid" | "guideline" | "either" | "both"
FIT_RULE = "either"
MIN_CLUSTER_SIZE = 5
MIN_TOPIC_N = 5
MIN_TO_CLUSTER = 5
PER_SEED = 15
PER_PROBE = 15
MIN_OPENING = 25

"""
MIN_SIM = 0.60 — Cosine similarity to the best prototype must be at least 0.60. Below that, the nearest label is treated as a weak match (too far from any known playbook item).

MIN_MARGIN = 0.10 — The best prototype must beat the second-best by at least 0.10. That blocks “close call” assignments (e.g. almost as similar to account_access as to order_issue). With only one prototype, margin is just sim.

FIT_RULE = "either" — How the two matchers vote:

rule	           assigned if
"centroid"         conversation looks like the seed-conversation centroid
"guideline"        conversation looks like the KB guideline text
"either" (current) either matcher fits
"both"             both must fit

Unassigned tickets go to clustering (candidate new intent / new subflow). If both matchers fit but disagree on the label, resolve_label keeps the one with the higher similarity.

These are the sampling and clustering knobs at the top of scratch_multi_intent_eda.ipynb. Three build the fake week; three decide when BERTopic is allowed to call something a topic.

Sampling the week

MIN_OPENING = 25 — Drop conversations whose first customer message is shorter than 25 characters ("hi", "ok", etc.). pick() only keeps tickets that pass this, then takes the first N by convo_id.

PER_SEED = 6 — How many conversations to take from each known playbook subflow (3 account-access + 4 order-issue). That is 7 × 6 = 42 seed tickets.

PER_PROBE = 8 — How many conversations to take for each held-out probe: the new order subflow (status_payment_method) and the new intent (troubleshoot_site / slow_speed).

Clustering / “is this a real topic?”

MIN_TO_CLUSTER = 5 — Do not run BERTopic at all if the leftover pile is smaller than this. Those items stay unclustered (topic = -1). Used when discovering candidate new intents or new subflows.

MIN_CLUSTER_SIZE = 5 — HDBSCAN’s min_cluster_size: the smallest group BERTopic is allowed to form. Smaller blobs become noise.

MIN_TOPIC_N = 5 — After clustering, a topic is cohesive_enough only if it is not noise (topic != -1) and has at least 5 members. Only those become “candidate new intent” or “candidate new subflow.”
"""


SEED_SUBFLOWS = {
    "account_access": ["recover_username", "recover_password", "reset_2fa"],
    "order_issue": [
        "status_mystery_fee",
        "status_delivery_time",
        "manage_upgrade",
        "manage_cancel",
    ],
}
INTENT_NAMES = list(SEED_SUBFLOWS)
INTENT_GUIDELINE_TITLE = {
    "account_access": "Account Access",
    "order_issue": "Order Issue",
}
SUBFLOW_GUIDELINE_TITLE = {
    "recover_username": "Recover Username",
    "recover_password": "Recover Password",
    "reset_2fa": "Reset Two-Factor Auth",
    "status_mystery_fee": "Status Mystery Fee",
    "status_delivery_time": "Status Delivery Time",
    "manage_upgrade": "Manage Upgrade",
    "manage_cancel": "Manage Cancel",
}
PROBE_SUBFLOW = ("order_issue", "status_payment_method")
PROBE_INTENT = ("troubleshoot_site", "slow_speed")
NOISE_PAIRS = [
    ("shipping_issue", "missing"),
    ("shipping_issue", "cost"),
    ("product_defect", "return_stain"),
    ("product_defect", "refund_initiate"),
    ("purchase_dispute", "promo_code_invalid"),
    ("purchase_dispute", "out_of_stock_general"),
    ("subscription_inquiry", "manage_pay_bill"),
    ("manage_account", "manage_change_address"),
    ("storewide_query", "pricing_3"),
    ("single_item_query", "boots_how_2"),
]

print("ROOT", ROOT)
print("MIN_SIM", MIN_SIM, "MIN_MARGIN", MIN_MARGIN, "FIT_RULE", FIT_RULE)

# %% [markdown]
# ## 1. Sample
#
# Seed = all `account_access` subflows + four `order_issue` subflows. Then noise singletons, a held-out order subflow, and a held-out site-troubleshooting subflow. First N by `convo_id` after dropping short openings.

# %%
raw = json.loads(ABCD_JSON.read_text(encoding="utf-8"))
guidelines = json.loads(GUIDELINES_JSON.read_text(encoding="utf-8"))
kb = json.loads(KB_JSON.read_text(encoding="utf-8"))
all_convos = [item for split in raw.values() for item in split]


def first_customer(item: dict) -> str:
    for speaker, text in item["original"]:
        if speaker == "customer":
            return text.strip()
    return ""


def pick(flow: str, subflow: str, n: int) -> list[dict]:
    items = [
        c
        for c in all_convos
        if c["scenario"]["flow"] == flow
        and c["scenario"]["subflow"] == subflow
        and len(first_customer(c)) >= MIN_OPENING
    ]
    items = sorted(items, key=lambda c: int(c["convo_id"]))
    if len(items) < n:
        raise ValueError(f"{flow}/{subflow}: {len(items)} after filter, need {n}")
    return items[:n]


def true_label(item: dict) -> str:
    return f"{item['scenario']['flow']}/{item['scenario']['subflow']}"


seed_items: list[dict] = []
for flow, subflows in SEED_SUBFLOWS.items():
    for subflow in subflows:
        seed_items.extend(pick(flow, subflow, PER_SEED))
noise_items = [pick(flow, subflow, 1)[0] for flow, subflow in NOISE_PAIRS]
probe_subflow_items = pick(*PROBE_SUBFLOW, PER_PROBE)
probe_intent_items = pick(*PROBE_INTENT, PER_PROBE)
week_items = seed_items + noise_items + probe_subflow_items + probe_intent_items
by_id = {int(c["convo_id"]): c for c in week_items}

inventory = pd.DataFrame(
    [
        {
            "conversation_id": int(c["convo_id"]),
            "role": role,
            "true_flow": c["scenario"]["flow"],
            "true_subflow": c["scenario"]["subflow"],
            "opening": first_customer(c)[:80],
        }
        for role, group in (
            ("seed", seed_items),
            ("noise", noise_items),
            ("probe_subflow", probe_subflow_items),
            ("probe_intent", probe_intent_items),
        )
        for c in group
    ]
)
display(inventory.groupby(["role", "true_flow", "true_subflow"]).size().reset_index(name="n"))

# %%
inventory.describe()

# %% [markdown]
# ## 2. Shared helpers
#
# Fitting/discovery lives in `playbook.topics` / `playbook.actions`. The notebook keeps
# sampling, ABCD guideline-doc construction, display, and offline-label inspection.
# `discover_intent_topics` / `discover_subflow_topics` are the clustering entry points.

# %%
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = list(frozenset(ENGLISH_STOP_WORDS).union(EXTRA_STOP))
TOPIC_CONFIG = BertopicConfig(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_to_cluster=MIN_TO_CLUSTER,
    min_topic_n=MIN_TOPIC_N,
    n_neighbors=12,
    seed=42,
)
print("BERTopic config")
display(TOPIC_CONFIG.model_dump())
print("extra stopwords", sorted(EXTRA_STOP))


def as_records(items: list[dict]):
    return [parse_raw_conversation(item) for item in items]


def embed_items(items: list[dict]) -> np.ndarray:
    return embed_texts([conversation_document(parse_raw_conversation(c)) for c in items])


def matches_frame(matches, prefix: str) -> pd.DataFrame:
    return pd.DataFrame([match.model_dump() for match in matches]).add_prefix(prefix)


def topic_frame(result, items: list[dict] | None = None) -> pd.DataFrame:
    """Display helper. Offline majority/purity stay in the notebook."""
    rows = []
    by_id = {str(item["convo_id"]): item for item in (items or [])}
    for topic in result.topics:
        members = [by_id[member_id] for member_id in topic.member_ids if member_id in by_id]
        mix = Counter(true_label(item) for item in members) if members else Counter()
        majority, maj_n = mix.most_common(1)[0] if mix else ("", 0)
        rows.append(
            {
                "topic": topic.topic_id,
                "n": topic.size,
                "descriptor": topic.descriptor,
                "cohesive_enough": topic.cohesive_enough,
                "parent_intent": topic.parent_intent,
                "offline_majority": majority,
                "offline_purity": round(maj_n / topic.size, 2) if topic.size else None,
                "member_ids": [int(member_id) for member_id in topic.member_ids],
                "representative_ids": [int(member_id) for member_id in topic.representative_ids],
            }
        )
    return pd.DataFrame(rows)


def guideline_subflow_doc(flow: str, subflow: str) -> str:
    title = INTENT_GUIDELINE_TITLE[flow]
    sub_title = SUBFLOW_GUIDELINE_TITLE[subflow]
    node = guidelines[title]["subflows"][sub_title]
    instructions = " ".join(node.get("instructions") or [])
    bits = []
    for action in node.get("actions") or []:
        bits.append(action.get("text") or "")
        bits.extend(action.get("subtext") or [])
    return f"{title} / {sub_title}. {instructions} {' '.join(bits)}"


def guideline_intent_doc(flow: str) -> str:
    title = INTENT_GUIDELINE_TITLE[flow]
    desc = guidelines[title].get("description") or ""
    return f"{title}. {desc}"

# %% [markdown]
# ## 3. Build intent prototypes from the seed only

# %%
week_emb = embed_items(week_items)
seed_emb = embed_items(seed_items)

intent_centroid = centroid_matrix(
    seed_emb, [c["scenario"]["flow"] for c in seed_items], INTENT_NAMES
)
intent_guideline = embed_texts([guideline_intent_doc(flow) for flow in INTENT_NAMES])

print("intent centroid shape", intent_centroid.shape)
print("intent guideline docs:")
for flow in INTENT_NAMES:
    print(f"  {flow}: {guideline_intent_doc(flow)}")

subflow_centroid: dict[str, np.ndarray] = {}
subflow_guideline: dict[str, np.ndarray] = {}
for flow, subflows in SEED_SUBFLOWS.items():
    seeded = [c for c in seed_items if c["scenario"]["flow"] == flow]
    subflow_centroid[flow] = centroid_matrix(
        embed_items(seeded),
        [c["scenario"]["subflow"] for c in seeded],
        subflows,
    )
    subflow_guideline[flow] = embed_texts(
        [guideline_subflow_doc(flow, sf) for sf in subflows]
    )
    print(flow, "subflows", subflows)

# %% [markdown]
# ## 4. Intent classification
#
# Two matchers, then `FIT_RULE` decides who is assigned vs leftover. Leftovers go to clustering.

# %%
intent_c = matches_frame(
    classify_against(week_emb, intent_centroid, INTENT_NAMES, min_sim=MIN_SIM, min_margin=MIN_MARGIN),
    "intent_centroid_",
)
intent_g = matches_frame(
    classify_against(week_emb, intent_guideline, INTENT_NAMES, min_sim=MIN_SIM, min_margin=MIN_MARGIN),
    "intent_guideline_",
)

intent_df = inventory.copy()
intent_df = pd.concat([intent_df.reset_index(drop=True), intent_c, intent_g], axis=1)
intent_df["intent_assigned"] = [
    apply_fit_rule(bool(centroid_fits), bool(guideline_fits), FIT_RULE)
    for centroid_fits, guideline_fits in zip(
        intent_df["intent_centroid_fits"], intent_df["intent_guideline_fits"]
    )
]
intent_df["pred_intent"] = [
    resolve_label(
        bool(r.intent_centroid_fits),
        r.intent_centroid_pred,
        r.intent_centroid_sim,
        bool(r.intent_guideline_fits),
        r.intent_guideline_pred,
        r.intent_guideline_sim,
        FIT_RULE,
    )
    for r in intent_df.itertuples()
]

print("assignment counts (FIT_RULE=%s)" % FIT_RULE)
display(
    intent_df.groupby(
        ["true_flow", "pred_intent", "intent_centroid_pred", "intent_guideline_pred", "intent_assigned"],
        dropna=False,
    )
    .size()
    .reset_index(name="n")
)
display(
    intent_df[
        [
            "conversation_id",
            "role",
            "true_flow",
            "true_subflow",
            "intent_centroid_pred",
            "intent_centroid_sim",
            "intent_centroid_margin",
            "intent_centroid_fits",
            "intent_guideline_pred",
            "intent_guideline_sim",
            "intent_guideline_margin",
            "intent_guideline_fits",
            "pred_intent",
        ]
    ]
)

unmatched_ids = intent_df.loc[~intent_df["intent_assigned"], "conversation_id"].tolist()
unmatched_items = [by_id[i] for i in unmatched_ids]
print("unmatched n", len(unmatched_ids), unmatched_ids)

# %%
intent_df.head()

# %%
import numpy as np
def label_summary(df, pred_col, true_col):
    df["label_result"] = np.select(
        [
            df[pred_col].isna(),
            df[pred_col] == df[true_col],
        ],
        [
            "No label",
            "Correct",
        ],
        default="Mislabelled",
    )
    
    df["label_result"].value_counts(normalize=True).mul(100).round(0)
    
    summary = (
        df["label_result"]
        .value_counts()
        .rename("count")
        .to_frame()
    )
    
    summary["pct"] = (summary["count"] / len(df) * 100).round(1)
    display(summary)

label_summary(df=intent_df, pred_col="pred_intent", true_col="true_flow")

# %%
pd.crosstab(
    intent_df.loc[intent_df["pred_intent"].notna(), "true_flow"],
    intent_df.loc[intent_df["pred_intent"].notna(), "pred_intent"],
    rownames=["True"],
    colnames=["Predicted"]
)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def label_confusion(df, pred_col="pred_intent", true_col="true_flow"):
    mask = df[pred_col].notna()
    
    y_true = df.loc[mask, true_col]
    y_pred = df.loc[mask, pred_col]
    
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=sorted(set(y_true) | set(y_pred))
    )
    
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.show()

label_confusion(df=intent_df, pred_col="pred_intent", true_col="true_flow")

# %% [markdown]
# ## 5. Discover candidate new intents
#
# Clustering runs only on unmatched conversations.

# %%
intent_result = discover_intent_topics(as_records(unmatched_items), config=TOPIC_CONFIG)
if intent_result.skipped:
    print(f"skip clustering: n={len(unmatched_items)} < min_to_cluster={TOPIC_CONFIG.min_to_cluster}")
intent_topics = topic_frame(intent_result, unmatched_items)
display(intent_topics)
print("assignments", intent_result.assignments)

new_intent_clusters: list[dict] = []
for topic in intent_result.topics:
    if not topic.cohesive_enough:
        continue
    members = [by_id[int(member_id)] for member_id in topic.member_ids]
    new_intent_clusters.append(
        {
            "topic": topic.topic_id,
            "n": topic.size,
            "descriptor": topic.descriptor,
            "items": members,
        }
    )
print("candidate new intents", [(c["topic"], c["n"], c["descriptor"]) for c in new_intent_clusters])

# %% [markdown]
# ## 6. Subflow classification inside each assigned known intent
#
# Prototypes are that intent’s seeded subflows only.

# %%
assigned_df = intent_df[intent_df["pred_intent"].isin(INTENT_NAMES)].copy()
subflow_frames = []
for flow, subflows in SEED_SUBFLOWS.items():
    part = assigned_df[assigned_df["pred_intent"] == flow]
    if part.empty:
        print(flow, "n assigned=0")
        continue
    items = [by_id[i] for i in part["conversation_id"]]
    embs = embed_items(items)
    sf_c = matches_frame(
        classify_against(embs, subflow_centroid[flow], subflows, min_sim=MIN_SIM, min_margin=MIN_MARGIN),
        "subflow_centroid_",
    )
    sf_g = matches_frame(
        classify_against(embs, subflow_guideline[flow], subflows, min_sim=MIN_SIM, min_margin=MIN_MARGIN),
        "subflow_guideline_",
    )
    block = pd.concat([part.reset_index(drop=True), sf_c, sf_g], axis=1)
    block["subflow_assigned"] = [
        apply_fit_rule(bool(centroid_fits), bool(guideline_fits), FIT_RULE)
        for centroid_fits, guideline_fits in zip(
            block["subflow_centroid_fits"], block["subflow_guideline_fits"]
        )
    ]
    block["pred_subflow"] = [
        resolve_label(
            bool(r.subflow_centroid_fits),
            r.subflow_centroid_pred,
            r.subflow_centroid_sim,
            bool(r.subflow_guideline_fits),
            r.subflow_guideline_pred,
            r.subflow_guideline_sim,
            FIT_RULE,
        )
        for r in block.itertuples()
    ]
    subflow_frames.append(block)
    print(f"\n{flow} assigned n={len(block)} unclassified n={(~block.subflow_assigned).sum()}")
    display(
        block.groupby(
            ["true_subflow", "pred_subflow", "subflow_centroid_pred", "subflow_guideline_pred", "subflow_assigned"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
    )

subflow_df = pd.concat(subflow_frames, ignore_index=True) if subflow_frames else pd.DataFrame()
if not subflow_df.empty:
    display(
        subflow_df[
            [
                "conversation_id",
                "pred_intent",
                "true_subflow",
                "subflow_centroid_pred",
                "subflow_centroid_sim",
                "subflow_centroid_fits",
                "subflow_guideline_pred",
                "subflow_guideline_sim",
                "subflow_guideline_fits",
                "pred_subflow",
            ]
        ]
    )

# %%
subflow_df.head()

# %%
label_confusion(df=subflow_df, pred_col="pred_subflow", true_col="true_subflow")
label_summary(df=subflow_df, pred_col="pred_subflow", true_col="true_subflow")

# %% [markdown]
# ## 7. Discover candidate new subflows
#
# Same `discover_subflow_topics` on unclassified conversations **inside each known intent**.

# %%
new_subflow_clusters: list[dict] = []
for flow in INTENT_NAMES:
    if subflow_df.empty:
        continue
    leftover = subflow_df[(subflow_df["pred_intent"] == flow) & (~subflow_df["subflow_assigned"])]
    leftover_items = [by_id[i] for i in leftover["conversation_id"]]
    print(f"\n--- unclassified under {flow}: n={len(leftover_items)} ---")
    result = discover_subflow_topics(as_records(leftover_items), intent=flow, config=TOPIC_CONFIG)
    if result.skipped:
        print(f"skip clustering: n={len(leftover_items)} < min_to_cluster={TOPIC_CONFIG.min_to_cluster}")
        continue
    table = topic_frame(result, leftover_items)
    display(table)
    for topic in result.topics:
        if not topic.cohesive_enough:
            continue
        new_subflow_clusters.append(
            {
                "intent": flow,
                "topic": topic.topic_id,
                "n": topic.size,
                "descriptor": topic.descriptor,
                "member_ids": [int(member_id) for member_id in topic.member_ids],
            }
        )
print("candidate new subflows", new_subflow_clusters)

# %% [markdown]
# ## 8. Subflows inside each candidate new intent
#
# No seeded subflows. Run the same clustering on that topic’s members.

# %%
for cluster in new_intent_clusters:
    print(f"\n--- new intent topic {cluster['topic']} n={cluster['n']} ---")
    print("descriptor:", cluster["descriptor"])
    result = discover_subflow_topics(
        as_records(cluster["items"]),
        intent=f"new:{cluster['topic']}",
        config=TOPIC_CONFIG,
    )
    if result.skipped:
        print(f"skip clustering: n={cluster['n']} < min_to_cluster={TOPIC_CONFIG.min_to_cluster}")
        continue
    display(topic_frame(result, cluster["items"]))

# %% [markdown]
# ## 9. Action sequences on discovered clusters
#
# Inspection only. Exact sequence identity is usually noisy; button sets are listed too.

# %%
def show_action_paths(items: list[dict], title: str) -> None:
    result = discover_action_paths(as_records(items))
    print(f"\n{title}  n={result.n_conversations}  unique_seqs={result.n_unique_paths}")
    for path in result.paths[:5]:
        print(f"  {path.count:3d}  {tuple(path.actions)}")
    print("  button presence", result.button_counts)


for cluster in new_subflow_clusters:
    members = [by_id[i] for i in cluster["member_ids"]]
    show_action_paths(members, f"new subflow under {cluster['intent']} topic {cluster['topic']}")

for cluster in new_intent_clusters:
    show_action_paths(cluster["items"], f"new intent topic {cluster['topic']}")
    print("  canonical kb for probe intent subflow (offline)", kb.get(PROBE_INTENT[1]))

print("\ndone. change MIN_SIM / MIN_MARGIN / FIT_RULE / MIN_TOPIC_N and re-run.")

# %%
