"""End-to-end runs against the real agent, the real demo data and a real model.

Deselected by default. To run them:

    uv run pytest -m live

Needs `OPENAI_API_KEY` in `.env`. Each test drives the agent the same way the
Streamlit front end does — one threaded conversation over
`create_playbook_agent` — and then checks the working store, not just the
model's word for it.

The requests come from `evals/responsiveness.json`, the same rows the LangSmith
eval scores, so the two cannot drift apart.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from agent_target import (  # noqa: E402
    build_agent,
    grade,
    load_examples,
    run_request,
)

pytestmark = pytest.mark.live

EXAMPLES = load_examples()


@pytest.fixture(scope="module")
def live_runtime(tmp_path_factory):
    """One real runtime for the module: real embeddings, real 145-task store.

    Session-wide is safe because no test here writes the knowledge base without
    asking for it — discovery drafts and pathway drafts do not mutate the KB.
    The one test that does persist runs last and asserts its own before/after.
    """
    from playbook import configure_runtime
    from playbook.kb import DEFAULT_DATA_DIR

    workdir = tmp_path_factory.mktemp("live")
    playbook, store = configure_runtime(
        data_dir=DEFAULT_DATA_DIR,
        store_path=workdir / "run_store.sqlite",
        vectors_path=workdir / "embeddings.duckdb",
        method="bertopic",
    )
    task_count = len(store.fetchall("SELECT task_id FROM tasks"))
    assert task_count == 145, f"expected the 145-task demo set, got {task_count}"
    return playbook, store


@pytest.fixture
def agent(live_runtime):
    return build_agent()


def _thread(name: str) -> str:
    return f"live-{name}-{uuid.uuid4().hex[:6]}"


@pytest.mark.parametrize(
    "example",
    EXAMPLES,
    ids=[row["inputs"]["content"][:40] for row in EXAMPLES],
)
def test_agent_answers_the_request_with_the_right_tools(agent, example):
    """Tool choice on a real request, graded exactly as the LangSmith eval grades it."""
    trace = run_request(agent, example["inputs"]["content"], thread_id=_thread("eval"))
    result = grade(example["outputs"], trace)
    assert result["passed"], (
        f"request: {example['inputs']['content']}\n"
        f"called: {result['called']}\n"
        f"missing: {result['missing_tools']}\n"
        f"forbidden: {result['forbidden_used']}\n"
        f"args: {result['arg_failures']}"
    )
    assert trace["answer"], "the agent ran tools but never answered"


def test_classification_labels_real_tasks(agent, live_runtime):
    """Classification has to move the store, not just report that it did."""
    _playbook, store = live_runtime
    trace = run_request(
        agent,
        "Classify the 2026-08-25 to 2026-09-08 cohort with bertopic. "
        "Stop after classification.",
        thread_id=_thread("classify"),
    )
    assert "classify_intent" in trace["tools"]

    run_id = _latest_run_id(store)
    labelled = store.latest_intents(run_id)
    assert labelled, "no intent labels were written"
    named = [
        row for row in labelled.values() if dict(row).get("intent_id") not in (None, "", "unknown")
    ]
    assert named, "every task came back unknown; classification did nothing"


def test_discovery_persists_proposals_for_a_real_cohort(agent, live_runtime):
    """Discovery drafts candidates into the store without touching the KB."""
    playbook, store = live_runtime
    before = set(playbook.intent_ids())
    trace = run_request(
        agent,
        "Run discovery on tasks between 2026-09-08 and 2026-09-12",
        thread_id=_thread("discover"),
    )
    assert "discover_intent" in trace["tools"]

    run_id = _latest_run_id(store)
    proposals = store.list_proposals(run_id)
    assert proposals, "discovery returned but persisted no proposals"
    assert set(playbook.intent_ids()) == before, "a draft must not write the KB"


def test_recommend_then_approve_writes_a_pathway(agent, live_runtime):
    """The two-turn path: draft a pathway, then approve it into the KB."""
    playbook, store = live_runtime
    thread = _thread("pathway")
    drafted = run_request(
        agent,
        "Are there new pathways in the 2026-09-08 and 2026-09-12 cohort",
        thread_id=thread,
    )
    assert "recommend_pathway" in drafted["tools"]

    run_id = _latest_run_id(store)
    drafts = store.list_recommendations(run_id)
    assert drafts, "recommend_pathway produced no draft"

    before = playbook.version
    approved = run_request(
        agent,
        "Approve that pathway and write it to the knowledge base.",
        thread_id=thread,
    )
    assert "persist_recc" in approved["tools"], approved["tools"]
    assert playbook.version >= before


def _latest_run_id(store) -> str:
    from playbook import runtime as rt

    if rt.current_run_id:
        return rt.current_run_id
    rows = store.fetchall("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1")
    assert rows, "no run was recorded"
    return rows[0]["run_id"]
