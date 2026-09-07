"""Process-local playbook + store binding used by tools and graph nodes."""



from __future__ import annotations



import json

import os

from collections.abc import Callable

from pathlib import Path

from typing import Any



import numpy as np



from playbook.config import ROOT

from playbook.kb import DEFAULT_DATA_DIR, PlaybookKB, load_conversations, load_playbook

from playbook.store import TaskStore

from playbook.topics import BertopicConfig, FitRule, embed_texts, start_topic_stack_warmup

from playbook.vectors import VectorStore, sync_playbook_vectors, sync_task_vectors



playbook: PlaybookKB | None = None

store: TaskStore | None = None

vectors: VectorStore | None = None

embed_fn: Callable[..., np.ndarray] | None = None

current_run_id: str | None = None

method: str = "bertopic"

topic_config: BertopicConfig | None = None

min_sim: float = 0.70

min_margin: float = 0.05

fit_rule: FitRule = "either"

seed_examples: list[dict[str, Any]] = []





def load_dotenv(path: Path) -> None:

    if not path.exists():

        return

    for raw in path.read_text(encoding="utf-8").splitlines():

        line = raw.strip()

        if not line or line.startswith("#") or "=" not in line:

            continue

        key, _, value = line.partition("=")

        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))





def configure_runtime(

    *,

    data_dir: Path | None = None,

    store_path: Path | None = None,

    vectors_path: Path | None = None,

    playbook_kb: PlaybookKB | None = None,

    conversations: list[dict[str, Any]] | None = None,

    load_env: bool = True,

    method: str = method,

    topic_config: BertopicConfig | None = None,

    min_sim: float =min_sim,

    min_margin: float = min_margin,

    fit_rule: FitRule = "either",

    embed_fn_override: Callable[..., np.ndarray] | None = None,

) -> tuple[PlaybookKB, TaskStore]:

    """Load the seed playbook and rebuild the working store. Does not invoke the agent."""

    global playbook, store, current_run_id, vectors, embed_fn

    from playbook import runtime as rt

    current_run_id = None



    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    if load_env:

        load_dotenv(ROOT / ".env")

        os.environ.setdefault("LANGSMITH_TRACING", "true")

        os.environ.setdefault("LANGSMITH_PROJECT", "fresh-skills")

    playbook = playbook_kb or load_playbook(root)

    convos = conversations if conversations is not None else load_conversations(root)

    store = TaskStore(Path(store_path) if store_path is not None else root / "run_store.sqlite")

    store.load_tasks(convos)

    rt.method = method

    rt.topic_config = topic_config

    rt.min_sim = min_sim

    rt.min_margin = min_margin

    rt.fit_rule = fit_rule

    examples_path = root / "seed_examples.json"

    if examples_path.exists():

        rt.seed_examples = json.loads(examples_path.read_text(encoding="utf-8"))

    else:

        rt.seed_examples = []



    rt.embed_fn = embed_fn_override or embed_texts

    vpath = Path(vectors_path) if vectors_path is not None else root / "embeddings.duckdb"

    rt.vectors = VectorStore(vpath)

    sync_playbook_vectors(rt.vectors, playbook, rt.embed_fn)

    sync_task_vectors(rt.vectors, store.get_tasks([c["convo_id"] for c in convos]), rt.embed_fn)

    if embed_fn_override is None:
        start_topic_stack_warmup(background=True)

    return playbook, store

