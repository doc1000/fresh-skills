"""Process-local playbook + store binding used by tools and graph nodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from playbook.config import ROOT
from playbook.kb import DEFAULT_DATA_DIR, PlaybookKB, load_conversations, load_playbook
from playbook.store import TaskStore
from playbook.topics import BertopicConfig, FitRule

playbook: PlaybookKB | None = None
store: TaskStore | None = None
current_run_id: str | None = None
method: str = "bertopic"
topic_config: BertopicConfig | None = None
min_sim: float = 0.60
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
    playbook_kb: PlaybookKB | None = None,
    conversations: list[dict[str, Any]] | None = None,
    load_env: bool = True,
    method: str = "bertopic",
    topic_config: BertopicConfig | None = None,
    min_sim: float = 0.60,
    min_margin: float = 0.05,
    fit_rule: FitRule = "either",
) -> tuple[PlaybookKB, TaskStore]:
    """Load the seed playbook and rebuild the working store. Does not invoke the agent."""
    global playbook, store, current_run_id
    from playbook import runtime as rt
    current_run_id = None

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if load_env:
        load_dotenv(ROOT / ".env")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "fresh-skills")
    playbook = playbook_kb or load_playbook(root)
    store = TaskStore(Path(store_path) if store_path is not None else root / "run_store.sqlite")
    store.load_tasks(conversations if conversations is not None else load_conversations(root))
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
    return playbook, store
