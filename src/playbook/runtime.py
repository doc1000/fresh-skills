"""Process-local playbook + store binding used by tools and graph nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from playbook.kb import DEFAULT_DATA_DIR, PlaybookKB, load_conversations, load_playbook
from playbook.store import TaskStore

playbook: PlaybookKB | None = None
store: TaskStore | None = None


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
) -> tuple[PlaybookKB, TaskStore]:
    """Load the seed playbook and rebuild the working store."""
    global playbook, store
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if load_env:
        load_dotenv(root.parent / ".env")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "fresh-skills")
    playbook = playbook_kb or load_playbook(root)
    store = TaskStore(Path(store_path) if store_path is not None else root / "run_store.sqlite")
    store.load_tasks(conversations if conversations is not None else load_conversations(root))
    return playbook, store
