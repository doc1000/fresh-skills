#!/usr/bin/env python3
"""Build scratch_data/eda/embeddings.duckdb from seed playbook + conversations."""

from __future__ import annotations

from pathlib import Path

from playbook.kb import DEFAULT_DATA_DIR, load_conversations, load_playbook
from playbook.topics import embed_texts
from playbook.vectors import VectorStore, sync_playbook_vectors, sync_task_vectors


def main(data_dir: Path | None = None) -> Path:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    playbook = load_playbook(root)
    convos = load_conversations(root)
    out = root / "embeddings.duckdb"
    store = VectorStore(out)
    sync_playbook_vectors(store, playbook, embed_texts)
    task_rows = [
        {
            "task_id": str(c["convo_id"]),
            "turns_json": c["original"],
            "text": " ".join(t["text"] for t in c["original"]),
            "actions": [t["text"] for t in c["original"] if t["speaker"] == "action"],
        }
        for c in convos
    ]
    sync_task_vectors(store, task_rows, embed_texts)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
