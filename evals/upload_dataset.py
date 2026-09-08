"""Upload data/eval/responsiveness.json to LangSmith."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET_NAME = "playbook-responsiveness"
EXAMPLES_PATH = HERE / "responsiveness.json"


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is missing. Copy .env first.")

    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Tool-use: did the agent run the asked action",
        )
        existing = 0
    else:
        existing = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if existing:
            print(f"dataset already has {existing} examples: {dataset.url}")
            return

    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"uploaded {len(examples)} examples → {dataset.url}")


if __name__ == "__main__":
    main()
