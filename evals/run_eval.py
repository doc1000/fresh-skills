"""Score the responsiveness dataset in LangSmith.

    uv run python evals/run_eval.py

Needs `OPENAI_API_KEY` and `LANGSMITH_API_KEY` in `.env`, and the dataset
uploaded once with `upload_dataset.py`. Each example is one user request
against a fresh working store; the graders check which tools the agent chose,
which it avoided, and whether it carried the dates it was given.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from agent_target import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_MODEL,
    build_agent,
    grade,
    new_thread,
    run_request,
)


def _target(inputs: dict) -> dict:
    """One dataset row -> one agent run. The runtime is rebuilt per example so
    a KB write in one row cannot change the answer in the next."""
    from playbook import configure_runtime
    from playbook.kb import DEFAULT_DATA_DIR

    workdir = Path(tempfile.mkdtemp(prefix="eval-"))
    configure_runtime(
        data_dir=DEFAULT_DATA_DIR,
        store_path=workdir / "run_store.sqlite",
        vectors_path=workdir / "embeddings.duckdb",
        method=os.environ.get("FRESH_SKILLS_METHOD", "bertopic"),
    )
    agent = build_agent()
    return run_request(agent, inputs["content"], thread_id=new_thread("langsmith"))


def tools_called(outputs: dict, reference_outputs: dict) -> dict:
    result = grade(reference_outputs, outputs)
    return {
        "key": "expected_tools_called",
        "score": float(result["tools_ok"]),
        "comment": f"missing: {result['missing_tools']}" if result["missing_tools"] else "all present",
    }


def no_forbidden_tools(outputs: dict, reference_outputs: dict) -> dict:
    result = grade(reference_outputs, outputs)
    return {
        "key": "no_forbidden_tools",
        "score": float(result["no_forbidden"]),
        "comment": f"used: {result['forbidden_used']}" if result["forbidden_used"] else "clean",
    }


def args_carried(outputs: dict, reference_outputs: dict) -> dict:
    result = grade(reference_outputs, outputs)
    return {
        "key": "args_carried",
        "score": float(result["args_ok"]),
        "comment": "; ".join(result["arg_failures"]) or "ok",
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is missing. Copy .env first.")

    from langsmith import Client, evaluate

    client = Client()
    results = evaluate(
        _target,
        data=DATASET_NAME,
        evaluators=[tools_called, no_forbidden_tools, args_carried],
        experiment_prefix=f"responsiveness-{DEFAULT_MODEL.split(':')[-1]}",
        client=client,
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
