"""Shared target and grading for the responsiveness eval.

`run_eval.py` sends this to LangSmith; `tests/test_live_agent.py` asserts on it
locally. One dataset, one definition of what a correct run looks like.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXAMPLES_PATH = HERE / "responsiveness.json"
DATASET_NAME = "playbook-responsiveness"
DEFAULT_MODEL = "openai:gpt-4.1-mini"

# deepagents ships its own scratchpad tools. They are not domain tools, so a
# forbidden-tool check has to ignore them.
BUILTIN_TOOLS = frozenset(
    {
        "task",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "delete",
        "execute",
        "write_todos",
        "think_tool",
    }
)


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    """The eval dataset: rows of {"inputs": {"content": ...}, "outputs": {...}}."""
    return json.loads((path or EXAMPLES_PATH).read_text(encoding="utf-8"))


def build_agent(model: Any | None = None):
    """The same agent the Streamlit app builds."""
    from langchain.chat_models import init_chat_model

    from playbook import create_playbook_agent

    if model is None:
        model = init_chat_model(DEFAULT_MODEL, temperature=0)
    return create_playbook_agent(model=model)


def new_thread(prefix: str = "eval") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Every tool call the run made, in order, as {"name", "args"}."""
    from langchain_core.messages import AIMessage

    calls: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            calls.append({"name": call.get("name", ""), "args": call.get("args") or {}})
    return calls


def final_text(messages: list[Any]) -> str:
    from langchain_core.messages import AIMessage

    for message in reversed(messages or []):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str):
                return content.strip()
            return "".join(
                block.get("text", "")
                for block in content or []
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
    return ""


def run_request(agent, content: str, *, thread_id: str | None = None) -> dict[str, Any]:
    """Send one user request. Returns the trace the graders read."""
    config = {"configurable": {"thread_id": thread_id or new_thread()}}
    result = agent.invoke({"messages": [{"role": "user", "content": content}]}, config=config)
    messages = result.get("messages") or []
    calls = tool_calls(messages)
    return {
        "tool_calls": calls,
        "tools": [call["name"] for call in calls],
        "answer": final_text(messages),
    }


def domain_tools(names: list[str]) -> list[str]:
    return [name for name in names if name not in BUILTIN_TOOLS]


def grade(expected: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Score one run against one dataset row. All three checks must hold.

    * every expected tool ran
    * no forbidden tool ran (deepagents' own scratchpad tools do not count)
    * where the row pins arguments, some call carried them
    """
    called = domain_tools(trace.get("tools") or [])
    called_set = set(called)

    missing = [name for name in (expected.get("expected_tools") or []) if name not in called_set]
    used_forbidden = sorted(set(expected.get("forbidden_tools") or []) & called_set)

    arg_failures: list[str] = []
    for tool_name, pinned in (expected.get("expected_args") or {}).items():
        seen = [call["args"] for call in trace.get("tool_calls") or [] if call["name"] == tool_name]
        if not seen:
            arg_failures.append(f"{tool_name}: never called")
        elif not any(all(args.get(k) == v for k, v in pinned.items()) for args in seen):
            arg_failures.append(f"{tool_name}: no call matched {pinned}; saw {seen}")

    return {
        "called": called,
        "missing_tools": missing,
        "forbidden_used": used_forbidden,
        "arg_failures": arg_failures,
        "tools_ok": not missing,
        "no_forbidden": not used_forbidden,
        "args_ok": not arg_failures,
        "passed": not missing and not used_forbidden and not arg_failures,
    }
