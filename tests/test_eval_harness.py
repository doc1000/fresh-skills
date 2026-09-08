"""The eval harness itself, verified without a network call.

`tests/test_live_agent.py` needs an API key and a model download, so it is
deselected by default. Everything around the model — driving the agent loop,
pulling the tool calls back out, and grading them against the dataset — is
plain code, and this exercises it with a scripted model instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

from agent_target import (  # noqa: E402
    BUILTIN_TOOLS,
    final_text,
    grade,
    load_examples,
    run_request,
    tool_calls,
)


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of assistant turns, ignoring the prompt.

    Enough to drive the real tool loop: the graph executes whatever tool calls
    the script carries, against the real tools and the real store.
    """

    script: list[AIMessage]
    cursor: list[int] = [0]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(self.cursor[0], len(self.script) - 1)
        self.cursor[0] += 1
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------
# Trace extraction and grading
# --------------------------------------------------------------------------
def test_tool_calls_and_final_text_read_a_transcript():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "cohort", "args": {"start_date": "2026-09-01"}, "id": "1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "classify_intent", "args": {}, "id": "2", "type": "tool_call"}]),
        AIMessage(content="Classified 51 tasks."),
    ]
    calls = tool_calls(messages)
    assert [call["name"] for call in calls] == ["cohort", "classify_intent"]
    assert calls[0]["args"]["start_date"] == "2026-09-01"
    assert final_text(messages) == "Classified 51 tasks."


def test_grade_passes_a_clean_run():
    expected = {
        "expected_tools": ["cohort", "classify_intent"],
        "forbidden_tools": ["persist_recc"],
        "expected_args": {"cohort": {"start_date": "2026-09-01"}},
    }
    trace = {
        "tools": ["cohort", "classify_intent"],
        "tool_calls": [
            {"name": "cohort", "args": {"start_date": "2026-09-01", "end_date": "2026-09-07"}},
            {"name": "classify_intent", "args": {}},
        ],
    }
    assert grade(expected, trace)["passed"]


def test_grade_catches_each_kind_of_failure():
    expected = {
        "expected_tools": ["cohort", "classify_intent"],
        "forbidden_tools": ["persist_recc"],
        "expected_args": {"cohort": {"start_date": "2026-09-01"}},
    }

    missing = grade(expected, {"tools": ["cohort"], "tool_calls": [{"name": "cohort", "args": {"start_date": "2026-09-01"}}]})
    assert not missing["passed"] and missing["missing_tools"] == ["classify_intent"]

    forbidden = grade(
        expected,
        {
            "tools": ["cohort", "classify_intent", "persist_recc"],
            "tool_calls": [
                {"name": "cohort", "args": {"start_date": "2026-09-01"}},
                {"name": "classify_intent", "args": {}},
                {"name": "persist_recc", "args": {}},
            ],
        },
    )
    assert not forbidden["passed"] and forbidden["forbidden_used"] == ["persist_recc"]

    wrong_args = grade(
        expected,
        {
            "tools": ["cohort", "classify_intent"],
            "tool_calls": [
                {"name": "cohort", "args": {"start_date": "2026-08-01"}},
                {"name": "classify_intent", "args": {}},
            ],
        },
    )
    assert not wrong_args["passed"] and wrong_args["arg_failures"]


def test_builtin_scratchpad_tools_do_not_count_as_forbidden():
    """deepagents ships ls/read_file/write_todos. They are not domain tools."""
    expected = {"expected_tools": ["retrieve_guidance"], "forbidden_tools": ["cohort"]}
    trace = {
        "tools": sorted(BUILTIN_TOOLS) + ["retrieve_guidance"],
        "tool_calls": [{"name": "retrieve_guidance", "args": {}}],
    }
    result = grade(expected, trace)
    assert result["passed"]
    assert result["called"] == ["retrieve_guidance"]


def test_every_dataset_row_is_well_formed():
    """A typo in the dataset should fail here, not halfway through a paid run."""
    from playbook import PLAYBOOK_TOOLS

    known = {tool.name for tool in PLAYBOOK_TOOLS}
    rows = load_examples()
    assert rows
    for row in rows:
        assert row["inputs"]["content"].strip()
        outputs = row["outputs"]
        named = set(outputs.get("expected_tools", [])) | set(outputs.get("forbidden_tools", []))
        assert named <= known, f"unknown tool in dataset row: {named - known}"
        assert not (set(outputs.get("expected_tools", [])) & set(outputs.get("forbidden_tools", [])))
        assert set(outputs.get("expected_args", {})) <= known


# --------------------------------------------------------------------------
# The agent loop, driven by a scripted model
# --------------------------------------------------------------------------
def test_run_request_drives_the_real_tools_and_reports_them(runtime, stub_topic_fit):
    """No API key: a scripted model chooses the tools, the real ones execute."""
    from playbook import create_playbook_agent
    from playbook import runtime as rt

    model = ScriptedChatModel(
        script=[
            _call("cohort", {"start_date": "2026-09-01", "end_date": "2026-09-07", "method": "jaccard", "run_id": "harness"}, "1"),
            _call("classify_intent", {}, "2"),
            AIMessage(content="Classified the cohort."),
        ]
    )
    agent = create_playbook_agent(model=model)
    trace = run_request(agent, "classify that week", thread_id="harness-thread")

    assert trace["tools"][:2] == ["cohort", "classify_intent"]
    assert trace["answer"] == "Classified the cohort."

    # the tools really ran: the store has the cohort and the labels
    assert rt.store.cohort_ids("harness")
    assert rt.store.latest_intents("harness")

    result = grade(
        {
            "expected_tools": ["cohort", "classify_intent"],
            "forbidden_tools": ["persist_recc"],
            "expected_args": {"cohort": {"start_date": "2026-09-01"}},
        },
        trace,
    )
    assert result["passed"], result
