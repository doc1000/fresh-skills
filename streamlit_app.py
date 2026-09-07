"""Streamlit front end for the fresh-skills playbook agent.

Mirrors the notebook flow: configure the runtime once, build the deep agent
with `create_playbook_agent`, then hold one threaded conversation against it.

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    # Lets the app run from a checkout where the package was not pip-installed.
    sys.path.insert(0, str(SRC))

REPO_URL = "https://github.com/doc1000/fresh-skills"
DEFAULT_MODEL = os.environ.get("FRESH_SKILLS_MODEL", "openai:gpt-4.1-mini")

INTRO = """
Hello. I am ready to identify customer service call intentions (intent) and
sub-intentions (subflow). I can:

- Classify tasks against intents already in the knowledge base (KB).
- Discover new intents and subflows and add them to the KB.
- Recommend pathways — agent workflows that succeeded for a subflow — and write
  them as guidelines in the KB.
- Explore the current KB.
- Run the whole sequence: classify against the existing KB, discover new intents
  and subflows, then recommend workflows.

Pass a date range, or I will look for unclassified tasks.

### current seeded data
seed: ~Aug 25–Sep 8

noise: Sep 9–10

status_payment_method: Sep 10–12

slow_speed: Sep 12–14
"""

RESET_WARNING = """
**This starts over.** The conversation, the working task store and the knowledge
base all return to their seeded state — discovered intents, discovered subflows
and persisted pathways are lost. The agent is discarded along with its
checkpointer, so every conversation thread goes with it. The reset applies to
the whole app process, so anyone else viewing this demo starts over too.
"""

st.set_page_config(page_title="Fresh Skills Agent", page_icon="🧭", layout="centered")


# --------------------------------------------------------------------------
# Environment and runtime
# --------------------------------------------------------------------------
ENV_KEYS = (
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "FRESH_SKILLS_DATA_DIR",
)


def load_env() -> None:
    """`.env` and the process environment. Streamlit secrets are optional."""
    from dotenv import load_dotenv

    # override=True: a stale OPENAI_API_KEY in the shell must not beat `.env`.
    # `playbook.runtime.load_dotenv` uses setdefault, so it will not undo this.
    load_dotenv(ROOT / ".env", override=True)
    try:
        # Only reached when a secrets.toml exists; any failure means there is
        # none, and .env has already supplied the values.
        secrets = dict(st.secrets)
    except Exception:
        return
    for key in ENV_KEYS:
        if not os.environ.get(key) and key in secrets:
            os.environ[key] = str(secrets[key])


KB_FILES = ("seed_ontology.json", "seed_kb.json", "seed_guidelines.json")
TASK_FILES = ("incoming_conversations.json",)


def search_path() -> list[Path]:
    """Where to look, best first: the EDA fill output, then the repo seed."""
    from playbook.fill import EDA_DATA_DIR

    override = os.environ.get("FRESH_SKILLS_DATA_DIR")
    return ([Path(override)] if override else []) + [EDA_DATA_DIR, ROOT / "scratch_data"]


def resolve_sources() -> tuple[Path, Path]:
    """Directories holding the tasks and the KB. They need not be the same.

    `scratch_data/eda/` is the notebook fill output and wins when present. A
    folder with tasks but no KB of its own falls back to the repo seed KB.
    """
    candidates = search_path()

    def first_with(names: tuple[str, ...]) -> Path | None:
        for candidate in candidates:
            if all((candidate / name).exists() for name in names):
                return candidate
        return None

    tasks_dir = first_with(TASK_FILES)
    kb_dir = first_with(KB_FILES)
    if tasks_dir is None or kb_dir is None:
        looked = "\n".join(f"- {c}" for c in candidates)
        missing = "tasks" if tasks_dir is None else "knowledge base"
        raise FileNotFoundError(
            f"No seeded {missing} found. Looked in:\n{looked}\n"
            "Run the EDA fill notebook or set FRESH_SKILLS_DATA_DIR."
        )
    return tasks_dir, kb_dir


def show_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def release_store() -> None:
    """Close this thread's SQLite handle on the outgoing store."""
    from playbook import runtime as rt

    store = getattr(rt, "store", None)
    conn = getattr(getattr(store, "_local", None), "conn", None) if store else None
    if conn is not None:
        conn.close()
        store._local.conn = None


def sweep_stores(data_dir: Path, keep: Path) -> None:
    """Best-effort delete of stores from earlier generations."""
    for path in sorted(data_dir.glob("app_store*.sqlite*")):
        if path.name.startswith(keep.name):
            continue
        try:
            path.unlink()
        except OSError:
            # A LangGraph worker thread still holds it. Harmless: the file is
            # dead weight, and the next sweep will get it.
            pass


@st.cache_resource(show_spinner=False)
def bootstrap_runtime(tasks_dir: str, kb_dir: str, generation: int):
    """Load the seed playbook and rebuild the working store. Destructive.

    Each generation gets its own SQLite file. `TaskStore` unlinks its path on
    construction, and Windows refuses that while any thread still holds the
    handle — `TaskStore` opens one per thread and LangGraph runs tools on
    worker threads, so a shared filename cannot be reclaimed on reset.
    """
    from playbook import configure_runtime
    from playbook.kb import load_playbook

    root = Path(tasks_dir)
    store_path = root / f"app_store_{generation}.sqlite"
    release_store()
    sweep_stores(root, keep=store_path)
    # configure_runtime reads both from data_dir; pass the KB when it lives
    # somewhere else.
    playbook_kb = load_playbook(Path(kb_dir)) if kb_dir != tasks_dir else None
    return configure_runtime(
        data_dir=root,
        playbook_kb=playbook_kb,
        store_path=store_path,
    )


@st.cache_resource(show_spinner=False)
def build_agent(model_name: str, generation: int):
    from langchain.chat_models import init_chat_model

    from playbook import create_playbook_agent

    model = init_chat_model(model_name, temperature=0)
    return create_playbook_agent(model=model)


@st.cache_resource(show_spinner=False)
def app_state() -> dict:
    """Process-wide counter. The runtime is a module global, so a reset in one
    browser session has to move every session to the same generation."""
    return {"generation": 0}


def new_thread() -> str:
    return f"streamlit-{uuid.uuid4().hex[:8]}"


def reset_agent() -> None:
    """Drop the agent (and with it every MemorySaver thread), then rebuild."""
    from playbook import runtime as rt

    app_state()["generation"] += 1
    st.session_state.thread_id = new_thread()
    st.session_state.messages = [{"role": "assistant", "content": INTRO, "tools": []}]
    build_agent.clear()
    bootstrap_runtime.clear()
    rt.current_run_id = None


def is_streamlit_control_flow(exc: BaseException) -> bool:
    """Stop/rerun signals travel as exceptions and must not be swallowed."""
    return type(exc).__module__.startswith("streamlit.")


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
def chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def final_answer(agent, config) -> str:
    """The checkpointer's last assistant message, which is the turn's answer.

    The token stream is a preview: it carries every message the graph emits,
    including replayed input and subagent chatter. State is the source of truth.
    """
    from langchain_core.messages import AIMessage

    values = agent.get_state(config).values or {}
    for message in reversed(values.get("messages") or []):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return chunk_text(message).strip()
    return ""


def tool_lines(payload: dict) -> list[str]:
    """Tool calls and returns pulled from a LangGraph `updates` chunk."""
    from langchain_core.messages import AIMessage, ToolMessage

    lines: list[str] = []
    for update in (payload or {}).values():
        if not isinstance(update, dict):
            continue
        for msg in update.get("messages") or []:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls or []:
                    args = {k: v for k, v in (call.get("args") or {}).items() if v not in ("", None)}
                    lines.append(f"→ `{call.get('name')}` {args or ''}".rstrip())
            elif isinstance(msg, ToolMessage):
                body = str(msg.content).replace("\n", " ")
                lines.append(f"← `{msg.name}` {body[:300]}{'…' if len(body) > 300 else ''}")
    return lines


def stream_turn(agent, prompt: str, thread_id: str, record: dict) -> None:
    """Stream one turn into `record`, which is already in the transcript.

    Mutating in place means a Stop keeps whatever arrived before the interrupt.
    """
    from langchain_core.messages import AIMessage

    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [{"role": "user", "content": prompt}]}

    status = st.status("Working…", expanded=True)
    body = st.empty()

    for mode, chunk in agent.stream(payload, config=config, stream_mode=["updates", "messages"]):
        if mode == "messages":
            message, _meta = chunk
            # Assistant text only. This stream also replays human turns and
            # tool returns, which is what produced the run-on transcript.
            if not isinstance(message, AIMessage):
                continue
            piece = chunk_text(message)
            if piece:
                record["content"] += piece
                body.markdown(record["content"])
        elif mode == "updates":
            for line in tool_lines(chunk):
                record["tools"].append(line)
                status.write(line)

    # Reconcile: whatever the stream accumulated, the answer is the one the
    # checkpointer committed. Covers a turn that ended on a tool call, and a
    # stream polluted by replay or a subagent.
    settled = final_answer(agent, config)
    if settled:
        record["content"] = settled
    elif not record["content"].strip():
        record["content"] = "_The agent ran tools but returned no message._"
    body.markdown(record["content"])

    count = len(record["tools"])
    status.update(
        label=f"{count} tool events" if count else "No tools used",
        state="complete",
        expanded=False,
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.session_state.setdefault("thread_id", new_thread())
st.session_state.setdefault(
    "messages", [{"role": "assistant", "content": INTRO, "tools": []}]
)

st.title("Fresh Skills Agent")
st.subheader("Manage skill discovery for the ABCD Customer Support Agent")
st.caption(f"[{REPO_URL.removeprefix('https://')}]({REPO_URL})")

load_env()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("Set `OPENAI_API_KEY` in `.env` to run the agent.")
    st.stop()

generation = app_state()["generation"]
try:
    tasks_dir, kb_dir = resolve_sources()
    with st.spinner("Loading knowledge base and embeddings…"):
        playbook, store = bootstrap_runtime(str(tasks_dir), str(kb_dir), generation)
        agent = build_agent(DEFAULT_MODEL, generation)
except Exception as exc:  # surfaced instead of a stack trace in the console
    st.error(f"Startup failed: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("**Runtime**")
    st.caption(f"model `{DEFAULT_MODEL}`")
    st.caption(f"tasks `{show_path(tasks_dir)}`")
    if kb_dir != tasks_dir:
        st.caption(f"kb `{show_path(kb_dir)}`")
    subflows = sum(len(playbook.subflows_for(i)) for i in playbook.intent_ids())
    st.caption(
        f"kb version `{playbook.version}` · intents `{len(playbook.intent_ids())}`"
        f" · subflows `{subflows}`"
    )
    st.caption(f"tasks loaded `{len(store.fetchall('SELECT task_id FROM tasks'))}`")
    st.caption(f"thread `{st.session_state.thread_id}` · gen `{generation}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("tools"):
            with st.expander(f"{len(message['tools'])} tool events"):
                for line in message["tools"]:
                    st.markdown(line)
        st.markdown(message["content"])

prompt = st.chat_input("Classify, discover, recommend — or ask what the KB holds")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "tools": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # In the transcript before the first token, so an interrupt keeps the partial.
    record = {"role": "assistant", "content": "", "tools": []}
    st.session_state.messages.append(record)
    with st.chat_message("assistant"):
        try:
            stream_turn(agent, prompt, st.session_state.thread_id, record)
        except Exception as exc:
            if is_streamlit_control_flow(exc):
                record["content"] = (record["content"] + "\n\n_Stopped._").lstrip()
                raise
            record["content"] = f"Agent error: `{exc}`"
            st.error(record["content"])

# Controls sit under the transcript: the page scrolls to the newest turn, so
# this row stays in view where a header would have scrolled away.
stop, reset, _ = st.columns([1, 1, 4], vertical_alignment="center")
with stop:
    # A click interrupts a running script at its next widget write, which is how
    # this escapes a long turn or a wedged one. Nothing to do on this pass.
    st.button("Stop", width="stretch", help="Interrupt the current turn")
with reset:
    with st.popover("New agent", width="stretch"):
        st.warning(RESET_WARNING)
        st.button("Reset everything", type="primary", width="stretch", on_click=reset_agent)
