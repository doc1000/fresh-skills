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
METHODS = ("bertopic", "jaccard")
_env_method = os.environ.get("FRESH_SKILLS_METHOD", "")
DEFAULT_METHOD = _env_method if _env_method in METHODS else METHODS[0]

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
seed: ~Aug 25–Sep 8, 2026

noise: Sep 9–10, 2026

status_payment_method: Sep 10–12, 2026

slow_speed: Sep 12–14, 2026
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


DATA_FILES = (
    "incoming_conversations.json",
    "seed_ontology.json",
    "seed_kb.json",
    "seed_guidelines.json",
)


def resolve_data_dir() -> Path:
    """The seeded tasks and KB. `FRESH_SKILLS_DATA_DIR` overrides the repo demo set."""
    from playbook.kb import DEFAULT_DATA_DIR

    override = os.environ.get("FRESH_SKILLS_DATA_DIR")
    candidate = Path(override) if override else DEFAULT_DATA_DIR
    missing = [name for name in DATA_FILES if not (candidate / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{candidate} is missing {', '.join(missing)}. "
            "Point FRESH_SKILLS_DATA_DIR at a seeded folder."
        )
    return candidate


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
def bootstrap_runtime(data_dir: str, generation: int, method: str):
    """Load the seed playbook and rebuild the working store. Destructive.

    Each generation gets its own SQLite file. `TaskStore` unlinks its path on
    construction, and Windows refuses that while any thread still holds the
    handle — `TaskStore` opens one per thread and LangGraph runs tools on
    worker threads, so a shared filename cannot be reclaimed on reset.
    """
    from playbook import configure_runtime

    root = Path(data_dir)
    store_path = root / f"app_store_{generation}.sqlite"
    release_store()
    sweep_stores(root, keep=store_path)
    return configure_runtime(data_dir=root, store_path=store_path, method=method)


@st.cache_resource(show_spinner=False)
def build_agent(model_name: str, generation: int):
    from langchain.chat_models import init_chat_model

    from playbook import create_playbook_agent

    model = init_chat_model(model_name, temperature=0)
    return create_playbook_agent(model=model)


@st.cache_resource(show_spinner=False)
def app_state() -> dict:
    """Process-wide settings. The runtime is a module global, so a reset in one
    browser session has to move every session to the same generation."""
    return {"generation": 0, "method": DEFAULT_METHOD}


def new_thread() -> str:
    return f"streamlit-{uuid.uuid4().hex[:8]}"


def reset_agent() -> None:
    """Drop the agent (and with it every MemorySaver thread), then rebuild."""
    from playbook import runtime as rt

    state = app_state()
    state["generation"] += 1
    state["method"] = st.session_state.get("method_choice", DEFAULT_METHOD)
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


def task_window(store) -> tuple[str, str, int]:
    row = store.fetchall(
        "SELECT min(conversation_date) AS lo, max(conversation_date) AS hi,"
        " count(*) AS n FROM tasks"
    )[0]
    return row["lo"] or "", row["hi"] or "", row["n"]


def session_context(lo: str, hi: str, n: int) -> str:
    """Prefix for a thread's first message.

    Without it the model has to guess a year for a bare "Aug 25", and the
    seeded data sits in a year it will not guess.
    """
    return (
        f"[session context] The task store holds {n} customer-support tasks "
        f"dated {lo} through {hi}. Resolve any partial or relative date the "
        "user gives inside that window. Do not assume a different year."
    )


def thread_is_new(agent, config) -> bool:
    return not (agent.get_state(config).values or {}).get("messages")


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


def stream_turn(agent, prompt: str, thread_id: str, record: dict, context: str = "") -> None:
    """Stream one turn into `record`, which is already in the transcript.

    Mutating in place means a Stop keeps whatever arrived before the interrupt.
    """
    from langchain_core.messages import AIMessage

    config = {"configurable": {"thread_id": thread_id}}
    # The context rides on the first message only; after that it is in history.
    sent = f"{context}\n\n{prompt}" if context and thread_is_new(agent, config) else prompt
    payload = {"messages": [{"role": "user", "content": sent}]}

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

generation, method = app_state()["generation"], app_state()["method"]
try:
    data_dir = resolve_data_dir()
    with st.spinner("Loading knowledge base and embeddings…"):
        playbook, store = bootstrap_runtime(str(data_dir), generation, method)
        agent = build_agent(DEFAULT_MODEL, generation)
except Exception as exc:  # surfaced instead of a stack trace in the console
    st.error(f"Startup failed: {exc}")
    st.stop()

task_lo, task_hi, n_tasks = task_window(store)

with st.sidebar:
    st.markdown("**Runtime**")
    st.caption(f"model `{DEFAULT_MODEL}` · scoring `{method}`")
    st.caption(f"data `{show_path(data_dir)}`")
    subflows = sum(len(playbook.subflows_for(i)) for i in playbook.intent_ids())
    st.caption(
        f"kb version `{playbook.version}` · intents `{len(playbook.intent_ids())}`"
        f" · subflows `{subflows}`"
    )
    st.caption(f"tasks loaded `{n_tasks}` · `{task_lo}` → `{task_hi}`")
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
            stream_turn(
                agent,
                prompt,
                st.session_state.thread_id,
                record,
                context=session_context(task_lo, task_hi, n_tasks),
            )
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
        st.radio(
            "Scoring method",
            METHODS,
            index=METHODS.index(method),
            key="method_choice",
            horizontal=True,
            help="jaccard skips the BERTopic fit — faster, coarser.",
        )
        st.caption("Applied on reset.")
        st.warning(RESET_WARNING)
        st.button("Reset everything", type="primary", width="stretch", on_click=reset_agent)
