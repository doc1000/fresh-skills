# Planning Task: Minimal LangChain Agent Take-Home

You are the planning agent for a two-day Agent Engineer take-home exercise.

**Do not implement anything. Do not write application code. Do not modify source files.**

Your sole job is to inspect the repository, read the authoritative take-home exercise, inspect any existing planning material, and produce a concrete implementation plan in the `docs/` folder.

The implementation will later be performed by separate agents, often in parallel Git worktrees using `/worktree`.

The plan must therefore be specific enough that individual phases can be handed to fresh agents independently.

---

# 1. First understand the exercise

Read the take-home requirements in this repository first.

Treat them as authoritative.

The exercise is intentionally small. The goal is **not** to build the most sophisticated agent possible.

The goal is to produce a small agentic application that:

* clearly uses LangGraph for state and control flow;
* clearly uses LangChain for model calls, at least one tool, and RAG;
* clearly uses LangSmith for tracing and a small evaluation;
* solves one business problem end-to-end;
* is easy for another senior engineer to inspect;
* is easy for me to explain in a notebook and interview;
* demonstrates good engineering judgment.

This is an **exercise repo**, not the foundation of a production SaaS product.

Optimize in this order:

1. understandable;
2. working;
3. testable;
4. inspectable;
5. easy to demonstrate;
6. only then more robust or elegant.

Prefer standard LangChain/LangGraph/LangSmith examples and native APIs over custom abstractions.

Prefer the simplest implementation that genuinely satisfies the exercise.

---

# 2. Product purpose

The application is an:

**Emerging Workflow / Playbook Maintenance Agent**

It reviews small weekly batches of completed customer-support conversations and asks:

> Does our current support playbook adequately cover the recurring workflows appearing in these conversations?

When a repeated workflow is not adequately covered, the agent can propose a new operational guideline.

A human must approve or edit the proposal before the live knowledge base changes.

Approved knowledge becomes part of the RAG corpus for later weekly runs.

The purpose of the demo is therefore to show:

```text
observed customer-support behavior
        ↓
retrieve current operational knowledge
        ↓
judge whether behavior is already covered
        ↓
propose missing guidance when needed
        ↓
human review
        ↓
update live knowledge
        ↓
future agent behavior changes
```

This is not a customer-support chatbot.

It is a small agent that maintains the knowledge/workflow layer used by customer-support humans or downstream agents.

---

# 3. Deliberately narrow ABCD experiment

Use the public ABCD customer-support dataset.

Do **not** broadly sample across its entire intent space.

The data exploration phase must identify:

* exactly **one ABCD top-level flow** to use for the demo;
* at most **four clean subflows** within that flow.

For purposes of this project, ABCD `subflow` is the closest operational unit to what we may casually call an "intent."

Do not choose the subflows before inspecting the data.

The selection should be made visibly during Phase 1 EDA.

Favor subflows that have:

* enough conversations to demonstrate recurrence;
* reasonably distinct customer situations;
* recognizable action sequences;
* understandable ABCD guidelines;
* low enough ambiguity that the demo is easy to explain.

Target the following controlled experiment:

## Week 1

Use only a small subset of the selected subflows.

For example:

```text
A — represented in seed KB
B — represented in seed KB
C — deliberately missing from seed KB
```

Week 1 should make C visible as a recurring uncovered workflow.

## Week 2

Use:

```text
A and/or B
C — now covered if Week 1 was approved
D — a newly introduced missing subflow
```

Week 2 should show that the persistent KB mutation matters.

The system should not rediscover C as missing if C was approved in Week 1.

## Week 3

Use only already-known subflows from A/B/C/D.

Introduce no new subflow.

Correct behavior should be:

```text
no material action
```

Abstention is part of the demonstration.

Do not require a Week 4 revision scenario.

Mention workflow revision as a possible extension only if useful.

---

# 4. Data layer: DuckDB first

Prefer DuckDB and SQL over pandas.

A reasonable data path is:

```text
ABCD raw JSON
    ↓
small preparation step
    ↓
Parquet / simple normalized local artifacts
    ↓
DuckDB SQL
    ↓
small Python/Pydantic records
    ↓
agent
```

DuckDB may query Parquet directly; a persistent database is not required unless inspection shows a concrete benefit.

Avoid pandas except where an external API genuinely requires a DataFrame or it provides an obviously simpler one-line display operation.

Do not pass DataFrames through LangGraph state.

Use small typed Python records or IDs at application boundaries.

---

# 5. Phase notebooks are mandatory review gates

Every implementation phase must eventually produce a **phase-specific Jupyter notebook** that I can inspect in that phase's worktree **before I merge it**.

Design the plan around this requirement.

The notebook is not merely polished documentation.

It is a technical review artifact.

It should expose:

* what was tried;
* what assumptions were made;
* important alternatives;
* unresolved choices;
* intermediate outputs;
* test results;
* state or data structures;
* reasons behind decisions.

Do **not** smooth over decisions simply to make the notebook look clean.

The notebook should help me understand and challenge the implementation.

Each phase's "done" criteria must include:

1. tests pass;
2. the actual feature runs;
3. the phase notebook has been executed sufficiently to inspect the result;
4. I can review the notebook before merging the worktree.

Recommend a simple naming convention such as:

```text
notebooks/
    phase_01_data_eda.ipynb
    phase_02_kb_rag.ipynb
    phase_03_graph.ipynb
    phase_04_hitl_persistence.ipynb
    phase_05_eval_ui_integration.ipynb
```

Improve the names if the final phase decomposition changes.

Eventually the project may also have a polished `walkthrough.ipynb` or `demo_story.ipynb`, but do not treat that polished notebook as a substitute for the exploratory phase notebooks.

---

# 6. Phase 1 notebook is specifically EDA

The data notebook has a different purpose from later notebooks.

It is primarily **exploratory data analysis**.

Do not write the plan as though the correct subflows are already known.

The notebook should visibly inspect enough ABCD data to make a deliberate selection.

It should likely include SQL-driven exploration such as:

* high-level flow counts;
* subflow counts within promising flows;
* number of conversations per subflow;
* observed action sequences;
* guideline/reference availability;
* several representative conversation excerpts;
* comparison of candidate subflows;
* reasons to reject ambiguous or poor demo candidates;
* final choice of the one flow and up-to-four subflows.

The final fixture-selection decision should be visible as the result of the EDA.

It is acceptable, and desirable, for the notebook to contain genuine questions and alternatives.

The point is for me to participate in this decision rather than have an implementation agent silently make it.

---

# 7. Use test-driven development as the phase structure

Frame the implementation explicitly as:

```text
RED
↓
GREEN
↓
REVIEW
↓
REFACTOR when justified
```

For every phase specify:

### Starting failing tests

What behavioral contract should exist but does not yet exist?

Keep these tests few and meaningful.

Prefer deterministic boundary tests over exhaustive mocking.

### Simplest green implementation

What is the least complicated implementation that makes those tests pass while satisfying the actual product contract?

Do not build anticipated future abstractions.

### Notebook review gate

What should I inspect interactively before merging?

### Refactor decision

At the end of the phase, explicitly state one of:

* `NO REFACTOR — implementation is still too early`
* `LOCAL REFACTOR — simplify what was just learned`
* `INTEGRATION REFACTOR — wait until components meet`
* `FINAL REFACTOR — improve readability without changing architecture`

Do not assume every phase needs refactoring.

---

# 8. Expected refactor strategy

Avoid doing a major abstraction/refactor before we have one complete vertical slice.

Prefer roughly:

```text
data fixture
    ↓
RAG
    ↓
minimal working graph
    ↓
FIRST IMPORTANT REFACTOR
    ↓
HITL + persistence
    ↓
integration
    ↓
FINAL SMALL REFACTOR
```

The first meaningful refactor should usually happen only after:

* real ABCD fixture data exists;
* real KB retrieval exists;
* one real LangGraph run succeeds;
* LangSmith tracing shows the actual execution.

At that point we understand what the code wants to be.

Refactor for:

* legibility;
* smaller functions;
* clearer typed contracts;
* removal of duplicate transformations;
* deterministic code vs LLM responsibility separation;
* obvious node boundaries.

Do **not** refactor for hypothetical future extensibility.

Avoid:

* repository layers;
* service layers;
* factories;
* dependency-injection systems;
* generic agent frameworks;
* protocol-heavy architecture;
* unnecessary classes.

A second small refactor may occur after integration to remove mock-boundary artifacts and make the final walkthrough easier to understand.

---

# 9. Keep the agent graph very small

Do not assume a large graph.

Start from the smallest graph that honestly represents the workflow.

A plausible first target is:

```text
START
  ↓
analyze_batch
  ↓
retrieve_guidance
  ↓
decide
  ├── no_action → END
  └── proposal → human_review
                       ↓
                    update_kb
                       ↓
                      END
```

Only split nodes further when doing so makes state/control flow materially clearer.

Do not create bookkeeping nodes simply to make the graph look sophisticated.

RAG should be a genuine dependency of the decision:

```text
observed evidence
      ↓
retrieve CURRENT KB guidance
      ↓
compare evidence with retrieved guidance
      ↓
covered / uncovered
```

The agent should not classify something as missing without considering the relevant currently retrievable KB content.

---

# 10. Keep RAG minimal but real

Use a small human-readable operational KB.

Preferred implementation:

```text
JSON canonical live KB
      ↓
small LangChain vector store
      ↓
LangChain retriever
```

After human approval, update the JSON and rebuild/update the tiny retrieval index.

Reference ABCD guidelines used as evaluation truth must remain outside the runtime RAG corpus.

Plan deterministic tests proving at least:

* seed KB contains only intended known subflows;
* missing subflows are absent;
* retrieval works for known workflows;
* withheld references cannot leak into runtime retrieval;
* approving a new guideline changes subsequent retrieval.

Do not deploy external vector infrastructure unless there is an unexpectedly strong reason.

---

# 11. One explicit LangChain tool

The final implementation should include at least one unmistakable LangChain tool.

Prefer something simple and naturally useful, such as:

```text
inspect_conversation(conversation_id)
```

The agent can use it to retrieve a full representative ABCD conversation and its observed actions while investigating a candidate workflow.

Do not turn every deterministic helper into an agent tool.

Use tools only where agent-directed inspection actually makes sense.

---

# 12. LangGraph / HITL

Use native LangGraph features where practical:

* typed state;
* StateGraph;
* conditional routing;
* graph visualization;
* checkpointer;
* stable thread IDs;
* `interrupt()`;
* `Command(resume=...)`;
* state/history inspection.

Human review must happen before operational KB mutation.

The reviewer should be able to:

* approve;
* edit and approve;
* reject.

Approved changes must affect later weeks.

Rejected changes must not.

Do not emulate HITL using an ordinary Boolean condition if native LangGraph interrupts solve the problem.

---

# 13. LangSmith

Set up tracing early enough that the actual graph-development work is visible.

Use LangSmith for what it is good at.

Do not build a custom tracing dashboard.

Keep evaluation small.

The exercise only needs a small expected-outcome dataset.

A perfectly acceptable first deterministic evaluation target is:

```text
Week 1 initial KB → expected: add C
Week 2 after C    → expected: add D
Week 3 after C+D  → expected: no_action
```

A semantic guideline-quality evaluator can be added if it is simple and demonstrably useful.

Do not let evaluation infrastructure become a second project.

---

# 14. Plan for parallel worktrees

Implementation will use multiple IDE agent windows and `/worktree`.

Design phases so genuinely independent work can happen in parallel.

A likely topology is:

## Data/backend workstream

Owns:

* ABCD EDA;
* deterministic demo fixtures;
* DuckDB queries;
* KB;
* retrieval;
* graph;
* HITL;
* persistence.

Within this stream, prefer the sequence:

```text
Data → RAG → minimal graph → refactor → HITL
```

RAG and graph design should not drift apart independently because retrieval is part of the decision architecture.

## UI workstream

May proceed in parallel against mock typed contracts.

Its job is only presentation and interaction.

It should be possible to build:

* week selector;
* Run button;
* findings;
* retrieved evidence;
* proposed change;
* edit/approve/reject;
* current KB display;
* reset button;

using a small mock `RunResult` / `ReviewProposal`.

Later replace mocks with the real graph interface.

Do not duplicate business logic in the UI.

---

# 15. Define integration contracts early

The plan should identify the minimum shared typed contracts required to allow worktrees to proceed independently.

For example:

```text
WeeklyBatch
ReviewProposal
RunResult
KBEntry
```

Keep these very small.

The UI should not care how the graph reasons internally.

The graph should not care how Streamlit renders results.

Clearly identify:

* which worktree owns each contract;
* when the contract should be frozen;
* what changes would require coordination;
* the first intended merge/integration point.

---

# 16. Suggested TDD phases

Do not blindly use this decomposition if repo inspection suggests a better one, but aim for approximately five phases.

## Phase 1 — ABCD EDA + deterministic demo contract

Goal:

Choose one flow and at most four subflows through visible EDA and produce the three deterministic weekly fixtures.

Example RED tests:

```text
test_only_one_top_level_flow
test_no_more_than_four_demo_subflows
test_week2_introduces_exactly_one_new_subflow
test_week3_introduces_no_new_subflow
test_demo_fixture_is_deterministic
```

Phase notebook:

Primarily EDA.

Do not hide choices.

## Phase 2 — Operational KB + RAG

Goal:

Create deliberately incomplete seed KB and prove current-KB retrieval.

Example RED tests:

```text
test_seed_kb_contains_known_subflows
test_seed_kb_excludes_missing_subflows
test_known_workflow_is_retrievable
test_reference_truth_is_not_runtime_retrievable
test_adding_guideline_changes_retrieval
```

No graph required yet.

## Phase 3 — Minimal vertical LangGraph

Goal:

Run one real Week 1 batch through the smallest useful graph.

Example RED tests:

```text
test_graph_compiles
test_graph_renders
test_week1_reaches_proposal
test_decision_receives_retrieved_context
test_conversation_tool_is_callable
```

Enable real LangSmith tracing here if not already done.

After GREEN, explicitly schedule the **first substantial refactor**.

## Phase 4 — HITL + persistent three-week behavior

Goal:

Make the demo actually evolve.

Example RED tests:

```text
test_graph_interrupts_before_mutation
test_approval_updates_kb
test_rejection_does_not_update_kb
test_thread_resumes
test_week2_finds_D_not_already_approved_C
test_week3_abstains
```

This is the core end-to-end exercise.

When this phase is green, the underlying assignment should essentially be solved.

## Phase 5 — Eval + thin UI + integration/polish

Goal:

Expose the working system, run small LangSmith evaluation, integrate parallel UI work, and make the repo easy to demonstrate.

Keep Streamlit thin.

Keep eval small.

Perform only a final readability-oriented refactor.

---

# 17. Deliverables from this planning task

Again: **do not implement these phases.**

Your output for this planning task should only be planning documents.

Create:

```text
docs/
    PLAN.md
    phases/
        phase_01.md
        phase_02.md
        phase_03.md
        phase_04.md
        phase_05.md
```

Adjust the phase count only with a concrete reason.

`docs/PLAN.md` should contain:

1. concise purpose of the repo;
2. concise statement of the exercise constraints;
3. intentionally limited scope;
4. target architecture;
5. data experiment;
6. worktree strategy;
7. integration contracts;
8. phase dependency graph;
9. refactor points;
10. final acceptance/demo sequence.

Each `phase_XX.md` must be self-contained enough to hand directly to a fresh implementation agent in a worktree.

Each phase document must specify:

* objective;
* why this phase exists;
* dependencies;
* worktree/branch ownership;
* files expected to be touched;
* shared contracts;
* starting RED tests;
* minimum GREEN behavior;
* manual run required;
* notebook required;
* what the notebook should expose;
* LangSmith artifact expected, if relevant;
* explicit review gate;
* recommended refactor action;
* explicit DONE condition;
* what must be true before merge.

---

# 18. Planning attitude

Repeatedly challenge scope while planning.

For every mechanism ask:

> What is the simplest standard LangChain-family implementation that proves the requirement?

Prefer:

* functions;
* SQL;
* tiny typed schemas;
* native framework APIs;
* transparent JSON;
* small deterministic fixtures;
* direct imports;
* standard notebook examples.

Avoid:

* premature robustness;
* speculative scalability;
* production infrastructure;
* custom frameworks;
* large abstractions;
* hidden data choices;
* polished notebooks that obscure uncertainty.

The resulting repository should feel like:

> A senior engineer built the smallest thoughtful thing that clearly demonstrates how they think.

It should not feel like:

> A take-home exercise became a prototype startup platform.

---

# 19. Stop condition

Once the planning documents have been created, stop.

Do not begin Phase 1 implementation.

Do not create application code.

Do not download/process ABCD data beyond whatever lightweight inspection is necessary to make the plan coherent.

Do not create notebooks yet.

Do not create tests yet.

I will review the plan first and then launch implementation phases in separate worktrees.
