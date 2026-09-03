## Scratch Notes

modular meta agent notes

class MetaAgentState(TypedDict, total=False):
    run_id: str
    cohort_query: dict[str, Any]
    kb_version: int
    current_stage: str
    target_subflow: str
    intent_summary: dict[str, Any]
    subflow_summary: dict[str, Any]
    discovery_summary: dict[str, Any]
    recommendation_summary: dict[str, Any]
    pending_proposal_ids: list[str]
    approved_change_ids: list[str]
    errors: Annotated[list[dict[str, Any]], operator.add]
    
    # I think that the summaries could be passed as messages with differnt labels or roles.  more mutable. could be included in standard ChatMessages (or maybe another kind already exists).

### persist SQL in demo
    the current is SQLite... great for this.  will want it to partially persist or at least be built from files in the shipped demo agent.  I like the builder here - it can get dropped in code.  
    it makes it easier to potentially selecting different starting datasets easier.


### use rag to match similar queries vs centroid

### linear workflow
nothing branches or demonstrates agentic deicsion making explicitly.  its almost too clean.