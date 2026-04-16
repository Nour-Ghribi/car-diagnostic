## Hybrid Agent Migration Note

### What changed

The project now follows a cleaner hybrid-agent flow:

1. `nlp/llm_rewriter.py`
2. `nlp/semantic_normalizer.py`
3. `nlp/retriever.py`
4. `nlp/llm_resolver.py`
5. `validation/plan_validator.py`
6. `agent/orchestrator.py`
7. `tools/executor.py`
8. `agent/response_composer.py`

The main architectural change is that the execution plan is no longer constructed inside the orchestrator with business `if/else` branches. The resolver now emits a structured plan, the validator checks it, and the orchestrator executes it safely.

### New flow

```text
User Prompt
-> LLM Rewriter / Clarifier
-> Semantic Normalizer
-> SBERT Retrieval Top-K
-> LLM Resolver / Planner
-> JSON / Plan Validation
-> Orchestrator Execution
-> Tools
-> Final Diagnostic Response
```

### Mock mode

Mock mode remains first-class:

- mock OBD responses still come from `fake_data/tool_responses/`
- the resolver can run without network access
- the whole pipeline is testable locally

### Local testing

Run the full test suite:

```powershell
py -3.10 -m pytest -q
```

Run the interactive demo:

```powershell
py -3.10 demo/run_real_mode_demo.py
```

Run the prompt suite:

```powershell
py -3.10 demo/run_prompt_suite.py --category broad_health --vehicle-id veh_002
```
