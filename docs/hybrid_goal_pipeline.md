# Hybrid Goal Understanding Pipeline

## Overview

The intent layer now uses a hybrid, goal-oriented pipeline:

1. `semantic_normalizer`
2. SBERT-ready semantic retrieval over goal cards
3. deterministic reranking
4. confidence scoring with top-1, top-2, and score gap
5. ambiguity policy
6. optional LLM fallback only on low-confidence cases
7. planner/orchestrator mapping from broad goals to tool plans

The public project intents remain compatible with V2:
- `READ_DTC`
- `CHECK_CYLINDER`
- `CHECK_ENGINE_HEALTH`
- `CHECK_SIGNAL_STATUS`
- `EXPLAIN_WARNING_LIGHT`
- `GET_VEHICLE_CONTEXT`
- `UNKNOWN`

The internal layer adds broader semantic goals such as:
- `VEHICLE_HEALTH_CHECK`
- `BATTERY_CHECK`
- `ENGINE_TEMPERATURE_CHECK`
- `SIGNAL_STATUS_CHECK`
- `PERFORMANCE_ISSUE_CHECK`

Those goals are stored in `Intent.parameters.goal` and mapped back to the compatible public intent name.

## Why broad health prompts now work

Broad prompts such as:
- `je veux connaître le health de ma voiture`
- `je vx conaitre l etat de ma voiture`
- `check my car health`

no longer fall into `UNKNOWN` because the pipeline now:

- normalizes FR/EN and spelling noise into canonical automotive terms such as `vehicle` and `health`
- retrieves the broad goal card `VEHICLE_HEALTH_CHECK`
- accepts broad automotive requests through the ambiguity policy as `ACCEPT_BROAD_GOAL`
- returns the compatible public intent `CHECK_ENGINE_HEALTH`
- carries `goal=VEHICLE_HEALTH_CHECK` and `scope=broad` into the orchestrator

The orchestrator then builds a default diagnostic plan instead of treating the request as unsupported.

## Selected SBERT model

Default model:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Why this model:
- multilingual sentence embeddings suitable for French, English, and mixed FR/EN prompts
- compact enough for practical production retrieval
- well-suited to semantic similarity and top-k retrieval workloads

If `sentence-transformers` is unavailable locally, the code falls back safely to the deterministic mock embedding provider so offline tests still pass.

## When LLM fallback is triggered

The LLM is no longer the primary understanding engine.

It is triggered only when:
- confidence falls below `INTENT_LLM_FALLBACK_THRESHOLD`
- or the ambiguity policy returns `CLARIFICATION_NEEDED`

The LLM fallback receives:
- the original prompt
- the top retrieved goal candidates

and returns strict JSON only:

```json
{
  "goal": "...",
  "scope": "broad|specific",
  "confidence": 0.0,
  "reason": "...",
  "clarification_question": null
}
```

It never chooses tools directly.

## Ambiguity policy

The ambiguity layer now distinguishes:
- `ACCEPT`
- `ACCEPT_BROAD_GOAL`
- `CLARIFICATION_NEEDED`
- `UNKNOWN`

Important behavior changes:
- broad automotive health requests are valid and should be accepted
- vague automotive symptom prompts can trigger clarification instead of immediate `UNKNOWN`
- unrelated or unsupported action prompts like `repair everything automatically` still map to `UNKNOWN`

## Broad-goal planning

For `goal=VEHICLE_HEALTH_CHECK`, the orchestrator now builds a default plan:

1. `get_vehicle_context`
2. `get_dtcs`
3. `get_latest_signals`
4. `request_fresh_signals` if cache data is missing or stale

This means:
- broad request != unknown
- broad request => executable diagnostic plan

## Notes

- OBD tools remain mock-backed
- deterministic logic remains primary
- LLM use is optional and limited to disambiguation only
