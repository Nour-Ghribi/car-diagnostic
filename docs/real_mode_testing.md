# Real Mode Testing

## What Is Real Now

The intent pipeline can now run with real cloud NLP providers for:
- semantic embeddings
- candidate reranking

The orchestrator remains deterministic and still controls:
- intent-to-signal mapping
- tool execution
- confidence composition
- final diagnostic construction

## What Is Still Mocked

The OBD/data layer is still intentionally mocked:
- no STM32 connection
- no ELM327 connection
- no live vehicle reads
- no database integration

The project still uses:
- `agent/tools.py`
- `fake_data/tool_responses/`

This means the system can be tested realistically for NLP/orchestration while keeping the vehicle/backend side stable and offline.

## Environment Variables

Supported variables:

- `OPENAI_API_KEY`
- `INTENT_EMBEDDING_PROVIDER=mock|openai`
- `INTENT_RERANKER_PROVIDER=mock|openai`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`
- `OPENAI_RERANKER_MODEL=gpt-5.1`
- `INTENT_RETRIEVAL_TOP_K=3`
- `AGENT_DEBUG=false`

Recommended first step:

```powershell
copy .env.example .env
```

Then set variables in your shell or your local environment manager.

Example PowerShell session:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:INTENT_EMBEDDING_PROVIDER="openai"
$env:INTENT_RERANKER_PROVIDER="openai"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
$env:OPENAI_RERANKER_MODEL="gpt-5.1"
```

## Running The Demo

From the project root:

```powershell
py -3.10 demo/run_real_mode_demo.py
```

The demo asks for:
- `vehicle_id`
- a natural language query

Then it prints:
- detected intent
- intent confidence
- final diagnosis
- final confidence
- evidence
- missing data
- actions taken
- recommendations

It also clearly shows:
- embedding provider in use
- reranker provider in use
- that the OBD/data layer is still mocked

## Switching Between Mock And OpenAI

Default offline-safe mode:

```powershell
$env:INTENT_EMBEDDING_PROVIDER="mock"
$env:INTENT_RERANKER_PROVIDER="mock"
```

Real NLP mode:

```powershell
$env:INTENT_EMBEDDING_PROVIDER="openai"
$env:INTENT_RERANKER_PROVIDER="openai"
```

You can also enable only one real layer at a time:
- real embeddings + mock reranker
- mock embeddings + real reranker

## Common Failure Cases

### Missing `OPENAI_API_KEY`

If an OpenAI provider is requested but no key is available:
- embeddings fall back to mock safely
- reranking falls back to mock safely
- the pipeline does not crash

### Provider Runtime Failure

If the OpenAI SDK raises an exception or the provider call fails:
- reranking falls back to the local mock reranker
- the parser still returns a valid `Intent` or `UNKNOWN`

### No Network Wanted In Tests

Default test execution uses mock providers.
No network access is required unless you explicitly set:
- `INTENT_EMBEDDING_PROVIDER=openai`
- or `INTENT_RERANKER_PROVIDER=openai`

## Why OBD Remains Mocked

OBD remains mocked on purpose because the current milestone is focused on:
- robust NLP understanding
- strict structured outputs
- safe orchestration
- confidence and ambiguity handling

This keeps the system testable and stable while the real backend/hardware path can be integrated later without changing the agent architecture.
