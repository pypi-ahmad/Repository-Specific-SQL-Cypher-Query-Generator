# 04 - Schema Retrieval and Baselines

## What it is

A prompt-only baseline stage that generates SQL and Cypher from local Ollama models, optionally using lexical schema retrieval to reduce prompt context.

## Why it is used

Before training adapters, you need a baseline reference to measure whether fine-tuning actually improves behavior.

## How it appears in code

- Baseline module: `src/repo_query_gen/baselines.py`
- Schema retrieval module: `src/repo_query_gen/schema_retrieval.py`
- Baseline script: `scripts/run_baselines.py`

Important functions:
- `parse_schema_context(...)`
- `select_schema_context(...)`
- `_build_prompt(...)`
- `_generate(...)`
- `run_baseline_generation(...)`

The retrieval strategy and top-k tables are controlled by:
- `schema_retrieval_mode`
- `schema_retrieval_top_k`

## Practical explanation

Run:

```bash
python scripts/run_baselines.py --profile tutorial
```

Artifacts:
- `artifacts/baseline/tutorial/baseline_predictions.csv`
- `artifacts/baseline/tutorial/baseline_predictions.json`

Inference JSON artifacts created by batch inference:
- `artifacts/inference/tutorial/baseline_granite.json`
- `artifacts/inference/tutorial/baseline_qwen.json`

Real inference sample sizes:
- fast: 2 records per model JSON
- tutorial: 6 records per model JSON

The baseline stage also records retrieval-selected tables and columns, so you can inspect grounding behavior row-by-row.
