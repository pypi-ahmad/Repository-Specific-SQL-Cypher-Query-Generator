# 06 - Inference Pipeline

## What it is

A query-generation layer for single-request and batch inference, supporting both Ollama baseline models and fine-tuned adapters.

## Why it is used

It standardizes prompt construction, schema retrieval, postprocessing, validation, and output contracts so evaluation can compare models consistently.

## How it appears in code

- Module: `src/repo_query_gen/inference.py`
- Script: `scripts/infer.py`

Key functions:
- `infer_single(...)`
- `run_batch_inference(...)`
- `generate_with_ollama(...)`
- `generate_with_finetuned(...)`
- `postprocess_generated_query(...)`
- `validate_generated_query(...)`

Important quality behavior:
- fenced-code and preamble stripping (`_strip_fenced_block`, `_extract_sql_statement`, `_extract_cypher_statement`)
- schema retrieval for prompt reduction (`_select_schema_for_prompt`)
- parse + grounding checks in validation output.

## Practical explanation

Single query:

```bash
python scripts/infer.py \
  --task sql \
  --mode ollama \
  --question "How many countries exist?" \
  --schema-context "CREATE TABLE countries (id INT, name TEXT);"
```

Batch files used by evaluation:
- `artifacts/inference/tutorial/baseline_granite.json` (6 records)
- `artifacts/inference/tutorial/baseline_qwen.json` (6 records)
- `artifacts/inference/tutorial/finetuned.json` (6 records)

Every record includes:
- generated query,
- latency,
- retrieval metadata,
- schema references,
- validation report.
