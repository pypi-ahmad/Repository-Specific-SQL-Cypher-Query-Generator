# 00 - Orientation and Workflow

## What it is

This project is a full local pipeline that converts natural-language questions into SQL and Cypher, then evaluates those generations with both text metrics and execution checks.

## Why it is used

Prompt-only query generation often fails silently (wrong joins, missing filters, invalid syntax). This system adds profile-based reproducibility, schema-grounded prompting, training, and measurable evaluation.

## How it appears in code

- End-to-end orchestration: `src/repo_query_gen/pipeline.py`
- CLI entrypoint: `scripts/run_end_to_end.py`
- Stage modules:
  - `data_prep.py`
  - `cypher.py`
  - `baselines.py`
  - `training.py`
  - `inference.py`
  - `evaluation.py`
  - `neo4j_demo.py`

## Practical explanation

Pipeline order executed by `run_end_to_end`:
1. Prepare split data (`train/val/test`).
2. Build Cypher labels from SQL.
3. Run baseline generation.
4. Optionally fine-tune adapters.
5. Optionally run batch inference.
6. Evaluate metrics, judging, and Spider execution.
7. Optionally load Neo4j graph and run demo queries.

The manifest JSON is the source of truth for stage outputs. Example: `artifacts/manifest_tutorial_real_run.json`.
