# 09 - Real Runs and Artifact Reading

## What it is

A guide to reading completed run outputs directly from `artifacts/` and `data/processed/` so conclusions are based on files, not assumptions.

## Why it is used

This project has multiple run scales and repeated attempts. Correct reporting requires selecting completed artifacts and distinguishing partial runs.

## How it appears in code

- End-to-end manifest writing: `save_manifest(...)` in `src/repo_query_gen/pipeline.py`
- Stage artifact outputs are returned by each module and aggregated in `run_end_to_end(...)`.

## Practical explanation

### A) Completed tutorial end-to-end run

Primary manifest:
- `artifacts/manifest_tutorial_real_run.json`

Key files from that manifest:
- training run dir: `artifacts/training/tutorial_2026-06-20T19-14-21.185341+00-00_trl`
- inference: `artifacts/inference/tutorial/*.json`
- evaluation: `artifacts/evaluation/tutorial/*.json`
- neo4j outputs: `artifacts/neo4j_outputs/tutorial/*.json`

### B) Fast profile artifacts

Present:
- `artifacts/baseline/fast/*`
- `artifacts/inference/fast/baseline_granite.json`
- `artifacts/inference/fast/baseline_qwen.json`
- `artifacts/evaluation/fast/summary_*.json`
- `artifacts/neo4j_outputs/fast/*`

### C) Full profile current state in repository

Present:
- `data/processed/full/train.csv` (`209,766` rows)
- `data/processed/full/val.csv` (`26,221` rows)
- `data/processed/full/test.csv` (`26,221` rows)
- `data/processed/full/train_cypher.csv` (`209,765` rows)
- `data/processed/full/val_cypher.csv` (`26,221` rows)
- `data/processed/full/test_cypher.csv` (`26,221` rows)

Not present:
- `artifacts/baseline/full/*`
- `artifacts/inference/full/*`
- `artifacts/evaluation/full/*`

Interpretation:
- full profile data preparation and Cypher extension are complete,
- full profile downstream stages are not materialized in current artifacts.

### D) Choosing the right training run directory

Under `artifacts/training/` there are multiple timestamped directories.
Only some contain full output files (`train_result.json`, `eval_result.json`, `training_metadata.json`).
The tutorial real-run report uses:
- `tutorial_2026-06-20T19-14-21.185341+00-00_trl`
