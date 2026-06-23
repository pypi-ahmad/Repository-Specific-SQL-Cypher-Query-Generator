# Repository-Specific SQL & Cypher Query Generator

A local, tutorial-first system for schema-aware Text-to-SQL and Text-to-Cypher generation with:
- dataset preparation and schema parsing,
- deterministic SQL-to-Cypher label extension,
- prompt-only baselines,
- QLoRA fine-tuning,
- evaluation (string, syntax, retrieval, judge, execution),
- Neo4j graph loading and demo queries.

## Verified Project Status (Real Artifacts)

All values below are read from existing files in this repository.

### Processed data status

| Profile | train.csv | val.csv | test.csv | train_cypher.csv | val_cypher.csv | test_cypher.csv |
|---|---:|---:|---:|---:|---:|---:|
| fast | 9,000 | 1,200 | 1,200 | 9,000 | 1,200 | 1,200 |
| tutorial | 15,000 | 2,000 | 2,000 | 15,000 | 2,000 | 2,000 |
| full | 209,766 | 26,221 | 26,221 | 209,765 | 26,221 | 26,221 |

Notes:
- Full profile Cypher train split has one fewer row (`209,765`) than SQL train split (`209,766`) because invalid SQL rows are dropped before conversion in `src/repo_query_gen/cypher.py`.
- Total prepared SQL rows across full profile = `262,208`.
- These counts come from current CSV artifacts and may differ from profile YAML defaults if artifacts were produced in earlier runs.

### Tutorial profile end-to-end run snapshot

Source files:
- `artifacts/manifest_tutorial_real_run.json`
- `artifacts/evaluation/tutorial/summary_*.json`
- `artifacts/evaluation/tutorial/spider_exec_*.json`
- `artifacts/training/tutorial_2026-06-20T19-14-21.185341+00-00_trl/*`
- `artifacts/neo4j_outputs/tutorial/graph_summary.json`

Training (`tutorial_2026-06-20T19-14-21.185341+00-00_trl`):
- backend: requested `auto`, effective `trl`
- train loss: `1.7538`
- eval loss: `1.4786`
- train runtime: `87.9225s`
- eval runtime: `33.5015s`

Evaluation summaries:

| Label | exact_match | syntax_success | schema_grounding | retrieval_table_recall | latency_ms_p50 | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_granite | 0.0000 | 1.0000 | 0.8333 | 1.0000 | 908.83 | 14,389.47 |
| baseline_qwen | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5,206.34 | 20,683.04 |
| finetuned | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 33,922.16 | 111,392.80 |

Spider execution benchmark (`n=3`):
- baseline_granite: execution match `2/3` (`0.667`), predicted SQL executable `3/3`
- finetuned: execution match `2/3` (`0.667`), predicted SQL executable `3/3`

Neo4j tutorial graph summary:
- sources: `15`
- questions: `1000`
- tables: `910`

## Project Layout

```text
configs/                 profile configs (fast/tutorial/full)
data/raw/                downloaded/cached raw data
data/processed/{profile}/
  train.csv, val.csv, test.csv
  train_cypher.csv, val_cypher.csv, test_cypher.csv
scripts/                 CLI entrypoints per stage
src/repo_query_gen/      implementation modules
artifacts/               baselines, training, inference, evaluation, neo4j outputs
notebooks/               tutorial notebooks
docs/                    handbook and tutorial documentation
```

## Quick Start

### 1) Environment

```bash
cd /home/ahmad/AI/Repository-Specific-SQL-Cypher-Query-Generator
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12.10
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

### 2) Services and models

```bash
docker compose -f docker/docker-compose.neo4j.yml up -d
ollama pull granite4.1:3b
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

### 3) Pipeline commands

Fast profile:

```bash
source .venv/bin/activate
python scripts/run_end_to_end.py --profile fast --trainer-backend auto
```

Tutorial profile (the complete measured run path):

```bash
source .venv/bin/activate
python scripts/run_end_to_end.py \
  --profile tutorial \
  --with-inference \
  --with-judge \
  --with-spider \
  --trainer-backend auto
```

Full profile (long run):

```bash
source .venv/bin/activate
python scripts/run_end_to_end.py --profile full
```

## Pipeline Stages and Code Map

1. Data preparation
- Script: `scripts/prepare_data.py`
- Module: `src/repo_query_gen/data_prep.py`

2. SQL-to-Cypher label extension
- Script: `scripts/build_cypher_labels.py`
- Module: `src/repo_query_gen/cypher.py`

3. Prompt-only baselines
- Script: `scripts/run_baselines.py`
- Module: `src/repo_query_gen/baselines.py`

4. QLoRA fine-tuning
- Script: `scripts/train_qlora.py`
- Module: `src/repo_query_gen/training.py`

5. Batch inference
- Script: `scripts/infer.py`
- Module: `src/repo_query_gen/inference.py`

6. Evaluation + judging + Spider execution
- Script: `scripts/evaluate.py`
- Module: `src/repo_query_gen/evaluation.py`

7. Neo4j demo graph
- Script: `scripts/run_neo4j_demo.py`
- Module: `src/repo_query_gen/neo4j_demo.py`

8. End-to-end orchestration
- Script: `scripts/run_end_to_end.py`
- Module: `src/repo_query_gen/pipeline.py`

## Documentation

- Main handbook: `docs/documentation.md`
- PDF handbook: `docs/documentation.pdf`
- Tutorial index: `docs/tutorials/README.md`
- Real measured run details: `docs/tutorial_real_run.md`

## Reproducibility Notes

- Python version pin: `.python-version` (`3.12.10`)
- Package lock: `uv.lock`
- Config profiles: `configs/profiles.yaml`
- Runtime settings model: `src/repo_query_gen/config.py`
- Seed setup: `src/repo_query_gen/utils.py` (`set_global_seed`)
