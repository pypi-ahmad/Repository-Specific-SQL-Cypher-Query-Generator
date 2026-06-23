# Tutorial Profile Real Run Report

Date: June 21, 2026  
Project: `Repository-Specific-SQL-Cypher-Query-Generator`  
Environment: local Ubuntu, `uv` venv, Ollama, Neo4j (Docker), CUDA GPU

## 1. Final Tutorial Profile Used

Configured in `configs/profiles.yaml`:

- `dataset_rows: 20000`
- `train_rows: 15000`, `val_rows: 2500`, `test_rows: 2500`
- `max_train_steps: 10`
- `batch_size: 1`
- `gradient_accumulation_steps: 4`
- `learning_rate: 0.0002`
- `lora_r: 8`, `lora_alpha: 16`, `lora_dropout: 0.05`
- `max_seq_len: 512`
- `eval_sample_size: 3`
- `spider_eval_rows: 3`
- `train_max_examples: 256`, `val_max_examples: 64`

## 2. Execution Commands (Real Artifacts)

```bash
cd /home/ahmad/AI/Repository-Specific-SQL-Cypher-Query-Generator
source .venv/bin/activate

# 0) Neo4j service
docker compose -f docker/docker-compose.neo4j.yml up -d

# 1) Full tutorial pipeline (with training + inference + judge + spider)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MPLCONFIGDIR=/tmp/matplotlib-cache \
uv run python scripts/run_end_to_end.py \
  --profile tutorial \
  --with-inference \
  --with-judge \
  --with-spider \
  --trainer-backend auto

# 2) If only spider/neo4j need refresh after code updates
uv run python scripts/evaluate.py --profile tutorial --with-spider
uv run python scripts/run_neo4j_demo.py --profile tutorial
```

Notes:
- Pipeline code now performs best-effort `ollama stop` before training in `run_end_to_end` to reduce GPU OOM risk.
- Spider downloader now supports official Drive warning flow (`confirm` + `uuid`) and resolves both `spider/` and `spider_data/` extraction roots.
- Generation output is post-processed to strip model preamble text and keep query-only SQL/Cypher.
- Ollama calls now use configurable timeout/retry settings (`ollama_timeout_seconds`, `ollama_max_retries`) to reduce long-call timeouts.

## 3. Produced Artifacts

Training:
- `artifacts/training/tutorial_2026-06-20T19-14-21.185341+00-00_trl/`

Run manifest:
- `artifacts/manifest_tutorial_real_run.json`

Inference:
- `artifacts/inference/tutorial/baseline_granite.json`
- `artifacts/inference/tutorial/baseline_qwen.json`
- `artifacts/inference/tutorial/finetuned.json`

Evaluation:
- `artifacts/evaluation/tutorial/summary_baseline_granite.json`
- `artifacts/evaluation/tutorial/summary_baseline_qwen.json`
- `artifacts/evaluation/tutorial/summary_finetuned.json`
- `artifacts/evaluation/tutorial/judge_baseline_granite.json`
- `artifacts/evaluation/tutorial/judge_baseline_qwen.json`
- `artifacts/evaluation/tutorial/judge_finetuned.json`
- `artifacts/evaluation/tutorial/spider_exec_baseline_granite.json`
- `artifacts/evaluation/tutorial/spider_exec_finetuned.json`
- `artifacts/evaluation/tutorial/metrics_per_example.csv`
- `artifacts/evaluation/tutorial/plots/*.png`

Neo4j:
- `artifacts/neo4j_outputs/tutorial/graph_summary.json`
- `artifacts/neo4j_outputs/tutorial/demo_query_results.json`

## 4. Key Results

### 4.1 Training Metadata

From `training_metadata.json`:
- `requested_backend: auto`
- `effective_backend: trl`
- `fallback_reason: null`
- `unsloth: not_installed`

From training results:
- `train_loss: 1.7538`
- `eval_loss: 1.4786`
- `train_runtime: 87.92s`
- `eval_runtime: 33.50s`

### 4.2 Inference Summary Metrics

From `summary_*.json`:

| Label | exact_match | syntax_success | schema_grounding | retrieval_table_recall | latency_ms_p50 | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_granite | 0.0000 | 1.0000 | 0.8333 | 1.0000 | 908.830 | 14389.468 |
| baseline_qwen | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5206.345 | 20683.038 |
| finetuned | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 33922.155 | 111392.802 |

Inference row counts:
- `baseline_granite.json`: 6
- `baseline_qwen.json`: 6
- `finetuned.json`: 6

### 4.3 Judge Aggregates (n=6 each)

Computed means across both judge models and sampled rows:

| Label | correctness | completeness | schema_grounding | hallucination_risk |
|---|---:|---:|---:|---:|
| baseline_granite | 2.833 | 2.500 | 3.000 | 2.500 |
| baseline_qwen | 2.833 | 2.667 | 3.000 | 2.500 |
| finetuned | 3.000 | 2.500 | 3.000 | 2.500 |

### 4.4 Spider Execution Benchmark (n=3)

From `spider_exec_*.json`:
- baseline_granite: `execution_match = 2/3 (0.667)`, `pred_exec_success = 3/3 (1.000)`
- finetuned: `execution_match = 2/3 (0.667)`, `pred_exec_success = 3/3 (1.000)`

### 4.5 Neo4j Demo

From `graph_summary.json`:
- `sources: 15`
- `questions: 1000`
- `tables: 910`

## 5. Practical Conclusions

- End-to-end real artifacts are now reproducible for tutorial profile with local tooling.
- Training path is stable under current profile constraints (`max_seq_len=512`, low-rank LoRA, bounded examples).
- Finetuned SQL post-processing fixed prior non-SQL preamble issues and restored Spider SQL executability.
- On this sampled Spider execution check, baseline granite and finetuned both achieved `2/3` execution match.
