# 07 - Evaluation and Judging

## What it is

A combined evaluation stack that computes text metrics, syntax metrics, schema grounding, retrieval recall, latency stats, optional LLM-as-a-judge outputs, and Spider execution benchmarks.

## Why it is used

String similarity alone cannot validate query quality. This stage provides multiple views:
- exactness,
- parseability,
- schema use,
- runtime behavior on real databases.

## How it appears in code

- Module: `src/repo_query_gen/evaluation.py`
- Script: `scripts/evaluate.py`

Main functions:
- `evaluate_inference_json(...)`
- `evaluate_baseline_csv(...)`
- `run_llm_judging(...)`
- `run_spider_execution_benchmark(...)`
- `run_evaluation_bundle(...)`
- `create_metric_plots(...)`

Metric helpers:
- `exact_match(...)`
- `sql_parse_success(...)`
- `cypher_parse_success(...)`
- `schema_grounding_accuracy(...)`
- `_retrieval_table_recall(...)`
- `text_metrics(...)` (BLEU, ROUGE-L, METEOR, optional BERTScore)

## Practical explanation

Run:

```bash
python scripts/evaluate.py --profile tutorial --with-judge --with-spider
```

Tutorial summary metrics from repository artifacts:

| Label | exact_match | syntax_success | schema_grounding | retrieval_table_recall | latency_p50_ms | latency_p95_ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline_granite | 0.0000 | 1.0000 | 0.8333 | 1.0000 | 908.83 | 14389.47 |
| baseline_qwen | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 5206.34 | 20683.04 |
| finetuned | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 33922.16 | 111392.80 |

Spider execution files:
- `artifacts/evaluation/tutorial/spider_exec_baseline_granite.json`
- `artifacts/evaluation/tutorial/spider_exec_finetuned.json`

Both have:
- rows: `3`
- execution match: `2`
- predicted SQL execution success: `3`

Judge output caveat:
- each tutorial judge file has 6 rows, with 3 timeout fallback rows (`reasoning` starts with `judge_error`).
