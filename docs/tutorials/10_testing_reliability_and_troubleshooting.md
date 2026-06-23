# 10 - Testing, Reliability, and Troubleshooting

## What it is

A set of tests and operational checks that protect core pipeline behavior and explain how to handle common runtime issues.

## Why it is used

Without guardrails, pipeline changes can silently break retrieval, conversion, or inference parsing.

## How it appears in code

Test files:
- `tests/test_data_prep.py`
- `tests/test_cypher.py`
- `tests/test_schema_retrieval.py`
- `tests/test_inference_validation.py`
- `tests/test_training_backend.py`

Operationally important modules:
- retries/timeouts: `baselines.py`, `inference.py`, `evaluation.py`
- backend fallback logic: `training.py`
- model unload before training: `_unload_ollama_models()` in `pipeline.py`

## Practical explanation

Run tests:

```bash
uv run pytest -q
```

Common issues and fixes:

1. CUDA OOM during training
- Cause: active Ollama workers hold VRAM.
- Code-side mitigation: `pipeline._unload_ollama_models()` runs `ollama stop` before training.

2. Long Ollama calls timing out
- Increase settings in `config.py`:
  - `ollama_timeout_seconds`
  - `ollama_max_retries`
  - `ollama_retry_backoff_seconds`

3. Spider dataset download differences
- `evaluation.py` handles legacy URL fallback and Google Drive confirmation-token flow.
- Also resolves extracted roots (`spider/` or `spider_data/`).

4. No fine-tuned adapter for finetuned inference/eval
- Ensure a training run produced an `adapter` directory under `artifacts/training/<run>/adapter`.

Reliability recommendation:
- treat `manifest` + JSON artifacts as canonical run evidence,
- avoid reporting metrics from directories that do not contain full result files.
