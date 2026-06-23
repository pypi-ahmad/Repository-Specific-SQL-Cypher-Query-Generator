# 01 - Configuration and Profiles

## What it is

Configuration is split between:
- runtime environment settings (`Settings`), and
- profile-level run scales (`fast`, `tutorial`, `full`).

## Why it is used

It lets you run the same code in different budgets:
- fast smoke checks,
- tutorial-scale reproducible demonstrations,
- full-scale expensive runs.

## How it appears in code

- Settings model: `src/repo_query_gen/config.py`
  - model names
  - Neo4j credentials
  - schema retrieval mode
  - Ollama timeout/retry controls
- Profile definitions: `configs/profiles.yaml`
- Profile loader: `load_profile(...)` in `config.py`

## Practical explanation

Key profiles from `configs/profiles.yaml`:
- `fast`: small sample with short training budget.
- `tutorial`: 20k dataset subsample with balanced speed/coverage.
- `full`: all rows (`262,208`) and high training/eval limits.

Commands:

```bash
python scripts/run_end_to_end.py --profile fast
python scripts/run_end_to_end.py --profile tutorial
python scripts/run_end_to_end.py --profile full
```

If you hit slow/timeout behavior in judge or generation stages, tune:
- `ollama_timeout_seconds`
- `ollama_max_retries`
- `ollama_retry_backoff_seconds`

These are defined in `Settings` and used in baseline/inference/evaluation calls.
