# 02 - Data Preparation Pipeline

## What it is

A schema-aware preprocessing stage that transforms raw dataset rows into structured train/val/test splits with metadata needed for later training and evaluation.

## Why it is used

Raw instruction datasets are not directly safe for training:
- they need normalized SQL text,
- parsed schema context,
- reproducible splits,
- complexity/source metadata for analysis.

## How it appears in code

- Main module: `src/repo_query_gen/data_prep.py`
- Entry script: `scripts/prepare_data.py`

Core functions:
- `load_raw_clinton_dataset(...)`
- `build_processed_dataframe(...)`
- `_parse_create_table_blocks(...)`
- `_extract_complexity_tags(...)`
- `stratified_split(...)`
- `save_processed_splits(...)`

Important implementation detail:
- `_safe_text(...)` prevents null/NaN leakage into text fields.

## Practical explanation

Run:

```bash
python scripts/prepare_data.py --profile tutorial
```

Outputs under `data/processed/<profile>/`:
- `train.csv`
- `val.csv`
- `test.csv`
- `manifest.json`

Real processed row counts in this repository:

| Profile | train | val | test |
|---|---:|---:|---:|
| fast | 9,000 | 1,200 | 1,200 |
| tutorial | 15,000 | 2,000 | 2,000 |
| full | 209,766 | 26,221 | 26,221 |

Total full rows (`train+val+test`) = `262,208`.

The split strategy is source+complexity stratification with handling for tiny strata to reduce split failures.
