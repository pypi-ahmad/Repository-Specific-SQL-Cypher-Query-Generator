# 03 - SQL-to-Cypher Labeling

## What it is

A deterministic conversion stage that generates Cypher supervision from SQL rows, with lightweight validation and quality scoring.

## Why it is used

The dataset is SQL-native. To train/evaluate Cypher generation, the project builds aligned Cypher labels without relying on opaque one-shot model outputs.

## How it appears in code

- Main module: `src/repo_query_gen/cypher.py`
- Entry script: `scripts/build_cypher_labels.py`

Core functions:
- `sql_to_cypher_deterministic(...)`
- `validate_cypher_text(...)`
- `build_cypher_labels_for_split(...)`
- `run_cypher_extension(...)`

Key behavior:
- SQL is parsed with `sqlglot`.
- Query components are mapped into `MATCH/WHERE/RETURN` structure.
- Invalid SQL tokens are dropped before conversion.
- Optional model refinement exists (`enable_cypher_refinement`) but is off by default.

## Practical explanation

Run:

```bash
python scripts/build_cypher_labels.py --profile tutorial
```

Outputs:
- `train_cypher.csv`
- `val_cypher.csv`
- `test_cypher.csv`

Real row counts in this repository:

| Profile | train_cypher | val_cypher | test_cypher |
|---|---:|---:|---:|
| fast | 9,000 | 1,200 | 1,200 |
| tutorial | 15,000 | 2,000 | 2,000 |
| full | 209,765 | 26,221 | 26,221 |

Note: full train lost one row vs `train.csv` because `build_cypher_labels_for_split` drops invalid SQL values before conversion.
