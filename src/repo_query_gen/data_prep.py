"""Dataset exploration and preparation pipeline for SQL/Cypher training."""

from __future__ import annotations

import re
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from repo_query_gen.config import ProfileConfig, Settings
from repo_query_gen.types import QueryExample, SchemaColumn, TableSchema, TrainingSample
from repo_query_gen.utils import ensure_dir, normalize_ws, save_json

CREATE_TABLE_RE = re.compile(r"CREATE TABLE\s+([^\s(]+)\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)
COMPLEXITY_PATTERNS = {
    "join": re.compile(r"\bjoin\b", re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b", re.IGNORECASE),
    "having": re.compile(r"\bhaving\b", re.IGNORECASE),
    "nested": re.compile(r"\(\s*select\b", re.IGNORECASE),
    "limit": re.compile(r"\blimit\b", re.IGNORECASE),
    "union": re.compile(r"\bunion\b", re.IGNORECASE),
}


def _safe_text(value: object, default: str = "") -> str:
    """Convert nullable values to clean text without leaking NaN literals."""

    if value is None:
        return default
    # pandas NaN is not equal to itself
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return default
    return text


def _parse_create_table_blocks(input_text: str) -> list[TableSchema]:
    """Parse CREATE TABLE declarations from dataset input text.

    Args:
        input_text: The `input` field from the source dataset.

    Returns:
        Parsed table schema list.
    """

    tables: list[TableSchema] = []
    for match in CREATE_TABLE_RE.finditer(input_text):
        table_name = match.group(1).strip().strip("`\"")
        column_blob = match.group(2)
        columns: list[SchemaColumn] = []
        for line in column_blob.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.lower().startswith(("primary key", "foreign key", "constraint")):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) != 2:
                continue
            col_name = parts[0].strip("`\"")
            col_type = parts[1].split()[0]
            columns.append(SchemaColumn(name=col_name, dtype=col_type))

        tables.append(TableSchema(table_name=table_name, columns=columns))

    return tables


def _extract_complexity_tags(sql: str) -> list[str]:
    tags = [name for name, pattern in COMPLEXITY_PATTERNS.items() if pattern.search(sql)]
    if not tags:
        tags.append("simple")
    return tags


def load_raw_clinton_dataset(settings: Settings, profile: ProfileConfig) -> pd.DataFrame:
    """Load the primary dataset and optionally downsample for the fast profile."""

    logger.info("Loading dataset {}", settings.dataset_name)
    cache_dir = ensure_dir(settings.raw_data_dir / "hf_cache")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    ds = load_dataset(settings.dataset_name, split="train", cache_dir=str(cache_dir))
    df = ds.to_pandas()

    if profile.dataset_rows is not None and profile.dataset_rows < len(df):
        logger.info("Subsampling dataset for profile: {} rows", profile.dataset_rows)
        df = df.sample(n=profile.dataset_rows, random_state=settings.seed).reset_index(drop=True)

    return df


def build_processed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build schema-aware processed dataframe from raw source rows.

    Args:
        df: Raw dataset dataframe.

    Returns:
        Processed dataframe with parsed schema, tags, and canonical text.
    """

    records: list[dict] = []

    for idx, row in df.iterrows():
        sql = normalize_ws(_safe_text(row.get("response", ""), default=""))
        parsed_tables = _parse_create_table_blocks(_safe_text(row.get("input", ""), default=""))
        table_names = [t.table_name for t in parsed_tables]
        columns = [f"{t.table_name}.{c.name}" for t in parsed_tables for c in t.columns]

        source = _safe_text(row.get("source", "unknown"), default="unknown")
        if not source:
            source = "unknown"
        complexity_tags = _extract_complexity_tags(sql)
        complexity_bucket = "complex" if len(complexity_tags) >= 2 else complexity_tags[0]

        records.append(
            {
                "example_id": f"clinton_{idx}",
                "source": source,
                "instruction": _safe_text(row.get("instruction", "Generate SQL query."), default="Generate SQL query."),
                "question_or_context": _safe_text(row.get("text", ""), default=""),
                "input_schema_context": _safe_text(row.get("input", ""), default=""),
                "sql": sql,
                "tables": table_names,
                "columns": columns,
                "schema_json": [t.model_dump() for t in parsed_tables],
                "complexity_tags": complexity_tags,
                "complexity_bucket": complexity_bucket,
            }
        )

    return pd.DataFrame.from_records(records)


def stratified_split(
    df: pd.DataFrame,
    settings: Settings,
    profile: ProfileConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create source-stratified and complexity-aware train/val/test splits."""

    split_key = df["source"].astype(str) + "|" + df["complexity_bucket"].astype(str)

    # Collapse tiny strata to avoid split failures.
    counts = Counter(split_key)
    safe_key = split_key.map(lambda k: k if counts[k] >= 3 else "other|other")

    train_df, tmp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=settings.seed,
        stratify=safe_key,
    )

    tmp_key = (
        tmp_df["source"].astype(str) + "|" + tmp_df["complexity_bucket"].astype(str)
    ).map(lambda k: k if Counter(tmp_df["source"].astype(str) + "|" + tmp_df["complexity_bucket"].astype(str))[k] >= 2 else "other|other")

    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=settings.seed,
        stratify=tmp_key,
    )

    # Optional profile caps while preserving deterministic behavior.
    if profile.train_rows:
        train_df = train_df.sample(min(profile.train_rows, len(train_df)), random_state=settings.seed)
    if profile.val_rows:
        val_df = val_df.sample(min(profile.val_rows, len(val_df)), random_state=settings.seed)
    if profile.test_rows:
        test_df = test_df.sample(min(profile.test_rows, len(test_df)), random_state=settings.seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def to_query_examples(rows: Iterable[dict]) -> list[QueryExample]:
    """Convert dataframe rows to typed QueryExample objects."""

    return [
        QueryExample(
            example_id=row["example_id"],
            source=row["source"],
            instruction=row["instruction"],
            question_or_context=row["question_or_context"],
            sql=row["sql"],
            tables=row["tables"],
            columns=row["columns"],
            complexity_tags=row["complexity_tags"],
        )
        for row in rows
    ]


def build_instruction_samples(df: pd.DataFrame, task: str, cypher_column: str = "cypher") -> list[TrainingSample]:
    """Create instruction-response pairs for SQL or Cypher."""

    samples: list[TrainingSample] = []
    for row in df.to_dict(orient="records"):
        schema_context = row["input_schema_context"]
        if task == "sql":
            prompt = (
                "Generate SQL for this question with strict schema grounding.\n\n"
                f"Schema and context:\n{schema_context}\n\n"
                "Return SQL only."
            )
            response = row["sql"]
        else:
            prompt = (
                "Generate Cypher for this question with strict schema grounding.\n\n"
                f"Relational schema context:\n{schema_context}\n\n"
                "Return Cypher only."
            )
            response = row.get(cypher_column, "")

        samples.append(
            TrainingSample(
                task=task,
                prompt=prompt,
                response=response,
                example_id=row["example_id"],
            )
        )

    return samples


def save_processed_splits(
    settings: Settings,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    profile_name: str,
) -> dict[str, Path]:
    """Persist split CSV and JSON artifacts for reproducibility."""

    out_dir = ensure_dir(settings.processed_data_dir / profile_name)
    paths = {
        "train_csv": out_dir / "train.csv",
        "val_csv": out_dir / "val.csv",
        "test_csv": out_dir / "test.csv",
        "manifest_json": out_dir / "manifest.json",
    }

    for frame in (train_df, val_df, test_df):
        for col in ("instruction", "question_or_context", "input_schema_context", "sql", "source"):
            if col in frame.columns:
                frame[col] = frame[col].map(lambda v: _safe_text(v, default=""))

    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)

    manifest = {
        "profile": profile_name,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "sources_train": train_df["source"].value_counts().to_dict(),
        "sources_val": val_df["source"].value_counts().to_dict(),
        "sources_test": test_df["source"].value_counts().to_dict(),
    }
    save_json(paths["manifest_json"], manifest)
    return paths


def run_data_preparation(profile_name: str) -> dict[str, Path]:
    """Execute full data preparation pipeline.

    Args:
        profile_name: Execution profile (`fast`, `tutorial`, or `full`).

    Returns:
        Path mapping for generated artifacts.

    Example:
        >>> _ = run_data_preparation("fast")
    """

    settings = Settings()
    from repo_query_gen.config import load_profile

    profile = load_profile(profile_name)

    df = load_raw_clinton_dataset(settings, profile)
    processed_df = build_processed_dataframe(df)
    train_df, val_df, test_df = stratified_split(processed_df, settings, profile)

    paths = save_processed_splits(settings, train_df, val_df, test_df, profile_name)
    logger.info("Saved processed splits to {}", settings.processed_data_dir / profile_name)
    return paths


__all__ = [
    "run_data_preparation",
    "build_processed_dataframe",
    "build_instruction_samples",
    "to_query_examples",
]
