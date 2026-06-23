"""SQL-to-Cypher conversion and validation utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
from loguru import logger

from repo_query_gen.config import Settings
from repo_query_gen.utils import ensure_dir


def _clean_identifier(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not token:
        token = "entity"
    if token[0].isdigit():
        token = f"t_{token}"
    return token


def _sql_literal_to_cypher(value_sql: str) -> str:
    value_sql = value_sql.strip()
    if value_sql.lower() in {"true", "false", "null"}:
        return value_sql.lower()
    return value_sql


def sql_to_cypher_deterministic(sql: str) -> tuple[str, dict[str, Any]]:
    """Convert SQL query to approximate Cypher using deterministic rules.

    Args:
        sql: Input SQL string.

    Returns:
        Tuple of `(cypher_query, metadata)`.

    Notes:
        This conversion targets practical training labels and is not a perfect
        theorem-preserving SQL-to-Cypher compiler.
    """

    normalized = " ".join(sql.strip().split())
    try:
        expr = sqlglot.parse_one(normalized)
    except Exception as exc:
        fallback = f"// parse_error: {exc}\nMATCH (n) RETURN n LIMIT 5"
        return fallback, {"status": "parse_error", "error": str(exc)}

    tables = []
    table_aliases: dict[str, str] = {}
    for t in expr.find_all(sqlglot.expressions.Table):
        table_name = _clean_identifier(t.name)
        alias = _clean_identifier(t.alias_or_name)
        if table_name not in tables:
            tables.append(table_name)
        table_aliases[alias] = table_name

    if not tables:
        return "MATCH (n) RETURN n LIMIT 5", {"status": "no_table_detected"}

    match_parts: list[str] = []
    for alias, table in table_aliases.items():
        match_parts.append(f"({alias}:{table.title()})")

    # Translate simple JOIN conditions into relationships if possible.
    join_conditions = []
    for join in expr.find_all(sqlglot.expressions.Join):
        on_expr = join.args.get("on")
        if on_expr is not None:
            join_conditions.append(on_expr.sql())

    where_expr = expr.args.get("where")
    where_sql = where_expr.this.sql() if where_expr is not None else ""

    # Keep join logic in WHERE when relationship mapping is uncertain.
    all_filters = [cond for cond in join_conditions if cond]
    if where_sql:
        all_filters.append(where_sql)

    select_exprs = []
    select = expr.args.get("expressions") or []
    for s in select:
        select_exprs.append(s.sql())

    if not select_exprs:
        select_exprs = ["*"]

    order_clause = expr.args.get("order")
    limit_clause = expr.args.get("limit")
    group_clause = expr.args.get("group")

    cypher_lines = [f"MATCH {', '.join(match_parts)}"]
    if all_filters:
        rendered = " AND ".join(_sql_literal_to_cypher(f) for f in all_filters)
        cypher_lines.append(f"WHERE {rendered}")

    if group_clause is not None:
        cypher_lines.append(f"WITH {', '.join(select_exprs)}")

    cypher_lines.append(f"RETURN {', '.join(select_exprs)}")

    if order_clause is not None:
        cypher_lines.append(f"ORDER BY {order_clause.sql().replace('ORDER BY', '').strip()}")

    if limit_clause is not None:
        cypher_lines.append(f"LIMIT {limit_clause.expression.sql()}")

    cypher = "\n".join(cypher_lines)
    metadata = {
        "status": "ok",
        "tables": tables,
        "n_joins": len(join_conditions),
        "has_group": group_clause is not None,
    }
    return cypher, metadata


def validate_cypher_text(cypher: str) -> tuple[bool, list[str]]:
    """Run lightweight Cypher validity checks for offline gating."""

    issues: list[str] = []
    if "MATCH" not in cypher.upper():
        issues.append("missing_match")
    if "RETURN" not in cypher.upper():
        issues.append("missing_return")
    if cypher.count("(") != cypher.count(")"):
        issues.append("unbalanced_parentheses")
    if cypher.count("[") != cypher.count("]"):
        issues.append("unbalanced_brackets")

    return len(issues) == 0, issues


def maybe_refine_cypher_with_llm(sql: str, draft_cypher: str, settings: Settings) -> str:
    """Optionally refine deterministic conversion using a local model.

    This function is disabled by default for deterministic offline reproducibility.
    """

    if not settings.enable_cypher_refinement:
        return draft_cypher

    try:
        import ollama
    except Exception:
        logger.warning("ollama package unavailable; skipping Cypher refinement")
        return draft_cypher

    prompt = (
        "You are a strict SQL-to-Cypher translator. Improve the Cypher query without adding tables/labels not present in SQL."
        " Return only Cypher.\n\n"
        f"SQL:\n{sql}\n\n"
        f"Draft Cypher:\n{draft_cypher}\n"
    )

    try:
        response = ollama.chat(
            model=settings.qwen_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        text = response["message"]["content"].strip()
        return text or draft_cypher
    except Exception as exc:
        logger.warning("Cypher refinement failed: {}", exc)
        return draft_cypher


def build_cypher_labels_for_split(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Generate Cypher labels and quality signals for a dataframe split."""

    out = df.copy()
    out["sql"] = out["sql"].fillna("").astype(str).str.strip()
    invalid_mask = out["sql"].isin(["", "nan", "None", "none", "NaN"])
    dropped = int(invalid_mask.sum())
    if dropped:
        logger.warning("Dropping {} rows with invalid SQL before Cypher conversion", dropped)
        out = out[~invalid_mask].copy()

    cyphers: list[str] = []
    statuses: list[str] = []
    quality_scores: list[float] = []
    issues_col: list[list[str]] = []

    for row in out.to_dict(orient="records"):
        draft, meta = sql_to_cypher_deterministic(row["sql"])
        refined = maybe_refine_cypher_with_llm(row["sql"], draft, settings)
        ok, issues = validate_cypher_text(refined)

        score = 1.0
        if meta.get("status") != "ok":
            score -= 0.4
        if not ok:
            score -= 0.4
        if "join" in row.get("complexity_tags", []) and "AND" not in refined:
            score -= 0.2

        cyphers.append(refined)
        statuses.append(meta.get("status", "unknown"))
        quality_scores.append(max(0.0, score))
        issues_col.append(issues)

    out["cypher"] = cyphers
    out["cypher_status"] = statuses
    out["cypher_quality"] = quality_scores
    out["cypher_issues"] = [json.dumps(x) for x in issues_col]
    return out


def run_cypher_extension(profile_name: str) -> dict[str, Path]:
    """Run SQL-to-Cypher extension pipeline across train/val/test splits."""

    settings = Settings()
    base_dir = settings.processed_data_dir / profile_name
    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Parse list-like columns serialized via CSV.
    for df in [train_df, val_df, test_df]:
        df["complexity_tags"] = df["complexity_tags"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) and x.startswith("[") else ["simple"]
        )

    train_c = build_cypher_labels_for_split(train_df, settings)
    val_c = build_cypher_labels_for_split(val_df, settings)
    test_c = build_cypher_labels_for_split(test_df, settings)

    out_dir = ensure_dir(base_dir)
    out_paths = {
        "train_cypher_csv": out_dir / "train_cypher.csv",
        "val_cypher_csv": out_dir / "val_cypher.csv",
        "test_cypher_csv": out_dir / "test_cypher.csv",
    }
    train_c.to_csv(out_paths["train_cypher_csv"], index=False)
    val_c.to_csv(out_paths["val_cypher_csv"], index=False)
    test_c.to_csv(out_paths["test_cypher_csv"], index=False)

    logger.info("Cypher labels built for profile {}", profile_name)
    return out_paths


__all__ = [
    "sql_to_cypher_deterministic",
    "validate_cypher_text",
    "run_cypher_extension",
]
