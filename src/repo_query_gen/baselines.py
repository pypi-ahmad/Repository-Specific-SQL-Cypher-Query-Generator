"""Prompt-only baseline generation for SQL and Cypher tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from repo_query_gen.config import Settings, load_profile
from repo_query_gen.schema_retrieval import parse_schema_context, select_schema_context
from repo_query_gen.utils import ensure_dir, save_json


def _build_prompt(task: str, schema_context: str, question_or_context: str) -> str:
    if task == "sql":
        return (
            "You are an expert SQL generator. Use only provided schema context. "
            "Do not invent tables or columns. Return SQL only.\n\n"
            f"Schema:\n{schema_context}\n\n"
            f"Question:\n{question_or_context}\n"
        )

    return (
        "You are an expert Cypher generator for graph databases. "
        "Respect schema context and avoid hallucinated labels/relations. Return Cypher only.\n\n"
        f"Relational Schema Context:\n{schema_context}\n\n"
        f"Question:\n{question_or_context}\n"
    )


def _generate(model_name: str, prompt: str, settings: Settings) -> str:
    try:
        import ollama
    except Exception as exc:
        raise RuntimeError("ollama package is required for baseline generation") from exc

    retries = max(settings.ollama_max_retries, 0)
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            client = ollama.Client(timeout=settings.ollama_timeout_seconds)
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 256},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                raise
            logger.warning(
                "Baseline generation failed for model {} (attempt {}/{}): {}",
                model_name,
                attempt + 1,
                retries + 1,
                exc,
            )
            time.sleep(settings.ollama_retry_backoff_seconds * (attempt + 1))

    raise RuntimeError(f"Baseline generation failed unexpectedly: {last_exc}")


def run_baseline_generation(profile_name: str) -> dict[str, Path]:
    """Run prompt-only baselines for both SQL and Cypher outputs."""

    settings = Settings()
    profile = load_profile(profile_name)

    test_path = settings.processed_data_dir / profile_name / "test_cypher.csv"
    test_df = pd.read_csv(test_path)

    # Ensure complexity tags are usable list values in downstream analysis.
    if "complexity_tags" in test_df:
        test_df["complexity_tags"] = test_df["complexity_tags"].astype(str)

    n_rows = min(profile.eval_sample_size, len(test_df))
    sample_df = test_df.sample(n=n_rows, random_state=settings.seed).reset_index(drop=True)

    models = [settings.granite_model, settings.qwen_model]
    rows: list[dict] = []

    for model_name in models:
        logger.info("Running baseline model {} on {} rows", model_name, len(sample_df))
        for row in sample_df.to_dict(orient="records"):
            full_schema_context = str(row["input_schema_context"])[:4000]
            question_ctx = row["question_or_context"]
            if settings.schema_retrieval_mode == "lexical":
                retrieval = select_schema_context(
                    question=question_ctx,
                    schema_context=full_schema_context,
                    top_k_tables=settings.schema_retrieval_top_k,
                )
                schema_context = str(retrieval["selected_schema_context"])
            else:
                parsed = parse_schema_context(full_schema_context)
                retrieval = {
                    "strategy": "full",
                    "selected_schema_context": full_schema_context,
                    "selected_tables": [tbl.table_name for tbl in parsed],
                    "selected_columns": [f"{tbl.table_name}.{col}" for tbl in parsed for col in tbl.columns],
                }
                schema_context = full_schema_context

            sql_prompt = _build_prompt("sql", schema_context, question_ctx)
            cypher_prompt = _build_prompt("cypher", schema_context, question_ctx)

            sql_started = time.perf_counter()
            sql_pred = _generate(model_name, sql_prompt, settings)
            sql_latency_ms = (time.perf_counter() - sql_started) * 1000.0
            cypher_started = time.perf_counter()
            cypher_pred = _generate(model_name, cypher_prompt, settings)
            cypher_latency_ms = (time.perf_counter() - cypher_started) * 1000.0

            rows.append(
                {
                    "example_id": row["example_id"],
                    "model_name": model_name,
                    "question_or_context": question_ctx,
                    "schema_context": full_schema_context,
                    "used_schema_context": schema_context,
                    "sql_reference": row["sql"],
                    "sql_pred": sql_pred,
                    "sql_latency_ms": round(sql_latency_ms, 2),
                    "cypher_reference": row.get("cypher", ""),
                    "cypher_pred": cypher_pred,
                    "cypher_latency_ms": round(cypher_latency_ms, 2),
                    "retrieval_strategy": retrieval["strategy"],
                    "retrieval_selected_tables": json.dumps(retrieval["selected_tables"]),
                    "retrieval_selected_columns": json.dumps(retrieval["selected_columns"]),
                    "source": row["source"],
                    "complexity_tags": row.get("complexity_tags", "[]"),
                }
            )

    out_dir = ensure_dir(settings.artifacts_dir / "baseline" / profile_name)
    csv_path = out_dir / "baseline_predictions.csv"
    json_path = out_dir / "baseline_predictions.json"

    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    save_json(json_path, out_df.to_dict(orient="records"))

    logger.info("Saved baseline outputs to {}", out_dir)
    return {"baseline_csv": csv_path, "baseline_json": json_path}


__all__ = ["run_baseline_generation"]
