"""Inference pipeline for schema-aware SQL and Cypher generation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from repo_query_gen.config import Settings, load_profile
from repo_query_gen.cypher import validate_cypher_text
from repo_query_gen.schema_retrieval import select_schema_context
from repo_query_gen.utils import normalize_ws, save_json

TASK = Literal["sql", "cypher"]
SQL_START_RE = re.compile(
    r"\b(with|select|insert|update|delete|create|drop|alter)\b",
    re.IGNORECASE,
)
CYPHER_START_RE = re.compile(
    r"\b(optional\s+match|match|with|call|unwind|merge|create|return|set|delete)\b",
    re.IGNORECASE,
)


def extract_schema_refs(schema_context: str) -> dict[str, list[str]]:
    """Extract table and column names from CREATE TABLE schema context."""

    table_pat = re.compile(r"CREATE TABLE\s+([^\s(]+)", re.IGNORECASE)
    col_pat = re.compile(r"\n\s*([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z]", re.IGNORECASE)

    tables = [m.group(1).strip("`\"") for m in table_pat.finditer(schema_context)]
    columns = [m.group(1).strip("`\"") for m in col_pat.finditer(schema_context)]

    return {"tables": sorted(set(tables)), "columns": sorted(set(columns))}


def _build_prompt(task: TASK, question: str, schema_context: str) -> str:
    prompt_task = "SQL" if task == "sql" else "Cypher"
    return (
        f"You are a schema-aware {prompt_task} assistant. "
        "Use only the provided schema. Do not hallucinate schema objects.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question:\n{question}\n\n"
        f"Return only {prompt_task}."
    )


def _select_schema_for_prompt(question: str, schema_context: str, settings: Settings) -> dict[str, object]:
    """Select schema context for prompt using configured retrieval mode."""

    if settings.schema_retrieval_mode == "full":
        refs = extract_schema_refs(schema_context)
        return {
            "strategy": "full",
            "selected_schema_context": schema_context,
            "selected_tables": refs["tables"],
            "selected_columns": refs["columns"],
            "all_tables": refs["tables"],
        }

    return select_schema_context(
        question=question,
        schema_context=schema_context,
        top_k_tables=settings.schema_retrieval_top_k,
    )


def _strip_fenced_block(text: str) -> str:
    """Extract content from first markdown fenced code block when present."""

    text = str(text)
    fence = re.search(r"```(?:sql|cypher)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _extract_sql_statement(text: str) -> str:
    """Extract the first parseable SQL statement from free-form model text."""

    cleaned = _strip_fenced_block(text)
    starts = [m.start() for m in SQL_START_RE.finditer(cleaned)]
    if 0 not in starts:
        starts = [0, *starts]

    for start in starts:
        candidate = cleaned[start:].strip()
        if not candidate:
            continue

        semicolons = [idx for idx, ch in enumerate(candidate) if ch == ";"]
        chunks = [candidate[: idx + 1].strip() for idx in semicolons] + [candidate]
        for chunk in chunks:
            if not chunk:
                continue
            try:
                import sqlglot

                sqlglot.parse_one(chunk)
                return chunk.rstrip(";").strip()
            except Exception:
                continue

    match = SQL_START_RE.search(cleaned)
    if match:
        return cleaned[match.start() :].strip().rstrip(";")
    return cleaned.strip()


def _extract_cypher_statement(text: str) -> str:
    """Extract likely Cypher query from free-form model text."""

    cleaned = _strip_fenced_block(text)
    cleaned = re.sub(r"^\s*cypher\s*:?\s*", "", cleaned, flags=re.IGNORECASE)

    match = CYPHER_START_RE.search(cleaned)
    if match:
        cleaned = cleaned[match.start() :]

    tail = re.search(r"\n\s*(?:explanation|reasoning|note)\s*[:\-]", cleaned, flags=re.IGNORECASE)
    if tail:
        cleaned = cleaned[: tail.start()]

    return cleaned.strip()


def postprocess_generated_query(task: TASK, text: str) -> str:
    """Normalize model output to query-only text for SQL/Cypher tasks."""

    if task == "sql":
        return _extract_sql_statement(text)
    return _extract_cypher_statement(text)


def generate_with_ollama(task: TASK, question: str, schema_context: str, model_name: str) -> str:
    """Generate SQL/Cypher using local Ollama model."""

    import ollama

    settings = Settings()
    prompt = _build_prompt(task, question, schema_context)
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
            return postprocess_generated_query(task, response["message"]["content"])
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                raise
            backoff_seconds = settings.ollama_retry_backoff_seconds * (attempt + 1)
            logger.warning(
                "Ollama generation failed for model {} (attempt {}/{}): {}. Retrying in {:.1f}s",
                model_name,
                attempt + 1,
                retries + 1,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)

    raise RuntimeError(f"Ollama generation failed unexpectedly: {last_exc}")


def load_finetuned_model(adapter_dir: Path, base_model: str):
    """Load PEFT adapter over base model for local inference."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return tokenizer, model


def generate_with_finetuned(
    task: TASK,
    question: str,
    schema_context: str,
    adapter_dir: Path,
    base_model: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate SQL/Cypher using fine-tuned adapter model."""

    import torch

    tokenizer, model = load_finetuned_model(adapter_dir, base_model)
    prompt = _build_prompt(task, question, schema_context)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return postprocess_generated_query(task, text)


def validate_generated_query(task: TASK, query: str, schema_context: str) -> dict:
    """Validate generated query for syntax and schema grounding."""

    schema = extract_schema_refs(schema_context)
    query_norm = query.lower()
    issues: list[str] = []

    if task == "sql":
        try:
            import sqlglot

            sqlglot.parse_one(query)
            parse_ok = True
        except Exception as exc:
            parse_ok = False
            issues.append(f"sql_parse_error:{exc}")
    else:
        parse_ok, cypher_issues = validate_cypher_text(query)
        issues.extend(cypher_issues)

    schema_ok = True
    for table in schema["tables"]:
        if table.lower() in query_norm:
            continue
    # Lightweight hallucination check: ensure at least one known table appears.
    if schema["tables"] and not any(tbl.lower() in query_norm for tbl in schema["tables"]):
        schema_ok = False
        issues.append("no_known_table_used")

    return {
        "parse_success": parse_ok,
        "schema_grounded": schema_ok,
        "issues": issues,
    }


def infer_single(
    question: str,
    schema_context: str,
    model_mode: Literal["ollama", "finetuned"],
    task: TASK,
    model_name: str | None = None,
    adapter_dir: Path | None = None,
) -> dict:
    """Run one query generation request end-to-end."""

    settings = Settings()
    retrieval = _select_schema_for_prompt(question, schema_context, settings)
    prompt_schema_context = str(retrieval["selected_schema_context"])

    started = time.perf_counter()
    if model_mode == "ollama":
        if not model_name:
            model_name = settings.granite_model
        generated = generate_with_ollama(task, question, prompt_schema_context, model_name)
        used_model = model_name
    else:
        if adapter_dir is None:
            raise ValueError("adapter_dir is required when model_mode='finetuned'")
        generated = generate_with_finetuned(
            task,
            question,
            prompt_schema_context,
            adapter_dir,
            settings.hf_granite_base_model,
        )
        used_model = f"finetuned::{adapter_dir.name}"
    latency_ms = (time.perf_counter() - started) * 1000.0

    generated = postprocess_generated_query(task, generated)
    validation = validate_generated_query(task, generated, schema_context)
    schema_refs = extract_schema_refs(schema_context)

    explanation = (
        f"Generated {task.upper()} with model {used_model}. "
        f"Referenced {len(schema_refs['tables'])} candidate tables and {len(schema_refs['columns'])} columns from schema context."
    )

    return {
        "task": task,
        "model": used_model,
        "question": question,
        "generated_query": normalize_ws(generated),
        "latency_ms": round(latency_ms, 2),
        "schema_context": schema_context,
        "used_schema_context": prompt_schema_context,
        "retrieval": retrieval,
        "schema_references": schema_refs,
        "validation": validation,
        "explanation": explanation,
    }


def run_batch_inference(profile_name: str, output_name: str, mode: Literal["baseline_granite", "baseline_qwen", "finetuned"]) -> Path:
    """Run batch inference on processed test split and save JSONL results."""

    settings = Settings()
    profile = load_profile(profile_name)
    df = pd.read_csv(settings.processed_data_dir / profile_name / "test_cypher.csv")
    n_rows = min(profile.eval_sample_size, len(df))
    df = df.sample(n=n_rows, random_state=settings.seed).reset_index(drop=True)

    if mode == "baseline_granite":
        model_mode = "ollama"
        model_name = settings.granite_model
        adapter_dir = None
    elif mode == "baseline_qwen":
        model_mode = "ollama"
        model_name = settings.qwen_model
        adapter_dir = None
    else:
        model_mode = "finetuned"
        # Latest adapter run is used by convention.
        train_runs = sorted((settings.artifacts_dir / "training").glob("*/adapter"))
        if not train_runs:
            raise FileNotFoundError("No fine-tuned adapter found. Run training first.")
        adapter_dir = train_runs[-1]
        model_name = None

    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        question = row["question_or_context"]
        schema_context = row["input_schema_context"]

        for task in ("sql", "cypher"):
            out = infer_single(
                question=question,
                schema_context=schema_context,
                model_mode=model_mode,  # type: ignore[arg-type]
                task=task,  # type: ignore[arg-type]
                model_name=model_name,
                adapter_dir=adapter_dir,
            )
            out["example_id"] = row["example_id"]
            out["source"] = row["source"]
            out["sql_reference"] = row["sql"]
            out["cypher_reference"] = row.get("cypher", "")
            out["question_or_context"] = question
            records.append(out)

    out_dir = settings.artifacts_dir / "inference" / profile_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name}.json"
    save_json(out_path, records)
    logger.info("Saved batch inference to {}", out_path)
    return out_path


__all__ = [
    "infer_single",
    "run_batch_inference",
    "extract_schema_refs",
    "validate_generated_query",
    "postprocess_generated_query",
]
