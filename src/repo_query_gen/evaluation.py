"""Evaluation suite for SQL/Cypher generation quality and execution."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import sqlglot
from loguru import logger
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import sentence_bleu

from repo_query_gen.config import Settings, load_profile
from repo_query_gen.cypher import validate_cypher_text
from repo_query_gen.inference import generate_with_finetuned, generate_with_ollama
from repo_query_gen.utils import ensure_dir, normalize_ws, save_json


TABLE_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def exact_match(pred: str, ref: str) -> float:
    pred = str(pred)
    ref = str(ref)
    return 1.0 if normalize_ws(pred).lower() == normalize_ws(ref).lower() else 0.0


def sql_parse_success(sql: str) -> float:
    sql = str(sql)
    try:
        sqlglot.parse_one(sql)
        return 1.0
    except Exception:
        return 0.0


def cypher_parse_success(cypher: str) -> float:
    cypher = str(cypher)
    ok, _ = validate_cypher_text(cypher)
    return 1.0 if ok else 0.0


def schema_grounding_accuracy(pred: str, schema_context: str) -> float:
    pred = str(pred)
    schema_context = str(schema_context)
    tables = []
    for line in schema_context.splitlines():
        if line.strip().lower().startswith("create table"):
            token = line.split()[2].strip("`\"(")
            tables.append(token.lower())
    if not tables:
        return 0.0
    return 1.0 if any(tbl in pred.lower() for tbl in tables) else 0.0


def _extract_sql_tables(sql_text: str) -> set[str]:
    sql_text = str(sql_text)
    try:
        tree = sqlglot.parse_one(sql_text)
        return {tbl.name.lower() for tbl in tree.find_all(sqlglot.exp.Table) if tbl.name}
    except Exception:
        # Fallback lexical extraction if parser fails.
        tokens = [tok.lower() for tok in TABLE_TOKEN_RE.findall(sql_text)]
        tables: set[str] = set()
        prev = ""
        for tok in tokens:
            if prev in {"from", "join"}:
                tables.add(tok)
            prev = tok
        return tables


def _retrieval_table_recall(reference_sql: str, retrieved_tables: list[str]) -> float | None:
    ref_tables = _extract_sql_tables(reference_sql)
    if not ref_tables:
        return None
    retrieved = {tbl.lower() for tbl in retrieved_tables}
    return float(len(ref_tables & retrieved) / len(ref_tables))


def text_metrics(pred: str, ref: str) -> dict[str, float]:
    pred = str(pred)
    ref = str(ref)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(rouge.score(ref, pred)["rougeL"].fmeasure)

    bleu = float(sentence_bleu(pred, [ref]).score / 100.0)

    try:
        meteor = float(meteor_score([ref.split()], pred.split()))
    except Exception:
        meteor = 0.0

    bert_f1 = 0.0
    if os.getenv("ENABLE_BERTSCORE", "0") == "1":
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(
                cands=[pred],
                refs=[ref],
                lang="en",
                model_type="microsoft/deberta-base-mnli",
                verbose=False,
            )
            bert_f1 = float(f1.mean().item())
        except Exception:
            bert_f1 = 0.0

    return {
        "bleu": bleu,
        "rouge_l": rouge_l,
        "meteor": meteor,
        "bertscore_f1": bert_f1,
    }


def evaluate_inference_json(inference_json: Path, label: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate one inference output JSON file."""

    rows = json.loads(inference_json.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    for row in rows:
        task = row["task"]
        pred = row["generated_query"]
        schema_context = row.get("schema_context", row.get("input_schema_context", ""))
        retrieval = row.get("retrieval", {})
        retrieved_tables = retrieval.get("selected_tables", []) if isinstance(retrieval, dict) else []
        if task == "sql":
            ref = row.get("sql_reference", "")
            syntax = sql_parse_success(pred)
        else:
            ref = row.get("cypher_reference", "")
            syntax = cypher_parse_success(pred)

        metrics = text_metrics(pred, ref)
        record = {
            "label": label,
            "example_id": row["example_id"],
            "task": task,
            "model": row["model"],
            "exact_match": exact_match(pred, ref),
            "syntax_success": syntax,
            "schema_grounding": schema_grounding_accuracy(pred, schema_context),
            "retrieval_table_recall": _retrieval_table_recall(row.get("sql_reference", ""), list(retrieved_tables)),
            "generation_latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
            **metrics,
        }
        records.append(record)

    df = pd.DataFrame(records)
    summary = {
        "exact_match": float(df["exact_match"].mean()),
        "syntax_success": float(df["syntax_success"].mean()),
        "schema_grounding": float(df["schema_grounding"].mean()),
        "bleu": float(df["bleu"].mean()),
        "rouge_l": float(df["rouge_l"].mean()),
        "meteor": float(df["meteor"].mean()),
        "bertscore_f1": float(df["bertscore_f1"].mean()),
        "retrieval_table_recall": float(df["retrieval_table_recall"].dropna().mean())
        if df["retrieval_table_recall"].notna().any()
        else 0.0,
        "latency_ms_p50": float(df["generation_latency_ms"].quantile(0.5)),
        "latency_ms_p95": float(df["generation_latency_ms"].quantile(0.95)),
    }
    return df, summary


def evaluate_baseline_csv(baseline_csv: Path, label: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate baseline CSV output without regenerating inference JSON."""

    src = pd.read_csv(baseline_csv)
    records: list[dict[str, Any]] = []

    for row in src.to_dict(orient="records"):
        try:
            retrieved_tables = json.loads(str(row.get("retrieval_selected_tables", "[]")))
            if not isinstance(retrieved_tables, list):
                retrieved_tables = []
        except Exception:
            retrieved_tables = []

        for task in ("sql", "cypher"):
            pred = row["sql_pred"] if task == "sql" else row["cypher_pred"]
            ref = row["sql_reference"] if task == "sql" else row["cypher_reference"]
            syntax = sql_parse_success(pred) if task == "sql" else cypher_parse_success(pred)
            latency_ms = row.get("sql_latency_ms", 0.0) if task == "sql" else row.get("cypher_latency_ms", 0.0)
            metrics = text_metrics(pred, ref)
            records.append(
                {
                    "label": label,
                    "example_id": row["example_id"],
                    "task": task,
                    "model": row["model_name"],
                    "exact_match": exact_match(pred, ref),
                    "syntax_success": syntax,
                    "schema_grounding": schema_grounding_accuracy(pred, row.get("schema_context", "")),
                    "retrieval_table_recall": _retrieval_table_recall(row.get("sql_reference", ""), retrieved_tables),
                    "generation_latency_ms": float(latency_ms or 0.0),
                    **metrics,
                }
            )

    df = pd.DataFrame(records)
    summary = {
        "exact_match": float(df["exact_match"].mean()),
        "syntax_success": float(df["syntax_success"].mean()),
        "schema_grounding": float(df["schema_grounding"].mean()),
        "bleu": float(df["bleu"].mean()),
        "rouge_l": float(df["rouge_l"].mean()),
        "meteor": float(df["meteor"].mean()),
        "bertscore_f1": float(df["bertscore_f1"].mean()),
        "retrieval_table_recall": float(df["retrieval_table_recall"].dropna().mean())
        if df["retrieval_table_recall"].notna().any()
        else 0.0,
        "latency_ms_p50": float(df["generation_latency_ms"].quantile(0.5)),
        "latency_ms_p95": float(df["generation_latency_ms"].quantile(0.95)),
    }
    return df, summary


def _judge_once(
    judge_model: str,
    question: str,
    reference: str,
    candidate: str,
    task: str,
    settings: Settings,
) -> dict[str, Any]:
    """LLM-as-a-judge scoring for one generated query."""

    import ollama

    prompt = (
        "You are an expert evaluator for query generation. "
        "Score from 1-5 for correctness, completeness, schema grounding, and hallucination risk. "
        "Return strict JSON keys: correctness, completeness, schema_grounding, hallucination_risk, reasoning.\n\n"
        f"Task: {task}\n"
        f"Question/Context:\n{question}\n\n"
        f"Reference:\n{reference}\n\n"
        f"Candidate:\n{candidate}\n"
    )
    retries = max(settings.ollama_max_retries, 0)
    last_exc: Exception | None = None
    response: dict[str, Any] | None = None

    for attempt in range(retries + 1):
        try:
            client = ollama.Client(timeout=settings.ollama_timeout_seconds)
            response = client.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                format="json",
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                raise
            logger.warning(
                "Judge call failed for {} (attempt {}/{}): {}",
                judge_model,
                attempt + 1,
                retries + 1,
                exc,
            )
    if response is None:
        raise RuntimeError(f"Judge call failed unexpectedly: {last_exc}")

    try:
        payload = json.loads(response["message"]["content"])
    except Exception:
        payload = {
            "correctness": 1,
            "completeness": 1,
            "schema_grounding": 1,
            "hallucination_risk": 5,
            "reasoning": "judge_parse_failure",
        }
    return payload


def run_llm_judging(
    inference_json: Path,
    output_path: Path,
    sample_size: int = 200,
) -> Path:
    """Run LLM-as-a-judge with Granite and Qwen."""

    settings = Settings()
    rows = json.loads(inference_json.read_text(encoding="utf-8"))
    np.random.seed(settings.seed)

    if sample_size < len(rows):
        idx = np.random.choice(len(rows), size=sample_size, replace=False)
        sample = [rows[i] for i in idx]
    else:
        sample = rows

    judge_models = [settings.granite_model, settings.qwen_model]
    out_rows: list[dict[str, Any]] = []

    for judge_model in judge_models:
        logger.info("Running judge model {} on {} items", judge_model, len(sample))
        for row in sample:
            task = row["task"]
            ref = row["sql_reference"] if task == "sql" else row.get("cypher_reference", "")
            try:
                payload = _judge_once(
                    judge_model=judge_model,
                    question=row.get("question", row.get("question_or_context", "")),
                    reference=ref,
                    candidate=row["generated_query"],
                    task=task,
                    settings=settings,
                )
            except Exception as exc:
                payload = {
                    "correctness": 1,
                    "completeness": 1,
                    "schema_grounding": 1,
                    "hallucination_risk": 5,
                    "reasoning": f"judge_error:{exc}",
                }
            out_rows.append(
                {
                    "judge_model": judge_model,
                    "example_id": row["example_id"],
                    "task": task,
                    **payload,
                }
            )

    save_json(output_path, out_rows)
    return output_path


def _download_spider_dataset(root: Path) -> Path:
    """Download and extract official Spider benchmark bundle if missing."""

    def _resolve_spider_root(base: Path) -> Path | None:
        candidates = [base / "spider", base / "spider_data"]
        for candidate in candidates:
            if candidate.exists() and (candidate / "database").exists() and (candidate / "dev.json").exists():
                return candidate
        for child in base.iterdir():
            if child.is_dir() and (child / "database").exists() and (child / "dev.json").exists():
                return child
        return None

    existing = _resolve_spider_root(root) if root.exists() else None
    if existing is not None:
        return existing

    ensure_dir(root)
    zip_path = root / "spider.zip"

    # Legacy URL used in older scripts.
    primary_url = "https://github.com/taoyds/spider/raw/master/spider.zip"
    # Official Spider 1.0 dataset link from https://yale-lily.github.io/spider.
    drive_file_id = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
    drive_url = "https://docs.google.com/uc?export=download"

    data: bytes | None = None
    logger.info("Downloading Spider dataset from {}", primary_url)
    try:
        response = requests.get(primary_url, timeout=120)
        response.raise_for_status()
        if response.content.startswith(b"PK"):
            data = response.content
        else:
            logger.warning("Primary Spider URL did not return a zip payload; trying official Drive link")
    except Exception as exc:
        logger.warning("Primary Spider URL failed: {}. Trying official Drive link", exc)

    if data is None:
        session = requests.Session()
        response = session.get(drive_url, params={"id": drive_file_id}, timeout=120)
        response.raise_for_status()
        token = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
        if token:
            response = session.get(drive_url, params={"id": drive_file_id, "confirm": token}, timeout=120)
            response.raise_for_status()
        elif not response.content.startswith(b"PK"):
            # Handle Drive "virus scan warning" flow that returns a form with confirm+uuid.
            action_match = re.search(r'<form[^>]+action="([^"]+)"', response.text)
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', response.text)
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', response.text)
            if confirm_match:
                action_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
                params = {
                    "id": drive_file_id,
                    "export": "download",
                    "confirm": confirm_match.group(1),
                }
                if uuid_match:
                    params["uuid"] = uuid_match.group(1)
                response = session.get(action_url, params=params, timeout=120)
                response.raise_for_status()
        if not response.content.startswith(b"PK"):
            raise RuntimeError("Spider download did not return a valid zip payload")
        data = response.content

    zip_path.write_bytes(data)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    spider_root = _resolve_spider_root(root)
    if spider_root is None:
        raise RuntimeError("Spider archive extracted but dataset root could not be resolved")
    return spider_root


def _sqlite_schema_context(db_path: Path) -> str:
    con = sqlite3.connect(db_path)
    try:
        q = "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        rows = con.execute(q).fetchall()
        return "\n".join(r[1] for r in rows if r[1])
    finally:
        con.close()


def _execute_sql(db_path: Path, query: str) -> tuple[bool, list[tuple[Any, ...]] | str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(query)
        rows = cur.fetchall()
        return True, rows
    except Exception as exc:
        return False, str(exc)
    finally:
        con.close()


def run_spider_execution_benchmark(
    mode: Literal["baseline_granite", "baseline_qwen", "finetuned"],
    output_path: Path,
    n_rows: int,
) -> Path:
    """Run SQL execution accuracy benchmark on Spider with real SQLite DBs."""

    settings = Settings()
    spider_root = _download_spider_dataset(settings.raw_data_dir)

    dev_path = spider_root / "dev.json"
    dev_rows = json.loads(dev_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(settings.seed)
    idx = rng.choice(len(dev_rows), size=min(n_rows, len(dev_rows)), replace=False)
    subset = [dev_rows[int(i)] for i in idx]

    if mode == "finetuned":
        adapters = sorted((settings.artifacts_dir / "training").glob("*/adapter"))
        if not adapters:
            raise FileNotFoundError("No fine-tuned adapter found for Spider benchmark")
        adapter = adapters[-1]
    else:
        adapter = None

    out_rows = []

    for row in subset:
        db_id = row["db_id"]
        db_path = spider_root / "database" / db_id / f"{db_id}.sqlite"
        schema_context = _sqlite_schema_context(db_path)
        question = row["question"]
        ref_sql = row["query"]

        if mode == "baseline_granite":
            pred_sql = generate_with_ollama("sql", question, schema_context, settings.granite_model)
        elif mode == "baseline_qwen":
            pred_sql = generate_with_ollama("sql", question, schema_context, settings.qwen_model)
        else:
            assert adapter is not None
            pred_sql = generate_with_finetuned(
                "sql",
                question,
                schema_context,
                adapter,
                settings.hf_granite_base_model,
            )

        ref_ok, ref_rows = _execute_sql(db_path, ref_sql)
        pred_ok, pred_rows = _execute_sql(db_path, pred_sql)

        exec_match = False
        if ref_ok and pred_ok:
            exec_match = sorted(ref_rows) == sorted(pred_rows)  # type: ignore[arg-type]

        out_rows.append(
            {
                "db_id": db_id,
                "question": question,
                "reference_sql": ref_sql,
                "predicted_sql": pred_sql,
                "reference_exec_success": ref_ok,
                "pred_exec_success": pred_ok,
                "execution_match": exec_match,
            }
        )

    save_json(output_path, out_rows)
    return output_path


def create_metric_plots(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Generate evaluation charts for report and notebook use."""

    ensure_dir(out_dir)
    paths: dict[str, Path] = {}

    summary = (
        df.groupby(["label", "task"])[
            [
                "exact_match",
                "syntax_success",
                "schema_grounding",
                "retrieval_table_recall",
                "bleu",
                "rouge_l",
                "meteor",
            ]
        ]
        .mean()
        .reset_index()
    )

    for metric in [
        "exact_match",
        "syntax_success",
        "schema_grounding",
        "retrieval_table_recall",
        "bleu",
        "rouge_l",
        "meteor",
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=summary, x="task", y=metric, hue="label")
        plt.title(f"{metric} by task")
        if metric != "retrieval_table_recall":
            plt.ylim(0, 1)
        plt.tight_layout()
        path = out_dir / f"{metric}_by_task.png"
        plt.savefig(path)
        plt.close()
        paths[metric] = path

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="task", y="generation_latency_ms", hue="label")
    plt.title("Generation latency (ms) by task")
    plt.tight_layout()
    latency_path = out_dir / "latency_ms_by_task.png"
    plt.savefig(latency_path)
    plt.close()
    paths["latency_ms"] = latency_path

    return paths


def run_evaluation_bundle(
    profile_name: str,
    run_judging: bool = False,
    run_spider: bool = False,
) -> dict[str, Path]:
    """Run the complete evaluation bundle over available inference artifacts."""

    settings = Settings()
    profile = load_profile(profile_name)

    inf_dir = settings.artifacts_dir / "inference" / profile_name
    eval_dir = ensure_dir(settings.artifacts_dir / "evaluation" / profile_name)

    artifact_map: dict[str, Path] = {}
    all_frames: list[pd.DataFrame] = []

    for label, file_name in [
        ("baseline_granite", "baseline_granite.json"),
        ("baseline_qwen", "baseline_qwen.json"),
        ("finetuned", "finetuned.json"),
    ]:
        path = inf_dir / file_name
        if path.exists():
            df, summary = evaluate_inference_json(path, label)
            if run_judging:
                judge_path = eval_dir / f"judge_{label}.json"
                run_llm_judging(path, judge_path, sample_size=min(100, profile.eval_sample_size))
                artifact_map[f"judge_{label}"] = judge_path
        elif label.startswith("baseline"):
            baseline_csv = settings.artifacts_dir / "baseline" / profile_name / "baseline_predictions.csv"
            if not baseline_csv.exists():
                logger.warning("Skipping {}: no inference JSON or baseline CSV", label)
                continue
            df_all, _ = evaluate_baseline_csv(baseline_csv, "baseline_all")
            target_model = settings.granite_model if label == "baseline_granite" else settings.qwen_model
            df = df_all[df_all["model"] == target_model].copy()
            summary = {
                "exact_match": float(df["exact_match"].mean()),
                "syntax_success": float(df["syntax_success"].mean()),
                "schema_grounding": float(df["schema_grounding"].mean()),
                "bleu": float(df["bleu"].mean()),
                "rouge_l": float(df["rouge_l"].mean()),
                "meteor": float(df["meteor"].mean()),
                "bertscore_f1": float(df["bertscore_f1"].mean()),
            }
        else:
            logger.warning("Skipping missing inference file: {}", path)
            continue

        all_frames.append(df)
        summary_path = eval_dir / f"summary_{label}.json"
        save_json(summary_path, summary)
        artifact_map[f"summary_{label}"] = summary_path

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_csv = eval_dir / "metrics_per_example.csv"
        combined.to_csv(combined_csv, index=False)
        artifact_map["metrics_per_example"] = combined_csv

        plot_paths = create_metric_plots(combined, eval_dir / "plots")
        artifact_map.update({f"plot_{k}": v for k, v in plot_paths.items()})

    if run_spider:
        spider_out = eval_dir / "spider_exec_baseline_granite.json"
        run_spider_execution_benchmark("baseline_granite", spider_out, n_rows=min(profile.spider_eval_rows, 120))
        artifact_map["spider_exec_baseline_granite"] = spider_out

        # Fine-tuned Spider execution if adapter exists.
        try:
            spider_ft = eval_dir / "spider_exec_finetuned.json"
            run_spider_execution_benchmark("finetuned", spider_ft, n_rows=min(profile.spider_eval_rows, 120))
            artifact_map["spider_exec_finetuned"] = spider_ft
        except FileNotFoundError:
            logger.warning("Fine-tuned Spider benchmark skipped (adapter missing)")

    logger.info("Evaluation bundle finished for profile {}", profile_name)
    return artifact_map


__all__ = ["run_evaluation_bundle", "run_spider_execution_benchmark", "evaluate_inference_json"]
