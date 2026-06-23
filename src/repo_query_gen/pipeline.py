"""Orchestration pipeline for end-to-end project execution."""

from __future__ import annotations

import subprocess
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from repo_query_gen.baselines import run_baseline_generation
from repo_query_gen.config import Settings
from repo_query_gen.cypher import run_cypher_extension
from repo_query_gen.data_prep import run_data_preparation
from repo_query_gen.evaluation import run_evaluation_bundle
from repo_query_gen.inference import run_batch_inference
from repo_query_gen.neo4j_demo import run_neo4j_demo
from repo_query_gen.training import run_finetuning
from repo_query_gen.utils import save_json, utc_now_iso


class PipelineFailure(Exception):
    """Raised when a required stage fails."""


def _unload_ollama_models() -> None:
    """Best-effort unload of active Ollama model workers before training."""

    settings = Settings()
    for model_name in (settings.granite_model, settings.qwen_model, settings.embed_model):
        try:
            result = subprocess.run(
                ["ollama", "stop", model_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Unloaded Ollama model before training: {}", model_name)
            else:
                logger.debug(
                    "Ollama stop returned non-zero for {}: {}",
                    model_name,
                    (result.stderr or result.stdout).strip() or f"exit={result.returncode}",
                )
        except FileNotFoundError:
            logger.warning("Ollama CLI not found; skipping model unload before training")
            break



def run_end_to_end(
    profile_name: str,
    include_training: bool = True,
    include_neo4j: bool = True,
    include_inference: bool = False,
    run_judging: bool = False,
    run_spider_eval: bool = False,
    training_backend: str = "auto",
    allow_backend_fallback: bool = True,
) -> dict[str, Any]:
    """Run full end-to-end pipeline and return artifact manifest.

    Args:
        profile_name: Execution profile (`fast`, `tutorial`, or `full`).
        include_training: Whether to run QLoRA training stage.
        include_neo4j: Whether to run Neo4j demonstration stage.
        training_backend: Training backend (`auto`, `hf`, `trl`, `unsloth`).
        allow_backend_fallback: Whether training backend fallback is allowed.

    Returns:
        Stage-to-artifact mapping.

    Example:
        >>> manifest = run_end_to_end("fast", include_training=False, include_neo4j=False)
        >>> "data_preparation" in manifest
        True
    """

    manifest: dict[str, Any] = {
        "profile": profile_name,
        "started_at": utc_now_iso(),
        "stages": {},
    }

    try:
        manifest["stages"]["data_preparation"] = {k: str(v) for k, v in run_data_preparation(profile_name).items()}
        manifest["stages"]["cypher_extension"] = {k: str(v) for k, v in run_cypher_extension(profile_name).items()}

        # Baseline prompt-only runs before fine-tuning.
        manifest["stages"]["baselines"] = {k: str(v) for k, v in run_baseline_generation(profile_name).items()}
        if include_inference:
            manifest["stages"]["inference_baseline_granite"] = str(
                run_batch_inference(profile_name, "baseline_granite", "baseline_granite")
            )
            manifest["stages"]["inference_baseline_qwen"] = str(
                run_batch_inference(profile_name, "baseline_qwen", "baseline_qwen")
            )

        if include_training:
            _unload_ollama_models()
            manifest["stages"]["training"] = {
                k: str(v)
                for k, v in run_finetuning(
                    profile_name,
                    backend=training_backend,  # type: ignore[arg-type]
                    allow_fallback=allow_backend_fallback,
                ).items()
            }
            if include_inference:
                manifest["stages"]["inference_finetuned"] = str(
                    run_batch_inference(profile_name, "finetuned", "finetuned")
                )

        manifest["stages"]["evaluation"] = {
            k: str(v)
            for k, v in run_evaluation_bundle(
                profile_name,
                run_judging=run_judging,
                run_spider=run_spider_eval,
            ).items()
        }

        if include_neo4j:
            manifest["stages"]["neo4j_demo"] = {k: str(v) for k, v in run_neo4j_demo(profile_name).items()}

        manifest["status"] = "success"
    except Exception as exc:
        logger.error("Pipeline failed: {}", exc)
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_at"] = utc_now_iso()

    return manifest


def save_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    save_json(output_path, manifest)
    return output_path


__all__ = ["run_end_to_end", "save_manifest", "PipelineFailure"]
