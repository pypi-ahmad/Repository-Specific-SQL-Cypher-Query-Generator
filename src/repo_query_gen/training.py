"""QLoRA fine-tuning pipeline with optional TRL and conditional Unsloth support."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from repo_query_gen.config import ProfileConfig, Settings, load_profile
from repo_query_gen.utils import ensure_dir, save_json, set_global_seed, utc_now_iso


TrainerBackend = Literal["auto", "hf", "trl", "unsloth"]
EffectiveTrainerBackend = Literal["hf", "trl", "unsloth"]


@dataclass
class PreparedDatasets:
    train: Dataset
    val: Dataset


@dataclass
class BackendResolution:
    requested: TrainerBackend
    effective: EffectiveTrainerBackend
    fallback_reason: str | None


def _gpu_snapshot() -> dict[str, Any]:
    """Capture best-effort GPU telemetry."""

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "nvidia_smi": "unavailable",
        }

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)

    nvidia_text = ""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        nvidia_text = out.strip()
    except Exception:
        nvidia_text = "nvidia-smi unavailable"

    return {
        "cuda_available": True,
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "nvidia_smi": nvidia_text,
    }


def _load_split(profile_name: str, split_name: str) -> pd.DataFrame:
    settings = Settings()
    path = settings.processed_data_dir / profile_name / f"{split_name}_cypher.csv"
    return pd.read_csv(path)


def _to_instruction_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in df.to_dict(orient="records"):
        schema_context = sample["input_schema_context"]
        question_ctx = sample["question_or_context"]

        sql_prompt = (
            "[TASK] Generate SQL with strict schema grounding. Return SQL only.\n"
            f"[SCHEMA]\n{schema_context}\n"
            f"[QUESTION]\n{question_ctx}\n"
        )
        sql_response = sample["sql"]

        cypher_prompt = (
            "[TASK] Generate Cypher with strict schema grounding. Return Cypher only.\n"
            f"[SCHEMA]\n{schema_context}\n"
            f"[QUESTION]\n{question_ctx}\n"
        )
        cypher_response = sample.get("cypher", "")

        rows.append(
            {
                "text": f"<|user|>\n{sql_prompt}\n<|assistant|>\n{sql_response}",
                "prompt": sql_prompt,
                "completion": sql_response,
                "task": "sql",
                "example_id": sample["example_id"],
            }
        )

        if cypher_response:
            rows.append(
                {
                    "text": f"<|user|>\n{cypher_prompt}\n<|assistant|>\n{cypher_response}",
                    "prompt": cypher_prompt,
                    "completion": cypher_response,
                    "task": "cypher",
                    "example_id": sample["example_id"],
                }
            )

    return rows


def _prepare_datasets(profile_name: str, profile: ProfileConfig, settings: Settings) -> PreparedDatasets:
    train_df = _load_split(profile_name, "train")
    val_df = _load_split(profile_name, "val")

    if profile.train_max_examples is not None and profile.train_max_examples < len(train_df):
        train_df = train_df.sample(n=profile.train_max_examples, random_state=settings.seed).reset_index(drop=True)
    if profile.val_max_examples is not None and profile.val_max_examples < len(val_df):
        val_df = val_df.sample(n=profile.val_max_examples, random_state=settings.seed).reset_index(drop=True)

    train_rows = _to_instruction_rows(train_df)
    val_rows = _to_instruction_rows(val_df)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    return PreparedDatasets(train=train_ds, val=val_ds)


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_len: int) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _unsloth_model_supported(model_name: str) -> bool:
    """Conservative compatibility gate for Unsloth model loading.

    We only allow families explicitly documented in current Unsloth tutorials.
    Granite-4.1 is intentionally treated as unsupported until explicitly validated.
    """

    model_name_l = model_name.lower()
    if "granite-4.1" in model_name_l:
        return False

    supported_markers = (
        "qwen3.5",
        "qwen3",
        "granite-4.0",
        "llama",
        "gemma",
        "mistral",
        "phi",
        "unsloth/",
    )
    return any(marker in model_name_l for marker in supported_markers)


def _next_fallback_backend() -> EffectiveTrainerBackend:
    if _is_module_available("trl"):
        return "trl"
    return "hf"


def _resolve_backend(
    requested_backend: TrainerBackend,
    settings: Settings,
    allow_fallback: bool,
) -> BackendResolution:
    """Resolve the effective training backend with guarded fallbacks."""

    if requested_backend == "auto":
        if _is_module_available("trl"):
            return BackendResolution(requested="auto", effective="trl", fallback_reason=None)
        return BackendResolution(requested="auto", effective="hf", fallback_reason="trl_not_installed")

    if requested_backend == "hf":
        return BackendResolution(requested="hf", effective="hf", fallback_reason=None)

    if requested_backend == "trl":
        if _is_module_available("trl"):
            return BackendResolution(requested="trl", effective="trl", fallback_reason=None)
        if not allow_fallback:
            raise RuntimeError("TRL backend requested but `trl` is not installed.")
        return BackendResolution(requested="trl", effective="hf", fallback_reason="trl_not_installed")

    # requested_backend == "unsloth"
    if not _is_module_available("unsloth"):
        if not allow_fallback:
            raise RuntimeError("Unsloth backend requested but `unsloth` is not installed.")
        return BackendResolution(
            requested="unsloth",
            effective=_next_fallback_backend(),
            fallback_reason="unsloth_not_installed",
        )

    if not _unsloth_model_supported(settings.hf_granite_base_model):
        if not allow_fallback:
            raise RuntimeError(
                f"Unsloth backend requested but base model {settings.hf_granite_base_model!r} "
                "is not in the validated support set for this project.",
            )
        return BackendResolution(
            requested="unsloth",
            effective=_next_fallback_backend(),
            fallback_reason=f"unsloth_model_incompatible:{settings.hf_granite_base_model}",
        )

    return BackendResolution(requested="unsloth", effective="unsloth", fallback_reason=None)


def _build_lora_config(profile: ProfileConfig) -> LoraConfig:
    """Create PEFT LoRA config using QLoRA-style `all-linear` targeting when possible."""

    return LoraConfig(
        r=profile.lora_r,
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )


def _attach_lora(model: AutoModelForCausalLM, profile: ProfileConfig) -> AutoModelForCausalLM:
    """Attach LoRA adapters, with a safe fallback to explicit projection modules."""

    lora_cfg = _build_lora_config(profile)
    try:
        model = get_peft_model(model, lora_cfg)
    except Exception as exc:
        logger.warning("LoRA all-linear attachment failed ({}). Falling back to explicit projections.", exc)
        fallback_cfg = LoraConfig(
            r=profile.lora_r,
            lora_alpha=profile.lora_alpha,
            lora_dropout=profile.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, fallback_cfg)
    model.print_trainable_parameters()
    return model


def _build_model_and_tokenizer(settings: Settings, profile: ProfileConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Build quantized model and tokenizer for HF/TRL backends."""

    model_name = settings.hf_granite_base_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for fine-tuning in this project.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
    except Exception as exc:
        logger.warning(
            "Primary 4-bit single-GPU load failed ({}). Falling back to 8-bit CPU offload mode.",
            exc,
        )
        fallback_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=fallback_config,
            device_map="auto",
            max_memory={0: "1700MiB", "cpu": "48GiB"},
        )

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as exc:
        logger.warning("prepare_model_for_kbit_training failed: {}", exc)

    model = _attach_lora(model, profile)
    return model, tokenizer


def _run_hf_training(
    datasets: PreparedDatasets,
    out_dir: Path,
    settings: Settings,
    profile: ProfileConfig,
) -> tuple[Any, dict[str, Any], AutoTokenizer]:
    """Run legacy HF Trainer training path with PEFT model."""

    model, tokenizer = _build_model_and_tokenizer(settings, profile)

    train_tok = _tokenize_dataset(datasets.train, tokenizer, profile.max_seq_len)
    val_tok = _tokenize_dataset(datasets.val, tokenizer, profile.max_seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=profile.max_train_steps,
        per_device_train_batch_size=profile.batch_size,
        per_device_eval_batch_size=profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=profile.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=60,
        eval_steps=60,
        eval_strategy="steps",
        save_strategy="steps",
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    return trainer, {"train": train_result.metrics, "eval": eval_result}, tokenizer


def _build_sft_config(
    out_dir: Path,
    profile: ProfileConfig,
) -> Any:
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=profile.max_train_steps,
        per_device_train_batch_size=profile.batch_size,
        per_device_eval_batch_size=profile.batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        learning_rate=profile.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=60,
        eval_steps=60,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to=[],
        dataset_text_field="text",
        max_length=profile.max_seq_len,
        completion_only_loss=False,
    )


def _run_trl_training(
    datasets: PreparedDatasets,
    out_dir: Path,
    settings: Settings,
    profile: ProfileConfig,
) -> tuple[Any, dict[str, Any], AutoTokenizer]:
    """Run TRL SFTTrainer path with PEFT adapters."""

    from trl import SFTTrainer

    model, tokenizer = _build_model_and_tokenizer(settings, profile)
    sft_args = _build_sft_config(out_dir, profile)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": datasets.train,
        "eval_dataset": datasets.val,
    }

    # TRL switched from `tokenizer` to `processing_class` in newer versions.
    init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    return trainer, {"train": train_result.metrics, "eval": eval_result}, tokenizer


def _run_unsloth_training(
    datasets: PreparedDatasets,
    out_dir: Path,
    settings: Settings,
    profile: ProfileConfig,
) -> tuple[Any, dict[str, Any], Any]:
    """Run Unsloth + TRL training path.

    This backend is intentionally gated by compatibility checks before invocation.
    """

    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.hf_granite_base_model,
        max_seq_length=profile.max_seq_len,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=profile.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=settings.seed,
        max_seq_length=profile.max_seq_len,
    )

    sft_args = _build_sft_config(out_dir, profile)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_args,
        "train_dataset": datasets.train,
        "eval_dataset": datasets.val,
    }

    init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    return trainer, {"train": train_result.metrics, "eval": eval_result}, tokenizer


def _package_versions() -> dict[str, str]:
    """Capture key package versions used in this run for reproducibility."""

    versions: dict[str, str] = {}
    for module_name in ["torch", "transformers", "peft", "trl", "unsloth", "bitsandbytes"]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "not_installed"
    return versions


def run_finetuning(
    profile_name: str,
    backend: TrainerBackend | None = None,
    allow_fallback: bool | None = None,
) -> dict[str, Path]:
    """Execute fine-tuning run for selected profile and backend.

    Args:
        profile_name: Profile name (`fast`, `tutorial`, or `full`).
        backend: Training backend (`auto`, `hf`, `trl`, `unsloth`).
        allow_fallback: Whether backend fallback is allowed.
    """

    settings = Settings()
    profile = load_profile(profile_name)

    requested_backend = backend or settings.training_backend
    fallback_enabled = settings.allow_backend_fallback if allow_fallback is None else allow_fallback

    resolved = _resolve_backend(
        requested_backend=requested_backend,
        settings=settings,
        allow_fallback=fallback_enabled,
    )

    set_global_seed(settings.seed)

    run_id = f"{profile_name}_{utc_now_iso().replace(':', '-')}_{resolved.effective}"
    out_dir = ensure_dir(settings.artifacts_dir / "training" / run_id)

    mlruns_dir = ensure_dir(settings.project_root / "mlruns")
    db_path = mlruns_dir / "mlflow.db"
    ensure_dir(mlruns_dir / "artifacts")
    os.environ.setdefault("MLFLOW_ARTIFACTS_DESTINATION", str(mlruns_dir / "artifacts"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("repo_sql_cypher_qlora")

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(
            {
                "profile": profile_name,
                "requested_backend": resolved.requested,
                "effective_backend": resolved.effective,
                "fallback_reason": resolved.fallback_reason or "none",
                "allow_fallback": fallback_enabled,
                "max_train_steps": profile.max_train_steps,
                "batch_size": profile.batch_size,
                "gradient_accumulation_steps": profile.gradient_accumulation_steps,
                "learning_rate": profile.learning_rate,
                "lora_r": profile.lora_r,
                "lora_alpha": profile.lora_alpha,
                "max_seq_len": profile.max_seq_len,
                "base_model": settings.hf_granite_base_model,
            }
        )

        datasets = _prepare_datasets(profile_name, profile, settings)
        logger.info("Prepared training datasets: train_rows={} val_rows={}", len(datasets.train), len(datasets.val))

        if resolved.effective == "hf":
            trainer, result_bundle, tokenizer = _run_hf_training(datasets, out_dir, settings, profile)
        elif resolved.effective == "trl":
            trainer, result_bundle, tokenizer = _run_trl_training(datasets, out_dir, settings, profile)
        else:
            trainer, result_bundle, tokenizer = _run_unsloth_training(datasets, out_dir, settings, profile)

        train_metrics = result_bundle["train"]
        eval_metrics = result_bundle["eval"]

        adapter_dir = ensure_dir(out_dir / "adapter")
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        gpu_state = _gpu_snapshot()
        save_json(out_dir / "gpu_snapshot.json", gpu_state)
        save_json(out_dir / "train_result.json", train_metrics)
        save_json(out_dir / "eval_result.json", eval_metrics)

        metadata = {
            "requested_backend": resolved.requested,
            "effective_backend": resolved.effective,
            "fallback_reason": resolved.fallback_reason,
            "allow_fallback": fallback_enabled,
            "base_model": settings.hf_granite_base_model,
            "package_versions": _package_versions(),
        }
        save_json(out_dir / "training_metadata.json", metadata)

        mlflow.log_metrics(
            {
                "train_loss": float(train_metrics.get("train_loss", 0.0)),
                "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                "train_runtime": float(train_metrics.get("train_runtime", 0.0)),
            }
        )

        modelfile_path = out_dir / "Modelfile.adapter.template"
        modelfile_path.write_text(
            "\n".join(
                [
                    f"FROM {settings.granite_model}",
                    "# ADAPTER path requires runtime compatibility with serving backend.",
                    f"ADAPTER {adapter_dir}",
                    'SYSTEM "You are a schema-aware SQL and Cypher assistant."',
                ]
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(out_dir / "gpu_snapshot.json"))
        mlflow.log_artifact(str(out_dir / "train_result.json"))
        mlflow.log_artifact(str(out_dir / "eval_result.json"))
        mlflow.log_artifact(str(out_dir / "training_metadata.json"))

    logger.info("Training finished with backend {}. Artifacts stored at {}", resolved.effective, out_dir)
    return {
        "run_dir": out_dir,
        "adapter_dir": out_dir / "adapter",
        "train_result": out_dir / "train_result.json",
        "eval_result": out_dir / "eval_result.json",
        "metadata": out_dir / "training_metadata.json",
    }


__all__ = ["run_finetuning"]
