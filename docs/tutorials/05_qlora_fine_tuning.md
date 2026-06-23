# 05 - QLoRA Fine-Tuning

## What it is

A PEFT/QLoRA training stage for `ibm-granite/granite-4.1-3b`, with backend resolution across HF Trainer, TRL, and conditional Unsloth.

## Why it is used

Fine-tuning adapts the model to project-specific schema/query patterns while keeping hardware requirements manageable with 4-bit quantization and LoRA adapters.

## How it appears in code

- Module: `src/repo_query_gen/training.py`
- Script: `scripts/train_qlora.py`

Core implementation blocks:
- backend resolution: `_resolve_backend(...)`
- LoRA config/attachment: `_build_lora_config(...)`, `_attach_lora(...)`
- model loading: `_build_model_and_tokenizer(...)`
- backend runners: `_run_hf_training(...)`, `_run_trl_training(...)`, `_run_unsloth_training(...)`
- main entry: `run_finetuning(...)`

Artifacts produced per run:
- adapter directory
- `train_result.json`
- `eval_result.json`
- `gpu_snapshot.json`
- `training_metadata.json`
- `Modelfile.adapter.template`

## Practical explanation

Run:

```bash
python scripts/train_qlora.py --profile tutorial --backend auto
```

Real measured tutorial run used for documentation:
- run dir: `artifacts/training/tutorial_2026-06-20T19-14-21.185341+00-00_trl`
- requested backend: `auto`
- effective backend: `trl`
- fallback reason: `null`
- train loss: `1.7538`
- eval loss: `1.4786`
- train runtime: `87.9225s`
- eval runtime: `33.5015s`

Package versions in that run metadata:
- torch `2.12.1+cu130`
- transformers `5.12.1`
- peft `0.19.1`
- trl `1.6.0`
- unsloth `not_installed`
- bitsandbytes `0.49.2`
