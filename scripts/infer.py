#!/usr/bin/env python3
"""Run single-query inference for SQL/Cypher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repo_query_gen.inference import infer_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference entrypoint")
    parser.add_argument("--task", choices=["sql", "cypher"], required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--schema-context", required=True)
    parser.add_argument("--mode", choices=["ollama", "finetuned"], default="ollama")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    args = parser.parse_args()

    out = infer_single(
        question=args.question,
        schema_context=args.schema_context,
        model_mode=args.mode,
        task=args.task,
        model_name=args.model_name,
        adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
