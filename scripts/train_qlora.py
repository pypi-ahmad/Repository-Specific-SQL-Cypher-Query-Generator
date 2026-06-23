#!/usr/bin/env python3
"""Run QLoRA training stage."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.training import run_finetuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    parser.add_argument("--backend", choices=["auto", "hf", "trl", "unsloth"], default="auto")
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    out = run_finetuning(
        args.profile,
        backend=args.backend,
        allow_fallback=not args.no_fallback,
    )
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
