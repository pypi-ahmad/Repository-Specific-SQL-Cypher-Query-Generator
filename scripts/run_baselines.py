#!/usr/bin/env python3
"""Run prompt-only baseline generation."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.baselines import run_baseline_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline SQL/Cypher generation")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    args = parser.parse_args()

    out = run_baseline_generation(args.profile)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
