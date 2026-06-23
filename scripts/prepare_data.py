#!/usr/bin/env python3
"""Run dataset preparation stage."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.data_prep import run_data_preparation


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for SQL/Cypher training")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    args = parser.parse_args()

    out = run_data_preparation(args.profile)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
