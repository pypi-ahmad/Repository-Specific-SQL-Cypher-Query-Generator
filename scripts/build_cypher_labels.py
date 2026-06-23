#!/usr/bin/env python3
"""Run SQL-to-Cypher extension stage."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.cypher import run_cypher_extension


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cypher labels from SQL splits")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    args = parser.parse_args()

    out = run_cypher_extension(args.profile)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
