#!/usr/bin/env python3
"""Run evaluation bundle."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.evaluation import run_evaluation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--with-spider", action="store_true")
    args = parser.parse_args()

    out = run_evaluation_bundle(
        args.profile,
        run_judging=args.with_judge,
        run_spider=args.with_spider,
    )
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
