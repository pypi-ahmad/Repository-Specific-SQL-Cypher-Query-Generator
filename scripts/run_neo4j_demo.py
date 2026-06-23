#!/usr/bin/env python3
"""Run Neo4j demo stage."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES
from repo_query_gen.neo4j_demo import run_neo4j_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neo4j demonstration")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    args = parser.parse_args()

    out = run_neo4j_demo(args.profile)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
