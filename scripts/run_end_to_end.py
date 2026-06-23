#!/usr/bin/env python3
"""Run full end-to-end pipeline."""

from __future__ import annotations

import argparse
import json

from repo_query_gen.config import PROFILE_CHOICES, Settings
from repo_query_gen.pipeline import run_end_to_end, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SQL/Cypher project pipeline")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="fast")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--with-inference", action="store_true")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--with-spider", action="store_true")
    parser.add_argument("--trainer-backend", choices=["auto", "hf", "trl", "unsloth"], default="auto")
    parser.add_argument("--no-backend-fallback", action="store_true")
    args = parser.parse_args()

    manifest = run_end_to_end(
        profile_name=args.profile,
        include_training=not args.skip_training,
        include_neo4j=not args.skip_neo4j,
        include_inference=args.with_inference,
        run_judging=args.with_judge,
        run_spider_eval=args.with_spider,
        training_backend=args.trainer_backend,
        allow_backend_fallback=not args.no_backend_fallback,
    )

    settings = Settings()
    out_path = settings.artifacts_dir / f"manifest_{args.profile}.json"
    save_manifest(manifest, out_path)

    print(json.dumps(manifest, indent=2))
    print(f"Manifest saved: {out_path}")


if __name__ == "__main__":
    main()
