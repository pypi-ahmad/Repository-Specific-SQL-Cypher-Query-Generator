"""Neo4j demonstration pipeline for graph-based query validation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from neo4j import GraphDatabase

from repo_query_gen.config import Settings
from repo_query_gen.utils import save_json


def _driver(settings: Settings):
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def wait_for_neo4j(settings: Settings, timeout_s: int = 60) -> None:
    """Wait until Neo4j accepts simple queries."""

    start = time.time()
    while True:
        try:
            with _driver(settings).session() as session:
                session.run("RETURN 1").consume()
            return
        except Exception:
            if time.time() - start > timeout_s:
                raise
            time.sleep(2)


def reset_graph(settings: Settings) -> None:
    """Clear all existing graph content."""

    with _driver(settings).session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()


def load_dataset_graph(profile_name: str, max_rows: int = 1000) -> dict[str, Any]:
    """Load processed data into Neo4j graph model for tutorial demo."""

    settings = Settings()
    df = pd.read_csv(settings.processed_data_dir / profile_name / "train_cypher.csv")
    df = df.head(max_rows)

    wait_for_neo4j(settings)
    reset_graph(settings)

    with _driver(settings).session() as session:
        for row in df.to_dict(orient="records"):
            example_id = row["example_id"]
            source = row["source"]
            question = row["question_or_context"]
            sql = row["sql"]
            cypher = row.get("cypher", "")

            # Parse tables list from CSV-serialized value.
            try:
                tables = json.loads(str(row.get("tables", "[]")).replace("'", '"'))
            except Exception:
                tables = []

            session.run(
                """
                MERGE (s:Source {name: $source})
                MERGE (q:Question {id: $example_id})
                SET q.text = $question
                MERGE (sqlq:SqlQuery {id: $example_id})
                SET sqlq.text = $sql
                MERGE (cyq:CypherQuery {id: $example_id})
                SET cyq.text = $cypher
                MERGE (q)-[:FROM_SOURCE]->(s)
                MERGE (q)-[:HAS_SQL]->(sqlq)
                MERGE (q)-[:HAS_CYPHER]->(cyq)
                """,
                {
                    "source": source,
                    "example_id": example_id,
                    "question": question,
                    "sql": sql,
                    "cypher": cypher,
                },
            ).consume()

            for table in tables:
                session.run(
                    """
                    MERGE (t:Table {name: $table})
                    MERGE (sqlq:SqlQuery {id: $example_id})
                    MERGE (cyq:CypherQuery {id: $example_id})
                    MERGE (sqlq)-[:USES_TABLE]->(t)
                    MERGE (cyq)-[:MATCHES_TABLE]->(t)
                    """,
                    {"table": table, "example_id": example_id},
                ).consume()

    with _driver(settings).session() as session:
        summary = {
            "sources": session.run("MATCH (s:Source) RETURN count(s) AS c").single()["c"],
            "questions": session.run("MATCH (q:Question) RETURN count(q) AS c").single()["c"],
            "tables": session.run("MATCH (t:Table) RETURN count(t) AS c").single()["c"],
        }

    return summary


def run_demo_queries(output_path: Path) -> Path:
    """Run practical Cypher demo queries and save results."""

    settings = Settings()
    queries = {
        "top_sources": "MATCH (s:Source)<-[:FROM_SOURCE]-(:Question) RETURN s.name AS source, count(*) AS n ORDER BY n DESC LIMIT 10",
        "most_used_tables": "MATCH (:SqlQuery)-[:USES_TABLE]->(t:Table) RETURN t.name AS table, count(*) AS n ORDER BY n DESC LIMIT 10",
        "questions_with_many_tables": "MATCH (q:Question)-[:HAS_SQL]->(sq:SqlQuery)-[:USES_TABLE]->(t:Table) WITH q, count(DISTINCT t) AS n WHERE n >= 3 RETURN q.id AS question_id, n ORDER BY n DESC LIMIT 20",
    }

    results: dict[str, list[dict[str, Any]]] = {}

    with _driver(settings).session() as session:
        for name, query in queries.items():
            rows = [dict(r) for r in session.run(query)]
            results[name] = rows

    save_json(output_path, results)
    return output_path


def validate_generated_cypher_on_neo4j(cypher_query: str) -> dict[str, Any]:
    """Validate generated Cypher using EXPLAIN in Neo4j."""

    settings = Settings()
    try:
        with _driver(settings).session() as session:
            session.run(f"EXPLAIN {cypher_query}").consume()
        return {"valid": True, "error": ""}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def run_neo4j_demo(profile_name: str) -> dict[str, Path]:
    """Execute full Neo4j demo workflow."""

    settings = Settings()
    out_dir = settings.artifacts_dir / "neo4j_outputs" / profile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_dataset_graph(profile_name)
    summary_path = out_dir / "graph_summary.json"
    save_json(summary_path, summary)

    query_path = run_demo_queries(out_dir / "demo_query_results.json")
    return {"summary": summary_path, "demo_queries": query_path}


__all__ = ["run_neo4j_demo", "validate_generated_cypher_on_neo4j"]
