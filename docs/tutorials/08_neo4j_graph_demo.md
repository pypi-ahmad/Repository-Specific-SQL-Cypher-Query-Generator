# 08 - Neo4j Graph Demo

## What it is

A graph-loading stage that projects processed query examples into Neo4j nodes and relationships for graph exploration and Cypher validation demos.

## Why it is used

It demonstrates that the SQL/Cypher dataset can be represented and queried as a graph, making lineage, source distribution, and table usage patterns easier to inspect.

## How it appears in code

- Module: `src/repo_query_gen/neo4j_demo.py`
- Script: `scripts/run_neo4j_demo.py`
- Compose service: `docker/docker-compose.neo4j.yml`

Core functions:
- `wait_for_neo4j(...)`
- `reset_graph(...)`
- `load_dataset_graph(...)`
- `run_demo_queries(...)`
- `validate_generated_cypher_on_neo4j(...)`
- `run_neo4j_demo(...)`

Graph model includes:
- `Source`, `Question`, `SqlQuery`, `CypherQuery`, `Table` nodes
- relationships such as `FROM_SOURCE`, `HAS_SQL`, `HAS_CYPHER`, `USES_TABLE`, `MATCHES_TABLE`

## Practical explanation

Start service:

```bash
docker compose -f docker/docker-compose.neo4j.yml up -d
```

Run demo:

```bash
python scripts/run_neo4j_demo.py --profile tutorial
```

Tutorial graph summary artifact:
- `artifacts/neo4j_outputs/tutorial/graph_summary.json`
- values: sources `15`, questions `1000`, tables `910`

Demo query outputs:
- `artifacts/neo4j_outputs/tutorial/demo_query_results.json`
