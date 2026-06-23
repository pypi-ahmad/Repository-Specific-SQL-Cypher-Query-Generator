from repo_query_gen.inference import infer_single, postprocess_generated_query, validate_generated_query


def test_validate_generated_sql() -> None:
    schema = "CREATE TABLE startups (id INT, name TEXT);"
    query = "SELECT name FROM startups"
    report = validate_generated_query("sql", query, schema)

    assert report["parse_success"] is True
    assert report["schema_grounded"] is True


def test_infer_single_includes_retrieval_and_latency(monkeypatch) -> None:
    def _mock_generate(task, question, schema_context, model_name):  # noqa: ANN001
        if task == "sql":
            return "SELECT name FROM startups"
        return "MATCH (s:Startup) RETURN s.name"

    monkeypatch.setattr("repo_query_gen.inference.generate_with_ollama", _mock_generate)

    schema = """
CREATE TABLE startups (
  id INTEGER,
  name TEXT,
  founded_year INTEGER
);
CREATE TABLE investors (
  id INTEGER,
  name TEXT
);
"""
    out = infer_single(
        question="List startup names",
        schema_context=schema,
        model_mode="ollama",
        task="sql",
        model_name="granite4.1:3b",
    )

    assert out["retrieval"]["strategy"] in {"lexical", "full"}
    assert isinstance(out["latency_ms"], float)
    assert out["used_schema_context"]


def test_postprocess_generated_query_sql_strips_preamble() -> None:
    raw = "No additional text.\nSELECT name FROM startups WHERE id > 10;"
    assert postprocess_generated_query("sql", raw) == "SELECT name FROM startups WHERE id > 10"


def test_postprocess_generated_query_sql_from_fenced_block() -> None:
    raw = "```sql\nSELECT COUNT(*) FROM startups;\n```\nExplanation: returns count."
    assert postprocess_generated_query("sql", raw) == "SELECT COUNT(*) FROM startups"
