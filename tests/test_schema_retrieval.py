from repo_query_gen.schema_retrieval import select_schema_context


def test_select_schema_context_picks_relevant_table() -> None:
    schema = """
CREATE TABLE startups (
  id INTEGER,
  name TEXT,
  founded_year INTEGER
);

CREATE TABLE investors (
  id INTEGER,
  name TEXT,
  location TEXT
);
"""
    out = select_schema_context(
        question="Which startups were founded after 2020?",
        schema_context=schema,
        top_k_tables=1,
    )

    assert out["strategy"] == "lexical"
    assert out["selected_tables"] == ["startups"]
    assert "CREATE TABLE startups" in out["selected_schema_context"]
