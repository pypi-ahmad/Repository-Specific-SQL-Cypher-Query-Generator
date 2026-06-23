from repo_query_gen.cypher import sql_to_cypher_deterministic, validate_cypher_text


def test_sql_to_cypher_basic() -> None:
    sql = "SELECT name FROM users WHERE age > 18 ORDER BY age DESC LIMIT 5"
    cypher, meta = sql_to_cypher_deterministic(sql)

    assert "MATCH" in cypher
    assert "RETURN" in cypher
    assert meta["status"] in {"ok", "parse_error", "no_table_detected"}


def test_validate_cypher_text() -> None:
    ok, issues = validate_cypher_text("MATCH (u:User) RETURN u")
    assert ok
    assert issues == []
