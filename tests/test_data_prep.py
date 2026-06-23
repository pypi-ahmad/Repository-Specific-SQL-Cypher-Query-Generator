import pandas as pd

from repo_query_gen.data_prep import build_processed_dataframe


def test_build_processed_dataframe_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "instruction": "Generate SQL",
                "input": "CREATE TABLE users (id INT, name TEXT)",
                "response": "SELECT name FROM users",
                "source": "unit",
                "text": "list names",
            }
        ]
    )

    out = build_processed_dataframe(df)
    assert len(out) == 1
    assert out.iloc[0]["example_id"] == "clinton_0"
    assert out.iloc[0]["source"] == "unit"
    assert "sql" in out.columns
