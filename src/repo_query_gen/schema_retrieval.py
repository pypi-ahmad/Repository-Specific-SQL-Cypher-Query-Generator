"""Schema retrieval helpers for schema-aware prompt construction and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass


CREATE_TABLE_RE = re.compile(r"CREATE TABLE\s+([^\s(]+)\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)
COLUMN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


@dataclass
class ParsedTable:
    table_name: str
    columns: list[str]
    create_stmt: str


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text)}


def parse_schema_context(schema_context: str) -> list[ParsedTable]:
    """Parse `CREATE TABLE` statements from schema context."""

    parsed: list[ParsedTable] = []
    for match in CREATE_TABLE_RE.finditer(schema_context):
        table_name = match.group(1).strip("`\"")
        block = match.group(2)
        cols: list[str] = []
        for line in block.splitlines():
            if line.strip().lower().startswith(("primary key", "foreign key", "constraint")):
                continue
            col_match = COLUMN_RE.match(line.strip().rstrip(","))
            if col_match:
                cols.append(col_match.group(1).strip("`\""))
        parsed.append(
            ParsedTable(
                table_name=table_name,
                columns=cols,
                create_stmt=f"CREATE TABLE {table_name} ({block})",
            )
        )
    return parsed


def select_schema_context(
    question: str,
    schema_context: str,
    top_k_tables: int = 6,
) -> dict[str, object]:
    """Select question-relevant schema subset using lexical overlap heuristics."""

    tables = parse_schema_context(schema_context)
    if not tables:
        return {
            "strategy": "lexical",
            "selected_schema_context": schema_context,
            "selected_tables": [],
            "selected_columns": [],
            "all_tables": [],
        }

    q_tokens = _tokenize(question)
    scored: list[tuple[float, ParsedTable]] = []
    for tbl in tables:
        tbl_tokens = _tokenize(tbl.table_name)
        col_tokens = _tokenize(" ".join(tbl.columns))
        overlap = len(q_tokens & tbl_tokens) * 3 + len(q_tokens & col_tokens)
        score = float(overlap)
        scored.append((score, tbl))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [t for s, t in scored if s > 0][:top_k_tables]
    if not selected:
        selected = [t for _, t in scored[: min(top_k_tables, len(scored))]]

    selected_schema = "\n\n".join(t.create_stmt for t in selected)
    selected_tables = [t.table_name for t in selected]
    selected_columns = sorted({f"{t.table_name}.{col}" for t in selected for col in t.columns})

    return {
        "strategy": "lexical",
        "selected_schema_context": selected_schema,
        "selected_tables": selected_tables,
        "selected_columns": selected_columns,
        "all_tables": [t.table_name for t in tables],
    }


__all__ = ["parse_schema_context", "select_schema_context"]
