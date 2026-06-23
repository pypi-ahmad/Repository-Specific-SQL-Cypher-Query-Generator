"""Typed contracts used across preparation, training, and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SchemaColumn(BaseModel):
    """Represents one column in a relational table schema."""

    name: str
    dtype: str


class TableSchema(BaseModel):
    """Represents one parsed SQL table schema."""

    table_name: str
    columns: list[SchemaColumn] = Field(default_factory=list)


class QueryExample(BaseModel):
    """One processed query example with schema grounding."""

    example_id: str
    source: str
    instruction: str
    question_or_context: str
    sql: str
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    complexity_tags: list[str] = Field(default_factory=list)


class TrainingSample(BaseModel):
    """Instruction-tuning sample for SQL or Cypher generation."""

    task: Literal["sql", "cypher"]
    prompt: str
    response: str
    example_id: str


class GeneratedQuery(BaseModel):
    """Output contract for model-generated queries."""

    task: Literal["sql", "cypher"]
    model_name: str
    example_id: str
    question: str
    generated_query: str
    reference_query: str
    schema_refs: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Validation details for generated SQL/Cypher query."""

    parse_success: bool
    execution_success: bool | None = None
    schema_grounded: bool | None = None
    issues: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Per-example evaluation result."""

    example_id: str
    task: Literal["sql", "cypher"]
    model_name: str
    exact_match: float
    syntax_success: float
    execution_success: float | None = None
    schema_grounding: float | None = None
    bleu: float | None = None
    rouge_l: float | None = None
    meteor: float | None = None
    bertscore_f1: float | None = None


class ErrorCase(BaseModel):
    """Structured error case for detailed failure analysis."""

    example_id: str
    task: Literal["sql", "cypher"]
    model_name: str
    error_type: str
    question: str
    reference_query: str
    generated_query: str
    notes: str
