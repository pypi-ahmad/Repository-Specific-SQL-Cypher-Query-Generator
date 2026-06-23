#!/usr/bin/env python3
"""Build tutorial-first notebooks for the project."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_notebook(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    path = NB_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def technique_section(
    technique_name: str,
    definition: str,
    why_developed: str,
    rag_limitations: str,
    workflow: str,
    components: str,
    usage_guidance: str,
    pros_cons: str,
    comparison: str,
    project_design: str,
) -> str:
    """Return standardized deep tutorial section required across notebooks."""

    return f"""
## What Is This Technique?

### What is this technique?
{technique_name}

### Definition and core concepts.
{definition}

### Why was this technique developed?
{why_developed}

### What limitations of traditional RAG does it solve?
{rag_limitations}

### Architecture and workflow diagram explanation.
{workflow}

### Component-by-component breakdown.
{components}

### When should it be used in real-world systems?
{usage_guidance}

### Advantages and disadvantages.
{pros_cons}

### Comparison against standard RAG and other implemented RAG variants.
{comparison}

### Implementation details and design decisions used in this project.
{project_design}
"""


def post_run_analysis_section(technique_name: str) -> str:
    return f"""
## Post-Run Analysis ({technique_name})

After executing the real pipeline, use this section to document measured outcomes.

- Analyze actual outputs, metrics, retrieval quality, latency, and generation quality.
- Explain how the technique changed measured behavior vs baseline.
- Interpret every metric in business and engineering terms.
- Capture failure modes, lessons learned, and practical takeaways.
- Explain why specific outputs were produced and what they demonstrate.
- Conclude effectiveness based on measured evidence, not assumptions.
"""


def build_00_intro() -> None:
    cells = [
        md(
            """
# 00 - Zero-to-Hero Introduction: Repository-Specific SQL & Cypher Query Generation

This project is a practical textbook plus production-minded pipeline:
- complete local execution,
- schema-aware SQL/Cypher generation,
- baseline-vs-finetuned comparisons,
- reusable modules for GraphRAG and enterprise analytics.
"""
        ),
        md(
            """
## Learning Outcomes

By the end of this notebook series you will be able to:
1. Explain why text-to-SQL and text-to-Cypher are difficult in production.
2. Build and evaluate schema-aware query generators.
3. Fine-tune local models with PEFT/TRL and conditionally Unsloth.
4. Interpret quality, grounding, and latency metrics with practical rigor.
"""
        ),
        md(
            technique_section(
                technique_name="Problem framing for schema-aware NL-to-query systems",
                definition=(
                    "A structured process for converting user questions into executable SQL/Cypher while enforcing schema grounding, "
                    "validation, and measurable quality gates."
                ),
                why_developed=(
                    "Generic prompting often returns syntactically plausible but semantically unsafe queries. Production systems need grounded outputs "
                    "with explicit failure visibility."
                ),
                rag_limitations=(
                    "Traditional RAG improves answer grounding for prose tasks, but does not guarantee executable query construction. "
                    "This project extends grounding to schema retrieval, query structure control, and parser/execution validation."
                ),
                workflow=(
                    "Workflow: Question -> schema retrieval/selection -> query generation -> syntax check -> schema-grounding check -> evaluation. "
                    "You can visualize this as a left-to-right pipeline with validation gates between each stage."
                ),
                components=(
                    "Data preparation, SQL-to-Cypher label extension, baseline generation, fine-tuning backend, inference, metric evaluation, and Neo4j demo."
                ),
                usage_guidance=(
                    "Use this architecture for BI copilots, graph analytics assistants, and agentic systems that must issue executable DB queries."
                ),
                pros_cons=(
                    "Advantages: stronger grounding, reproducible evaluation, lower hallucination risk. "
                    "Disadvantages: more engineering complexity and data/compute requirements than prompt-only setups."
                ),
                comparison=(
                    "Compared to standard RAG QA, this pipeline adds executable query constraints and schema-level metrics. "
                    "Compared to prompt-only query generation, it introduces explicit supervision and stronger validation."
                ),
                project_design=(
                    "We keep everything local (`uv`, Python 3.12, Ollama models), with profile-based fast/full runs and artifact-driven reproducibility."
                ),
            )
        ),
        code(
            """
from repo_query_gen.config import Settings, load_profile

settings = Settings()
print("Project root:", settings.project_root)
print("Training backend default:", settings.training_backend)
print("Schema retrieval mode:", settings.schema_retrieval_mode)
print("Fast profile:", load_profile("fast"))
"""
        ),
    ]
    write_notebook("00_intro_and_problem_framing.ipynb", cells)


def build_01_foundations() -> None:
    cells = [
        md(
            """
# 01 - Text-to-SQL and Text-to-Cypher Foundations

This notebook builds intuition for query-generation failure modes and why schema awareness is mandatory.
"""
        ),
        md(
            """
## Why Text-to-SQL Is Hard
- Schema understanding and join path selection.
- Aggregations and nested query composition.
- Ambiguity in natural-language business questions.
- Hallucinated tables/columns under weak grounding.

## Why Text-to-Cypher Is Harder
- Relationship directionality and multi-hop traversal planning.
- Pattern matching semantics differ from relational joins.
- Graph schema mismatch is often silent but severe.
"""
        ),
        md(
            technique_section(
                technique_name="Schema-aware query planning",
                definition=(
                    "A planning approach where generation is conditioned on a known schema context and constrained by parser- and schema-level validation."
                ),
                why_developed=(
                    "Prompt-only models frequently produce structurally valid but non-executable or semantically incorrect queries."
                ),
                rag_limitations=(
                    "Standard RAG retrieves documents; it does not guarantee accurate table/relationship selection. "
                    "Schema-aware planning focuses retrieval on schema entities and measures grounding directly."
                ),
                workflow=(
                    "Question tokens are matched to schema entities, candidate schema context is assembled, query is generated, then validated."
                ),
                components=(
                    "Token overlap retrieval, schema subset builder, generator, SQL/Cypher parser checks, grounding metrics."
                ),
                usage_guidance=(
                    "Use when wrong joins or relationship direction errors can cause high business risk."
                ),
                pros_cons=(
                    "Advantages: higher reliability and interpretability. Disadvantages: extra preprocessing and evaluation engineering."
                ),
                comparison=(
                    "Compared with generic RAG answer generation, this method adds executable constraints. "
                    "Compared with full-schema prompting, it can reduce context noise and latency."
                ),
                project_design=(
                    "This project supports `full` and `lexical` schema retrieval modes to compare quality vs simplicity."
                ),
            )
        ),
        code(
            """
import pandas as pd
from repo_query_gen.data_prep import run_data_preparation

paths = run_data_preparation("fast")
pd.read_csv(paths["train_csv"])[["example_id", "source", "complexity_bucket"]].head()
"""
        ),
        md(post_run_analysis_section("schema-aware query planning")),
    ]
    write_notebook("01_text2sql_and_text2cypher_foundations.ipynb", cells)


def build_02_dataset() -> None:
    cells = [
        md(
            """
# 02 - Dataset Deep Dive and Split Strategy

Primary dataset: `Clinton/Text-to-sql-v1`.

Goal: understand structure, complexity, limitations, and robust split design.
"""
        ),
        md(
            technique_section(
                technique_name="Schema-aware dataset curation and stratified splitting",
                definition=(
                    "Transform raw instruction-input-response rows into structured training examples with parsed schema metadata and complexity tags."
                ),
                why_developed=(
                    "Mixed-source SQL datasets can hide distribution skew and complexity imbalance, causing misleading model quality signals."
                ),
                rag_limitations=(
                    "Traditional RAG evaluation rarely addresses data split leakage for executable query tasks. "
                    "This curation pipeline enforces source+complexity-aware split behavior."
                ),
                workflow=(
                    "Raw rows -> schema parsing -> SQL normalization -> complexity tagging -> stratified train/val/test split -> persisted manifests."
                ),
                components=(
                    "Schema parser, normalization utilities, complexity heuristics, stratified splitter, manifest writer."
                ),
                usage_guidance=(
                    "Use for any text-to-query fine-tuning setup where schema distribution varies across domains."
                ),
                pros_cons=(
                    "Advantages: reliable evaluation and better generalization diagnostics. "
                    "Disadvantages: extra preprocessing and edge-case handling for malformed schemas."
                ),
                comparison=(
                    "Compared with random splits, stratified splits reduce variance and leakage risk. "
                    "Compared with document-RAG corpora, query datasets require executable-target consistency checks."
                ),
                project_design=(
                    "We parse CREATE TABLE blocks directly from dataset context and preserve source labels for downstream error slicing."
                ),
            )
        ),
        code(
            """
from datasets import load_dataset

ds = load_dataset("Clinton/Text-to-sql-v1", split="train")
print(ds)
print(ds.features)
print(ds[0])
"""
        ),
        code(
            """
import re
from collections import Counter
import numpy as np

sources = Counter(ds["source"])
print("Top sources:", sources.most_common(10))

sqls = ds["response"]
patterns = {
    "join": r"\\bjoin\\b",
    "group_by": r"\\bgroup\\s+by\\b",
    "nested": r"\\(\\s*select\\b",
}
for name, pat in patterns.items():
    c = sum(1 for q in sqls if re.search(pat, q, flags=re.I))
    print(name, c)

lengths = [len(x) for x in sqls]
print("p95 chars", np.percentile(lengths, 95))
"""
        ),
        code(
            """
import pandas as pd
from repo_query_gen.data_prep import run_data_preparation

paths = run_data_preparation("fast")
train_df = pd.read_csv(paths["train_csv"])
val_df = pd.read_csv(paths["val_csv"])
test_df = pd.read_csv(paths["test_csv"])

print(len(train_df), len(val_df), len(test_df))
print(train_df[["source", "complexity_bucket"]].head())
"""
        ),
        md(post_run_analysis_section("dataset curation and split strategy")),
    ]
    write_notebook("02_dataset_deep_dive_and_splitting.ipynb", cells)


def build_03_cypher() -> None:
    cells = [
        md(
            """
# 03 - SQL-to-Cypher Hybrid Labeling Pipeline

The source dataset is SQL-centric, so we construct Cypher supervision using deterministic translation plus validation.
"""
        ),
        md(
            technique_section(
                technique_name="Hybrid SQL-to-Cypher supervision",
                definition=(
                    "A deterministic translation-first pipeline that maps SQL structure to Cypher templates and validates syntax/quality before training usage."
                ),
                why_developed=(
                    "Native Cypher-labeled datasets with broad schema coverage are limited; direct LLM generation alone is hard to audit."
                ),
                rag_limitations=(
                    "RAG retrieval alone cannot provide graph query supervision labels. "
                    "This hybrid pipeline creates executable graph-query targets from relational supervision."
                ),
                workflow=(
                    "SQL parse -> clause mapping -> Cypher draft -> syntax validation -> quality scoring -> persisted labels."
                ),
                components=(
                    "SQL parser, mapping rules, Cypher validator, optional refiner, quality metadata."
                ),
                usage_guidance=(
                    "Use when extending relational query corpora to graph-query tasks for Neo4j/GraphRAG systems."
                ),
                pros_cons=(
                    "Advantages: auditable conversion and reproducible labels. "
                    "Disadvantages: some SQL semantics need manual mapping rules for graph equivalents."
                ),
                comparison=(
                    "Compared with direct model-generated Cypher labels, deterministic mapping improves control. "
                    "Compared with traditional RAG, this is supervised label generation rather than context retrieval."
                ),
                project_design=(
                    "This project stores Cypher quality scores and keeps SQL references for error analysis traceability."
                ),
            )
        ),
        code(
            """
from repo_query_gen.cypher import sql_to_cypher_deterministic, validate_cypher_text, run_cypher_extension

sql = "SELECT name FROM students WHERE age > 18 ORDER BY age DESC LIMIT 5"
cypher, meta = sql_to_cypher_deterministic(sql)
print(cypher)
print(meta)
print(validate_cypher_text(cypher))
"""
        ),
        code(
            """
paths = run_cypher_extension("fast")
print(paths)
"""
        ),
        code(
            """
import pandas as pd

df = pd.read_csv("../data/processed/fast/train_cypher.csv")
print(df[["sql", "cypher", "cypher_quality"]].head(5))
print(df["cypher_quality"].describe())
"""
        ),
        md(post_run_analysis_section("SQL-to-Cypher hybrid labeling")),
    ]
    write_notebook("03_sql_to_cypher_hybrid_labeling.ipynb", cells)


def build_04_baselines() -> None:
    cells = [
        md(
            """
# 04 - Baselines: Prompt-Only SQL/Cypher Generation

This notebook establishes the non-fine-tuned reference point.
"""
        ),
        md(
            technique_section(
                technique_name="Prompt-only baseline benchmarking",
                definition=(
                    "Generate SQL/Cypher directly from local instruction models without task-specific fine-tuning and measure schema-aware metrics."
                ),
                why_developed=(
                    "A baseline is required to prove whether domain-specific fine-tuning provides real gains."
                ),
                rag_limitations=(
                    "Traditional RAG benchmarks often score answer text only; here we score executability and schema grounding."
                ),
                workflow=(
                    "Sample test set -> build schema-aware prompts -> generate SQL/Cypher -> log latency -> compare with references."
                ),
                components=(
                    "Prompt templates, local model calls, latency capture, structured prediction artifacts."
                ),
                usage_guidance=(
                    "Use as a minimum viability gate before allocating compute to fine-tuning."
                ),
                pros_cons=(
                    "Advantages: fast setup and low training cost. Disadvantages: weaker grounding and higher hallucination risk."
                ),
                comparison=(
                    "Compared with fine-tuned variants, prompt-only is simpler but usually less reliable for schema-specific tasks."
                ),
                project_design=(
                    "Baselines now include schema retrieval metadata and per-task latency to support deeper post-run comparisons."
                ),
            )
        ),
        code(
            """
from repo_query_gen.baselines import run_baseline_generation

out = run_baseline_generation("fast")
out
"""
        ),
        code(
            """
import pandas as pd

df = pd.read_csv("../artifacts/baseline/fast/baseline_predictions.csv")
print(df.head(3))
print(df[["model_name", "sql_latency_ms", "cypher_latency_ms"]].groupby("model_name").mean())
"""
        ),
        md(post_run_analysis_section("prompt-only baselines")),
    ]
    write_notebook("04_baselines_prompt_only.ipynb", cells)


def build_05_training() -> None:
    cells = [
        md(
            """
# 05 - Fine-Tuning with PEFT, TRL, and Conditional Unsloth

This notebook explains and runs the training stack used in this project.
"""
        ),
        md(
            """
## Framework Coverage: PEFT, TRL, Unsloth

### PEFT
- Definition: Adapter-based parameter-efficient fine-tuning (LoRA).
- Why used: local memory-efficient tuning for 3B-scale models.
- Where used: all training backends in this project.

### TRL
- Definition: SFT-oriented training stack (`SFTTrainer`, `SFTConfig`).
- Why used: instruction-tuning ergonomics and explicit completion loss configuration.
- Where used: primary backend when available.

### Unsloth
- Definition: accelerated training runtime integrated with TRL workflows.
- Why used: potential speed/VRAM gains only under verified compatibility.
- Where used: optional backend behind strict compatibility gating; not forced.
"""
        ),
        md(
            technique_section(
                technique_name="Hybrid backend fine-tuning (HF/TRL/Unsloth) with guarded fallback",
                definition=(
                    "A backend router that selects the best available training path while preserving correctness and reproducibility."
                ),
                why_developed=(
                    "Training environments vary; a single rigid backend can fail due dependency/runtime incompatibilities."
                ),
                rag_limitations=(
                    "RAG-only systems avoid training but remain limited by prompt behavior. "
                    "This technique adds domain adaptation while retaining local deployability."
                ),
                workflow=(
                    "Resolve backend -> prepare instruction dataset -> attach PEFT adapters -> train -> evaluate -> persist adapters + metadata."
                ),
                components=(
                    "Backend resolver, PEFT config, TRL/HF trainers, optional Unsloth gate, MLflow logging, artifact writer."
                ),
                usage_guidance=(
                    "Use in production pipelines where reliability matters more than forcing one framework choice."
                ),
                pros_cons=(
                    "Advantages: resilient execution and explicit traceability. Disadvantages: extra orchestration complexity."
                ),
                comparison=(
                    "Compared with single-backend training, this design is more robust across environments. "
                    "Compared with standard RAG-only setups, it enables schema-specific behavioral adaptation."
                ),
                project_design=(
                    "Granite4.1 remains the primary model. Unsloth is only used when compatibility checks pass; otherwise fallback is logged."
                ),
            )
        ),
        code(
            """
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        code(
            """
from repo_query_gen.training import run_finetuning

# Fast profile for reproducible tutorial run.
out = run_finetuning("fast", backend="auto", allow_fallback=True)
out
"""
        ),
        code(
            """
import json
from pathlib import Path

run_dir = Path(out["run_dir"])
print(json.loads((run_dir / "training_metadata.json").read_text()))
print(json.loads((run_dir / "train_result.json").read_text()))
print(json.loads((run_dir / "eval_result.json").read_text()))
"""
        ),
        md(post_run_analysis_section("PEFT + TRL + conditional Unsloth")),
    ]
    write_notebook("05_qlora_finetuning_granite3b.ipynb", cells)


def build_06_eval() -> None:
    cells = [
        md(
            """
# 06 - Evaluation, LLM-as-a-Judge, and Error Analysis

This notebook performs objective and judge-based evaluation with per-task latency and retrieval diagnostics.
"""
        ),
        md(
            technique_section(
                technique_name="Multi-axis query generation evaluation",
                definition=(
                    "A unified evaluation framework combining exact/syntax/grounding scores, text metrics, retrieval diagnostics, latency, and judge assessments."
                ),
                why_developed=(
                    "Single metrics can hide critical failures (for example, fluent but non-executable queries)."
                ),
                rag_limitations=(
                    "Traditional RAG evaluation focuses answer relevance/faithfulness. Here we require executable query correctness and schema alignment."
                ),
                workflow=(
                    "Load predictions -> compute metrics -> aggregate by task/model -> plot distributions -> run judge analysis."
                ),
                components=(
                    "SQL/Cypher parsers, overlap metrics, schema-grounding checks, retrieval recall metrics, latency aggregators, plotting utilities."
                ),
                usage_guidance=(
                    "Use for release gating, regression testing, and deciding whether fine-tuning quality justifies operational cost."
                ),
                pros_cons=(
                    "Advantages: high diagnostic coverage. Disadvantages: more compute and longer evaluation cycles."
                ),
                comparison=(
                    "Compared with standard RAG metrics, this evaluation explicitly measures executable query behavior and schema retrieval quality."
                ),
                project_design=(
                    "Evaluation now reports retrieval table recall and latency p50/p95 alongside classic SQL/Cypher metrics."
                ),
            )
        ),
        code(
            """
from repo_query_gen.inference import run_batch_inference

# Build inference artifacts for evaluation.
run_batch_inference("fast", "baseline_granite", "baseline_granite")
run_batch_inference("fast", "baseline_qwen", "baseline_qwen")
try:
    run_batch_inference("fast", "finetuned", "finetuned")
except FileNotFoundError:
    print("Finetuned adapter missing, continuing baseline-only.")
"""
        ),
        code(
            """
from repo_query_gen.evaluation import run_evaluation_bundle

artifacts = run_evaluation_bundle("fast")
artifacts
"""
        ),
        code(
            """
import json
import pandas as pd
from pathlib import Path

eval_dir = Path("../artifacts/evaluation/fast")
metrics = pd.read_csv(eval_dir / "metrics_per_example.csv")
print(metrics.groupby(["label", "task"])[["exact_match", "syntax_success", "schema_grounding", "retrieval_table_recall", "generation_latency_ms"]].mean())

for summary_path in sorted(eval_dir.glob("summary_*.json")):
    print(summary_path.name)
    print(json.loads(summary_path.read_text()))
"""
        ),
        md(post_run_analysis_section("evaluation and metric interpretation")),
    ]
    write_notebook("06_evaluation_and_llm_judging.ipynb", cells)


def build_07_infer() -> None:
    cells = [
        md(
            """
# 07 - Inference Pipeline and Real Query Examples

This notebook demonstrates practical inference outputs for SQL/Cypher with retrieval metadata and validation.
"""
        ),
        md(
            technique_section(
                technique_name="Schema-aware inference with retrieval-aware prompt construction",
                definition=(
                    "Inference pipeline that optionally selects a relevant schema subset before generation and logs retrieval + latency metadata."
                ),
                why_developed=(
                    "Passing full schema context can increase latency and context noise, especially in large enterprise schemas."
                ),
                rag_limitations=(
                    "Traditional RAG retrieves document chunks; this retrieves schema entities needed for executable query planning."
                ),
                workflow=(
                    "Question -> schema selection (`full` or `lexical`) -> query generation -> syntax/grounding validation -> structured response."
                ),
                components=(
                    "Schema retriever, prompt builder, model interface, validators, structured output formatter."
                ),
                usage_guidance=(
                    "Use for user-facing assistants and agent pipelines where query traceability and low-latency responses matter."
                ),
                pros_cons=(
                    "Advantages: lower context noise and better observability. Disadvantages: retrieval heuristics may miss weakly referenced tables."
                ),
                comparison=(
                    "Compared with full-schema prompting, retrieval-aware prompts can reduce latency and sometimes improve grounding precision."
                ),
                project_design=(
                    "Current implementation offers lexical retrieval mode plus full-schema fallback, both exposed through settings."
                ),
            )
        ),
        code(
            """
from repo_query_gen.inference import infer_single

schema = '''
CREATE TABLE startups (
  id INTEGER,
  name TEXT,
  founded_year INTEGER,
  total_funding REAL
);
CREATE TABLE investors (
  id INTEGER,
  name TEXT,
  location TEXT
);
CREATE TABLE investments (
  startup_id INTEGER,
  investor_id INTEGER,
  round TEXT,
  amount REAL
);
'''

question = "Which startup has received the highest funding from a venture capitalist based in New York?"

sql_out = infer_single(question, schema, model_mode="ollama", task="sql")
cy_out = infer_single(question, schema, model_mode="ollama", task="cypher")

sql_out, cy_out
"""
        ),
        md(post_run_analysis_section("inference pipeline")),
    ]
    write_notebook("07_inference_pipeline_and_examples.ipynb", cells)


def build_08_neo4j() -> None:
    cells = [
        md(
            """
# 08 - Neo4j Demonstration

This notebook demonstrates graph loading and Cypher execution on a local Neo4j instance.
"""
        ),
        md(
            technique_section(
                technique_name="Executable graph-query validation with Neo4j",
                definition=(
                    "Operational validation step where generated Cypher is tested in a real graph database environment."
                ),
                why_developed=(
                    "Text-only evaluation is insufficient for graph query workflows used in GraphRAG and knowledge graph systems."
                ),
                rag_limitations=(
                    "Traditional RAG does not execute graph traversals. This step validates that generated Cypher maps to real graph operations."
                ),
                workflow=(
                    "Load processed data into Neo4j graph -> execute representative Cypher queries -> collect graph and traversal summaries."
                ),
                components=(
                    "Neo4j connector, graph loader, query executor, result serializer."
                ),
                usage_guidance=(
                    "Use in GraphRAG and knowledge-graph assistants before production deployment."
                ),
                pros_cons=(
                    "Advantages: real execution confidence. Disadvantages: requires local database runtime and data loading overhead."
                ),
                comparison=(
                    "Compared with static Cypher linting, live Neo4j execution validates real traversal behavior."
                ),
                project_design=(
                    "Outputs are written under `artifacts/neo4j_outputs/<profile>` to avoid container ownership issues."
                ),
            )
        ),
        md(
            """
## Setup

Start Neo4j using Docker Compose:

```bash
docker compose -f docker/docker-compose.neo4j.yml up -d
```
"""
        ),
        code(
            """
from repo_query_gen.neo4j_demo import run_neo4j_demo

out = run_neo4j_demo("fast")
out
"""
        ),
        code(
            """
import json
from pathlib import Path

neo_dir = Path("../artifacts/neo4j_outputs/fast")
print(json.load(open(neo_dir / "graph_summary.json")))
print(json.load(open(neo_dir / "demo_query_results.json")))
"""
        ),
        md(post_run_analysis_section("Neo4j execution validation")),
    ]
    write_notebook("08_neo4j_graph_demo.ipynb", cells)


def main() -> None:
    build_00_intro()
    build_01_foundations()
    build_02_dataset()
    build_03_cypher()
    build_04_baselines()
    build_05_training()
    build_06_eval()
    build_07_infer()
    build_08_neo4j()
    print("Notebook build completed.")


if __name__ == "__main__":
    main()
