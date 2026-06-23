"""Configuration models for the Text-to-SQL and Text-to-Cypher project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ProfileName = Literal["fast", "tutorial", "full"]
PROFILE_CHOICES: tuple[ProfileName, ...] = ("fast", "tutorial", "full")


class ProfileConfig(BaseModel):
    """Tunable profile for fast, tutorial, or full execution modes."""

    dataset_rows: int | None = None
    train_rows: int | None = None
    val_rows: int | None = None
    test_rows: int | None = None
    max_train_steps: int = 120
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_len: int = 1024
    eval_sample_size: int = 400
    spider_eval_rows: int = 500
    train_max_examples: int | None = None
    val_max_examples: int | None = None


class Settings(BaseSettings):
    """Environment-driven runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset_name: str = "Clinton/Text-to-sql-v1"

    granite_model: str = "granite4.1:3b"
    qwen_model: str = "qwen3.5:4b"
    embed_model: str = "qwen3-embedding:4b"
    ollama_timeout_seconds: int = 600
    ollama_max_retries: int = 2
    ollama_retry_backoff_seconds: float = 2.0

    hf_granite_base_model: str = "ibm-granite/granite-4.1-3b"
    hf_qwen_base_model: str = "Qwen/Qwen3.5-4B"

    training_backend: Literal["auto", "hf", "trl", "unsloth"] = "auto"
    allow_backend_fallback: bool = True

    # Unsloth is opt-in and only used when compatibility checks pass.
    unsloth_model_name: str = "unsloth/granite-4.0-h-micro"

    # Schema retrieval mode for prompt construction during inference/baselines.
    schema_retrieval_mode: Literal["full", "lexical"] = "lexical"
    schema_retrieval_top_k: int = 6

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    enable_cypher_refinement: bool = False

    seed: int = 42

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to project root.

        Args:
            *parts: Relative path chunks.

        Returns:
            Absolute project path.
        """

        return self.project_root.joinpath(*parts)

    @property
    def raw_data_dir(self) -> Path:
        return self.path("data", "raw")

    @property
    def processed_data_dir(self) -> Path:
        return self.path("data", "processed")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("artifacts")


def load_profile(profile: ProfileName, config_path: Path | None = None) -> ProfileConfig:
    """Load a named profile from YAML config.

    Args:
        profile: Profile name.
        config_path: Optional alternate config path.

    Returns:
        Parsed profile config.

    Example:
        >>> cfg = load_profile("fast")
        >>> cfg.max_train_steps > 0
        True
    """

    cfg_path = config_path or Settings().path("configs", "profiles.yaml")
    with cfg_path.open("r", encoding="utf-8") as fp:
        payload: dict[str, Any] = yaml.safe_load(fp)
    return ProfileConfig.model_validate(payload["profiles"][profile])


__all__ = ["Settings", "ProfileConfig", "ProfileName", "PROFILE_CHOICES", "load_profile"]
