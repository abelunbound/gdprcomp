"""Load and validate application configuration.

Non-secret settings come from a YAML file. API keys come only from the
environment, so they are never stored in the config file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

SECRET_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class MissingSecretError(ConfigError):
    """Raised when the API key for the selected provider is missing."""


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(min_length=1)
    corpus_cache_dir: str = Field(min_length=1)
    result_cache_dir: str = Field(min_length=1)
    prompts_dir: str = Field(min_length=1)
    criteria_path: str = Field(min_length=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["claude", "openai"]
    name: str = Field(min_length=1)
    id: str = Field(min_length=1)
    gold_id: str = Field(min_length=1)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=3, ge=1, le=6)
    max_output_tokens: int = Field(default=800, ge=1, le=8000)
    max_input_tokens: int = Field(default=200_000, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def gold_model_differs(self) -> ModelConfig:
        if self.gold_id == self.id:
            raise ValueError("gold_id must be a different model from id")
        return self


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yes_score: float = Field(gt=0, le=1)
    somewhat_score: float = Field(gt=0, lt=1)
    no_score: float = 0
    high_band_min: float = Field(gt=0, le=100)
    mid_band_min: float = Field(ge=0, lt=100)

    @model_validator(mode="after")
    def scores_and_bands(self) -> ScoringConfig:
        if not self.yes_score > self.somewhat_score > self.no_score:
            raise ValueError("yes_score, somewhat_score, and no_score must be strictly decreasing")
        if self.high_band_min <= self.mid_band_min:
            raise ValueError("high_band_min must be greater than mid_band_min")
        return self


class ThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_similarity: float | None = Field(default=None, ge=0, le=1)


class EmbeddingConfig(BaseModel):
    """Legal-BERT settings for comparator methods. The whole-policy judge does not use them."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(default=512, ge=8, le=512)
    batch_size: int = Field(default=8, ge=1, le=64)


class SegmentationConfig(BaseModel):
    """Passage settings for comparator methods. The whole-policy judge does not use them."""

    model_config = ConfigDict(extra="forbid")

    passage_words: int = Field(ge=1)
    overlap_words: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_is_shorter_than_a_passage(self) -> SegmentationConfig:
        if self.overlap_words >= self.passage_words:
            raise ValueError("overlap_words must be less than passage_words")
        return self


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["equal"]
    value: float = Field(gt=0)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_suffix: str = Field(min_length=2)
    pre_start: date
    pre_end: date
    buffer_start: date
    buffer_end: date
    post_start: date
    post_end: date

    @field_validator("domain_suffix")
    @classmethod
    def suffix_starts_with_dot(cls, value: str) -> str:
        if not value.startswith("."):
            raise ValueError("domain_suffix must start with '.'")
        return value.lower()

    @model_validator(mode="after")
    def windows_do_not_overlap(self) -> SamplingConfig:
        if self.pre_start > self.pre_end:
            raise ValueError("pre_start must be on or before pre_end")
        if self.buffer_start > self.buffer_end:
            raise ValueError("buffer_start must be on or before buffer_end")
        if self.post_start > self.post_end:
            raise ValueError("post_start must be on or before post_end")
        if self.pre_end >= self.buffer_start:
            raise ValueError("pre window must end before the buffer")
        if self.buffer_end >= self.post_start:
            raise ValueError("buffer must end before the post window")
        return self


class FileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    truncation: Literal["error"]
    corpus_location: str = Field(min_length=1)
    paths: PathsConfig
    model: ModelConfig
    scoring: ScoringConfig
    thresholds: ThresholdsConfig
    weights: WeightsConfig
    sampling: SamplingConfig
    segmentation: SegmentationConfig
    embedding: EmbeddingConfig
    log_policy_text: bool = False


class AppConfig(FileConfig):
    api_key: SecretStr


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    load_env_file: bool = True,
) -> AppConfig:
    """Load config from YAML and attach the provider API key from the environment."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    file_config = _read_file_config(config_path)
    _require_existing_file(Path(file_config.paths.criteria_path), "paths.criteria_path")
    secret_name = SECRET_ENV_VARS[file_config.model.provider]
    secret = _environment(environ, load_env_file).get(secret_name, "").strip()
    if not secret:
        raise MissingSecretError(
            f"{secret_name} is not set. Add it to .env or export it in the environment. "
            f"It is required when model.provider is {file_config.model.provider!r}."
        )
    return AppConfig.model_validate({**file_config.model_dump(), "api_key": secret})


def _read_file_config(path: Path) -> FileConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}. Expected {DEFAULT_CONFIG_PATH}.")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Invalid configuration in {path}: the top level must be a mapping.")
    try:
        return FileConfig.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc


def _environment(environ: Mapping[str, str] | None, load_env_file: bool) -> dict[str, str]:
    if environ is not None:
        return {key: value for key, value in environ.items()}
    values = dict(os.environ)
    if not load_env_file:
        return values
    for key, value in dotenv_values(Path.cwd() / ".env").items():
        if value is not None and key not in values:
            values[key] = value
    return values


def _require_existing_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise ConfigError(f"Invalid configuration: {field_name} does not exist: {path}")


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid configuration in {path}:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(file)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)
