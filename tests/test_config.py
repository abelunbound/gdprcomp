from pathlib import Path

import pytest
import yaml

from gdpr_gap.config import ConfigError, MissingSecretError, load_config

KEY = "test-key-not-a-real-secret"


def _valid_payload(criteria_path: Path) -> dict[str, object]:
    return {
        "seed": 42,
        "truncation": "error",
        "log_policy_text": False,
        "corpus_location": "https://github.com/citp/privacy-policy-historical",
        "paths": {
            "data_dir": "data",
            "corpus_cache_dir": "data/corpus",
            "result_cache_dir": "data/result-cache",
            "prompts_dir": "prompts",
            "criteria_path": str(criteria_path),
            "calibration_dir": "configs/calibration",
        },
        "model": {
            "provider": "claude",
            "name": "Claude Haiku 5.5",
            "id": "claude-haiku-5-5",
            "gold_id": "claude-sonnet-5-5",
        },
        "scoring": {
            "yes_score": 1,
            "somewhat_score": 0.5,
            "no_score": 0,
            "high_band_min": 80,
            "mid_band_min": 50,
        },
        "thresholds": {"retrieval_similarity": None},
        "weights": {"mode": "equal", "value": 1},
        "sampling": {
            "domain_suffix": ".co.uk",
            "pre_start": "2017-01-01",
            "pre_end": "2017-12-31",
            "buffer_start": "2018-01-01",
            "buffer_end": "2018-12-31",
            "post_start": "2019-01-01",
            "post_end": "2019-12-31",
        },
        "segmentation": {"passage_words": 120, "overlap_words": 20},
        "embedding": {
            "model_name": "nlpaueb/legal-bert-base-uncased",
            "revision": "15b570cbf88259610b082a167dacc190124f60f6",
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_repository_config_loads() -> None:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    assert config.seed == 42
    assert config.paths.result_cache_dir == "data/result-cache"
    assert config.paths.calibration_dir == "configs/calibration"
    assert config.thresholds.retrieval_similarity is None
    assert config.model.id == "claude-haiku-5-5"
    assert config.model.gold_id == "claude-sonnet-5-5"
    assert config.model.max_input_tokens == 200_000
    assert config.model.max_output_tokens == 8000
    assert config.weights.mode == "equal"
    assert config.sampling.domain_suffix == ".co.uk"
    assert config.truncation == "error"
    assert config.embedding.model_name == "nlpaueb/legal-bert-base-uncased"
    assert config.embedding.revision == "15b570cbf88259610b082a167dacc190124f60f6"
    assert config.log_policy_text is False
    assert config.api_key.get_secret_value() == KEY
    assert KEY not in repr(config)


def test_missing_api_key_names_the_variable(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    path = _write_config(tmp_path, _valid_payload(criteria))

    with pytest.raises(MissingSecretError) as exc_info:
        load_config(path, environ={}, load_env_file=False)

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert ".env" in message
    assert "claude" in message


def test_openai_provider_requires_its_own_key(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    payload = _valid_payload(criteria)
    payload["model"] = {
        "provider": "openai",
        "name": "GPT",
        "id": "gpt-4o-mini",
        "gold_id": "gpt-4.1",
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(MissingSecretError, match="OPENAI_API_KEY"):
        load_config(path, environ={"ANTHROPIC_API_KEY": KEY}, load_env_file=False)


def test_blank_api_key_is_missing(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    path = _write_config(tmp_path, _valid_payload(criteria))

    with pytest.raises(MissingSecretError):
        load_config(path, environ={"ANTHROPIC_API_KEY": "   "}, load_env_file=False)


def test_invalid_config_names_the_field(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    payload = _valid_payload(criteria)
    payload["seed"] = -1
    payload["weights"] = {"mode": "equal", "value": 0}
    path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigError) as exc_info:
        load_config(path, environ={"ANTHROPIC_API_KEY": KEY}, load_env_file=False)

    message = str(exc_info.value)
    assert "seed" in message
    assert "weights.value" in message
    assert not isinstance(exc_info.value, MissingSecretError)


def test_secret_in_the_config_file_is_rejected(tmp_path: Path) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text("{}", encoding="utf-8")
    payload = _valid_payload(criteria)
    payload["api_key"] = KEY
    path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigError, match="api_key"):
        load_config(path, environ={"ANTHROPIC_API_KEY": KEY}, load_env_file=False)


def test_missing_criteria_file_is_a_config_error(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path / "missing.json")
    path = _write_config(tmp_path, payload)

    with pytest.raises(ConfigError, match="paths.criteria_path"):
        load_config(path, environ={"ANTHROPIC_API_KEY": KEY}, load_env_file=False)


def test_missing_config_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(
            tmp_path / "absent.yaml",
            environ={"ANTHROPIC_API_KEY": KEY},
            load_env_file=False,
        )
