import json
from pathlib import Path

import pytest

from gdpr_gap.config import load_config
from gdpr_gap.llm import LLMClient, RawCompletion
from gdpr_gap.manifest import ManifestError, load_run, record_run, reproduce, run_inputs
from gdpr_gap.orchestrator import PolicyDocument

KEY = "test-key-not-a-real-secret"
RUN_ID = "11111111-1111-4111-8111-111111111111"
POLICY = "We keep records of processing activities."


class ScriptedTransport:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        del model_id, timeout
        self.calls += 1
        if "suggestion must be" in prompt:
            body = {
                "criterion_id": _criterion_id(prompt),
                "suggestion": "Name the purposes of processing.",
                "evidence_quotes": [POLICY],
            }
            return RawCompletion(text=json.dumps(body), input_tokens=4, output_tokens=4)
        verdict = "Somewhat" if self.calls == 1 else "Yes"
        quotes = [POLICY] if verdict == "Somewhat" else []
        body = {
            "criterion_id": _criterion_id(prompt),
            "verdict": verdict,
            "justification": "The policy states the condition.",
            "evidence_quotes": quotes,
        }
        return RawCompletion(text=json.dumps(body), input_tokens=3, output_tokens=3)


def _criterion_id(prompt: str) -> str:
    marker = "criterion_id must be "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(".", maxsplit=1)[0]


def test_manifest_round_trip_reproduces_verdicts_feedback_and_settings(tmp_path: Path) -> None:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    document = PolicyDocument(policy_id="745", text=POLICY)
    manifest = record_run(
        [document],
        config,
        client=LLMClient(config, ScriptedTransport()),
        directory=tmp_path,
        run_id=RUN_ID,
    )
    path = tmp_path / f"{RUN_ID}.json"
    text = path.read_text(encoding="utf-8")
    loaded = load_run(path)
    restored = reproduce(loaded)
    review = restored[0].review

    assert loaded.model_dump(mode="json") == manifest.model_dump(mode="json")
    assert loaded.run_id == RUN_ID
    assert loaded.settings.seed == 42
    assert loaded.settings.method == "whole_policy_llm"
    assert loaded.settings.model_id == "claude-haiku-5-5"
    assert loaded.settings.embedding_revision == "15b570cbf88259610b082a167dacc190124f60f6"
    assert loaded.calibration is None
    assert {item.prompt_id: item.version for item in loaded.prompts} == {
        "feedback": "v1",
        "whole_policy_judge": "v1",
    }
    assert {item.name for item in loaded.libraries} >= {"gdpr-gap", "pydantic", "dash"}
    assert loaded.python_version.startswith("3.12")
    assert run_inputs(loaded) == [document]
    assert loaded.corpus[0].sha256
    assert review is not None
    assert review.judgement.verdicts[0].judgement.verdict == "Somewhat"
    assert review.judgement.verdicts[0].judgement.evidence_quotes == [POLICY]
    assert review.judgement.verdicts[0].usage.input_tokens == 3
    assert review.judgement.verdicts[0].usage.cost_usd is None
    assert loaded.results[0].cost_usd is None
    assert loaded.results[0].runtime_seconds >= 0
    assert review.feedback[0].suggestion == "Name the purposes of processing."
    assert review.feedback[0].evidence_quotes == [POLICY]
    assert KEY not in text
    assert "api_key" not in text

    with pytest.raises(ManifestError, match="overwrite"):
        record_run(
            [document],
            config,
            client=LLMClient(config, ScriptedTransport()),
            directory=tmp_path,
            run_id=RUN_ID,
        )
    assert path.read_text(encoding="utf-8") == text


def test_empty_run_and_bad_id_write_nothing(tmp_path: Path) -> None:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    with pytest.raises(ManifestError, match="no policies"):
        record_run([], config, directory=tmp_path, run_id=RUN_ID)
    with pytest.raises(ManifestError, match="UUID"):
        record_run(
            [PolicyDocument(policy_id="745", text=POLICY)],
            config,
            client=LLMClient(config, ScriptedTransport()),
            directory=tmp_path,
            run_id="not-a-uuid",
        )
    assert list(tmp_path.iterdir()) == []
