import json

import pytest

from gdpr_gap.config import AppConfig, load_config
from gdpr_gap.criteria import load_criteria
from gdpr_gap.feedback import evaluate_policy, suggest_fixes
from gdpr_gap.llm import LLMClient, ModelError, RawCompletion

KEY = "test-key-not-a-real-secret"


def _config() -> AppConfig:
    return load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )


class ScriptedTransport:
    def __init__(self, gaps: dict[str, str], *, reject_feedback: bool = False) -> None:
        self.prompts: list[str] = []
        self.gaps = gaps
        self.reject_feedback = reject_feedback

    def complete(self, *, model_id: str, prompt: str, timeout: float) -> RawCompletion:
        self.prompts.append(prompt)
        criterion_id = _criterion_id(prompt)
        if prompt.startswith("Suggest a fix"):
            body: dict[str, object] = {
                "criterion_id": criterion_id,
                "suggestion": "State this more precisely.",
                "evidence_quotes": ["w0"],
            }
            if self.reject_feedback:
                body["verdict"] = "Yes"
        else:
            body = {
                "criterion_id": criterion_id,
                "verdict": self.gaps.get(criterion_id, "Yes"),
                "justification": "Judged from the full policy.",
                "evidence_quotes": ["w0"],
            }
        return RawCompletion(text=json.dumps(body), input_tokens=4, output_tokens=4)


def _criterion_id(prompt: str) -> str:
    marker = "criterion_id must be "
    start = prompt.index(marker) + len(marker)
    return prompt[start:].split(".", maxsplit=1)[0]


def _gaps(config: AppConfig) -> dict[str, str]:
    ordered = [
        item.id
        for item in sorted(
            load_criteria(config.paths.criteria_path).criteria,
            key=lambda item: item.display_order,
        )
    ]
    return {ordered[0]: "Somewhat", ordered[1]: "No"}


def test_disabling_feedback_leaves_verdicts_and_scores_unchanged() -> None:
    config = _config()
    gaps = _gaps(config)
    policy = "We ask for consent before sending marketing email."
    plain_transport = ScriptedTransport(gaps)
    feedback_transport = ScriptedTransport(gaps)

    plain = evaluate_policy(policy, config, LLMClient(config, plain_transport), feedback=False)
    reviewed = evaluate_policy(policy, config, LLMClient(config, feedback_transport), feedback=True)

    assert plain.judgement.model_dump_json() == reviewed.judgement.model_dump_json()
    assert plain.score.model_dump_json() == reviewed.score.model_dump_json()
    assert plain.feedback == []
    assert [item.criterion_id for item in reviewed.feedback] == list(gaps)
    assert [item.suggestion for item in reviewed.feedback] == ["State this more precisely."] * 2
    assert not any(prompt.startswith("Suggest a fix") for prompt in plain_transport.prompts)
    feedback_prompts = [
        prompt for prompt in feedback_transport.prompts if prompt.startswith("Suggest a fix")
    ]
    assert len(feedback_prompts) == len(gaps)
    assert len(plain_transport.prompts) == 22


def test_feedback_that_includes_a_verdict_does_not_change_the_judgement() -> None:
    config = _config()
    gaps = _gaps(config)
    policy = "We ask for consent before sending marketing email."
    transport = ScriptedTransport(gaps, reject_feedback=True)
    client = LLMClient(config, transport)
    reviewed = evaluate_policy(policy, config, client, feedback=False)
    before = reviewed.judgement.model_dump_json()

    with pytest.raises(ModelError, match="feedback schema"):
        suggest_fixes(policy, reviewed.judgement, config, client)

    assert reviewed.judgement.model_dump_json() == before
    assert (
        reviewed.score.model_dump_json()
        == evaluate_policy(
            policy, config, LLMClient(config, ScriptedTransport(gaps)), feedback=False
        ).score.model_dump_json()
    )
