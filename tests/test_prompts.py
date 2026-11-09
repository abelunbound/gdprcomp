from pathlib import Path

import pytest

from gdpr_gap.prompts import PromptError, PromptRegistry

ORIGINAL = (
    "Judge this policy.\nCriterion:\n{condition}\nId: {criterion_id}\nPolicy:\n{policy_text}\n"
)
CHANGED = ORIGINAL.replace("Judge this policy.", "Judge this policy again.")


def test_repository_prompt_renders_without_rewriting_the_policy() -> None:
    registry = PromptRegistry("prompts")
    record = registry.load("whole_policy_judge")
    policy = "We ask for consent. {condition}"

    rendered = registry.render(
        record.prompt_id,
        record.version,
        criterion_id="consent_mechanisms",
        condition="The policy should say consent is requested.",
        policy_text=policy,
    )

    assert record.version == "v1"
    assert rendered.prompt_id == "whole_policy_judge"
    assert rendered.version == "v1"
    assert policy in rendered.text
    assert rendered.text.count("The policy should say consent is requested.") == 1


def test_changing_a_prompt_keeps_the_old_version(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path)
    first = registry.save("whole_policy_judge", ORIGINAL)
    same = registry.save("whole_policy_judge", ORIGINAL)
    second = registry.save("whole_policy_judge", CHANGED)

    assert first.version == same.version == "v1"
    assert second.version == "v2"
    assert registry.load("whole_policy_judge", "v1").text == ORIGINAL
    assert registry.load("whole_policy_judge").text == CHANGED
    assert (tmp_path / "whole_policy_judge" / "v1.txt").read_text(encoding="utf-8") == ORIGINAL
    assert list(registry.versions("whole_policy_judge")) == ["v1", "v2"]


def test_bad_prompt_id_and_missing_placeholder_fail(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path)

    with pytest.raises(PromptError, match="underscores"):
        registry.save("../secrets", ORIGINAL)

    registry.save("whole_policy_judge", "no placeholders")
    with pytest.raises(PromptError, match="criterion_id"):
        registry.render(
            "whole_policy_judge",
            "v1",
            criterion_id="consent_mechanisms",
            condition="A condition.",
            policy_text="A policy.",
        )

