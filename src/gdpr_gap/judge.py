"""Judge a whole privacy policy against every criterion.

The full policy text is sent once per criterion. Passage retrieval is not
consulted, and a criterion is never marked No because no passage was found.
A prompt over the configured input limit raises JudgeError. The text is not
shortened and is not sent.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gdpr_gap.config import AppConfig
from gdpr_gap.criteria import load_criteria
from gdpr_gap.llm import LLMClient, ModelResult
from gdpr_gap.logging import log_event
from gdpr_gap.prompts import PromptRecord, PromptRegistry
from gdpr_gap.result_cache import ResultCache, make_cache_key

WHOLE_POLICY_PROMPT_ID = "whole_policy_judge"


class JudgeError(Exception):
    """Raised when a whole-policy judgement cannot be completed."""


class PolicyJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(pattern=r"^[a-z0-9_]+$")
    prompt_version: str = Field(pattern=r"^v[1-9][0-9]*$")
    model_id: str = Field(min_length=1)
    verdicts: list[ModelResult] = Field(min_length=22, max_length=22)

    @model_validator(mode="after")
    def one_distinct_criterion_each(self) -> PolicyJudgement:
        ids = [item.judgement.criterion_id for item in self.verdicts]
        if len(set(ids)) != len(ids):
            raise ValueError("each verdict must use a distinct criterion_id")
        for item in self.verdicts:
            if item.prompt_id != self.prompt_id or item.prompt_version != self.prompt_version:
                raise ValueError("every verdict must record the prompt used for the run")
            if item.model_id != self.model_id:
                raise ValueError("every verdict must use the configured model")
        return self


def judge_policy(
    policy_text: str,
    config: AppConfig,
    client: LLMClient,
    *,
    cache: ResultCache | None = None,
    logger: logging.Logger | None = None,
) -> PolicyJudgement:
    """Return one verdict for each of the 22 criteria. The policy text is not shortened."""
    if not policy_text.strip():
        raise JudgeError("Cannot judge an empty policy.")
    prompt = PromptRegistry(config.paths.prompts_dir).load(WHOLE_POLICY_PROMPT_ID)
    catalog = load_criteria(config.paths.criteria_path)
    ordered = sorted(catalog.criteria, key=lambda item: item.display_order)
    verdicts = [
        _verdict(
            policy_text=policy_text,
            criterion_id=criterion.id,
            condition=criterion.condition,
            prompt=prompt,
            config=config,
            client=client,
            cache=cache,
            logger=logger,
        )
        for criterion in ordered
    ]
    judgement = PolicyJudgement(
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        model_id=config.model.id,
        verdicts=verdicts,
    )
    if logger is not None:
        log_event(
            logger,
            "policy_judged",
            prompt_id=judgement.prompt_id,
            prompt_version=judgement.prompt_version,
            model_id=judgement.model_id,
            criterion_count=len(judgement.verdicts),
            word_count=len(policy_text.split()),
        )
    return judgement


def estimated_input_tokens(text: str) -> int:
    """Estimate input tokens from characters. This is not the provider tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _require_within_limit(
    text: str,
    policy_text: str,
    config: AppConfig,
    logger: logging.Logger | None,
) -> None:
    tokens = estimated_input_tokens(text)
    limit = config.model.max_input_tokens
    if tokens <= limit:
        return
    if logger is not None:
        log_event(
            logger,
            "policy_over_limit",
            model_id=config.model.id,
            input_tokens=tokens,
            max_input_tokens=limit,
            word_count=len(policy_text.split()),
        )
    raise JudgeError(
        "Policy exceeds the model input limit and was not sent: "
        f"{tokens} estimated input tokens > {limit}."
    )


def _verdict(
    *,
    policy_text: str,
    criterion_id: str,
    condition: str,
    prompt: PromptRecord,
    config: AppConfig,
    client: LLMClient,
    cache: ResultCache | None,
    logger: logging.Logger | None,
) -> ModelResult:
    rendered = PromptRegistry(config.paths.prompts_dir).render(
        prompt.prompt_id,
        prompt.version,
        criterion_id=criterion_id,
        condition=condition,
        policy_text=policy_text,
    )
    if policy_text not in rendered.text:
        raise JudgeError("The prompt does not contain the full policy text.")
    _require_within_limit(rendered.text, policy_text, config, logger)

    def produce() -> ModelResult:
        result = client.judge(
            rendered.text,
            prompt_id=rendered.prompt_id,
            prompt_version=rendered.version,
            model_id=config.model.id,
        )
        if result.judgement.criterion_id != criterion_id:
            raise JudgeError(
                f"Model returned {result.judgement.criterion_id}, expected {criterion_id}."
            )
        return result

    if cache is None:
        return produce()
    key = make_cache_key(
        policy_text=policy_text,
        criterion_id=criterion_id,
        prompt_id=rendered.prompt_id,
        prompt_version=rendered.version,
        model_id=config.model.id,
    )
    return cache.judgement(key, produce)
