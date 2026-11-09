"""Single-policy Dash page.

The method is the whole-policy LLM. Condition met and relevant text found are
separate lines. The not-legal-advice notice is always visible. A status line
above the results advances after each model call.
"""

from __future__ import annotations

import dash
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from gdpr_gap.config import AppConfig, load_config
from gdpr_gap.criteria import Criterion, load_criteria
from gdpr_gap.embeddings import EmbeddingClient, TextEncoder, load_legal_bert
from gdpr_gap.feedback import (
    CriterionFeedback,
    PolicyReview,
    check_citations,
    score_verdicts,
    suggest_fix,
)
from gdpr_gap.gaps import is_gap
from gdpr_gap.judge import WHOLE_POLICY_PROMPT_ID, PolicyJudgement, _verdict
from gdpr_gap.legal_bert import LegalBertBaseline, LegalBertCriterion
from gdpr_gap.llm import LLMClient, ModelResult, build_transport
from gdpr_gap.orchestrator import run_policy, whole_policy_result
from gdpr_gap.prompts import PromptRegistry
from gdpr_gap.single_policy import (
    NOT_LEGAL_ADVICE,
    CriterionDisplay,
    SinglePolicyDisplay,
    build_display,
    sample_review,
)

app = dash.Dash(__name__)
app.title = "Single policy"
app.layout = html.Main(
    [
        html.H1("Single policy"),
        html.P(NOT_LEGAL_ADVICE, id="notice"),
        html.P("Method: whole-policy LLM", id="method"),
        dcc.Textarea(
            id="policy-text",
            placeholder="Paste a privacy policy",
            style={"width": "100%", "height": "160px"},
        ),
        html.Button("Judge", id="judge", n_clicks=0),
        html.Button("Show sample", id="show-sample", n_clicks=0),
        html.Button("Show retrieval signal", id="show-retrieval", n_clicks=0),
        html.P(id="progress"),
        html.Div(
            html.P(
                "Paste a policy and press Judge, show the sample, or show the retrieval signal."
            ),
            id="result",
        ),
        dcc.Store(id="job"),
        dcc.Interval(id="tick", interval=1000, disabled=True),
    ]
)


def progress_message(kind: str, finished: int, total: int, name: str) -> str:
    """Name the model call that is about to start. `finished` is how many are already done."""
    step = finished + 1
    if kind == "judge":
        return f"Judging {step} of {total}: {name}"
    return f"Writing feedback {step} of {total}: {name}"


WHOLE_POLICY_METHOD = "Method: whole-policy LLM"
RETRIEVAL_METHOD = "Method: legal-bert only"
_encoder: TextEncoder | None = None


def render_retrieval(baseline: LegalBertBaseline, names: dict[str, str]) -> list[Component]:
    """Show similarity ranks. This is not a verdict and has no condition-met line."""
    header: list[Component] = [
        html.P("Signal: baseline/retrieval signal", id="signal"),
        html.P("This is a retrieval signal, not an LLM verdict."),
        html.P(f"Embedding: {baseline.model_name} @ {baseline.model_version}"),
    ]
    rows = [
        _render_retrieval_row(index, row, names)
        for index, row in enumerate(baseline.criteria, start=1)
    ]
    return header + rows


def _render_retrieval_row(index: int, row: LegalBertCriterion, names: dict[str, str]) -> Component:
    ranked = [html.Li(f"{passage.similarity:.3f} — {passage.text}") for passage in row.passages]
    return html.Article(
        [
            html.H3(f"{index}. {names[row.criterion_id]}"),
            html.P(f"Best similarity: {row.best_similarity:.3f}"),
            html.P("Ranked passages:"),
            html.Ol(ranked),
        ],
        id=f"retrieval-{row.criterion_id}",
    )


def render_result(display: SinglePolicyDisplay) -> list[Component]:
    """Turn a display model into Dash components."""
    header = [
        html.P(f"Score: {display.score:.1f} / 100", id="score"),
        html.P(f"220-point score: {display.points:.1f} / 220", id="points"),
        html.P(f"Band: {display.band}", id="band"),
    ]
    return header + [render_criterion(row) for row in display.criteria]


def render_criterion(row: CriterionDisplay) -> Component:
    """Render one criterion with condition, cited text, and gap feedback kept apart."""
    relevant: list[Component] | Component
    if row.relevant_text:
        relevant = html.Ul([html.Li(quote) for quote in row.relevant_text])
    else:
        relevant = html.P("None cited")
    children: list[Component] = [
        html.H3(f"{row.order}. {row.name}"),
        html.P(["Condition met: ", html.Strong(row.condition_met)]),
        html.P("Relevant text found:"),
        relevant,
    ]
    if row.feedback is not None:
        children.extend([html.P("Feedback:"), html.P(row.feedback)])
    return html.Article(children, id=f"criterion-{row.criterion_id}")


@callback(
    Output("job", "data", allow_duplicate=True),
    Output("progress", "children", allow_duplicate=True),
    Output("result", "children", allow_duplicate=True),
    Output("judge", "disabled", allow_duplicate=True),
    Output("show-sample", "disabled", allow_duplicate=True),
    Output("show-retrieval", "disabled", allow_duplicate=True),
    Output("tick", "disabled", allow_duplicate=True),
    Output("method", "children", allow_duplicate=True),
    Input("judge", "n_clicks"),
    Input("show-sample", "n_clicks"),
    State("policy-text", "value"),
    prevent_initial_call=True,
)
def start_run(
    judge_clicks: int,
    sample_clicks: int,
    policy_text: str | None,
) -> tuple[object, object, object, bool, bool, bool, bool, str]:
    """Show the status line immediately, before any model call starts."""
    del judge_clicks, sample_clicks
    try:
        if ctx.triggered_id == "show-sample":
            sample = _sample()
            return (*sample, sample[3], True, WHOLE_POLICY_METHOD)
        started = _start(policy_text)
    except Exception as exc:
        message = html.P(f"Could not judge the policy: {exc}")
        return None, "", message, False, False, False, True, WHOLE_POLICY_METHOD
    return (*started, started[3], started[0] is None, WHOLE_POLICY_METHOD)


@callback(
    Output("job", "data", allow_duplicate=True),
    Output("progress", "children", allow_duplicate=True),
    Output("result", "children", allow_duplicate=True),
    Output("judge", "disabled", allow_duplicate=True),
    Output("show-sample", "disabled", allow_duplicate=True),
    Output("show-retrieval", "disabled", allow_duplicate=True),
    Input("tick", "n_intervals"),
    State("job", "data"),
    running=[(Output("tick", "disabled", allow_duplicate=True), True, False)],
    prevent_initial_call=True,
)
def continue_run(
    tick: int,
    job: dict[str, object] | None,
) -> tuple[object, object, object, bool, bool, bool]:
    """Judge one criterion or write one suggestion, then update the status line."""
    del tick
    if not job:
        raise PreventUpdate
    try:
        advanced = _advance(job)
    except Exception as exc:
        message = html.P(f"Could not judge the policy: {exc}")
        return None, "", message, False, False, False
    return (*advanced, advanced[3])


@callback(
    Output("result", "children", allow_duplicate=True),
    Output("progress", "children", allow_duplicate=True),
    Output("method", "children", allow_duplicate=True),
    Input("show-retrieval", "n_clicks"),
    State("policy-text", "value"),
    running=[
        (Output("show-retrieval", "disabled", allow_duplicate=True), True, False),
        (Output("judge", "disabled", allow_duplicate=True), True, False),
        (Output("show-sample", "disabled", allow_duplicate=True), True, False),
    ],
    prevent_initial_call=True,
)
def show_retrieval(
    clicks: int,
    policy_text: str | None,
) -> tuple[object, str, str]:
    """Score topical similarity. This callback never calls the LLM judge."""
    del clicks
    if policy_text is None or not policy_text.strip():
        return html.P("Paste a policy first."), "", RETRIEVAL_METHOD
    try:
        config = load_config()
        embeddings = EmbeddingClient(config.embedding, _cached_encoder(config))
        recorded = run_policy(
            policy_text,
            config,
            "legal_bert_only",
            policy_id="pasted",
            embeddings=embeddings,
        )
        if recorded.retrieval is None:
            raise RuntimeError("legal_bert_only result has no retrieval signal")
        names = {item.id: item.name for item in _ordered(config)}
    except Exception as exc:
        message = html.P(f"Could not score the retrieval signal: {exc}")
        return message, "", RETRIEVAL_METHOD
    return render_retrieval(recorded.retrieval, names), "", RETRIEVAL_METHOD


def _cached_encoder(config: AppConfig) -> TextEncoder:
    global _encoder
    settings = config.embedding
    loaded = _encoder
    if loaded is None or loaded.name != settings.model_name or loaded.version != settings.revision:
        loaded = load_legal_bert(settings, confirm=True)
        _encoder = loaded
    return loaded


@callback(
    Output("tick", "disabled", allow_duplicate=True),
    Input("job", "data"),
    prevent_initial_call=True,
)
def stop_timer(job: dict[str, object] | None) -> bool:
    """Stop the timer once the run has cleared its job."""
    if job:
        raise PreventUpdate
    return True


def _sample() -> tuple[None, str, list[Component] | Component, bool, bool]:
    try:
        config = load_config()
        display = build_display(sample_review(config), config)
    except Exception as exc:
        return None, "", html.P(f"Could not judge the policy: {exc}"), False, False
    return None, "", render_result(display), False, False


def _start(policy_text: str | None) -> tuple[object, object, object, bool, bool]:
    if policy_text is None or not policy_text.strip():
        return None, "", html.P("Paste a policy first."), False, False
    config = load_config()
    first = _ordered(config)[0]
    job = {"policy": policy_text, "phase": "judge", "index": 0, "verdicts": [], "feedback": []}
    return job, progress_message("judge", 0, 22, first.name), [], True, True


def _advance(job: dict[str, object]) -> tuple[object, object, object, bool, bool]:
    config = load_config()
    client = LLMClient(config, build_transport(config))
    criteria = _ordered(config)
    policy = str(job["policy"])
    verdicts = _records(job["verdicts"])
    feedback = _records(job["feedback"])
    index = int(str(job["index"]))
    if job["phase"] == "judge":
        prompt = PromptRegistry(config.paths.prompts_dir).load(WHOLE_POLICY_PROMPT_ID)
        criterion = criteria[index]
        result = _verdict(
            policy_text=policy,
            criterion_id=criterion.id,
            condition=criterion.condition,
            prompt=prompt,
            config=config,
            client=client,
            cache=None,
            logger=None,
        )
        verdicts.append(result.model_dump(mode="json"))
        nxt = index + 1
        if nxt < len(criteria):
            updated = {**job, "index": nxt, "verdicts": verdicts}
            message = progress_message("judge", nxt, len(criteria), criteria[nxt].name)
            return updated, message, no_update, True, True
        gaps = [item for item in verdicts if is_gap(str(_judgement(item)["verdict"]))]
        if gaps:
            name = _name(criteria, str(_judgement(gaps[0])["criterion_id"]))
            updated = {**job, "phase": "feedback", "index": 0, "verdicts": verdicts, "feedback": []}
            return updated, progress_message("feedback", 0, len(gaps), name), no_update, True, True
        return _finish(config, policy, verdicts, [])
    gaps = [item for item in verdicts if is_gap(str(_judgement(item)["verdict"]))]
    current = _judgement(gaps[index])
    fix = suggest_fix(
        policy,
        str(current["criterion_id"]),
        str(current["verdict"]),
        str(current["justification"]),
        config,
        client,
    )
    feedback.append(fix.model_dump(mode="json"))
    nxt = index + 1
    if nxt < len(gaps):
        name = _name(criteria, str(_judgement(gaps[nxt])["criterion_id"]))
        updated = {**job, "index": nxt, "feedback": feedback}
        return updated, progress_message("feedback", nxt, len(gaps), name), no_update, True, True
    return _finish(config, policy, verdicts, feedback)


def _finish(
    config: AppConfig,
    policy: str,
    verdicts: list[dict[str, object]],
    feedback: list[dict[str, object]],
) -> tuple[None, str, list[Component], bool, bool]:
    judgement = PolicyJudgement(
        prompt_id=str(verdicts[0]["prompt_id"]),
        prompt_version=str(verdicts[0]["prompt_version"]),
        model_id=str(verdicts[0]["model_id"]),
        verdicts=[ModelResult.model_validate(item) for item in verdicts],
    )
    items = [CriterionFeedback.model_validate(item) for item in feedback]
    review = PolicyReview(
        judgement=judgement,
        feedback=items,
        citations=check_citations(policy, items),
        score=score_verdicts(judgement, config),
    )
    recorded = whole_policy_result("pasted", review)
    if recorded.review is None:
        raise RuntimeError("whole_policy_llm result has no review")
    return None, "", render_result(build_display(recorded.review, config)), False, False


def _ordered(config: AppConfig) -> list[Criterion]:
    catalog = load_criteria(config.paths.criteria_path).criteria
    return sorted(catalog, key=lambda item: item.display_order)


def _judgement(record: dict[str, object]) -> dict[str, object]:
    judgement = record["judgement"]
    if not isinstance(judgement, dict):
        raise TypeError("Stored judgement is not an object.")
    return judgement


def _records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _name(criteria: list[Criterion], criterion_id: str) -> str:
    for item in criteria:
        if item.id == criterion_id:
            return item.name
    return criterion_id


def main() -> None:
    """Run the single-policy page."""
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
