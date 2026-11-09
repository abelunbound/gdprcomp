from app.single_policy import app, progress_message, render_result
from dash.development.base_component import Component

from gdpr_gap.config import AppConfig, load_config
from gdpr_gap.gaps import is_gap
from gdpr_gap.single_policy import NOT_LEGAL_ADVICE, build_display, sample_review

KEY = "test-key-not-a-real-secret"


def _config() -> AppConfig:
    return load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )


def test_display_separates_condition_from_cited_text_and_limits_feedback() -> None:
    config = _config()
    display = build_display(sample_review(config), config)

    assert display.notice == NOT_LEGAL_ADVICE
    assert display.method == "whole-policy LLM"
    assert display.band == "high"
    assert len(display.criteria) == 22

    somewhat, missed, cited_yes = display.criteria[:3]
    assert somewhat.condition_met == "Somewhat"
    assert somewhat.relevant_text == ["We ask for consent before sending marketing email."]
    assert somewhat.feedback == "Describe how consent can be withdrawn."
    assert missed.condition_met == "No"
    assert missed.feedback == "Name the purposes of processing."
    assert cited_yes.condition_met == "Yes"
    assert cited_yes.relevant_text == []
    assert cited_yes.feedback is None
    assert all(row.feedback is None for row in display.criteria if not is_gap(row.condition_met))
    assert all(row.feedback is not None for row in display.criteria if is_gap(row.condition_met))
    text = " ".join(_text(component) for component in render_result(display))
    assert "Condition met:" in text
    assert "Somewhat" in text
    assert "Relevant text found:" in text
    assert "None cited" in text
    assert "Feedback: Describe how consent can be withdrawn." in text
    assert "Feedback: Name the purposes of processing." in text


def test_progress_names_the_call_about_to_start_and_sits_above_results() -> None:
    assert progress_message("judge", 0, 22, "Data Processing Records") == (
        "Judging 1 of 22: Data Processing Records"
    )
    assert progress_message("feedback", 2, 4, "Purpose Limitation") == (
        "Writing feedback 3 of 4: Purpose Limitation"
    )
    ids = _component_ids(app.layout)
    assert ids.index("progress") < ids.index("result")


def _component_ids(component: Component) -> list[str]:
    found: list[str] = []
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        found.append(component_id)
    children = getattr(component, "children", None)
    if isinstance(children, Component):
        found.extend(_component_ids(children))
    elif isinstance(children, list):
        for child in children:
            if isinstance(child, Component):
                found.extend(_component_ids(child))
    return found


def _text(component: Component) -> str:
    chunks: list[str] = []
    if isinstance(component.children, str):
        chunks.append(component.children)
    elif isinstance(component.children, list):
        for child in component.children:
            if isinstance(child, str):
                chunks.append(child)
            elif isinstance(child, Component):
                chunks.append(_text(child))
    return " ".join(chunks)
