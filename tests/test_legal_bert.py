from app.single_policy import app, render_retrieval
from dash.development.base_component import Component

from gdpr_gap.config import AppConfig, EmbeddingConfig, SegmentationConfig, load_config
from gdpr_gap.embeddings import EmbeddingClient, HiddenStates
from gdpr_gap.legal_bert import score_legal_bert

KEY = "test-key-not-a-real-secret"
NAME = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"
_FORBIDDEN = {"verdict", "justification", "score", "band", "feedback"}


class FixedEncoder:
    name = NAME
    version = REVISION

    def encode(self, texts: list[str]) -> HiddenStates:
        states = [[[1.0, 0.0]] for _ in texts]
        mask = [[1] for _ in texts]
        return HiddenStates(states, mask)


def _config() -> AppConfig:
    config = load_config(
        "configs/config.yaml",
        environ={"ANTHROPIC_API_KEY": KEY},
        load_env_file=False,
    )
    return config.model_copy(
        update={"segmentation": SegmentationConfig(passage_words=4, overlap_words=0)}
    )


def test_legal_bert_output_is_a_retrieval_signal_without_verdicts() -> None:
    settings = EmbeddingConfig(model_name=NAME, revision=REVISION)
    baseline = score_legal_bert(
        "one two three four",
        _config(),
        EmbeddingClient(settings, FixedEncoder()),
    )

    assert baseline.method == "legal_bert_only"
    assert baseline.signal == "baseline/retrieval signal"
    assert baseline.model_version == REVISION
    assert len(baseline.criteria) == 22
    assert baseline.criteria[0].best_similarity == 1
    dumped = baseline.model_dump()
    assert _FORBIDDEN.isdisjoint(_keys(dumped))
    names = {row.criterion_id: row.criterion_id for row in baseline.criteria}
    text = " ".join(_text(component) for component in render_retrieval(baseline, names))
    assert "Signal: baseline/retrieval signal" in text
    assert "This is a retrieval signal, not an LLM verdict." in text
    assert "Best similarity:" in text
    assert "Condition met" not in text
    assert "Feedback:" not in text
    assert "show-retrieval" in _component_ids(app.layout)


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
    children = component.children
    if isinstance(children, str):
        chunks.append(children)
    elif isinstance(children, list):
        for child in children:
            if isinstance(child, str):
                chunks.append(child)
            elif isinstance(child, Component):
                chunks.append(_text(child))
    return " ".join(chunks)


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_keys(item))
    return found
