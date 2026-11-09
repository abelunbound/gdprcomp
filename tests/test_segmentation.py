from gdpr_gap.config import SegmentationConfig
from gdpr_gap.segmentation import passages_cover, segment_passages


def _settings(passage_words: int = 120, overlap_words: int = 20) -> SegmentationConfig:
    return SegmentationConfig(passage_words=passage_words, overlap_words=overlap_words)


def _words(count: int) -> str:
    return " ".join(f"w{index}" for index in range(count))


def test_tail_words_are_kept_when_the_last_window_is_short() -> None:
    text = _words(250)
    passages = segment_passages(text, _settings())

    assert passages_cover(text, passages)
    assert passages[-1].end_word == 250
    assert passages[-1].text.endswith("w249")
    assert passages[-1].use == "comparator"
    assert len(passages[-1].text.split()) == 50


def test_windows_overlap_by_the_configured_amount() -> None:
    passages = segment_passages(_words(200), _settings(passage_words=10, overlap_words=3))

    assert passages[0].end_word - passages[1].start_word == 3
    assert passages_cover(_words(200), passages)


def test_short_and_empty_text() -> None:
    short = _words(12)
    passages = segment_passages(short, _settings())

    assert len(passages) == 1
    assert passages[0].text == short
    assert passages_cover(short, passages)
    assert segment_passages("  \n\t ", _settings()) == []
    assert passages_cover("", [])


def test_repository_settings_cover_a_long_policy() -> None:
    text = _words(1000)
    passages = segment_passages(text, _settings())

    assert passages[0].end_word - passages[0].start_word == 120
    assert passages_cover(text, passages)
    assert passages[-1].text.endswith("w999")
