import logging

import pytest

from gdpr_gap.extract import ExtractionError, extract_text, extract_url

HTML_PAGE = """
<html><body>
<script>secret tracker</script>
<h1>Privacy policy</h1>
<p>We   collect   personal data.</p>
<h2>Your rights</h2>
<p>You may request access.</p>
</body></html>
"""

MARKDOWN = """# Privacy policy

We   collect   personal data.

## Your rights

You may request access.
"""

PASTED = "We   collect personal data.\n\n\nYou may request access.\n"


def test_html_keeps_sections_and_drops_script() -> None:
    extracted = extract_text(HTML_PAGE)

    assert extracted.kind == "html"
    assert extracted.section_count == 2
    assert "secret tracker" not in extracted.text
    assert extracted.text.startswith("# Privacy policy")
    assert "\n\n## Your rights\n" in extracted.text
    assert "We collect personal data." in extracted.text


def test_markdown_normalizes_spaces_and_keeps_headings() -> None:
    extracted = extract_text(MARKDOWN)

    assert extracted.kind == "markdown"
    assert extracted.section_count == 2
    assert extracted.text == (
        "# Privacy policy\n\nWe collect personal data.\n\n## Your rights\n\nYou may request access."
    )


def test_pasted_text_keeps_the_paragraph_break() -> None:
    extracted = extract_text(PASTED)

    assert extracted.kind == "text"
    assert extracted.section_count == 2
    assert extracted.text == "We collect personal data.\n\nYou may request access."


def test_empty_page_is_logged_and_rejected(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("gdpr_gap_test_extract")
    logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(ExtractionError, match="empty"):
            extract_text("<html><body><script>only script</script></body></html>", logger=logger)

    failed = [record for record in caplog.records if record.getMessage() == "extraction_failed"]
    assert failed
    assert getattr(failed[0], "fields")["reason"] == "empty"


def test_very_long_policy_is_not_truncated() -> None:
    paragraph = "Personal data is retained for this stated purpose. " * 20
    parts = [f"## Section {index}\n\n{paragraph}" for index in range(400)]
    source = "\n\n".join(parts)
    tail = "UNIQUE-TAIL-MARKER"

    extracted = extract_text(source + "\n\n" + tail)

    assert extracted.kind == "markdown"
    assert extracted.section_count == 400
    assert extracted.text.endswith(tail)
    assert "Section 399" in extracted.text
    assert len(extracted.text) >= len(source.replace("  ", " ")) - 500


def test_url_fetch_uses_the_page_and_logs_network_failure(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("gdpr_gap_test_extract_url")
    logger.setLevel(logging.INFO)

    extracted = extract_url("https://example.test/privacy", fetch=lambda url: HTML_PAGE)
    assert extracted.url == "https://example.test/privacy"
    assert "We collect personal data." in extracted.text

    def fail(url: str) -> str:
        raise TimeoutError(f"timed out for {url}")

    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(ExtractionError, match="timed out"):
            extract_url("https://example.test/missing", logger=logger, fetch=fail)

    assert any(record.getMessage() == "extraction_failed" for record in caplog.records)
