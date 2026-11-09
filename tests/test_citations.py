import json
import logging
from typing import Any, cast

from gdpr_gap.feedback import CitationReport, CriterionFeedback, check_citations

POLICY = "We ask for consent before sending marketing email. You can withdraw consent."
VALID = "We ask for consent before sending marketing email."
FABRICATED = "We sell personal data to advertisers."


def _feedback(criterion_id: str, quotes: list[str]) -> CriterionFeedback:
    return CriterionFeedback(
        criterion_id=criterion_id,
        suggestion="State this more precisely.",
        evidence_quotes=quotes,
    )


def test_valid_citation_is_supported() -> None:
    reports = check_citations(POLICY, [_feedback("consent_mechanisms", [VALID])])

    assert reports == [
        CitationReport(
            criterion_id="consent_mechanisms",
            citation_support="supported",
            unsupported_quotes=[],
        )
    ]


def test_fabricated_citation_is_flagged() -> None:
    logger = logging.getLogger("gdpr_gap_citation_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger.addHandler(Capture())
    item = _feedback("consent_mechanisms", [VALID, FABRICATED, "   "])

    reports = check_citations(POLICY, [item], logger=logger)

    assert reports[0].citation_support == "unsupported"
    assert reports[0].unsupported_quotes == [FABRICATED, "   "]
    assert item.suggestion == "State this more precisely."
    assert records[0].getMessage() == "citation_unsupported"
    fields = cast(dict[str, Any], getattr(records[0], "fields"))
    assert fields["criterion_id"] == "consent_mechanisms"
    assert fields["unsupported_count"] == 2
    assert fields["quote_count"] == 3
    assert FABRICATED not in json.dumps(fields)


def test_feedback_with_no_quotes_is_unsupported() -> None:
    reports = check_citations(POLICY, [_feedback("consent_mechanisms", [])])

    assert reports[0].citation_support == "unsupported"
    assert reports[0].unsupported_quotes == []
