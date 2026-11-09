"""Turn Markdown, pasted text, or HTML into cleaned policy text.

Whitespace inside lines is collapsed. A blank line remains between sections.
Nothing is truncated: a long policy is returned whole. An empty result is logged
and raised. HTML is only the input path for a pasted page or a URL.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.logging import log_event

SourceKind = Literal["markdown", "html", "text"]
_HEADING = re.compile(r"^#{1,6} ")
_HTML_TAG = re.compile(r"<(?:html|body|div|p|h[1-6]|article|section)\b", re.IGNORECASE)
_SKIP_TAGS = {"script", "style", "noscript"}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "tr",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
}


class ExtractionError(Exception):
    """Raised when a page or paste yields no policy text."""


class ExtractedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    kind: SourceKind
    section_count: int = Field(ge=1)
    url: str | None = None


def extract_text(
    source: str,
    *,
    kind: SourceKind | None = None,
    url: str | None = None,
    logger: logging.Logger | None = None,
) -> ExtractedText:
    """Clean Markdown, pasted text, or an HTML page. The full text is kept."""
    detected = kind or _detect_kind(source)
    if detected == "html":
        raw = _html_to_text(source)
    else:
        raw = source
    text = _normalize(raw)
    if not text:
        _fail("empty", detected, url, logger)
    sections = _section_count(text)
    if logger is not None:
        log_event(
            logger,
            "extraction_ok",
            kind=detected,
            chars=len(text),
            sections=sections,
            url=url,
        )
    return ExtractedText(text=text, kind=detected, section_count=sections, url=url)


def extract_url(
    url: str,
    *,
    logger: logging.Logger | None = None,
    fetch: Callable[[str], str] | None = None,
) -> ExtractedText:
    """Fetch one page and extract it as HTML. Network errors are logged."""
    getter = fetch or _fetch
    try:
        page = getter(url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _fail(str(exc), "html", url, logger)
    return extract_text(page, kind="html", url=url, logger=logger)


def _detect_kind(source: str) -> SourceKind:
    sample = source[:5000]
    if _HTML_TAG.search(sample):
        return "html"
    if _HEADING.search(source):
        return "markdown"
    return "text"


def _normalize(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if collapsed and not blank:
                collapsed.append("")
            blank = True
            continue
        blank = False
        collapsed.append(line)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def _section_count(text: str) -> int:
    headings = [line for line in text.splitlines() if _HEADING.match(line)]
    if headings:
        return len(headings)
    return len([block for block in re.split(r"\n\s*\n", text) if block.strip()])


def _html_to_text(page: str) -> str:
    parser = _PageParser()
    parser.feed(page)
    parser.close()
    return "".join(parser.parts)


def _fail(
    reason: str,
    kind: SourceKind,
    url: str | None,
    logger: logging.Logger | None,
) -> NoReturn:
    if logger is not None:
        log_event(logger, "extraction_failed", reason=reason, kind=kind, url=url)
    raise ExtractionError(f"No policy text extracted ({reason}).")


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "gdpr-gap"})
    with urllib.request.urlopen(request, timeout=20) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = cast(bytes, response.read())
        return payload.decode(charset, errors="replace")


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if name in _BLOCK_TAGS:
            self.parts.append("\n")
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("#" * int(name[1]) + " ")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if name in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.parts.append(data)
