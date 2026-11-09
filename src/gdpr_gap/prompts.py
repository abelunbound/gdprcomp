"""Versioned prompt files.

A prompt is stored at `prompts/{prompt_id}/v1.txt`. Saving changed text writes
the next version and leaves every older file untouched, so a past run can load
the exact text it used. The prompt id and version are recorded on each model
result.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_PROMPT_ID = re.compile(r"^[a-z0-9_]+$")
_VERSION_FILE = re.compile(r"^v([1-9][0-9]*)\.txt$")


class PromptError(Exception):
    """Raised when a prompt cannot be stored, loaded, or rendered."""


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(pattern=r"^[a-z0-9_]+$")
    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    text: str = Field(min_length=1)


class PromptRegistry:
    """Load and append versioned prompt files."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def load(self, prompt_id: str, version: str | None = None) -> PromptRecord:
        """Load one version. Omit version to load the newest file."""
        chosen = version if version is not None else self.latest_version(prompt_id)
        path = self._file(prompt_id, chosen)
        if not path.is_file():
            raise PromptError(f"Prompt {prompt_id} {chosen} was not found at {path}.")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise PromptError(f"Prompt {prompt_id} {chosen} is empty: {path}.")
        return PromptRecord(prompt_id=prompt_id, version=chosen, text=text)

    def save(self, prompt_id: str, text: str) -> PromptRecord:
        """Store text. Identical latest text is kept; changed text becomes the next version."""
        self._require_id(prompt_id)
        if not text.strip():
            raise PromptError("Prompt text is empty.")
        found = self.versions(prompt_id)
        if found:
            latest = self.load(prompt_id, found[-1])
            if latest.text == text:
                return latest
            version = f"v{_version_number(found[-1]) + 1}"
        else:
            version = "v1"
        path = self._file(prompt_id, version)
        if path.exists():
            raise PromptError(f"Refusing to overwrite prompt {prompt_id} {version}: {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return PromptRecord(prompt_id=prompt_id, version=version, text=text)

    def render(
        self,
        prompt_id: str,
        version: str,
        *,
        criterion_id: str,
        condition: str,
        policy_text: str,
        verdict: str | None = None,
        justification: str | None = None,
    ) -> PromptRecord:
        """Fill the stored template. The policy text is inserted last and is not scanned again."""
        record = self.load(prompt_id, version)
        values = {"criterion_id": criterion_id, "condition": condition}
        if verdict is not None:
            values["verdict"] = verdict
        if justification is not None:
            values["justification"] = justification
        values["policy_text"] = policy_text
        return record.model_copy(update={"text": _fill(record.text, values)})

    def versions(self, prompt_id: str) -> list[str]:
        """Return version names from oldest to newest."""
        self._require_id(prompt_id)
        folder = self._directory / prompt_id
        if not folder.exists():
            return []
        if not folder.is_dir():
            raise PromptError(f"Prompt path is not a directory: {folder}.")
        numbered: list[tuple[int, str]] = []
        for path in folder.iterdir():
            if not path.is_file():
                raise PromptError(f"Unexpected entry in prompt directory: {path}.")
            match = _VERSION_FILE.fullmatch(path.name)
            if match is None:
                raise PromptError(f"Prompt file name must look like v1.txt: {path}.")
            numbered.append((int(match.group(1)), path.name.removesuffix(".txt")))
        numbered.sort()
        return [version for _number, version in numbered]

    def latest_version(self, prompt_id: str) -> str:
        found = self.versions(prompt_id)
        if not found:
            raise PromptError(f"Prompt {prompt_id} has no versions in {self._directory}.")
        return found[-1]

    def _file(self, prompt_id: str, version: str) -> Path:
        self._require_id(prompt_id)
        if _VERSION_FILE.fullmatch(f"{version}.txt") is None:
            raise PromptError(f"Prompt version must look like v1, not {version!r}.")
        return self._directory / prompt_id / f"{version}.txt"

    @staticmethod
    def _require_id(prompt_id: str) -> None:
        if _PROMPT_ID.fullmatch(prompt_id) is None:
            raise PromptError(
                "Prompt id must contain only lowercase letters, digits, and underscores: "
                f"{prompt_id!r}."
            )


def _fill(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        token = "{" + name + "}"
        if rendered.count(token) != 1:
            raise PromptError(f"Prompt must contain {token} once.")
        rendered = rendered.replace(token, value, 1)
    return rendered


def _version_number(version: str) -> int:
    match = _VERSION_FILE.fullmatch(f"{version}.txt")
    if match is None:
        raise PromptError(f"Prompt version must look like v1, not {version!r}.")
    return int(match.group(1))
