"""Load Princeton-Leuven privacy policies from a local Markdown cache.

The published release is a SQLite database that also contains every policy's
HTML. This loader never clones the git repository and never writes HTML.
`summarize_index` counts `.co.uk` snapshots and their character length without
reading policy text. `materialize_markdown_cache` writes Markdown only after
`confirm=True`. `load_corpus` reads that cache and fails if it is missing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from gdpr_gap.config import AppConfig

PUBLISHED_SQLITE_URL = (
    "https://privacypolicies.cs.princeton.edu/data-release/data/release_db.sqlite.xz"
)
_MARKDOWN_DIR = "markdown"
_MANIFEST_NAME = "manifest.jsonl"
_META_NAME = "meta.json"


class CorpusError(Exception):
    """Raised when the corpus index or cache is missing or invalid."""


class PolicyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(pattern=r"^[0-9]+$")
    domain: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snapshot_url: str = Field(min_length=1)
    snapshot_date: date
    text: str = Field(min_length=1)


class CorpusIndexSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_suffix: str
    snapshot_count: int = Field(ge=0)
    site_count: int = Field(ge=0)
    text_characters: int = Field(ge=0)

    def refusal_message(self) -> str:
        """Explain the cache write that will not happen without confirmation."""
        return (
            f"Refusing to cache {self.snapshot_count} snapshots from "
            f"{self.site_count} sites ending in {self.domain_suffix} "
            f"({self.text_characters} Markdown characters). "
            "Re-run with confirm=True after reviewing this count and size."
        )


class PolicyCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_suffix: str
    records: list[PolicyRecord]

    def __len__(self) -> int:
        return len(self.records)


def load_corpus(cache_dir: str | Path, *, domain_suffix: str) -> PolicyCorpus:
    """Load cached Markdown policies. Raise CorpusError when the cache is absent."""
    suffix = _validated_suffix(domain_suffix)
    root = Path(cache_dir)
    meta_path = root / _META_NAME
    manifest_path = root / _MANIFEST_NAME
    if not meta_path.is_file() or not manifest_path.is_file():
        raise CorpusError(
            f"Corpus cache not found at {root}. Expected {_META_NAME} and {_MANIFEST_NAME}. "
            "Build it from a local copy of the Princeton-Leuven SQLite index with "
            "materialize_markdown_cache after reviewing summarize_index. "
            f"The published index is {PUBLISHED_SQLITE_URL}."
        )
    meta = _read_meta(meta_path)
    if meta["domain_suffix"] != suffix:
        raise CorpusError(
            f"Corpus cache at {root} is for {meta['domain_suffix']!r}, not {suffix!r}."
        )
    records = _read_manifest(root, manifest_path, suffix)
    declared = meta["snapshot_count"]
    if not isinstance(declared, int) or len(records) != declared:
        raise CorpusError(
            f"Corpus cache at {root} declares {declared} snapshots "
            f"but the manifest contains {len(records)}."
        )
    return PolicyCorpus(domain_suffix=suffix, records=records)


def load_corpus_from_config(config: AppConfig) -> PolicyCorpus:
    """Load the cache path and domain suffix stored in application config."""
    return load_corpus(config.paths.corpus_cache_dir, domain_suffix=config.sampling.domain_suffix)


def summarize_index(sqlite_path: str | Path, *, domain_suffix: str) -> CorpusIndexSummary:
    """Count matching snapshots and Markdown characters. Does not read or write policy text."""
    suffix = _validated_suffix(domain_suffix)
    connection = _open_index(sqlite_path)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT snapshots.site_id), COALESCE(SUM(texts.length), 0)
            FROM policy_snapshots AS snapshots
            JOIN sites ON sites.id = snapshots.site_id
            JOIN policy_texts AS texts ON texts.id = snapshots.policy_text_id
            WHERE lower(sites.domain) LIKE ? ESCAPE '\\'
            """,
            (_like_pattern(suffix),),
        ).fetchone()
    except sqlite3.Error as exc:
        raise CorpusError(f"Could not read the corpus index {sqlite_path}: {exc}") from exc
    finally:
        connection.close()
    if row is None:
        raise CorpusError(
            f"Could not read the corpus index {sqlite_path}: summary query returned no row."
        )
    return CorpusIndexSummary(
        domain_suffix=suffix,
        snapshot_count=int(row[0]),
        site_count=int(row[1]),
        text_characters=int(row[2]),
    )


def materialize_markdown_cache(
    sqlite_path: str | Path,
    cache_dir: str | Path,
    *,
    domain_suffix: str,
    confirm: bool = False,
) -> CorpusIndexSummary:
    """Write Markdown files for the suffix. Refuse unless confirm is true."""
    summary = summarize_index(sqlite_path, domain_suffix=domain_suffix)
    root = Path(cache_dir)
    if not confirm:
        raise CorpusError(summary.refusal_message())
    if root.exists():
        raise CorpusError(f"Corpus cache already exists at {root}. Refusing to overwrite it.")

    partial = root.with_name(root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    markdown_dir = partial / _MARKDOWN_DIR
    markdown_dir.mkdir(parents=True)
    try:
        records = list(_iter_index_rows(sqlite_path, summary.domain_suffix))
        lines: list[str] = []
        characters = 0
        for record in records:
            text = record.pop("text")
            destination = markdown_dir / f"{record['snapshot_id']}.md"
            destination.write_text(text, encoding="utf-8")
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            characters += len(text)
            lines.append(json.dumps({**record, "text_sha1": digest, "text_chars": len(text)}))
        meta = {
            "domain_suffix": summary.domain_suffix,
            "snapshot_count": len(lines),
            "text_characters": characters,
        }
        (partial / _META_NAME).write_text(json.dumps(meta), encoding="utf-8")
        manifest = "\n".join(lines)
        if lines:
            manifest += "\n"
        (partial / _MANIFEST_NAME).write_text(manifest, encoding="utf-8")
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(root)
    return CorpusIndexSummary(
        domain_suffix=summary.domain_suffix,
        snapshot_count=len(lines),
        site_count=summary.site_count,
        text_characters=characters,
    )


def _validated_suffix(domain_suffix: str) -> str:
    suffix = domain_suffix.strip().lower()
    if not suffix.startswith("."):
        raise CorpusError(f"domain suffix must start with '.', got {domain_suffix!r}.")
    return suffix


def _like_pattern(suffix: str) -> str:
    escaped = suffix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}"


def _open_index(sqlite_path: str | Path) -> sqlite3.Connection:
    path = Path(sqlite_path)
    if not path.is_file():
        raise CorpusError(
            f"Corpus index not found: {path}. Place a local copy of the Princeton-Leuven "
            f"SQLite database there. The published file is {PUBLISHED_SQLITE_URL} "
            "(about 3.1 GB compressed) and is not downloaded automatically."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _iter_index_rows(sqlite_path: str | Path, suffix: str) -> Iterator[dict[str, str]]:
    connection = _open_index(sqlite_path)
    try:
        try:
            cursor = connection.execute(
                """
                SELECT snapshots.id, sites.domain, snapshots.policy_url,
                       snapshots.policy_snapshot_url, texts.policy_text
                FROM policy_snapshots AS snapshots
                JOIN sites ON sites.id = snapshots.site_id
                JOIN policy_texts AS texts ON texts.id = snapshots.policy_text_id
                WHERE lower(sites.domain) LIKE ? ESCAPE '\\'
                ORDER BY snapshots.id
                """,
                (_like_pattern(suffix),),
            )
        except sqlite3.Error as exc:
            raise CorpusError(f"Could not read the corpus index {sqlite_path}: {exc}") from exc
        for row in cursor:
            yield _row_to_record(row, suffix)
    finally:
        connection.close()


def _row_to_record(row: Sequence[object], suffix: str) -> dict[str, str]:
    snapshot_id = str(row[0]).strip()
    domain = str(row[1]).strip().lower()
    url = str(row[2]).strip() if row[2] is not None else ""
    snapshot_url = str(row[3]).strip() if row[3] is not None else ""
    text = row[4] if isinstance(row[4], str) else ""
    if not snapshot_id.isdecimal():
        raise CorpusError(f"Snapshot id must be numeric, got {snapshot_id!r}.")
    if not domain.endswith(suffix):
        raise CorpusError(f"Snapshot {snapshot_id} domain {domain!r} does not end with {suffix!r}.")
    if not url or not snapshot_url or not text.strip():
        raise CorpusError(
            f"Snapshot {snapshot_id} is missing a required field (url, snapshot url, or text)."
        )
    try:
        snapshot_date = _snapshot_date(snapshot_url)
    except ValueError as exc:
        raise CorpusError(f"Snapshot {snapshot_id} has no crawl date in {snapshot_url!r}.") from exc
    return {
        "snapshot_id": snapshot_id,
        "domain": domain,
        "url": url,
        "snapshot_url": snapshot_url,
        "snapshot_date": snapshot_date.isoformat(),
        "text": text,
    }


def _snapshot_date(snapshot_url: str) -> date:
    marker = "/web/"
    start = snapshot_url.find(marker)
    if start < 0:
        raise ValueError(snapshot_url)
    stamp = snapshot_url[start + len(marker) : start + len(marker) + 14]
    if len(stamp) != 14 or not stamp.isdecimal():
        raise ValueError(snapshot_url)
    return date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))


def _read_meta(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Invalid corpus metadata {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CorpusError(f"Invalid corpus metadata {path}: expected a mapping.")
    meta = cast(dict[str, object], loaded)
    suffix = meta.get("domain_suffix")
    count = meta.get("snapshot_count")
    if not isinstance(suffix, str) or not isinstance(count, int):
        raise CorpusError(
            f"Invalid corpus metadata {path}: domain_suffix and snapshot_count are required."
        )
    return meta


def _read_manifest(root: Path, path: Path, suffix: str) -> list[PolicyRecord]:
    records: list[PolicyRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"Invalid manifest line {line_number} in {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise CorpusError(f"Invalid manifest line {line_number} in {path}: expected a mapping.")
        record = _record_from_manifest(
            root,
            cast(dict[str, object], loaded),
            suffix,
            line_number,
            path,
        )
        if record.snapshot_id in seen:
            raise CorpusError(f"Duplicate snapshot id {record.snapshot_id} in {path}.")
        seen.add(record.snapshot_id)
        records.append(record)
    return records


def _record_from_manifest(
    root: Path,
    payload: dict[str, object],
    suffix: str,
    line_number: int,
    path: Path,
) -> PolicyRecord:
    snapshot_id = payload.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.isdecimal():
        raise CorpusError(f"Invalid snapshot id on manifest line {line_number} in {path}.")
    text_path = root / _MARKDOWN_DIR / f"{snapshot_id}.md"
    if not text_path.is_file():
        raise CorpusError(f"Cached Markdown for snapshot {snapshot_id} is missing: {text_path}.")
    text = text_path.read_text(encoding="utf-8")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if payload.get("text_sha1") != digest:
        raise CorpusError(
            f"Cached Markdown for snapshot {snapshot_id} does not match its manifest digest."
        )
    domain = payload.get("domain")
    if not isinstance(domain, str) or not domain.endswith(suffix):
        raise CorpusError(f"Snapshot {snapshot_id} domain does not end with {suffix!r}.")
    try:
        return PolicyRecord.model_validate(
            {
                "snapshot_id": snapshot_id,
                "domain": domain,
                "url": payload.get("url"),
                "snapshot_url": payload.get("snapshot_url"),
                "snapshot_date": payload.get("snapshot_date"),
                "text": text,
            }
        )
    except ValueError as exc:
        raise CorpusError(f"Invalid snapshot {snapshot_id} in {path}: {exc}") from exc
