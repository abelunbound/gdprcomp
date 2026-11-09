import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from gdpr_gap.config import load_config
from gdpr_gap.corpus import (
    CorpusError,
    load_corpus,
    load_corpus_from_config,
    materialize_markdown_cache,
    summarize_index,
)

WAYBACK = "https://web.archive.org/web/20170615120000id_/http://www.bbc.co.uk/privacy"
LATER = "https://web.archive.org/web/20190301120000id_/http://shop.bbc.co.uk/privacy"


def _index(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sites (id INTEGER PRIMARY KEY, domain TEXT, categories TEXT);
        CREATE TABLE policy_texts (
            id INTEGER PRIMARY KEY, policy_text TEXT, length INTEGER
        );
        CREATE TABLE policy_snapshots (
            id INTEGER PRIMARY KEY,
            site_id INTEGER,
            policy_snapshot_url TEXT,
            policy_url TEXT,
            policy_text_id INTEGER
        );
        INSERT INTO sites VALUES (1, 'BBC.CO.UK', 'news');
        INSERT INTO sites VALUES (2, 'example.com', 'retail');
        INSERT INTO sites VALUES (3, 'shop.bbc.co.uk', 'retail');
        INSERT INTO policy_texts VALUES (10, 'BBC policy text.', 16);
        INSERT INTO policy_texts VALUES (11, 'Other site policy.', 18);
        INSERT INTO policy_texts VALUES (12, 'Shop policy text.', 17);
        """
    )
    connection.execute(
        "INSERT INTO policy_snapshots VALUES (100, 1, ?, 'bbc.co.uk/privacy', 10)",
        (WAYBACK,),
    )
    connection.execute(
        "INSERT INTO policy_snapshots VALUES (101, 2, ?, 'example.com/privacy', 11)",
        (WAYBACK,),
    )
    connection.execute(
        "INSERT INTO policy_snapshots VALUES (102, 3, ?, 'shop.bbc.co.uk/privacy', 12)",
        (LATER,),
    )
    connection.commit()
    connection.close()


def test_missing_cache_names_the_path(tmp_path: Path) -> None:
    cache = tmp_path / "corpus"

    with pytest.raises(CorpusError, match=str(cache)):
        load_corpus(cache, domain_suffix=".co.uk")


def test_missing_index_names_the_path_and_does_not_download(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"

    with pytest.raises(CorpusError, match="not downloaded automatically"):
        summarize_index(index, domain_suffix=".co.uk")


def test_summary_counts_only_matching_domains_and_writes_nothing(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    _index(index)

    summary = summarize_index(index, domain_suffix=".co.uk")

    assert summary.snapshot_count == 2
    assert summary.site_count == 2
    assert summary.text_characters == 33
    assert not (tmp_path / "corpus").exists()


def test_materialize_refuses_until_confirmed(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    cache = tmp_path / "corpus"
    _index(index)

    with pytest.raises(CorpusError, match="2 snapshots") as exc_info:
        materialize_markdown_cache(index, cache, domain_suffix=".co.uk")

    assert "33 Markdown characters" in str(exc_info.value)
    assert not cache.exists()
    assert not cache.with_name("corpus.partial").exists()


def test_confirmed_cache_loads_required_fields(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    cache = tmp_path / "corpus"
    _index(index)

    written = materialize_markdown_cache(index, cache, domain_suffix=".co.uk", confirm=True)
    corpus = load_corpus(cache, domain_suffix=".co.uk")

    assert written.snapshot_count == 2
    assert len(corpus) == 2
    assert [record.domain for record in corpus.records] == ["bbc.co.uk", "shop.bbc.co.uk"]
    first = corpus.records[0]
    assert first.snapshot_id == "100"
    assert first.url == "bbc.co.uk/privacy"
    assert first.snapshot_date == date(2017, 6, 15)
    assert first.text == "BBC policy text."
    assert not list(cache.rglob("*.html"))
    markdown = list((cache / "markdown").glob("*.md"))
    assert {path.name for path in markdown} == {"100.md", "102.md"}


def test_load_corpus_from_config_uses_configured_paths(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    cache = tmp_path / "corpus"
    _index(index)
    materialize_markdown_cache(index, cache, domain_suffix=".co.uk", confirm=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("configs/config.yaml").read_text(encoding="utf-8").replace("data/corpus", str(cache)),
        encoding="utf-8",
    )

    corpus = load_corpus_from_config(
        load_config(config_path, environ={"ANTHROPIC_API_KEY": "test-key"}, load_env_file=False)
    )

    assert len(corpus) == 2
    assert corpus.domain_suffix == ".co.uk"


def test_blank_policy_text_fails_without_leaving_a_cache(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    cache = tmp_path / "corpus"
    _index(index)
    connection = sqlite3.connect(index)
    connection.execute("UPDATE policy_texts SET policy_text = '   ' WHERE id = 10")
    connection.commit()
    connection.close()

    with pytest.raises(CorpusError, match="Snapshot 100"):
        materialize_markdown_cache(index, cache, domain_suffix=".co.uk", confirm=True)

    assert not cache.exists()


def test_tampered_markdown_and_wrong_suffix_fail(tmp_path: Path) -> None:
    index = tmp_path / "release_db.sqlite"
    cache = tmp_path / "corpus"
    _index(index)
    materialize_markdown_cache(index, cache, domain_suffix=".co.uk", confirm=True)
    (cache / "markdown" / "100.md").write_text("changed", encoding="utf-8")

    with pytest.raises(CorpusError, match="does not match"):
        load_corpus(cache, domain_suffix=".co.uk")

    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    meta["domain_suffix"] = ".com"
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (cache / "markdown" / "100.md").write_text("BBC policy text.", encoding="utf-8")

    with pytest.raises(CorpusError, match="not '.co.uk'"):
        load_corpus(cache, domain_suffix=".co.uk")
