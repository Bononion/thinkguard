"""thinkguard.data.download against a local file:// source (no network)."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from thinkguard.data import download

ROWS = [{"output": "b", "input": "a"}, {"input": "c", "output": "d"}]
TARGET = Path("data/raw/tiny/tiny.json")


@pytest.fixture
def src(tmp_path, monkeypatch):
    """A 2-row upstream file; the test runs with tmp_path as the repo root."""
    path = tmp_path / "upstream" / "tiny.json"
    path.parent.mkdir()
    path.write_text(json.dumps(ROWS))
    monkeypatch.chdir(tmp_path)
    return path


def run(src, fmt="json"):
    """Run the CLI on a one-source config and return the manifest."""
    source = {"name": "tiny", "url": src.as_uri(), "revision": "r1", "format": fmt, "split": "train"}
    Path("sources.yaml").write_text(yaml.safe_dump([source]))
    download.main(["--config", "sources.yaml"])
    return json.loads(download.MANIFEST.read_text())


def test_first_run_writes_file_and_entry(src):
    entry = run(src)["tiny"]
    assert TARGET.read_bytes() == src.read_bytes()
    assert entry["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert entry["bytes"] == src.stat().st_size
    assert entry["rows"] == 2
    assert entry["schema"] == {"type": "array", "row_keys": ["input", "output"]}
    assert not TARGET.with_name("tiny.json.part").exists()


def test_second_run_downloads_nothing(src, monkeypatch):
    first = run(src)

    def no_fetch(*_):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(download, "fetch", no_fetch)
    assert run(src) == first  # same single entry, same downloaded_at


def test_changed_source_fails(src):
    run(src)
    manifest = download.MANIFEST.read_text()
    src.write_text(json.dumps(ROWS[:1]))
    TARGET.unlink()  # a fresh clone: an existing matching file would be skipped without downloading
    with pytest.raises(download.HashMismatch, match="source tiny"):
        run(src)
    assert download.MANIFEST.read_text() == manifest
    assert not TARGET.exists() and not TARGET.with_name("tiny.json.part").exists()


def test_unknown_format_raises(src):
    with pytest.raises(ValueError, match="'csv'"):
        run(src, fmt="csv")
    assert not TARGET.parent.exists()
