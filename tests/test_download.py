"""thinkguard.data.download against a local file:// source (no network)."""

import gzip
import hashlib
import json
import lzma
from pathlib import Path

import pytest
import yaml

from thinkguard.data import download

ROWS = [{"output": "b", "input": "a"}, {"input": "c", "output": "d"}]
TARGET = Path("data/raw/tiny/tiny.json")
GZ_ROWS = [
    {"is_safe": True, "category": {"b": False, "a": False}},
    {"is_safe": False, "category": {"b": True, "a": False}},
    {"is_safe": False, "category": {"a": True, "b": False}},  # same keys, other order
]
GZ_STATS = {"unsafe": {"field": "is_safe", "equals": False}, "dict_keys": "category"}


@pytest.fixture
def src(tmp_path, monkeypatch):
    """A 2-row upstream file; the test runs with tmp_path as the repo root."""
    path = tmp_path / "upstream" / "tiny.json"
    path.parent.mkdir()
    path.write_text(json.dumps(ROWS))
    monkeypatch.chdir(tmp_path)
    return path


def write_gz(src, rows):
    """A gzipped JSONL upstream file next to src, ending in a blank line."""
    path = src.with_name("tiny.jsonl.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
        f.write("\n")
    return path


def run(src, fmt="json", **extra):
    """Run the CLI on a one-source config and return the manifest."""
    source = {"name": "tiny", "url": src.as_uri(), "revision": "r1", "format": fmt, "split": "train", **extra}
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
    assert "unsafe_rows" not in entry and "dict_keys" not in entry
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


def test_jsonl_gz_counts_unsafe_rows_and_dict_keys(src):
    entry = run(write_gz(src, GZ_ROWS), "jsonl.gz", **GZ_STATS)["tiny"]
    assert entry["rows"] == 3
    assert entry["schema"] == {"type": "jsonl", "row_keys": ["category", "is_safe"]}
    assert entry["unsafe_rows"] == 2
    assert entry["dict_keys"] == {"category": ["b", "a"]}  # first row's file order


def test_jsonl_gz_fails_when_dict_keys_differ(src):
    rows = [*GZ_ROWS, {"is_safe": True, "category": {"b": False}}]
    with pytest.raises(ValueError, match="row 3 category keys"):
        run(write_gz(src, rows), "jsonl.gz", **GZ_STATS)
    assert not download.MANIFEST.exists()
    assert not Path("data/raw/tiny/tiny.jsonl.gz").exists()


def test_jsonl_xz_counts_unsafe_rows_and_dict_keys(src):
    path = src.with_name("tiny.jsonl.xz")
    with lzma.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in GZ_ROWS)
        f.write("\n")
    entry = run(path, "jsonl.xz", **GZ_STATS)["tiny"]
    assert Path("data/raw/tiny/tiny.jsonl.xz").read_bytes() == path.read_bytes()
    assert (entry["rows"], entry["unsafe_rows"]) == (3, 2)
    assert entry["schema"] == {"type": "jsonl", "row_keys": ["category", "is_safe"]}
    assert entry["dict_keys"] == {"category": ["b", "a"]}
