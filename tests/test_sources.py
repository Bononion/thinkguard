"""thinkguard.data.sources.load on a tiny local source (no network)."""

import json

import pytest
import yaml

from thinkguard.data import sources

ROWS = [{"prompt": "a", "is_safe": True}, {"prompt": "b", "is_safe": False}]


def test_load_checks_rows_and_unsafe_rows(tmp_path):
    data = tmp_path / "tiny.json"
    data.write_text(json.dumps(ROWS))
    source = {"name": "tiny", "url": str(data), "revision": "r1", "format": "json", "split": "train", "rows": 2}
    source["unsafe"] = {"field": "is_safe", "equals": False}
    config = tmp_path / "sources.yaml"

    config.write_text(yaml.safe_dump([{**source, "unsafe_rows": 1}]))
    assert sources.load("tiny", config) == ROWS
    config.write_text(yaml.safe_dump([{**source, "unsafe_rows": 2}]))
    with pytest.raises(ValueError, match="source tiny: 2 rows, 1 unsafe"):
        sources.load("tiny", config)
