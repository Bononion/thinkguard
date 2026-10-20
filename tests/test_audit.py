"""thinkguard.data.audit: unit tests run in CI; the data test needs the downloaded critique file."""

import json
from pathlib import Path

import pytest

from thinkguard.data import audit

ROOT = Path(__file__).parents[1]
KEYS = audit.load_keys(ROOT / audit.MANIFEST)


def test_regroup_takes_the_longest_key():
    found, unknown = audit.regroup("violence,aiding_and_abetting,incitement,self_harm", KEYS)
    assert found == ["violence,aiding_and_abetting,incitement", "self_harm"]
    assert unknown == []


def test_regroup_reports_an_unknown_piece():
    assert audit.regroup("self_harm, cyberbullying", KEYS) == (["self_harm"], ["cyberbullying"])


def test_contradiction_regex():
    assert audit.CONTRADICTION.search("The conversation is safe").group(1) == "safe"
    assert audit.CONTRADICTION.search("This could be considered unsafe") is None


@pytest.mark.skipif(not (ROOT / audit.CRITIQUE).exists(), reason="critique file not downloaded")
def test_critique_release():
    with open(ROOT / audit.CRITIQUE, encoding="utf-8") as f:
        stats, _ = audit.audit(json.load(f), KEYS)
    s, cat = stats["counts"], stats["categories"]
    assert stats["rows"] == 15000
    assert (s["unsafe"], s["safe"]) == (8526, 6474)
    assert s["safe_misplaced_categories"] == s["unsafe_misplaced_categories"] == 0
    assert (cat["pieces"], len(cat["keys"]), sum(cat["unknown"].values())) == (27, 14, 0)
