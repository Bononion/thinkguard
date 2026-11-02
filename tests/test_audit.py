"""thinkguard.data.audit.regroup: the category parser behind training targets and category metrics."""

from pathlib import Path

from thinkguard.data import audit
from thinkguard.data.sources import SOURCES

KEYS = audit.load_keys(Path(__file__).parents[1] / SOURCES)


def test_regroup_takes_the_longest_key_and_reports_unknown_pieces():
    found, unknown = audit.regroup("violence,aiding_and_abetting,incitement,self_harm, cyberbullying", KEYS)
    assert found == ["violence,aiding_and_abetting,incitement", "self_harm"]
    assert unknown == ["cyberbullying"]
