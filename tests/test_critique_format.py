"""thinkguard.data.critique_format.regroup: the category parser behind training targets and category metrics."""

from thinkguard.data.critique_format import load_keys, regroup

KEYS = load_keys()


def test_regroup_takes_the_longest_key_and_reports_unknown_pieces():
    found, unknown = regroup("violence,aiding_and_abetting,incitement,self_harm, cyberbullying", KEYS)
    assert found == ["violence,aiding_and_abetting,incitement", "self_harm"]
    assert unknown == ["cyberbullying"]
