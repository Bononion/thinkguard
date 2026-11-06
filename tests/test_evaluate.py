"""thinkguard.evaluate: config checks, and the test-set firewall on a throwaway git repo in tmp_path."""

import subprocess

import pytest

from thinkguard.evaluate import EvalConfig, check_firewall


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)  # fmt: skip


@pytest.fixture
def frozen(tmp_path):
    """A repo whose frozen/ has ok.yaml (committed), edited.yaml (committed, then changed), new.yaml (never added) and
    ignored.yaml (ignored, so git status prints nothing for it); outside.yaml sits next to frozen/.
    """
    d = tmp_path / "frozen"
    d.mkdir()
    git(tmp_path, "init", "-q")
    for name in ("ok.yaml", "edited.yaml"):
        (d / name).write_text("a: 1\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "freeze")
    (d / "edited.yaml").write_text("a: 2\n")
    (tmp_path / ".git/info/exclude").write_text("frozen/ignored.yaml\n")
    for path in (d / "new.yaml", d / "ignored.yaml", tmp_path / "outside.yaml"):
        path.write_text("a: 1\n")
    return d


@pytest.mark.parametrize(
    "name, match",
    [("../outside.yaml", "needs a config under"), ("new.yaml", "not committed"), ("ignored.yaml", "not committed"),
     ("edited.yaml", "not committed")],
)  # fmt: skip
def test_firewall_blocks_a_test_split_without_a_committed_frozen_config(frozen, name, match):
    with pytest.raises(ValueError, match=match):
        check_firewall(frozen / name, "test_wildguardtest", frozen)


def test_firewall_lets_a_committed_frozen_config_and_other_splits_through(frozen):
    check_firewall(frozen / "ok.yaml", "test_wildguardtest", frozen)
    check_firewall(frozen.parent / "outside.yaml", "smoke", frozen)


GOOD = {"condition": "R", "split": "smoke", "mode": "decision", "max_new_tokens": 32, "batch_size": 16}


@pytest.mark.parametrize(
    "field, value, match",
    [("condition", "X", "condition 'X'"), ("split", "nope", "split 'nope'"), ("mode", "fast", "mode 'fast'"),
     ("max_new_tokens", 0, "must be positive"), ("batch_size", -1, "must be positive")],
)  # fmt: skip
def test_config_rejects_bad_values(field, value, match):
    EvalConfig(**GOOD)
    with pytest.raises(ValueError, match=match):
        EvalConfig(**GOOD | {field: value})
