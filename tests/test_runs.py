"""thinkguard.runs: each way a run ends, and the git and split fields of its manifest."""

import json
import signal
import subprocess
from dataclasses import dataclass

import pytest

from thinkguard.config import config_hash
from thinkguard.runs import interrupt, start_run


@dataclass(frozen=True)
class Toy:
    seed: int = 7


def manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_completed_run_records_config_splits_and_end(tmp_path):
    (tmp_path / "toy.json").write_text(json.dumps({"sha256": "ab" * 32}), encoding="utf-8")
    with start_run("eval", Toy(), splits=("toy",), parent="ckpt/C", runs=tmp_path / "runs", manifests=tmp_path) as d:
        assert manifest(d)["status"] == "running" and manifest(d)["end"] is None
    m = manifest(d)
    assert m["status"] == "completed" and m["end"] is not None
    assert m["run_id"] == d.name and d.name.endswith(f"-eval-{config_hash(Toy())[:8]}")
    assert m["config"] == {"seed": 7} and m["seed"] == 7 and m["parent"] == "ckpt/C"
    assert m["splits"] == {"toy": "ab" * 32}


def test_exception_marks_failed_and_is_reraised(tmp_path):
    with pytest.raises(RuntimeError, match="boom"), start_run("eval", Toy(), runs=tmp_path) as d:
        raise RuntimeError("boom")
    assert manifest(d)["status"] == "failed"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_marks_interrupted_with_an_end_time(tmp_path, signum):
    with pytest.raises(KeyboardInterrupt), start_run("train", Toy(), runs=tmp_path) as d:
        signal.raise_signal(signum)
    assert manifest(d)["status"] == "interrupted" and manifest(d)["end"] is not None
    assert signal.getsignal(signal.SIGTERM) is not interrupt  # the previous handler is back


def test_git_dirty_counts_tracked_changes_and_untracked_code(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    monkeypatch.chdir(repo)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], check=True)
    (repo / "src/a.py").write_text("a = 1\n", encoding="utf-8")
    subprocess.run([*git, "add", "src/a.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "a"], check=True)

    def dirty(seed):
        with start_run("eval", Toy(seed=seed), runs=repo / "runs") as d:
            pass
        return manifest(d)["git_dirty"]

    (repo / "data.json").write_text("{}", encoding="utf-8")  # untracked, but not code
    assert dirty(1) is False  # also: the run directory from this call is untracked and must not count next time
    assert dirty(2) is False

    new = repo / "src/new.py"
    new.write_text("b = 2\n", encoding="utf-8")
    assert dirty(3) is True
    new.unlink()

    (repo / "src/a.py").write_text("a = 2\n", encoding="utf-8")
    assert dirty(4) is True
