"""thinkguard.evaluate end to end with the real R on the smoke split (32 rows, 2 batches of 16), bf16 on CUDA.

Run on the GPU host after building data/interim/ (README, Evaluate): uv run pytest -m gpu -s, which prints the
measured numbers. R is loaded once; the uninterrupted run (full) is made once and shared.
"""

import json
from functools import partial

import pytest
import torch

from thinkguard import evaluate, registry, runs
from thinkguard.config import load
from thinkguard.data.manifest import load_split
from thinkguard.evaluate import EvalConfig
from thinkguard.evaluation import generate
from thinkguard.evaluation.generate import predict
from thinkguard.evaluation.score import forward_score
from thinkguard.models import load_condition
from thinkguard.paths import ROOT
from thinkguard.prompts import encode

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
CONFIG = ROOT / "configs/eval/smoke_R.yaml"


@pytest.fixture(scope="session")
def smoke():
    """(config, example ids, prompts) as evaluate.main builds them from CONFIG."""
    config = load(CONFIG, EvalConfig)
    examples = load_split(config.split)
    return config, [e.example_id for e in examples], [encode(e, "infer").input_ids for e in examples]


@pytest.fixture(scope="session")
def r_model():
    return load_condition(registry.load()["R"])


@pytest.fixture(scope="session")
def full(r_model, smoke, tmp_path_factory):
    """predictions.jsonl of one uninterrupted run."""
    config, ids, prompts = smoke
    path = tmp_path_factory.mktemp("full") / "predictions.jsonl"
    predict(r_model, ids, prompts, config, path)
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_free_scores_match_a_separate_forward(r_model, smoke, full):
    config, ids, prompts = smoke
    prompt = dict(zip(ids, prompts, strict=True))
    free = [r for r in rows(full) if r["score_path"] == "free"]
    assert free, "no row took the free path"  # how many do is R's behaviour, not the code's: printed, not tested
    forward = []
    for i in range(0, len(free), config.batch_size):
        forward += forward_score(r_model, [prompt[r["example_id"]] for r in free[i : i + config.batch_size]])
    diffs = [abs(r["unsafe_score"] - f) for r, f in zip(free, forward, strict=True)]
    print(f"\nR on smoke: {len(free)}/{len(ids)} rows free, max |free - forward| = {max(diffs):.4f} (bf16)")
    assert max(diffs) <= 0.5  # a wrong token or position is off by whole units


def test_two_runs_write_the_same_bytes(r_model, smoke, full, tmp_path):
    config, ids, prompts = smoke
    predict(r_model, ids, prompts, config, tmp_path / "predictions.jsonl")
    assert (tmp_path / "predictions.jsonl").read_bytes() == full.read_bytes()


def test_an_interrupted_evaluate_resumes_to_the_same_bytes(r_model, full, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "load_condition", lambda entry: r_model)  # two copies of R do not fit on an L4
    monkeypatch.setattr(runs, "start_run", partial(runs.start_run, runs=tmp_path))
    real, calls = generate.generate_batch, []

    def stop_at_batch_2(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) == 2:
            raise KeyboardInterrupt  # as a Colab disconnect's SIGTERM does (runs.interrupt)
        return real(*args, **kwargs)

    monkeypatch.setattr(generate, "generate_batch", stop_at_batch_2)
    with pytest.raises(KeyboardInterrupt):
        evaluate.main(["--config", str(CONFIG)])
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "interrupted"
    assert len(rows(run_dir / "predictions.jsonl")) == 16  # batch 1 kept

    evaluate.main(["--config", str(CONFIG), "--resume", str(run_dir)])
    assert len(calls) == 3  # the resume generated batch 2 only
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert (run_dir / "predictions.jsonl").read_bytes() == full.read_bytes()
