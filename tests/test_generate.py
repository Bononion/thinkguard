"""thinkguard.evaluation.generate on the tiny model (conftest.py) in float32, with outputs forced by Script."""

import json

import pytest
from test_score import Script

from thinkguard.evaluate import EvalConfig
from thinkguard.evaluation import generate
from thinkguard.evaluation.generate import EOS_ID, batches, generate_batch, predict
from thinkguard.evaluation.score import forward_score
from thinkguard.prompts import tokenizer

IDS = [f"e{i}" for i in range(5)]  # the example ids of the prompts fixture


def ids(text):
    return tokenizer().encode(text, add_special_tokens=False)


def run(model, prompts, scripts, mode, max_new_tokens):
    """generate_batch with row b forced to write scripts[b] first."""
    n = max(map(len, prompts))
    return generate_batch(model, prompts, mode, max_new_tokens, processors=[Script(n, scripts)])


def test_batches_are_fixed_by_the_whole_list_and_stable():
    # lengths 3, 1, 3, 2, 1: sorted by (length, index) -> 1, 4, 3, 0, 2
    assert batches([[0] * 3, [0], [0] * 3, [0] * 2, [0]], 2) == [[1, 4], [3, 0], [2]]


def config(**changes):
    fields = {"condition": "R", "split": "smoke", "mode": "explanation", "max_new_tokens": 6, "batch_size": 2}
    return EvalConfig(**fields | changes)


def generated(path):
    rows = map(json.loads, path.read_text(encoding="utf-8").splitlines())
    return {r["example_id"]: r["generated_ids"] for r in rows}


def test_batched_generation_equals_one_prompt_at_a_time(model, prompts, tmp_path):
    for size in (1, 2):  # batches of 2 are sorted and padded
        predict(model, IDS, prompts, config(batch_size=size), tmp_path / f"{size}.jsonl")
    assert generated(tmp_path / "2.jsonl") == generated(tmp_path / "1.jsonl")


@pytest.mark.parametrize("tail", [0, 10])  # batch 1 cut after its first row, or inside its second
def test_a_resumed_run_redoes_only_the_cut_batch_and_writes_the_same_bytes(model, prompts, tmp_path, monkeypatch, tail):
    full, cut = tmp_path / "full.jsonl", tmp_path / "cut.jsonl"
    predict(model, IDS, prompts, config(), full)
    lines = full.read_bytes().splitlines(keepends=True)  # batches of 2, 2 and 1 rows
    cut.write_bytes(b"".join(lines[:3]) + lines[3][:tail])
    sizes, real = [], generate.generate_batch
    monkeypatch.setattr(generate, "generate_batch", lambda m, p, *a: sizes.append(len(p)) or real(m, p, *a))
    predict(model, IDS, prompts, config(), cut)
    assert cut.read_bytes() == full.read_bytes()
    assert sizes == [2, 1]  # batches 1 and 2 only


@pytest.mark.parametrize("other", ["config", "example_ids"])
def test_a_file_from_another_run_raises_and_stays_as_it_was(model, prompts, tmp_path, other):
    path = tmp_path / "p.jsonl"
    predict(model, IDS, prompts, config(), path)
    before = path.read_bytes()
    args = (IDS, config(max_new_tokens=5)) if other == "config" else (IDS[::-1], config())
    with pytest.raises(ValueError, match="another run"):
        predict(model, args[0], prompts, args[1], path)
    assert path.read_bytes() == before


DECISION = "- Safety Assessment: unsafe\n- Categories: self_harm"


@pytest.mark.parametrize("mode", ["decision", "explanation"])
def test_decision_mode_stops_after_the_explanation_header(model, prompts, mode):
    script = ids(f"{DECISION}\n- Explanation: It is harmful.")
    (r,) = run(model, prompts[:1], [script], mode, max_new_tokens=len(script))
    if mode == "decision":
        assert r["text"] == f"{DECISION}\n- Explanation:"
    else:
        assert r["text"].startswith(f"{DECISION}\n- Explanation: It is harmful.")


def test_a_row_that_stops_early_keeps_no_padding(model, prompts):
    # row 0 stops at the stop string, row 1 at its own EOS, row 2 runs on: generate pads rows 0 and 1 with EOS
    stop, eos = ids(f"{DECISION}\n- Explanation: x"), ids("- Safety Assessment: safe") + [EOS_ID]
    long = ids(DECISION * 3)
    a, b, c = run(model, prompts[:3], [stop, eos, long], "decision", max_new_tokens=len(long))
    assert a["text"] == f"{DECISION}\n- Explanation:" and EOS_ID not in a["generated_ids"]
    assert b["generated_ids"] == eos  # one EOS, the model's own
    assert len(c["generated_ids"]) == len(long)


def test_each_row_gets_the_free_or_the_forward_score(model, prompts):
    free_row, forward_row = run(model, prompts[:2], [ids(DECISION), ids("unsafe\nS1")], "decision", max_new_tokens=8)
    assert (free_row["score_path"], forward_row["score_path"]) == ("free", "forward")
    # the free score comes from the logits recorded during generation, the forward one from a separate forward
    assert free_row["unsafe_score"] == pytest.approx(forward_score(model, prompts[:1])[0], abs=1e-4)
    assert forward_row["unsafe_score"] == pytest.approx(forward_score(model, prompts[1:2])[0], abs=1e-4)


def test_generation_stops_only_at_eot_not_at_the_repos_other_eos_ids(model, prompts):
    # the tiny model's generation_config.json stops at 128001, 128008 and 128009, as B's and R's do
    script = [12, 128001, 12, 128008, 12]
    (r,) = run(model, prompts[:1], [script], "explanation", max_new_tokens=len(script))
    assert r["generated_ids"] == script
