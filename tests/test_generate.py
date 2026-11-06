"""thinkguard.evaluation.generate on the tiny model (conftest.py) in float32, with outputs forced by Script."""

import pytest
from test_score import Script

from thinkguard.evaluation.generate import EOS_ID, batches, generate, generate_batch
from thinkguard.evaluation.score import forward_score
from thinkguard.prompts import tokenizer


def ids(text):
    return tokenizer().encode(text, add_special_tokens=False)


def run(model, prompts, scripts, mode, max_new_tokens):
    """generate_batch with row b forced to write scripts[b] first."""
    n = max(map(len, prompts))
    return generate_batch(model, prompts, mode, max_new_tokens, processors=[Script(n, scripts)])


def test_batches_are_fixed_by_the_whole_list_and_stable():
    # lengths 3, 1, 3, 2, 1: sorted by (length, index) -> 1, 4, 3, 0, 2
    assert batches([[0] * 3, [0], [0] * 3, [0] * 2, [0]], 2) == [[1, 4], [3, 0], [2]]


def test_batched_generation_equals_one_prompt_at_a_time(model, prompts):
    one_by_one = generate(model, prompts, "explanation", max_new_tokens=6, batch_size=1)
    batched = generate(model, prompts, "explanation", max_new_tokens=6, batch_size=2)  # sorted, padded batches
    assert [r["generated_ids"] for r in batched] == [r["generated_ids"] for r in one_by_one]


DECISION = "- Safety Assessment: unsafe\n- Categories: self_harm"


@pytest.mark.parametrize("mode", ["decision", "explanation"])
def test_decision_mode_stops_after_the_explanation_header(model, prompts, mode):
    script = ids(f"{DECISION}\n- Explanation: It is harmful.")
    (r,) = run(model, prompts[:1], [script], mode, max_new_tokens=len(script))
    if mode == "decision":
        assert r["text"] == f"{DECISION}\n- Explanation:"
    else:
        assert r["text"].startswith(f"{DECISION}\n- Explanation: It is harmful.")
    assert r["n_generated"] == len(r["generated_ids"])


def test_a_row_that_stops_early_keeps_no_padding(model, prompts):
    # row 0 stops at the stop string, row 1 at its own EOS, row 2 runs on: generate pads rows 0 and 1 with EOS
    stop, eos = ids(f"{DECISION}\n- Explanation: x"), ids("- Safety Assessment: safe") + [EOS_ID]
    long = ids(DECISION * 3)
    a, b, c = run(model, prompts[:3], [stop, eos, long], "decision", max_new_tokens=len(long))
    assert a["text"] == f"{DECISION}\n- Explanation:" and EOS_ID not in a["generated_ids"]
    assert b["generated_ids"] == eos  # one EOS, the model's own
    assert c["n_generated"] == len(long)


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
