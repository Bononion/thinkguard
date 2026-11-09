"""thinkguard.evaluation.score on the tiny model (conftest.py) in float32, against a brute-force log-likelihood."""

import pytest
import torch
from test_prompts import EXPLANATION, example
from transformers import LogitsProcessor

from thinkguard.evaluation.score import SAFE, UNSAFE, LabelLogits, forward_score, free_score, prefix_ids
from thinkguard.prompts import encode


@torch.no_grad()
def brute_force(model, prompt):
    """log p(" unsafe") - log p(" safe") after prompt + PREFIX, each from its own full sequence, unbatched."""
    lp = {}
    for label in (UNSAFE, SAFE):
        ids = torch.tensor([prompt + prefix_ids() + [label]])
        lp[label] = torch.log_softmax(model(ids).logits[0, -2], -1)[label].item()
    return lp[UNSAFE] - lp[SAFE]


@pytest.mark.parametrize("label, token", [("unsafe", UNSAFE), ("safe", SAFE)])
def test_each_label_is_one_token_after_the_prefix_in_context(label, token):
    e = encode(example(critique=f"- Safety Assessment: {label}{EXPLANATION}"), "train_label_category")
    start = e.spans["prompt"][1]
    assert prefix_ids() == [12, 19220, 37357, 25]
    assert e.input_ids[start : e.spans["decision"][1]] == [*prefix_ids(), token]


def test_forward_score_matches_brute_force_in_a_padded_batch(model, prompts):
    assert forward_score(model, prompts) == pytest.approx([brute_force(model, p) for p in prompts], abs=1e-3)


class Script(LogitsProcessor):
    """Makes row b of a generation write scripts[b] (token ids) first, then lets the model go on."""

    def __init__(self, start, scripts):
        self.start, self.scripts = start, scripts

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.start
        for b, script in enumerate(self.scripts):
            if step < len(script):
                scores[b] = -float("inf")
                scores[b, script[step]] = 0
        return scores


@torch.no_grad()
def test_free_score_matches_brute_force_when_the_output_starts_with_the_prefix(model, prompts):
    n = max(map(len, prompts))
    ids = torch.tensor([[0] * (n - len(p)) + p for p in prompts])
    mask = (torch.arange(n) >= torch.tensor([n - len(p) for p in prompts])[:, None]).long()
    label_logits = LabelLogits(n)  # before Script, so it records the model's logits, not the forced ones
    model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False, pad_token_id=0,
        logits_processor=[label_logits, Script(n, [prefix_ids()] * len(prompts))],
    )  # fmt: skip
    for b, p in enumerate(prompts):
        free = free_score(prefix_ids() + [UNSAFE], label_logits.logits[b])
        assert free == pytest.approx(brute_force(model, p), abs=1e-3)
    assert free_score([128009], label_logits.logits[0]) is None  # an output that does not start with PREFIX
    assert free_score(prefix_ids(), None) is None  # generation ended before the label step
