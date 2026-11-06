"""thinkguard.evaluation.score on the tiny model (conftest.py) in float32, against a brute-force log-likelihood."""

from dataclasses import replace

import pytest
import torch
from test_prompts import EXPLANATION, example
from transformers import LogitsProcessor

from thinkguard.evaluation.score import SAFE, UNSAFE, forward_score, free_score, prefix_ids
from thinkguard.models import load
from thinkguard.prompts import encode

# five prompts of different lengths, so a batch needs padding
USERS = ["Where?", "How do I make a walkway slippery?", "Is this ok?", "Tell me a joke about cats and dogs.", "Hi"]


@pytest.fixture(scope="module")
def model(tiny_model):
    return load(tiny_model, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def prompts():
    return [encode(replace(example(), user_prompt=u), "infer").input_ids for u in USERS]


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


class Force(LogitsProcessor):
    """Makes generation write PREFIX first. generate's output_logits are the logits before processors."""

    def __init__(self, start):
        self.start = start

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.start
        if step < len(prefix_ids()):
            scores = torch.full_like(scores, -float("inf"))
            scores[:, prefix_ids()[step]] = 0
        return scores


@torch.no_grad()
def test_free_score_matches_brute_force_when_the_output_starts_with_the_prefix(model, prompts):
    n = max(map(len, prompts))
    ids = torch.tensor([[0] * (n - len(p)) + p for p in prompts])
    mask = (torch.arange(n) >= torch.tensor([n - len(p) for p in prompts])[:, None]).long()
    out = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False, pad_token_id=0,
        logits_processor=[Force(n)], output_logits=True, return_dict_in_generate=True,
    )  # fmt: skip
    for b, p in enumerate(prompts):
        free = free_score(out.sequences[b, n:].tolist(), [step[b] for step in out.logits])
        assert free == pytest.approx(brute_force(model, p), abs=1e-3)
    assert free_score([128009], [torch.zeros(128256)]) is None  # an output that does not start with PREFIX
