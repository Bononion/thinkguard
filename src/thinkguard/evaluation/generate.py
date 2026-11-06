"""Greedy generation on prompts (token ids, as encode(..., "infer") gives) in fixed batches, with the unsafe score.

Batches depend only on the whole list of prompts, so a resumed run (#9) that skips finished batches gets the same
batches, and in bf16 the same outputs. Generation uses a fresh GenerationConfig: B's and R's generation_config.json
also stop at 128001 and 128008, while the contract stops only at <|eot_id|>.
"""

import torch
from transformers import GenerationConfig

from thinkguard.evaluation.score import LabelLogits, forward_score, free_score
from thinkguard.prompts import EXPLANATION, tokenizer

EOS_ID = 128009  # <|eot_id|>: the only stop token, and the padding (B's tokenizer has no pad token)


def batches(prompts, size):
    """Lists of indices into prompts, size each (the last may be shorter), sorted by (length, index) so a batch has
    prompts of similar length. A stable order of the whole list: rows of the same length keep their order.
    """
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    return [order[i : i + size] for i in range(0, len(order), size)]


def trim(ids, mode):
    """The generated ids up to whichever comes first: the first <|eot_id|> (kept), or in decision mode the token that
    completes "\\n- Explanation:". What generate pads after it, <|eot_id|> too, is dropped.
    """
    end = ids.index(EOS_ID) + 1 if EOS_ID in ids else len(ids)
    if mode == "decision":
        decode = tokenizer().decode
        end = next((i for i in range(1, end + 1) if EXPLANATION in decode(ids[:i])), end)
    return ids[:end]


@torch.no_grad()
def generate_batch(model, prompts, mode, max_new_tokens, processors=()):
    """One record per prompt, in order: {generated_ids, text, n_generated, unsafe_score, score_path}. text keeps
    special tokens and tokenization spaces. score_path is "free" (LabelLogits during generation) or "forward" (one
    extra batched forward for the rows whose output does not start with the prefix). processors run after
    LabelLogits (tests use them to force outputs).
    """
    n = max(map(len, prompts))
    ids = torch.tensor([[EOS_ID] * (n - len(p)) + list(p) for p in prompts], device=model.device)
    mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=model.device)
    config = GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=EOS_ID, pad_token_id=EOS_ID,
        stop_strings=[EXPLANATION] if mode == "decision" else None,
    )  # fmt: skip
    label_logits = LabelLogits(n)
    out = model.generate(
        input_ids=ids, attention_mask=mask, generation_config=config, logits_processor=[label_logits, *processors],
        tokenizer=tokenizer(),
    )  # fmt: skip
    generated = [trim(row[n:].tolist(), mode) for row in out]
    rows = label_logits.logits
    scores = [free_score(g, None if rows is None else rows[b]) for b, g in enumerate(generated)]
    forward = [b for b, s in enumerate(scores) if s is None]
    for b, s in zip(forward, forward_score(model, [prompts[b] for b in forward]) if forward else [], strict=True):
        scores[b] = s
    decode = tokenizer().decode
    return [
        {"generated_ids": g, "text": decode(g, clean_up_tokenization_spaces=False), "n_generated": len(g),
         "unsafe_score": scores[b], "score_path": "forward" if b in forward else "free"}
        for b, g in enumerate(generated)
    ]  # fmt: skip


def generate(model, prompts, mode, max_new_tokens, batch_size):
    """generate_batch over batches(prompts, batch_size); one record per prompt, in the prompts' order."""
    records = [None] * len(prompts)
    for batch in batches(prompts, batch_size):
        for i, r in zip(batch, generate_batch(model, [prompts[i] for i in batch], mode, max_new_tokens), strict=True):
            records[i] = r
    return records
