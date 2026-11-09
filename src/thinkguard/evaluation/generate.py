"""Greedy generation on prompts (token ids, as encode(..., "infer") gives) in fixed batches, with the unsafe score.

predict writes one JSON line per prompt, batch by batch, and a rerun on the same file goes on after the last complete
batch. Batches depend only on the whole list of prompts, so the rerun gets the same batches, and in bf16 the same
outputs. Generation uses a fresh GenerationConfig: B's and R's generation_config.json also stop at 128001 and 128008,
while the contract stops only at <|eot_id|>.
"""

import json
import os
from pathlib import Path

import torch
from transformers import GenerationConfig

from thinkguard.config import config_hash
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
    ponytail: an output that starts with "- Explanation:" stops on the prompt's last "\\n" but is not cut here, so it
    keeps one padding <|eot_id|>; it has no label line anyway, so parse marks it invalid.
    """
    end = ids.index(EOS_ID) + 1 if EOS_ID in ids else len(ids)
    if mode == "decision":
        decode = tokenizer().decode
        end = next((i for i in range(1, end + 1) if EXPLANATION in decode(ids[:i])), end)
    return ids[:end]


@torch.no_grad()
def generate_batch(model, prompts, mode, max_new_tokens, processors=()):
    """One record per prompt, in order: {generated_ids, text, unsafe_score, score_path}. text keeps special tokens
    and tokenization spaces. score_path is "free" (LabelLogits during generation) or "forward" (one
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
        {"generated_ids": g, "text": decode(g, clean_up_tokenization_spaces=False), "unsafe_score": scores[b],
         "score_path": "forward" if b in forward else "free"}
        for b, g in enumerate(generated)
    ]  # fmt: skip


def done_batches(path, plan, example_ids, h):
    """How many batches of plan path holds in full, from the first; what follows them (a batch cut short by a crash)
    is cut off. Raises ValueError, before changing path, for a row of another run: another config_hash, or a batch
    whose example ids are not plan's.
    """
    if not path.exists():
        return 0
    lines = path.read_bytes().splitlines(keepends=True)
    k = i = end = 0
    while k < len(plan) and i + len(plan[k]) <= len(lines):
        chunk = lines[i : i + len(plan[k])]
        if not chunk[-1].endswith(b"\n"):  # the last line was cut mid-write
            break
        rows = [json.loads(line) for line in chunk]
        same_config = all(r["config_hash"] == h for r in rows)
        same_ids = [r["example_id"] for r in rows] == [example_ids[j] for j in plan[k]]
        if not (same_config and same_ids):
            raise ValueError(f"{path}: batch {k} is from another run")
        i, k, end = i + len(chunk), k + 1, end + sum(map(len, chunk))
    with open(path, "r+b") as f:
        f.truncate(end)
    return k


def predict(model, example_ids, prompts, config, path):
    """Appends to path one JSON line per prompt, {example_id, batch, config_hash, **generate_batch's record}, for each
    batch of batches(prompts, config.batch_size) that path does not hold in full yet (done_batches), in batch order.
    Each batch is one write, flushed and fsynced, so an interrupted run loses at most the batch it was on.
    """
    path, plan, h = Path(path), batches(prompts, config.batch_size), config_hash(config)
    start = done_batches(path, plan, example_ids, h)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for k in range(start, len(plan)):
            records = generate_batch(model, [prompts[i] for i in plan[k]], config.mode, config.max_new_tokens)
            rows = (
                {"example_id": example_ids[i], "batch": k, "config_hash": h, **r}
                for i, r in zip(plan[k], records, strict=True)
            )
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            f.flush()
            os.fsync(f.fileno())
