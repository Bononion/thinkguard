"""Training rows and batches (#56): an example in a train mode of the prompt contract as input_ids and labels.

Run from the repo root: python -m thinkguard.training.data --mode train_critique [--split smoke] [--n 2]
Prints the first n rows of the split with their supervised tokens decoded, to check the masks by eye.
"""

import argparse

import torch

from thinkguard.data.manifest import load_split
from thinkguard.evaluation.generate import EOS_ID
from thinkguard.prompts import encode, tokenizer

IGNORE = -100  # the label the loss skips


def rows(examples, mode, max_length):
    """[{input_ids, labels, length}] of examples encoded in mode (train_label_category for L, train_critique for C).
    labels are input_ids with the prompt span set to IGNORE: the loss falls on the decision, the explanation in
    train_critique, and the final <|eot_id|>. length is the train_critique length in both modes, so L and C are grouped
    into the same batches in the same order. Raises ValueError for a row longer than max_length (protocol.md, Other
    choices: nothing is truncated).
    """
    out = []
    for e in examples:
        enc, length = encode(e, mode), len(encode(e, "train_critique").input_ids)
        if len(enc.input_ids) > max_length:
            raise ValueError(f"{e.example_id}: {len(enc.input_ids)} tokens in {mode}, max_length is {max_length}")
        prompt_end = enc.spans["prompt"][1]
        labels = [IGNORE] * prompt_end + enc.input_ids[prompt_end:]
        out.append({"input_ids": enc.input_ids, "labels": labels, "length": length})
    return out


def collate(batch):
    """rows padded on the right to the batch's longest: input_ids with <|eot_id|>, labels with IGNORE, attention_mask
    0 on padding. Returns {input_ids, labels, attention_mask} as long tensors.
    """
    n = max(len(r["input_ids"]) for r in batch)
    pad = [n - len(r["input_ids"]) for r in batch]
    return {
        "input_ids": torch.tensor([r["input_ids"] + [EOS_ID] * p for r, p in zip(batch, pad, strict=True)]),
        "labels": torch.tensor([r["labels"] + [IGNORE] * p for r, p in zip(batch, pad, strict=True)]),
        "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * p for r, p in zip(batch, pad, strict=True)]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print supervised tokens of training rows.")
    parser.add_argument("--mode", required=True, choices=("train_label_category", "train_critique"))
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args(argv)
    examples = load_split(args.split)[: args.n]
    for e, r in zip(examples, rows(examples, args.mode, max_length=2048), strict=True):
        supervised = [i for i, label in zip(r["input_ids"], r["labels"], strict=True) if label != IGNORE]
        print(f"== {e.example_id}: {len(supervised)} of {len(r['input_ids'])} tokens supervised")
        print(tokenizer().decode(supervised))


if __name__ == "__main__":
    main()
