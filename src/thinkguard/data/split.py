"""Grouped validation split of the training rows, and smoke and pilot subsets of its train side, by the rules in
protocol.md (Validation split).

Whole prompt_hash groups go to one side, so no prompt is in both train and validation. The same rows and seed always
give the same split, and the same train rows the same subsets.
"""

import random
from collections import Counter

FRAC = 0.05  # validation share of the rows, at least
MIN_PER_CATEGORY = 5  # validation rows every teacher category needs for a seed to pass
SEEDS = 100  # seeds 0 to 99 are tried; if none passes, choose_seed raises
SMOKE_SIZE = 32
PILOT_SIZE = 1000
LONGEST = 20  # longest train rows the pilot always includes, for truncation and OOM checks


def group_split(examples, frac=FRAC, seed=0):
    """(train, validation), each in input order. The sorted distinct prompt_hash values, shuffled with
    random.Random(seed), go to validation as whole groups, in that order, until it has at least frac of the rows.
    """
    sizes = Counter(e.prompt_hash for e in examples)
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    validation, n = set(), 0
    for h in order:
        if n >= frac * len(examples):
            break
        validation.add(h)
        n += sizes[h]
    train = [e for e in examples if e.prompt_hash not in validation]
    return train, [e for e in examples if e.prompt_hash in validation]


def choose_seed(examples, frac=FRAC, min_per_category=MIN_PER_CATEGORY):
    """(seed, failures): the first seed whose validation has min_per_category rows of every category in the examples'
    teacher_categories, and {rejected seed: {category: validation rows}} for each rejected seed's categories under the
    minimum. Raises ValueError if no seed below SEEDS passes.
    """
    categories = sorted({c for e in examples for c in e.teacher_categories})
    failures = {}
    for seed in range(SEEDS):
        _, validation = group_split(examples, frac, seed)
        counts = Counter(c for e in validation for c in e.teacher_categories)
        low = {c: counts[c] for c in categories if counts[c] < min_per_category}
        if not low:
            return seed, failures
        failures[seed] = low
    raise ValueError(f"no seed below {SEEDS} gives every category {min_per_category} validation rows")


def length(e):
    return len(e.user_prompt) + len(e.assistant_response) + len(e.critique)


def subset(train, size, longest=0):
    """size rows of train, in train order, with train's unsafe ratio: the longest rows by length (ties by example_id),
    then per label a random.Random(0).sample of that label's other rows sorted by example_id, up to its count.
    """
    n_unsafe = round(size * sum(e.teacher_safety_label == "unsafe" for e in train) / len(train))
    top = sorted(train, key=lambda e: (-length(e), e.example_id))[:longest]
    ids = {e.example_id for e in top}
    for label, n in (("unsafe", n_unsafe), ("safe", size - n_unsafe)):
        pool = sorted(e.example_id for e in train if e.teacher_safety_label == label and e.example_id not in ids)
        ids.update(random.Random(0).sample(pool, n - sum(e.teacher_safety_label == label for e in top)))
    return [e for e in train if e.example_id in ids]
