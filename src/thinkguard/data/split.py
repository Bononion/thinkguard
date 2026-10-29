"""Grouped validation split of the training rows, by the rules in protocol.md (Validation split).

Whole prompt_hash groups go to one side, so no prompt is in both train and validation. The same rows and seed always
give the same split.
"""

import random
from collections import Counter

FRAC = 0.05  # validation share of the rows, at least
MIN_PER_CATEGORY = 5  # validation rows every teacher category needs for a seed to pass
SEEDS = 100  # seeds 0 to 99 are tried; if none passes, choose_seed raises


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
