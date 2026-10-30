"""thinkguard.data.split on rows built here."""

from types import SimpleNamespace

from thinkguard.data.keys import prompt_hash
from thinkguard.data.split import LONGEST, PILOT_SIZE, SMOKE_SIZE, group_split, subset

# 40 prompts with 1 to 4 rows each
EXAMPLES = [SimpleNamespace(prompt_hash=prompt_hash(f"prompt {p}")) for p in range(40) for _ in range(1 + p % 4)]
TRAIN = [  # 60% unsafe, lengths 3 to 52 chars
    SimpleNamespace(
        example_id=f"critique:{i:06d}",
        teacher_safety_label="unsafe" if i % 5 < 3 else "safe",
        user_prompt="p" * (i % 50),
        assistant_response="r",
        critique="cc",
    )
    for i in range(2000)
]


def test_no_prompt_hash_in_both_train_and_validation():
    train, validation = group_split(EXAMPLES, frac=0.2, seed=0)
    assert train and validation
    assert not {e.prompt_hash for e in train} & {e.prompt_hash for e in validation}


def test_smoke_and_pilot_have_their_size_and_come_from_train():
    train_ids = {e.example_id for e in TRAIN}
    for size, longest in ((SMOKE_SIZE, 0), (PILOT_SIZE, LONGEST)):
        rows = subset(TRAIN, size, longest)
        assert len(rows) == size and {e.example_id for e in rows} <= train_ids
