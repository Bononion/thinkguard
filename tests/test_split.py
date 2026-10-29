"""thinkguard.data.split on rows built here: 40 prompts with 1 to 4 rows each."""

from types import SimpleNamespace

from thinkguard.data.keys import prompt_hash
from thinkguard.data.split import group_split

EXAMPLES = [SimpleNamespace(prompt_hash=prompt_hash(f"prompt {p}")) for p in range(40) for _ in range(1 + p % 4)]


def test_no_prompt_hash_in_both_train_and_validation():
    train, validation = group_split(EXAMPLES, frac=0.2, seed=0)
    assert train and validation
    assert not {e.prompt_hash for e in train} & {e.prompt_hash for e in validation}
