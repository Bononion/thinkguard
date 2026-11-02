"""The normalized example: one record format for every source, one JSON line each in data/interim/<source>.jsonl.

Text fields keep the raw text; prompt_hash and pair_hash come from thinkguard.data.keys. A field the source has no
value for is None. Category fields are tuples, written as JSON lists.
"""

import json
from dataclasses import asdict, dataclass

CATEGORY_FIELDS = ("gold_categories", "teacher_categories")


@dataclass(frozen=True)
class Example:
    example_id: str
    source_dataset: str
    source_revision: str
    source_split: str
    user_prompt: str
    assistant_response: str | None
    task_type: str  # prompt_only | prompt_response
    gold_safety_label: str | None
    gold_categories: tuple[str, ...] | None
    teacher_safety_label: str | None
    teacher_categories: tuple[str, ...] | None
    critique: str | None
    raw_record_id: int  # row index in the raw file
    prompt_hash: str
    pair_hash: str | None
    parse_status: str
    annotation_status: str
    exclusion_reason: str | None

    def to_json(self):
        """One JSON line, fields in declaration order, non-ASCII kept as is."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line):
        d = json.loads(line)
        for f in CATEGORY_FIELDS:
            if d[f] is not None:
                d[f] = tuple(d[f])
        return cls(**d)


def read_examples(path):
    """The Examples of a JSON-lines file, one per line, in file order."""
    with open(path, encoding="utf-8") as f:
        return [Example.from_json(line) for line in f]
