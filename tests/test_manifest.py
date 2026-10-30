"""thinkguard.data.manifest on rows built here."""

import json
from pathlib import Path

import pytest

from thinkguard.data.keys import prompt_hash
from thinkguard.data.manifest import load_split, write
from thinkguard.data.schema import Example
from thinkguard.data.split import PILOT_SIZE, SMOKE_SIZE

MANIFESTS = Path(__file__).parents[1] / "data/manifests"

ROWS = [
    Example(
        example_id=f"toy:{i:06d}",
        source_dataset="toy",
        source_revision="0" * 40,
        source_split="test",
        user_prompt=f"prompt {i}",
        assistant_response=None,
        task_type="prompt_only",
        gold_safety_label="safe",
        gold_categories=(),
        teacher_safety_label=None,
        teacher_categories=None,
        critique=None,
        raw_record_id=i,
        prompt_hash=prompt_hash(f"prompt {i}"),
        pair_hash=None,
        parse_status="ok",
        annotation_status="human",
        exclusion_reason=None,
    )
    for i in range(5)
]


def test_written_manifest_loads_and_an_edited_id_raises(tmp_path):
    source = tmp_path / "toy.jsonl"
    source.write_text("".join(e.to_json() + "\n" for e in ROWS), encoding="utf-8")
    write("toy", source, ROWS[1:4], tmp_path)
    assert load_split("toy", tmp_path) == ROWS[1:4]

    path = tmp_path / "toy.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["ids"][0] = ROWS[0].example_id  # still in the source, so only the sha256 catches it
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split toy"):
        load_split("toy", tmp_path)


def test_committed_splits_nest_and_have_their_sizes():
    ids = {}
    for split in ("train", "validation", "smoke", "pilot"):
        with open(MANIFESTS / f"{split}.json", encoding="utf-8") as f:
            ids[split] = json.load(f)["ids"]
    train = set(ids["train"])
    assert set(ids["smoke"]) <= train and set(ids["pilot"]) <= train
    assert not train & set(ids["validation"])
    assert len(ids["smoke"]) == SMOKE_SIZE and len(ids["pilot"]) == PILOT_SIZE
