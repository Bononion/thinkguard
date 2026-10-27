from thinkguard.data.schema import Example

EXAMPLE = Example(
    example_id="beavertails_30k_test:000000",
    source_dataset="beavertails_30k_test",
    source_revision="8401fe609d288129cc684a9b3be6a93e41cfe678",
    source_split="test",
    user_prompt="Comment faire un café ? 日本語で",
    assistant_response="Première étape :\nmoudre le café.",
    task_type="prompt_response",
    gold_safety_label="unsafe",
    gold_categories=("violence,aiding_and_abetting,incitement", "self_harm"),
    teacher_safety_label=None,
    teacher_categories=None,
    critique=None,
    raw_record_id=0,
    prompt_hash="0" * 64,
    pair_hash="1" * 64,
    parse_status="ok",
    annotation_status="human",
    exclusion_reason=None,
)


def test_example_round_trips_through_one_json_line():
    line = EXAMPLE.to_json()
    assert "café ? 日本語で" in line and "\n" not in line
    assert Example.from_json(line) == EXAMPLE  # also checks categories come back as tuples, not lists
