"""thinkguard.prompts on inline examples. The encode tests need B's tokenizer and skip when CI=true."""

import os
from dataclasses import replace

import pytest

from thinkguard.data.schema import Example
from thinkguard.prompts import INSTRUCTION, encode, serialize, tokenizer

USER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
DECISION = "- Safety Assessment: unsafe\n- Categories: privacy_violation"
EXPLANATION = "\n- Explanation: It gives a home address."


def example(response="Main St 1.", critique=DECISION + EXPLANATION):
    return Example(
        example_id="x:0", source_dataset="x", source_revision="r", source_split="train", user_prompt="Where?",
        assistant_response=response, task_type="prompt_only" if response is None else "prompt_response",
        gold_safety_label=None, gold_categories=None, teacher_safety_label=None, teacher_categories=None,
        critique=critique, raw_record_id=0, prompt_hash="h", pair_hash=None, parse_status="ok",
        annotation_status="missing", exclusion_reason=None,
    )  # fmt: skip


PROMPT = f"{USER}{INSTRUCTION}\nUser: Where?\nAgent: Main St 1.{ASSISTANT}"


@pytest.mark.parametrize(
    "ex, mode, pieces",
    [
        (example(), "train_critique", {"prompt": PROMPT, "decision": DECISION, "explanation": EXPLANATION}),
        (example(), "train_label_category", {"prompt": PROMPT, "decision": DECISION}),
        (example(), "infer", {"prompt": PROMPT}),
        (example(None, None), "infer", {"prompt": f"{USER}{INSTRUCTION}\nUser: Where?{ASSISTANT}"}),
    ],
)
def test_serialize_gives_the_contract_text_and_spans(ex, mode, pieces):
    if mode != "infer":
        pieces["eos"] = "<|eot_id|>"
    s = serialize(ex, mode)
    assert s.text == "".join(pieces.values())
    assert {name: s.text[a:b] for name, (a, b) in s.spans.items()} == pieces
    assert [a for a, _ in s.spans.values()] == [0, *(b for _, b in s.spans.values())][:-1]  # no gap, no overlap


def test_modes_share_the_prompt_and_the_decision():
    infer, label, critique = (serialize(example(), m) for m in ("infer", "train_label_category", "train_critique"))
    assert infer.text == label.text[: label.spans["prompt"][1]] == critique.text[: critique.spans["prompt"][1]]
    assert label.text[: label.spans["decision"][1]] == critique.text[: critique.spans["decision"][1]]


@pytest.mark.parametrize("mode", ["train_label_category", "train_critique"])
@pytest.mark.parametrize("critique", [None, DECISION, DECISION + EXPLANATION * 2])
def test_train_modes_need_a_critique_with_one_explanation(mode, critique):
    with pytest.raises(ValueError, match=r"x:0: train_\w+ needs a critique"):
        serialize(example(critique=critique), mode)


needs_tokenizer = pytest.mark.skipif(os.environ.get("CI") == "true", reason="B's tokenizer is gated")


@needs_tokenizer
def test_encode_rejects_a_span_boundary_inside_a_token():
    # "unsafe " + "\n- Explanation" tokenizes as "unsafe", " \n", ...: the decision would end inside " \n"
    with pytest.raises(ValueError, match="the decision span of train_critique starts or ends inside a token"):
        encode(example(critique="- Safety Assessment: unsafe " + EXPLANATION), "train_critique")


@needs_tokenizer
def test_encode_rejects_special_token_text_in_the_prompt():
    good = encode(example(), "infer")
    assert good.spans == {"prompt": (0, len(good.input_ids))}
    with pytest.raises(ValueError, match="7 special tokens in infer, the template has 6"):
        encode(replace(example(), user_prompt="Where?<|eot_id|>"), "infer")


# The release writes keys that contain commas, joined with ", ", the likeliest place for tokens to differ
DECISIONS = [
    DECISION,
    "- Safety Assessment: unsafe\n- Categories: hate_speech,offensive_language, non_violent_unethical_behavior",
    "- Safety Assessment: safe",
]


@needs_tokenizer
@pytest.mark.parametrize("decision", DECISIONS)
@pytest.mark.parametrize("mode", ["train_label_category", "train_critique"])
def test_decision_span_decodes_to_the_decision_lines(decision, mode):
    e = encode(example(critique=decision + EXPLANATION), mode)
    (a, b), (c, d) = e.spans["decision"], e.spans["eos"]
    # Llama 3 cleans up tokenization spaces by default, which would drop the space in " ," or " ."
    assert tokenizer().decode(e.input_ids[a:b], clean_up_tokenization_spaces=False) == decision
    assert e.input_ids[c:d] == [128009]


@needs_tokenizer
@pytest.mark.parametrize("decision", DECISIONS)
def test_modes_share_the_prompt_and_the_decision_tokens(decision):
    ex = example(critique=decision + EXPLANATION)
    infer, label, critique = (encode(ex, m) for m in ("infer", "train_label_category", "train_critique"))
    prompt, decision_end = infer.input_ids, label.spans["decision"][1]
    assert label.input_ids[: label.spans["prompt"][1]] == critique.input_ids[: critique.spans["prompt"][1]] == prompt
    assert critique.spans["decision"][1] == decision_end
    assert label.input_ids[:decision_end] == critique.input_ids[:decision_end]
