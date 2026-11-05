"""thinkguard.evaluation.parse on the #52 cases. Each case checks every field, so one string covers several cases."""

import pytest

from thinkguard.evaluation.parse import parse

SAFE, UNSAFE, EOS = "- Safety Assessment: safe", "- Safety Assessment: unsafe", "<|eot_id|>"
HATE, NONVIOLENT = "hate_speech,offensive_language", "non_violent_unethical_behavior"


@pytest.mark.parametrize(
    "text, label, categories, explanation, flags",
    [
        # 'unsafe' contains 'safe'; an explanation that mentions safe without a verdict
        (f"{UNSAFE}\n- Categories: self_harm\n- Explanation: It is not safe.{EOS}", "unsafe", {"self_harm"},
         "It is not safe.", set()),
        # leading whitespace (R's output); keys with commas joined with ", " as in the release; EOS at the end
        (f"\n\n{UNSAFE}\n- Categories: {HATE}, {NONVIOLENT}{EOS}", "unsafe", {HATE, NONVIOLENT}, None, set()),
        # a repeated category, another order, a piece that is not a key
        (f"{UNSAFE}\n- Categories: self_harm, S1, privacy_violation, self_harm{EOS}", "unsafe",
         {"self_harm", "S1", "privacy_violation"}, None, {"unknown_category"}),
        # safe with categories: they are ignored
        (f"{SAFE}\n- Categories: self_harm{EOS}", "safe", set(), None, set()),
        # an explanation with a verdict for the other label: flagged, the label stays
        (f"{SAFE}\n- Explanation: The conversation is unsafe.{EOS}", "safe", set(), "The conversation is unsafe.",
         {"contradiction"}),
        # malformed: the template copied
        (f"- Safety Assessment: safe/unsafe{EOS}", None, set(), None, {"invalid_label"}),
        # Llama Guard style
        (f"unsafe\nS1{EOS}", None, set(), None, {"invalid_label"}),
        # EOS first, so the output is empty
        (f"{EOS}{SAFE}", None, set(), None, {"invalid_label"}),
        # an empty Categories line
        (f"{UNSAFE}\n- Categories: {EOS}", "unsafe", set(), None, set()),
        # decision-only generation stopped at "\n- Explanation:": not truncated
        (f"{UNSAFE}\n- Categories: self_harm\n- Explanation:", "unsafe", {"self_harm"}, "", set()),
        # no EOS and no stop: generation hit max_new_tokens
        (f"{SAFE}\n- Explanation: It is", "safe", set(), "It is", {"truncated"}),
    ],
)  # fmt: skip
def test_parse(text, label, categories, explanation, flags):
    p = parse(text)
    assert (p.label, p.categories, p.explanation, p.flags) == (label, categories, explanation, flags)
