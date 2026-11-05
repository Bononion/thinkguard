"""thinkguard.evaluation.parse on the #52 cases (each case checks every field, so one string covers several cases),
and the label rule on random strings.
"""

import random
from collections import Counter

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


# Short pieces that a label check could mix up; random strings join up to 8 of them
PIECES = ["- Safety Assessment:", " safe", " unsafe", "safe", "Safe", "/unsafe", ".", " ", "\n", "\t", EOS,
          "- Categories: self_harm", "- Explanation: ", "S1"]  # fmt: skip


def test_parse_reads_a_label_only_from_an_exact_first_line():
    rng, seen = random.Random(0), Counter()
    for _ in range(5000):
        start = rng.choice(["", "- Safety Assessment:", SAFE, UNSAFE])  # most strings start near a label line
        text = start + "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 8)))
        first = text.split(EOS, 1)[0].lstrip().split("\n", 1)[0].rstrip(" ")  # protocol.md, Parsing
        expected = {SAFE: "safe", UNSAFE: "unsafe"}.get(first)
        assert parse(text).label == expected, repr(text)
        seen[expected] += 1
    assert seen["safe"] and seen["unsafe"] and seen[None], seen  # the strings reach every outcome
