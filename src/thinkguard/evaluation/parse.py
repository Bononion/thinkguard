"""Parse a model's generated text into its label, categories, explanation and flags (protocol.md, Parsing)."""

from dataclasses import dataclass

from thinkguard.data.critique_format import CONTRADICTION, load_keys, regroup
from thinkguard.prompts import EOS, EXPLANATION

LABELS = {"- Safety Assessment: safe": "safe", "- Safety Assessment: unsafe": "unsafe"}  # the only valid label lines
CATEGORIES = "- Categories:"
KEYS = load_keys()


@dataclass(frozen=True)
class Parsed:
    label: str | None  # "safe", "unsafe", or None for an invalid output
    categories: frozenset[str]  # BeaverTails keys and any piece that is not one; empty unless label is "unsafe"
    explanation: str | None
    flags: frozenset[str]  # some of invalid_label, unknown_category, contradiction, truncated


def parse(text):
    """The Parsed reading of generated text (the prompt excluded), by protocol.md, Parsing.

    The text is cut at the first <|eot_id|>. label is None unless the first line, after leading whitespace and without
    trailing spaces, is one of LABELS. categories come from a Categories line right after an unsafe label line.
    explanation is the stripped text after the first "\\n- Explanation:", or None without one. Flags never change
    the label: truncated means the text has no <|eot_id|> and does not end at the decision-only stop.
    """
    ended = EOS in text
    text = text.split(EOS, 1)[0]
    lines = text.lstrip().split("\n")
    label = LABELS.get(lines[0].rstrip(" "))
    flags = set() if label else {"invalid_label"}
    categories = frozenset()
    if label == "unsafe" and len(lines) > 1 and lines[1].startswith(CATEGORIES):
        keys, unknown = regroup(lines[1][len(CATEGORIES) :], KEYS)
        unknown = [p for p in unknown if p]
        categories = frozenset(keys + unknown)
        if unknown:
            flags.add("unknown_category")
    explanation = text.split(EXPLANATION, 1)[1].strip() if EXPLANATION in text else None
    if label and explanation and any(m.group(1).lower() != label for m in CONTRADICTION.finditer(explanation)):
        flags.add("contradiction")
    if not ended and not text.endswith(EXPLANATION):
        flags.add("truncated")
    return Parsed(label, categories, explanation, frozenset(flags))
