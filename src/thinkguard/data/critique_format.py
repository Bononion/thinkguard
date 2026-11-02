"""Parse the critique release's text: the User/Agent input, the safety label line and the categories line."""

import re

from thinkguard.data.sources import SOURCES, load_sources

INPUT = re.compile(r"^User: (.*?)\nAgent: (.*)$", re.DOTALL)
LABEL = re.compile(r"^- Safety Assessment: (safe|unsafe)\b", re.MULTILINE)
CATEGORIES = re.compile(r"^- Categories:(.*)$", re.MULTILINE)


def load_keys(path=SOURCES):
    """The 14 BeaverTails category keys, as frozen from the 30k_test release into sources.yaml."""
    return next(s for s in load_sources(path) if s["name"] == "beavertails_30k_test")["dict_keys"]["category"]


def split_pieces(value):
    return [p.strip() for p in value.split(",")]


def regroup(value, keys):
    """Split a Categories value on "," and regroup the pieces left to right by longest match against the keys.

    A key such as "violence,aiding_and_abetting,incitement" contains commas itself. Returns (keys, unknown pieces).
    """
    pieces = split_pieces(value)
    key_pieces = sorted((k.split(",") for k in keys), key=len, reverse=True)
    found, unknown, i = [], [], 0
    while i < len(pieces):
        for kp in key_pieces:
            if pieces[i : i + len(kp)] == kp:
                found.append(",".join(kp))
                i += len(kp)
                break
        else:
            unknown.append(pieces[i])
            i += 1
    return found, unknown
