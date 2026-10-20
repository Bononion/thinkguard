"""Match keys for conversations across files: whitespace- and Unicode-normalized hashes.

Keys only: model inputs always keep the raw text.
"""

import hashlib
import json
import unicodedata


def normalize(text):
    """NFC, then every run of whitespace (tabs, newlines, NBSP, ...) collapsed to one space, ends stripped."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_hash(prompt, response):
    return sha256(json.dumps([normalize(prompt), normalize(response)], ensure_ascii=False))


def prompt_hash(prompt):
    return sha256(normalize(prompt))
