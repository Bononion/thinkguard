import json
import re
from pathlib import Path

MANIFEST = Path(__file__).parents[1] / "configs/frozen/source_manifest.json"
KEYS = {"name", "kind", "url", "revision", "license", "accessed_at", "used_for"}
KINDS = {"paper", "repo", "model", "dataset"}
REVISION = re.compile(r"[0-9a-f]{40}|10\.\d{4,9}/\S+")  # git/HF commit sha, or DOI


def test_source_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        sources = json.load(f)
    for s in sources:
        assert set(s) == KEYS, s.get("name")
        assert s["kind"] in KINDS, s["name"]
        assert REVISION.fullmatch(s["revision"]), s["name"]
