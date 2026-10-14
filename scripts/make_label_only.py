"""Build the label-only training set: the critique data with each Explanation removed.

Run from the repo root after scripts/get_data.sh: python scripts/make_label_only.py
Writes data/label_only.json. Rows differ from the critique rows only in the dropped Explanation.
"""
import json
import re
import sys
from collections import Counter

SRC = "data/beavertails_critiques_aug.json"
OUT = "data/label_only.json"
LABEL = re.compile(r"^- Safety Assessment: (safe|unsafe)\b", re.M)

src = json.load(open(SRC, encoding="utf-8"))
out = []
for n, row in enumerate(src):
    lines = row["output"].split("\n")
    cut = next((i for i, line in enumerate(lines) if line.startswith("- Explanation:")), None)
    if cut is None:
        sys.exit(f"row {n}: no '- Explanation:' line")
    out.append({**row, "output": "\n".join(lines[:cut]).rstrip()})

counts = Counter(m.group(1) if (m := LABEL.search(r["output"])) else None for r in out)
assert len(out) == 15000, f"{len(out)} rows, expected 15000"
assert counts == {"unsafe": 8526, "safe": 6474}, f"label counts {dict(counts)}"
assert not any("Explanation" in r["output"] for r in out), "an output still contains 'Explanation'"
for n, (s, r) in enumerate(zip(src, out)):
    truncated = re.split(r"^- Explanation:", s["output"], maxsplit=1, flags=re.M)[0].rstrip()
    assert r["output"] == truncated and s["output"].startswith(r["output"]), f"row {n}: output is not the truncated source"
    assert r["instruction"] == s["instruction"] and r["input"] == s["input"], f"row {n}: instruction/input changed"

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=4, ensure_ascii=False)
print(f"wrote {OUT}: {len(out)} rows ({counts['unsafe']} unsafe, {counts['safe']} safe)")
