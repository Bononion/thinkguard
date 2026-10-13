"""Check the ThinkGuard critique training data against the known counts.

Run from the repo root after scripts/get_data.sh: python scripts/check_data.py
Exits 1 if any count differs. Safe outputs have no Categories line.
"""
import json
import re
import sys

PATH = "data/beavertails_critiques_aug.json"
HEADERS = ["Safety Assessment", "Categories", "Explanation"]
EXPECTED = {
    "unsafe": {"rows": 8526, "Safety Assessment": 8526, "Categories": 8526, "Explanation": 8526},
    "safe": {"rows": 6474, "Safety Assessment": 6474, "Categories": 0, "Explanation": 6474},
}
EXPECTED_ROWS = 15000

with open(PATH, encoding="utf-8") as f:
    rows = json.load(f)
counts = {label: dict.fromkeys(EXPECTED[label], 0) for label in EXPECTED}
unparsed = 0
for row in rows:
    m = re.search(r"^- Safety Assessment: (safe|unsafe)\b", row["output"], re.M)
    if not m:
        unparsed += 1
        continue
    c = counts[m.group(1)]
    c["rows"] += 1
    for h in HEADERS:
        c[h] += bool(re.search(rf"^- {h}:", row["output"], re.M))

ok = len(rows) == EXPECTED_ROWS and unparsed == 0 and counts == EXPECTED
print(f"rows: {len(rows)} (expected {EXPECTED_ROWS}), unparsed labels: {unparsed}")
for label, c in counts.items():
    print(f"{label:>6}: " + " | ".join(f"{k} {v}" + ("" if v == EXPECTED[label][k] else f" (expected {EXPECTED[label][k]})")
                                       for k, v in c.items()))
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
