"""CPU test for infer.py's resume; loads no model. Run from the repo root inside .venv: python eval/test_infer.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import resume


def test_resume():
    rows = [{"id": f"t-{i}", "label": "safe"} for i in range(5)]
    # raw holds U+2028, which str.splitlines() would split on; json.dumps(ensure_ascii=False) leaves it unescaped
    written = [{"id": f"t-{i}", "label": "safe", "pred": "safe", "p_unsafe": 0.1, "raw": "safe "} for i in range(3)]
    whole = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written).encode("utf-8")
    # The 4th row cut off mid-write, inside a multi-byte character
    cut = '{"id": "t-3", "label": "safe", "pred": "safe", "p_unsafe": 0.1, "raw": "安'.encode("utf-8")[:-1]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "preds.jsonl")
        assert resume(out, rows) == ([], rows), "no file yet: every row runs"
        with open(out, "wb") as f:
            f.write(whole + cut)
        done, todo = resume(out, rows)
        assert open(out, "rb").read() == whole, "the cut-off last line is removed and the 3 whole rows are kept"
        assert done == written and [r["id"] for r in todo] == ["t-3", "t-4"], "the 3 written ids are skipped"
        assert resume(out, rows) == (done, todo) and open(out, "rb").read() == whole, "a whole file is left as is"


if __name__ == "__main__":
    test_resume()
    print("test_resume passed: cut-off last line removed, 3 written ids skipped, 2 rows to run")
