"""Download raw data sources at pinned revisions and record them in data/raw_manifest.json.

Run from the repo root: python -m thinkguard.data.download --config configs/data/sources.yaml
A source whose file already matches its manifest sha256 is skipped. A file that differs from its manifest entry
is an error, and the manifest is left unchanged.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

RAW_DIR = Path("data/raw")
MANIFEST = Path("data/raw_manifest.json")


class HashMismatch(Exception):
    pass


def read_json(path):
    """Rows of a file holding a top-level JSON array."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a top-level JSON array, got {type(data).__name__}")
    return data


def read_jsonl_gz(path):
    """Rows of a gzipped file with one JSON object per non-empty line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


READERS = {"json": (read_json, "array"), "jsonl.gz": (read_jsonl_gz, "jsonl")}  # format -> (reader, schema type)


def describe(source, rows):
    """Manifest fields computed from the rows: rows, schema, and unsafe_rows / dict_keys if the source asks."""
    schema = {"type": READERS[source["format"]][1]}
    if rows and isinstance(rows[0], dict):
        schema["row_keys"] = sorted(rows[0])
    info = {"rows": len(rows), "schema": schema}
    if "unsafe" in source:
        field, equals = source["unsafe"]["field"], source["unsafe"]["equals"]
        info["unsafe_rows"] = sum(r[field] == equals for r in rows)
    if "dict_keys" in source:
        field = source["dict_keys"]
        keys = list(rows[0][field])
        for i, r in enumerate(rows):
            if set(r[field]) != set(keys):
                raise ValueError(f"source {source['name']}: row {i} {field} keys {list(r[field])} != row 0's {keys}")
        info["dict_keys"] = {field: keys}
    return info


def load_sources(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest, path):
    """Write via a temp file and rename, so a crash never leaves a half-written manifest."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)


def check(name, expected, actual, path):
    if expected != actual:
        raise HashMismatch(f"source {name}: {path} has sha256 {actual}, manifest has {expected}")


def freeze(source, raw_dir=RAW_DIR, manifest_path=MANIFEST):
    """Download one source unless its file already matches the manifest, then record it there."""
    name, fmt = source["name"], source["format"]
    if fmt not in READERS:
        raise ValueError(f"source {name}: unknown format {fmt!r} (known: {', '.join(READERS)})")
    url = source["url"].format(revision=source["revision"])
    target = raw_dir / name / Path(urlparse(url).path).name
    manifest = load_manifest(manifest_path)
    expected = manifest.get(name, {}).get("sha256")

    if expected and target.exists():
        check(name, expected, sha256(target), target)
        print(f"{name}: {target} matches the manifest, skipping")
        return

    # Download next to the target and rename only once it checks out, so a cut-off file never looks complete
    part = target.with_name(target.name + ".part")
    print(f"{name}: downloading {url}", flush=True)
    fetch(url, part)
    try:
        digest = sha256(part)
        if expected:
            check(name, expected, digest, url)
        info = describe(source, READERS[fmt][0](part))
    except Exception:
        part.unlink()
        raise
    os.replace(part, target)

    manifest[name] = {
        "url": url,
        "revision": source["revision"],
        "sha256": digest,
        "bytes": target.stat().st_size,
        **info,
        "split": source["split"],
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_manifest(manifest, manifest_path)
    print(f"{name}: saved {target} ({info['rows']} rows, sha256 {digest})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Download raw data sources and record them in the manifest.")
    p.add_argument("--config", required=True, help="YAML list of sources, e.g. configs/data/sources.yaml")
    args = p.parse_args(argv)
    for source in load_sources(args.config):
        freeze(source)


if __name__ == "__main__":
    main()
