"""Run directories: runs/<YYYYMMDD-HHMM>-<stage>-<hash8>/manifest.json records what a run used and how it ended.

    with start_run("train", config, splits=("train", "validation")) as run_dir:
        ...

The manifest is written when the run starts, with status "running", and rewritten when the block exits: "completed"
if it returns, "interrupted" on SIGINT or SIGTERM (a Colab disconnect), "failed" on any other exception. The
exception is re-raised. A run killed without a signal it can catch keeps "running" and no end time.
The time in the run ID and in start and end is UTC; hash8 is the first 8 hex digits of the config hash.
"""

import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import platform
import signal
import subprocess

from thinkguard.config import config_hash
from thinkguard.data.manifest import MANIFESTS
from thinkguard.paths import ROOT

RUNS = ROOT / "runs"
CODE = ("src", "configs", "scripts")  # an untracked file here can change what a run does


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def packages_sha256():
    """sha256 of the sorted name==version lines of the installed distributions, the list pip freeze prints."""
    lines = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in importlib.metadata.distributions()})
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def gpu():
    """Name and driver of each GPU from nvidia-smi, or None without one."""
    try:
        query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        return subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    env = {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu()}
    for name in ("torch", "transformers", "peft"):
        try:
            env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env[name] = None
    return env


def now():
    return datetime.datetime.now(datetime.UTC)


def write(run_dir, manifest):
    with open(run_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


@contextlib.contextmanager
def start_run(stage, config, splits=(), parent=None, runs=RUNS, manifests=MANIFESTS):
    """A new run directory, yielded while the block runs. splits names the split manifests the run reads (their sha256
    is recorded), parent the checkpoint it starts from. git_dirty is true if a tracked file has changes or a file under
    CODE is untracked; other untracked files, such as earlier run directories, do not count. Raises FileExistsError if
    the run ID exists.
    Must be entered from the main thread, where SIGTERM can be handled.
    """
    start, h = now(), config_hash(config)
    manifest = {
        "run_id": f"{start:%Y%m%d-%H%M}-{stage}-{h[:8]}",
        "stage": stage,
        "status": "running",
        "start": start.isoformat(timespec="seconds"),
        "end": None,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(
            git("status", "--porcelain", "--untracked-files=no")
            or git("ls-files", "--others", "--exclude-standard", "--", *CODE)
        ),
        "config": dataclasses.asdict(config),
        "config_hash": h,
        "splits": {s: json.loads((manifests / f"{s}.json").read_text(encoding="utf-8"))["sha256"] for s in splits},
        "seed": getattr(config, "seed", None),
        "parent": parent,
        "environment": environment(),
        "packages_sha256": packages_sha256(),
    }
    run_dir = runs / manifest["run_id"]
    run_dir.mkdir(parents=True)
    write(run_dir, manifest)
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield run_dir
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        manifest["end"] = now().isoformat(timespec="seconds")
        write(run_dir, manifest)
