"""Run directories: runs/<YYYYMMDD-HHMM>-<stage>-<hash8>/manifest.json records what a run used and how it ended.

    with start_run("train", config, splits=("train", "validation")) as run_dir:
        ...

The manifest is written when the run starts, with status "running", and rewritten when the block exits: "completed"
if it returns, "interrupted" on SIGINT or SIGTERM (a Colab disconnect), "failed" on any other exception. The
exception is re-raised. A run killed without a signal it can catch keeps "running" and no end time.
start_run(..., run_dir=<an earlier run>) goes on in that directory instead (an evaluation resumed after a disconnect).
The time in the run ID and in start and end is UTC; hash8 is the first 8 hex digits of the config hash.
"""

import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
from pathlib import Path

from thinkguard.config import config_hash
from thinkguard.data.manifest import MANIFESTS
from thinkguard.paths import ROOT

RUNS = Path(os.environ.get("THINKGUARD_RUNS", ROOT / "runs"))  # point it at persistent storage on Colab
CODE = ("src", "configs", "scripts")  # an untracked file here can change what a run does


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def dirty():
    """True if a tracked file has changes or a file under CODE is untracked; other untracked files, such as earlier
    run directories, do not count.
    """
    return bool(
        git("status", "--porcelain", "--untracked-files=no")
        or git("ls-files", "--others", "--exclude-standard", "--", *CODE)
    )


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


def same_outputs(manifest):
    """The fields two runs must share for the same outputs: config, code, packages and GPU model. Not the GPU driver
    or the platform: cuBLAS ships in the torch wheel, and two Colab VMs can differ there and still agree.
    """
    gpus = manifest["environment"]["gpu"]
    return {
        "config_hash": manifest["config_hash"],
        "git_commit": manifest["git_commit"],
        "packages_sha256": manifest["packages_sha256"],
        "gpu": gpus and [line.split(",")[0] for line in gpus.splitlines()],
    }


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


@contextlib.contextmanager
def start_run(stage, config, splits=(), parent=None, runs=RUNS, manifests=MANIFESTS, run_dir=None):
    """A new run directory, yielded while the block runs. splits names the split manifests the run reads (their sha256
    is recorded), parent the checkpoint it starts from. git_dirty is dirty(). Raises FileExistsError if the run ID
    exists.
    With run_dir, that earlier run's directory instead: its manifest keeps run_id and start and is otherwise rewritten.
    Raises ValueError, before writing, if either run has git_dirty or a field of same_outputs differs.
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
        "git_dirty": dirty(),
        "config": dataclasses.asdict(config),
        "config_hash": h,
        "splits": {s: json.loads((manifests / f"{s}.json").read_text(encoding="utf-8"))["sha256"] for s in splits},
        "seed": getattr(config, "seed", None),
        "parent": parent,
        "environment": environment(),
        "packages_sha256": packages_sha256(),
    }
    if run_dir is None:
        run_dir = runs / manifest["run_id"]
        run_dir.mkdir(parents=True)
    else:
        old = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if old["git_dirty"] or manifest["git_dirty"]:
            raise ValueError(f"{run_dir}: resuming needs committed code in both runs (git_dirty)")
        was, now_ = same_outputs(old), same_outputs(manifest)
        differ = [k for k in was if was[k] != now_[k]]
        if differ:
            raise ValueError(f"{run_dir}: cannot resume, {differ} differ: {[(was[k], now_[k]) for k in differ]}")
        manifest |= {"run_id": old["run_id"], "start": old["start"]}
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
