"""The repo root, so every path resolves the same from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/thinkguard/paths.py -> repo root


def rel(path):
    """path as written into reports and manifests: relative to ROOT (posix) if inside it, else as given."""
    path = Path(path)
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
