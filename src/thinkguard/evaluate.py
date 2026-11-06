"""Evaluation configs and the test-set firewall (protocol.md, Test-set firewall) for the evaluate command (#9)."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from thinkguard import registry
from thinkguard.data.manifest import MANIFESTS
from thinkguard.paths import ROOT

FROZEN = ROOT / "configs/frozen"
MODES = ("decision", "explanation")  # where generation stops; the prompt is always prompts' infer mode


@dataclass(frozen=True)
class EvalConfig:
    condition: str  # an entry of configs/experiment_registry.yaml
    split: str  # a manifest in data/manifests/
    mode: str
    max_new_tokens: int
    batch_size: int

    def __post_init__(self):
        """Raises ValueError for a condition not in the registry, a split without a manifest, a mode not in MODES,
        or a max_new_tokens or batch_size that is not positive.
        """
        if self.condition not in registry.load():
            raise ValueError(f"condition {self.condition!r} is not in {registry.REGISTRY}")
        if not (MANIFESTS / f"{self.split}.json").is_file():
            raise ValueError(f"split {self.split!r} has no manifest in {MANIFESTS}")
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} is not one of {MODES}")
        if self.max_new_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("max_new_tokens and batch_size must be positive")


def check_firewall(config_path, split, frozen=FROZEN):
    """Returns if split is not a test split (test_*), or if config_path is under frozen and committed unchanged: git
    tracks it and git status prints nothing for it. Raises ValueError otherwise.
    """
    if not split.startswith("test_"):
        return
    path = Path(config_path).resolve()
    if not path.is_relative_to(Path(frozen).resolve()):
        raise ValueError(f"{split}: a test split needs a config under {frozen}, not {config_path}")
    git = ["git", "-C", str(frozen)]
    tracked = subprocess.run([*git, "ls-files", "--error-unmatch", str(path)], capture_output=True).returncode == 0
    status = subprocess.run([*git, "status", "--porcelain", "--", str(path)], capture_output=True, text=True)
    if not tracked or status.returncode or status.stdout.strip():
        raise ValueError(f"{split}: {config_path} is not committed unchanged")
