"""The training config (protocol.md, Training): what may differ between runs. Fixed settings are in thinkguard.train."""

from dataclasses import dataclass

from thinkguard.data.manifest import MANIFESTS

MODES = {"train_label_category": "L", "train_critique": "C"}  # prompt contract mode -> condition


@dataclass(frozen=True)
class TrainConfig:
    mode: str  # a key of MODES
    train_split: str
    eval_split: str  # scored for checkpoint selection (protocol.md, Checkpoint selection)
    max_length: int
    lr: float
    epochs: int
    evals_per_epoch: int
    microbatch: int
    grad_accum: int
    seed: int

    def __post_init__(self):
        """Raises ValueError for a mode not in MODES, a split without a manifest, a test split, an lr that is not a
        float (YAML reads 2e-4 as a string; write 2.0e-4), or a count that is not positive.
        """
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} is not one of {list(MODES)}")
        for split in (self.train_split, self.eval_split):
            if split.startswith("test_") or not (MANIFESTS / f"{split}.json").exists():
                raise ValueError(f"{split!r} is a test split or has no manifest")
        if not isinstance(self.lr, float):
            raise ValueError(f"lr {self.lr!r} is not a float")
        counts = (self.max_length, self.epochs, self.evals_per_epoch, self.microbatch, self.grad_accum)
        if min(counts) < 1:
            raise ValueError(f"max_length, epochs, evals_per_epoch, microbatch, grad_accum must be positive: {counts}")
