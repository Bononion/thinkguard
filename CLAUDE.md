# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction of ThinkGuard (critique-augmented safety guard on Llama Guard 3-8B, LoRA instead of the paper's full
fine-tuning), extended with LayerSkip-style early exit and self-speculative decoding. Work is tracked as GitHub issues
`[P<phase>-<nn>]` (phases P1–P9); each issue has Tasks, Acceptance criteria and **Depends on**. Check the issue before
starting work on it.

Conditions (defined in #27): B = Llama Guard 3-8B; L = B + `label_category` LoRA (keeps categories, drops only the
critique; always named `label_category` in code); C = B + critique LoRA; R = released ThinkGuard (reference);
E-AR / E-SSD = C after early-exit adaptation, decoded greedy vs self-speculative. E-AR and E-SSD are ONE checkpoint,
two decoding modes. Optional C-FT = C + ordinary continued training.

Only `thinkguard.data` is implemented so far; `prompts`, `evaluation`, `training`, `models`, `decoding` are empty
packages waiting for P3+.

## Commands

Setup is Linux + NVIDIA GPU (`bash scripts/setup_env.sh`, needs `HF_HOME`, `CKPT_DIR`, and `HF_TOKEN` for the gated
Llama Guard / WildGuardTest). The data code and tests run anywhere with `pip install -e .`.

```bash
ruff check                                   # line length 120, py311
pytest -m "not gpu"                          # what CI runs; `gpu` marker = needs CUDA
pytest tests/test_manifest.py::test_written_manifest_loads_and_an_edited_id_raises
python -m thinkguard.data.verify_readme --committed-only   # also in CI
```

Data pipeline, run from the repo root in this order (each module's docstring says what it reads and writes):

1. `python -m thinkguard.data.normalize` → `data/interim/<source>.jsonl` (not in git)
2. `python -m thinkguard.data.filter_train` → `data/interim/critique_filtered.jsonl`, `reports/label_conflicts.jsonl`
3. `python -m thinkguard.data.manifest` → `data/manifests/<split>.json` (in git)
4. `python -m thinkguard.data.audit` → `reports/data_audit.md`
5. `python -m thinkguard.data.verify_readme [--write]` → checks/regenerates the numbers block in `data/README.md`

## Architecture and invariants

- **Sources** are pinned in `configs/data/sources.yaml` (url with `{revision}`, expected `rows` / `unsafe_rows`).
  `sources.load` reads them via `datasets.load_dataset` and raises if the counts differ. Never load a source another
  way. `configs/frozen/source_manifest.json` pins the papers and repos used.
- **`Example`** (`data/schema.py`) is the one record format for every source. Model inputs keep raw text;
  `prompt_hash` / `pair_hash` (`data/keys.py`, NFC + whitespace-collapsed) are only match keys.
- **Labels:** `teacher_*` comes from the critique, `gold_*` from human annotations (critique rows get gold from
  `beavertails_30k_train`). One is never filled from the other. Training uses the teacher label.
- **Rules live in `protocol.md`, code applies them.** Filtering (first matching `exclusion_reason` wins) and the
  grouped validation split (by `prompt_hash`, seed rule) are specified there; change the protocol first, then the code.
- **Splits are loaded only through `thinkguard.data.manifest.load_split(name)`**, which checks the sha256 of the
  rows' content (not just the ids). The four test sets are `sources.TEST_SETS`.
- **Determinism:** generated files have no timestamps, use `newline="\n"`, and a rerun must be byte-identical.
- **Test-set firewall (planned, #29/#9):** test splits may only be evaluated from configs under `configs/frozen/`.
- Tests use tiny inline fixtures and `tmp_path`; a test that hits the network is skipped when `CI=true`.

## Conventions

- Docstrings state exactly what a function returns or raises; module docstrings start with what it does, then
  "Run from the repo root: python -m ..." and what it writes. Match that style.
- Commits: `[#N] Imperative title`, body `Closes #N` or `Part of #N` (with what's left). Branches: `p<phase>-<topic>`.
  PR titles list the issues (`[#44, #45] ...`); only `Closes #N` lines in the PR body close issues.
