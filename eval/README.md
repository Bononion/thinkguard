# Evaluation

`eval/infer.py` classifies each test row as safe/unsafe and writes `results/preds/<tag>/<benchmark>__<prompt>__<mode>.jsonl` with `{id, label, pred, p_unsafe, raw}`. See its docstring for usage. `--show-prompt` needs only the tokenizer, so it runs on Windows without vLLM.

## Prompt formats

### `--prompt lg3`

- The text is `PROMPT_TEMPLATE_3` with all 14 Llama Guard 3 categories and their policy text (`with_policy=True`), the prompt the authors' ThinkGuard.py builds.
- It is copied as a plain string from llama-recipes 0.0.5.post2, which forwards to llama-cookbook 0.0.5.post1. That is what `pip install llama-recipes` gave when ThinkGuard.py was written (Feb 2025). The older 0.0.4.post1 differs by one character (S9 says "I models" instead of "AI models").
- Agent mode: role `Agent`, conversation `User: {prompt}\n\nAgent: {response}`.
- User mode: role `User`, conversation `User: {prompt}`.
- It is not the tokenizer's chat template. Llama-Guard-3-8B's template lists category names only (no policy text), puts `\n\n` after the user header, has no space before `<|eot_id|>`, and starts with a space before `<|begin_of_text|>`.

### `--prompt train`

- This is LLaMA-Factory's `llama3` template applied to one Alpaca row, with content = instruction + `\n` + input:
  `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
- Instruction: all 15,000 training rows share one instruction (631 characters). It is read from `data/beavertails_critiques_aug.json`, and the script asserts it is still unique.
- Input: agent mode uses `User: {prompt}\nAgent: {response}`; user mode uses `User: {prompt}` with no Agent line.

## Default prefixes

The prefix is appended to the prompt. Generation and the label tokens start right after it.

| `--prompt` | Default `--prefix` | Why |
|---|---|---|
| `lg3` | `"\n\n"` | Both the llama_recipes prompt and the chat template end at `assistant<|end_header_id|>` with no newline. The Llama Guard 3 model card shows the output as `"\n\nsafe"`, so without this prefix the first generated token would be `\n\n` (id 271): the `"\n"` stop string would fire at once and the first-token logprobs would not be the label's. **This comes from the model card and is unverified until `--inspect` runs on a GPU.** |
| `train` | `"- Safety Assessment:"` | Every training output starts with `- Safety Assessment: safe` or `- Safety Assessment: unsafe`. |

## BOS

- Every prompt holds exactly one BOS, at position 0.
- How: the rendered text starts with `<|begin_of_text|>`, it is tokenized with `add_special_tokens=False`, and the ids go to vLLM as `prompt_token_ids`, so vLLM never re-tokenizes.
- The authors' ThinkGuard.py also ends up with one BOS. It passes the same `<|begin_of_text|>...` string to `tokenizer()`, but the Llama Guard 3 tokenizer (and ThinkGuard's copy of it) never adds a BOS on its own: its post-processor is `ByteLevel` only. This holds under transformers 4.56.2 and 5.17.0.

## CPU checks (2026-10-15)

All checks ran on Windows with the `meta-llama/Llama-Guard-3-8B` tokenizer only (no weights).

**A. lg3 rendering vs llama_recipes: pass.**
- The reference is `build_custom_prompt(..., PROMPT_TEMPLATE_3, with_policy=True)` from llama-recipes 0.0.5.post2.
- `--show-prompt --prefix ""` prints the reference character for character, for row 0 of each benchmark and mode: beavertails agent/user, toxicchat agent/user, openai_mod user.
- With the default prefix it prints the reference + `"\n\n"`.
- Rendering every row the same way: 0 mismatches over 17,874 prompts.

**B. train rendering vs LLaMA-Factory: pass.**
- Training row 0 was run through LLaMA-Factory's own Alpaca converter (`critique` entry in `data/dataset_info.json`) and `llama3` template (`encode_multiturn`). Our rendering matches it: string identical (909 characters), token ids identical (180).
- Tokenizing prompt + prefix as one string gives LLaMA-Factory's prompt ids followed by the first 4 ids of the training target. The target's next id is the `unsafe` label token (20451).

**C. Exactly one BOS: pass.**
- Every rendered prompt plus its default prefix has exactly one BOS, at position 0.
- This covers both formats and every benchmark × mode combination above.

**D. First label tokens differ: pass.** The Llama-Guard-3-8B and Rakancorle1/ThinkGuard tokenizers give the same ids, and their whole vocabularies are identical.

| Prefix | `safe` | `unsafe` |
|---|---|---|
| lg3 `"\n\n"` | 19193 (`safe`) | 39257 (`unsafe`) |
| train `"- Safety Assessment:"` | 6220 (`Ġsafe`) | 20451 (`Ġunsafe`) |

## Colab T4

These are the GPU acceptance checks on a free Colab T4 (16 GB, no bf16), set up the same way as the smoke test in the top-level README.

Before you start:
- Set Runtime → Change runtime type → T4 GPU.
- Add `HF_TOKEN` in Secrets (the key icon) and allow notebook access. The token's account needs access to `meta-llama/Llama-Guard-3-8B`.
- Push `phase-3-eval` first, because the cell clones it from GitHub.

Cell 1:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Cell 2:

```bash
%%bash
set -euo pipefail
export HF_HOME=/content/hf_cache CKPT_DIR=/content/checkpoints   # /content is wiped with the runtime; fine for checks
cd /content
[ -d thinkguard ] || git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
git checkout phase-3-eval && git pull --ff-only
bash scripts/get_data.sh
[ -d .venv ] || bash scripts/setup_env.sh   # if setup ever fails midway: rm -rf .venv, then re-run
source .venv/bin/activate
for b in beavertails toxicchat openai_mod; do python scripts/prep_eval.py $b; done   # data/test/ is gitignored

T4="--dtype half --cpu-offload-gb 6 --enforce-eager"
LG3=meta-llama/Llama-Guard-3-8B
TG=Rakancorle1/ThinkGuard
python eval/infer.py --model $LG3 --tag lg3_zeroshot --benchmark beavertails --prompt lg3 --inspect 5 $T4
python eval/infer.py --model $TG --tag thinkguard_released --benchmark beavertails --prompt lg3 --inspect 5 $T4
python eval/infer.py --model $TG --tag thinkguard_released --benchmark beavertails --prompt train --inspect 5 $T4
python eval/infer.py --model $TG --tag thinkguard_released --benchmark beavertails --prompt lg3 --prefix $'\n\n- Safety Assessment:' --limit 50 $T4
python eval/infer.py --model $TG --tag thinkguard_released --benchmark beavertails --prompt train --limit 50 $T4

python - <<'EOF'
import json
for p in ("lg3", "train"):
    path = f"results/preds/thinkguard_released/beavertails__{p}__agent.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    parsed = [r for r in rows if r["pred"] is not None and r["p_unsafe"] is not None]
    bad = [r["id"] for r in parsed if (r["pred"] == "unsafe") != (r["p_unsafe"] > 0.5)]
    print(f"{path}: {len(rows)} rows, {len(rows) - len(parsed)} unparsed, pred/p_unsafe disagree on {bad}")
    assert len(rows) == 50 and len(parsed) == 50 and not bad
EOF
```

What to look for:
- **`--inspect`, `lg3`:** each raw output should start with `safe` or `unsafe`. That confirms the `"\n\n"` prefix.
- **`--inspect`, `train`:** each raw output should start with ` safe` or ` unsafe`, then continue on the next line.
- **The 50-row runs:** the unparsed count should be 0 or close to it.

## Zero-shot run on Colab (#11)

Run this after the Colab T4 checks pass, since their `--inspect` output confirms the `"\n\n"` prefix. Same T4 runtime and `HF_TOKEN` secret. Budget about 3–4 GPU-hours, likely over 2 sessions.

- Predictions go to Google Drive, through a link from `results/preds/zero_shot`. `infer.py` appends every 256 rows, so a disconnect loses at most one batch.
- Weights stay in local `/content/hf_cache` (~16 GB, too big for free Drive). They download again after a disconnect.
- After a disconnect, run Cells 1 and 2 again. Each `infer.py` command prints `skipping N rows already written` and runs only the rest.
- Resume goes by file name only. A run with the same `--tag`, `--benchmark` and `--prompt` but a different `--model` or `--prefix` would append to the old rows, so delete the file to start over.

Cell 1:

```python
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Cell 2:

```bash
%%bash
set -euo pipefail
export HF_HOME=/content/hf_cache CKPT_DIR=/content/checkpoints   # weights on local disk, not Drive
cd /content
[ -d thinkguard ] || git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
git checkout phase-3-eval && git pull --ff-only
bash scripts/get_data.sh
[ -d .venv ] || bash scripts/setup_env.sh
source .venv/bin/activate
for b in beavertails toxicchat openai_mod; do python scripts/prep_eval.py $b; done

DRIVE=/content/drive/MyDrive/thinkguard/preds/zero_shot
mkdir -p "$DRIVE" results/preds
if [ -e results/preds/zero_shot ] && [ ! -L results/preds/zero_shot ]; then
  echo "ERROR: results/preds/zero_shot is a folder, not the Drive link. Move it (mv results/preds/zero_shot results/preds/zero_shot.local) and rerun." >&2
  exit 1
fi
ln -sfn "$DRIVE" results/preds/zero_shot   # infer.py writes results/preds/zero_shot/*.jsonl straight to Drive
T4="--dtype half --cpu-offload-gb 6 --enforce-eager"
LG3=meta-llama/Llama-Guard-3-8B
python eval/infer.py --model $LG3 --tag zero_shot --benchmark openai_mod --prompt lg3 $T4               # user mode: no responses
python eval/infer.py --model $LG3 --tag zero_shot --benchmark beavertails --prompt lg3 $T4              # agent mode
python eval/infer.py --model $LG3 --tag zero_shot --benchmark toxicchat --prompt lg3 --mode user $T4    # its label rates the user prompt

python - <<'EOF'
import json
for b, mode in [("openai_mod", "user"), ("beavertails", "agent"), ("toxicchat", "user")]:
    path = f"results/preds/zero_shot/{b}__lg3__{mode}.jsonl"
    n = sum(1 for _ in open(f"data/test/{b}.jsonl", encoding="utf-8"))
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    parsed = [r for r in rows if r["pred"] is not None and r["p_unsafe"] is not None]
    bad = [r["id"] for r in parsed if (r["pred"] == "unsafe") != (r["p_unsafe"] > 0.5)]
    print(f"{path}: {len(rows)} of {n} rows, {len(rows) - len(parsed)} unparsed, "
          f"{len(bad)} pred/p_unsafe disagreements {bad[:5]}")
    assert len(rows) == n and len(parsed) == n and not bad
EOF
```

Then, on your machine in the repo root:

1. Download the three files from Drive folder `MyDrive/thinkguard/preds/zero_shot/` into `results/preds/zero_shot/`.
2. Run `python eval/metrics.py --tag zero_shot`. It writes the zero-shot rows of `results/scores.csv`.

## Versions (pins for the training environment)

Check B ran against these versions:
- `llamafactory==0.9.5`
- `transformers==4.56.2`

They match `requirements.txt`. The check environment was otherwise CPU-only: torch 2.8.0+cpu, torchaudio 2.8.0, tokenizers 0.22.2, Python 3.11.
