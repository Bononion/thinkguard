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

## Versions (pins for the training environment)

Check B ran against these versions:
- `llamafactory==0.9.5`
- `transformers==4.56.2`

They match `requirements.txt`. The check environment was otherwise CPU-only: torch 2.8.0+cpu, torchaudio 2.8.0, tokenizers 0.22.2, Python 3.11.
