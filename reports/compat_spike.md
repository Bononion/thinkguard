# Compatibility spike

What the target GPU and the pinned stack actually do, measured on Colab. The sections are written from the output of
the one-off `scripts/spike_*.py`, deleted after the spike; they are in git at `3b863db`.

## Environment (#3)

`bash scripts/setup_env.sh` on Colab L4 (`uv sync --locked --group gpu`), last lines:

```
torch 2.8.0+cu128
transformers 4.57.1
peft 0.18.1
cuda available True
```

## B load

`uv run python scripts/spike_load.py` (#34).

Run `20261103-1717-spike-bfbf4008`, `meta-llama/Llama-Guard-3-8B` at `7327bd9f6efbbe6101dc6cc4736302b3cbb6e425`, bf16.

| Field | Value |
|---|---|
| python | 3.11.17 |
| platform | Linux-6.6.159+-x86_64-with-glibc2.39 |
| gpu | NVIDIA L4, 580.82.07 |
| torch | 2.8.0 |
| transformers | 4.57.1 |
| peft | 0.18.1 |
| cuda | 12.8 |
| vram | 22.0 GiB |
| num_hidden_layers | 32 |
| hidden_size | 4096 |
| vocab_size | 128256 |
| attn_implementation | sdpa |
| prompt tokens | 170 |
| generated tokens | 32 |

Peak memory, allocated / reserved GiB: load 14.96 / 14.96, forward 15.03 / 15.06, generate 15.01 / 15.07.

All 32 layers have the 7 LoRA targets. Layer 0, as printed by `named_modules()`:

- `model.layers.0.self_attn.q_proj`
- `model.layers.0.self_attn.k_proj`
- `model.layers.0.self_attn.v_proj`
- `model.layers.0.self_attn.o_proj`
- `model.layers.0.mlp.gate_proj`
- `model.layers.0.mlp.up_proj`
- `model.layers.0.mlp.down_proj`

Generated text (greedy):

```
 This information is unsafe as it violates the privacy of the individual.

safe

safe

safe
```

Notes:

- The weights take 15.0 GiB of the L4's 22.0 GiB; a 170-token forward and a 32-token generation add under 0.1 GiB.
  LoRA training must fit its activations, LoRA gradients and optimizer state in the remaining ~7 GiB (#35).
- The prompt is the raw instruction and input of critique row 0, not Llama Guard's native template, so B's output
  mixes an unsafe sentence with repeated `safe` lines. This is expected before the prompt contract (#48) and is why
  #11 runs B under both our contract and its native template.

## LoRA

`uv run python scripts/spike_lora.py --model llama-guard-3-1b`, then without `--model` (8B) (#35), in one Colab L4
session with the same environment table as the B load section. Runs `20261103-1854-spike-6012d398` (1B,
`meta-llama/Llama-Guard-3-1B` at `acf7aafa60f0410f8f42b1fa35e077d705892029`) and `20261103-1855-spike-f191e8ba` (8B,
revision as above), bf16 base, r 16, alpha 32, dropout 0.05 on the 7 targets, one AdamW step (lr 1e-3) on critique row 0
with the prompt masked.

| Field | 1B | 8B |
|---|---|---|
| trainable parameters (config gives the same) | 11,272,192 | 41,943,040 |
| adapter dtype | float32 | float32 |
| step tokens, prompt + output | 171 + 276 | 170 + 276 |
| loss | 3.1542 | 1.6492 |
| lora_A grad all zero | True | True |
| adapter effect, max abs logit diff | 13.44 | 12.72 |
| reload vs trained, max abs logit diff | 0 | 0 |
| peak GiB, load | 2.83 / 2.85 | 15.11 / 15.12 |
| peak GiB, train step | 4.72 / 4.95 | 20.41 / 21.34 |
| peak GiB, reload | 2.89 / 2.91 | 15.29 / 15.40 |

Peak memory is allocated / reserved. Both runs passed every check: only LoRA parameters are trainable, the two base
weights checked (first and last LoRA-wrapped layer) are unchanged by the step, lora_B moved, and the saved adapter
reloads to bit-identical logits on rows 0-2.

Notes:

- PEFT 0.18.1 keeps the adapter in float32 on a bf16 base, so small updates are not rounded away.
- The 8B train step on a 446-token sequence peaks at 21.34 GiB reserved of the L4's 22.0 GiB, leaving under 1 GiB, so
  sequences much longer than this are unlikely to fit without gradient checkpointing; the GPU for L and C training is
  chosen in #61.
- The same critique row 0 prompt is 171 tokens with the 1B tokenizer and 170 with the 8B one.
