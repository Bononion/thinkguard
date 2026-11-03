# Compatibility spike

What the target GPU and the pinned stack actually do, measured on Colab. The sections are written from the output of
the scripts in `scripts/`, run from this branch's commits.

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
