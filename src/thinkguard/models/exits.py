"""Early-exit logits: the state after block h through the final norm and the shared LM head (LayerSkip's exit, and
what transformers' generate(assistant_early_exit=h) drafts with).
"""


def exit_logits(model, hidden_states, exit, positions=None):
    """The logits of exit `exit` (1..num_hidden_layers), at positions (an index or mask over the batch and sequence
    dims) or everywhere. model is a Llama causal LM, PEFT-wrapped or not; hidden_states is its forward's
    output.hidden_states (output_hidden_states=True). In transformers 4.57.1 that is the embeddings, the state after
    each block before the final norm, and last the final state after it, so the last exit is not normalized again.
    Raises ValueError for an exit outside 1..num_hidden_layers.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap PEFT
    n = base.config.num_hidden_layers
    if not 1 <= exit <= n:
        raise ValueError(f"exit {exit} is not in 1..{n}")
    state = hidden_states[exit] if positions is None else hidden_states[exit][positions]
    return base.lm_head(state if exit == n else base.model.norm(state))
