"""
src/ddm/activations.py

Loads a causal LM + tokenizer and captures per-token hidden-state
activations for a prompt. This is the REAL "raw signal" source that
`src/ddm/evidence.py` wraps with `ActivationEvidenceExtractor` to build an
evidence stream for `DriftDiffusionModel` (see `src/ddm/drift.py`) -- i.e.
this module answers "where do the numbers come from?", `evidence.py`
answers "how do the numbers become one scalar per step?".

Everything here is wrapped in functions (model loading, forward pass,
per-token iteration) rather than executed at import time. This matters
because `src/ddm/evidence.py` now imports directly from this module: if
loading the model happened at import time (as it used to, in a bare
top-level script), then `import src.ddm.evidence` -- and therefore
`import src.ddm` -- would trigger a multi-GB model load/download as a side
effect of merely importing a package. Loading is now deferred until
`load_model_and_tokenizer()` (or a function that calls it) is actually
invoked, and results are cached so repeated calls (e.g. one per DDM trial)
don't reload the model or re-run an identical forward pass.

Run this file directly for the same standalone smoke test as before:
    python -m src.ddm.activations
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # torch/transformers are only needed to load a real model
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# ^ THE model used to source real activations. Swap this out for whatever
#   model is actually being monitored elsewhere (e.g. in
#   src/model-testing/runs/test_agent.py).

DEFAULT_LAYER_INDEX = -1
# ^ Which transformer layer's hidden state to read (-1 = last layer).
#   `outputs.hidden_states` has one entry per layer (plus the embedding
#   layer), each of shape (batch_size, sequence_length, hidden_size).

DEFAULT_PROMPT = "Explain recursion in one paragraph."
# ^ Placeholder prompt; can be imported from a prompts file later (see
#   src/model-testing/runs/prompts.jsonl for the format used elsewhere).


@lru_cache(maxsize=4)
def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load (and cache) a tokenizer + causal LM pair by name. `lru_cache` means
    calling this repeatedly with the same `model_name` (e.g. once per DDM
    trial) only loads the model once, rather than on every call.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers and torch are required to load a real model "
            "(see requirements.txt). Use a synthetic evidence stream "
            "instead if they aren't installed."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=32)
def get_hidden_states(prompt_text: str = DEFAULT_PROMPT, model_name: str = DEFAULT_MODEL_NAME):
    """
    Run one forward pass over `prompt_text` and return the full tuple of
    per-layer hidden states -- `hidden_states[layer][0, token_index, :]` for
    any layer/token combination -- alongside the tokenizer's tokenized
    inputs (useful for mapping token_index back to actual tokens/text).

    Cached by (prompt_text, model_name) since the forward pass is
    deterministic (`model.eval()`, no sampling) and this may be called once
    per DDM trial in `evidence.py` / `runner.py`.
    """
    tokenizer, model = load_model_and_tokenizer(model_name)
    tokenized_inputs = tokenizer(prompt_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **tokenized_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    return outputs.hidden_states, tokenized_inputs


def iter_token_activations(
    hidden_states: Tuple["torch.Tensor", ...],
    layer_index: int = DEFAULT_LAYER_INDEX,
) -> Iterable[np.ndarray]:
    """
    Yield one activation vector per token position, in order, from a chosen
    layer of `hidden_states` (as returned by `get_hidden_states`) -- i.e.
    `hidden_states[layer_index][0, token_index, :]` for each token_index in
    the sequence. This is what lets a `TimeScale.TOKEN` DDM step correspond
    to walking through a real prompt's tokens one at a time (see
    `src/ddm/evidence.py`'s `real_activation_stream`).
    """
    layer_states = hidden_states[layer_index][0]
    # ^ Index 0 selects the (only) batch element, leaving shape
    #   (sequence_length, hidden_size).
    for token_index in range(layer_states.shape[0]):
        vector = layer_states[token_index].detach().to("cpu")
        yield vector.float().numpy()
        # ^ .float() casts down from the model's native dtype (e.g.
        #   bfloat16, which numpy cannot represent directly) to float32
        #   before conversion.


if __name__ == "__main__":
    hidden_states, tokenized_inputs = get_hidden_states(DEFAULT_PROMPT)
    token_index = -1  # last token in the sequence; use if time scale is token-based

    print(hidden_states[DEFAULT_LAYER_INDEX][0, token_index].shape)
