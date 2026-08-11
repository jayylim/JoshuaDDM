"""
src/ddm/evidence.py

Evidence-source implementations: these convert raw signals captured from an
LLM (or an agent built on top of one) into a single scalar "evidence" value
that `DriftDiffusionModel.step()` / `.run()` (see `src/ddm/drift.py`) can
accumulate.

Each class below implements the `EvidenceSource` interface from
`src/ddm/drift.py`, i.e. `Callable[[object], float]`: call it with a raw
signal, get back one float. That means any instance of the classes here can
be passed directly as the `evidence_fn` argument of `DriftDiffusionModel`.

Two concrete extractors are provided, matching two very different kinds of
raw signal described in project_idea.md ("Evidence Signals"):

1. `ActivationEvidenceExtractor` -- the raw signal IS already numeric: a
   specific transformer activation vector, explicitly selected by layer and
   token position elsewhere (e.g. in `src/ddm/activations.py`, via
   `hidden_states[layer_index][0, token_index, :]`). Turning it into a
   scalar here means projecting/pooling a vector down to one float.

2. `ToolCallEvidenceExtractor` -- the raw signal is NOT numeric at all: a
   tool/function call such as {"tool": "delete_file", "arguments": {...}}.
   Turning it into a scalar first requires "vectorizing" the structured
   action into numeric features, then reducing those features to one float.

Both extractors are deliberately simple/linear so their behaviour is easy to
reason about, easy to unit test, and easy to swap out later for a properly
trained probe or learned risk model once labeled data exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import numpy as np

try:
    import torch
except ImportError:  # torch is only needed if activations are torch.Tensors
    torch = None  # type: ignore[assignment]


ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float]]
# ^ Any of these are accepted wherever a raw activation vector is expected,
#   so callers don't have to convert types before invoking an extractor.

def _to_numpy_vector(value: ArrayLike) -> np.ndarray:
    """
    Normalize an activation vector coming from numpy, torch, or a plain
    list/tuple into a flat 1-D numpy float array, so downstream math
    (dot products, norms) never has to special-case the input type.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().to("cpu").float().numpy()
        # ^ .detach() drops any autograd graph, .to("cpu") ensures the tensor
        #   isn't still on a GPU device, and .float().numpy() converts it
        #   into a plain numpy array so the rest of this module never needs
        #   to branch on torch vs. numpy again.

    array = np.asarray(value, dtype=np.float64)

    if array.ndim > 1:
        array = array.reshape(-1)
        # ^ Flatten defensively in case a shape like (1, hidden_size) or
        #   (1, 1, hidden_size) is passed in instead of a plain 1-D vector
        #   (e.g. forgetting to index out the batch/sequence dimensions).

    return array


# ---------------------------------------------------------------------------
# 1. Activation-based evidence: raw_signal is already a numeric vector.
# ---------------------------------------------------------------------------
@dataclass
class ActivationEvidenceExtractor:
    """
    Converts an explicitly-identified activation vector (e.g. one
    transformer layer's hidden state for one token) into a single scalar
    evidence value for the DDM.

    Expected `raw_signal` when called: a 1-D vector of length `hidden_size`
    (or any shape flattenable to it) -- e.g., using `src/ddm/activations.py`:

        raw_signal = hidden_states[layer_index][0, -1, :]   # last token,
                                                              # chosen layer

    Mechanism: a linear "probe" --
        evidence = (probe_direction . activation_vector) + probe_bias
    This is exactly one logit of a trained linear/logistic-regression probe
    (see project_idea.md, "Evidence Signals" -> `P(malicious | hidden state)`).
    Until a real probe has been trained, `probe_direction=None` falls back to
    a simple mean of the vector's components, purely so this extractor is
    runnable end-to-end before any training has happened.
    """

    probe_direction: Optional[ArrayLike] = None
    # ^ THE trained probe weight vector (shape: (hidden_size,)). Replace this
    #   with real fitted weights once a probe (e.g. sklearn
    #   LogisticRegression.coef_, or a single trained linear layer) exists.
    #   Positive values in the projection below should indicate evidence
    #   toward the DDM's "positive" decision (e.g. malicious), matching the
    #   sign convention in DriftDiffusionModel._check_boundary_crossing.

    probe_bias: float = 0.0
    # ^ Scalar bias/intercept paired with `probe_direction`
    #   (e.g. sklearn LogisticRegression.intercept_).

    scale: float = 1.0
    # ^ Post-hoc multiplier on the raw projection. Use this to control how
    #   strongly one activation-based evidence step can move the DDM
    #   accumulator relative to other evidence sources feeding the same DDM.

    normalize_input: bool = False
    # ^ If True, L2-normalize the activation vector before projecting, so
    #   evidence depends on activation *direction* rather than *magnitude*.
    #   Useful if raw activation norms vary a lot across tokens/layers for
    #   reasons unrelated to the property being probed.

    def __call__(self, raw_signal: ArrayLike) -> float:
        vector = _to_numpy_vector(raw_signal)
        # ^ THIS line is what makes the extractor agnostic to whether the
        #   caller passes a torch.Tensor straight from a Hugging Face model
        #   (see src/ddm/activations.py) or a plain numpy array / list.

        if vector.size == 0:
            return 0.0

        if self.normalize_input:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
                # ^ Rescales the vector to unit length in place of its
                #   original magnitude, isolating direction-only evidence.

        if self.probe_direction is None:
            projection = float(np.mean(vector))
            # ^ Placeholder behaviour before a real probe exists: average
            #   the activation components. Replace by fitting
            #   `probe_direction` / `probe_bias` on labeled hidden states.
        else:
            direction = _to_numpy_vector(self.probe_direction)
            if direction.shape != vector.shape:
                raise ValueError(
                    f"probe_direction shape {direction.shape} does not match "
                    f"activation vector shape {vector.shape}."
                )

            # THE CORE PROB COMPUTATINON    
            projection = float(np.dot(direction, vector)) + self.probe_bias
            # ^ THE core probe computation: one dot product turns a
            #   high-dimensional activation vector into one signed number.

        return projection * self.scale


# ---------------------------------------------------------------------------
# 2. Action/tool-call-based evidence: raw_signal must be vectorized first.
# ---------------------------------------------------------------------------
@dataclass
class ToolCallEvidenceExtractor:
    """
    Converts a structured, non-numeric action (typically a tool/function
    call issued by an LLM agent) into a scalar evidence value. Unlike
    `ActivationEvidenceExtractor`, the raw signal here is NOT already a
    vector -- it looks like:

        raw_signal = {
            "tool": "delete_file",
            "arguments": {"path": "/etc/passwd"},
        }

    so this extractor's work is split into two explicit stages:

        1. `vectorize()`  -- turn the structured action into a numeric,
           named feature dict (one tool-identity risk feature + one binary
           keyword-hit feature per configured keyword).
        2. `__call__()`   -- reduce those features to one scalar via a
           weighted sum, mirroring the same "vector -> scalar" reduction
           `ActivationEvidenceExtractor` performs with a dot product.

    This keeps the *shape* of the two extractors symmetric (both end by
    collapsing a vector of numbers into one float) while acknowledging that
    turning an action into numbers in the first place is a distinct,
    non-trivial step that raw activations do not need.
    """

    tool_risk_weights: Mapping[str, float] = field(default_factory=dict)
    # ^ THE main configurable knob: maps a tool/function name to a base risk
    #   score, e.g. {"delete_file": 0.8, "send_email": 0.2, "read_file": 0.0}.
    #   Unknown tool names fall back to `default_tool_risk` below. Extend or
    #   replace this dict (or fit it from labeled trajectories) for your
    #   actual tool set.

    default_tool_risk: float = 0.0
    # ^ Risk score used for any tool name not present in `tool_risk_weights`.

    keyword_risk_weights: Mapping[str, float] = field(default_factory=dict)
    # ^ Maps a keyword/substring (checked case-insensitively against the
    #   stringified tool arguments) to an additional risk contribution if
    #   that keyword is present, e.g. {"rm -rf": 1.0, "password": 0.3}.

    bias: float = 0.0
    # ^ Constant offset added after combining tool + keyword features,
    #   analogous to `probe_bias` in ActivationEvidenceExtractor.

    scale: float = 1.0
    # ^ Same role as in ActivationEvidenceExtractor: scales the final scalar
    #   so tool-call evidence can be weighted relative to other evidence
    #   sources feeding the same DDM.

    def vectorize(self, raw_signal: Mapping[str, object]) -> dict[str, float]:
        """
        Stage 1: turn a structured action (tool name + arguments) into a
        named numeric feature dict. Kept as its own method (rather than
        inlined in __call__) so the intermediate feature representation can
        be inspected, logged, or unit-tested independently of the final
        scalar reduction.

        Returned features:
            "tool_risk"          -- looked up from `tool_risk_weights`
            "keyword:<keyword>"  -- 1.0 if that keyword occurs in the
                                     stringified arguments, else 0.0
        """
        tool_name = str(raw_signal.get("tool", "")).strip()
        arguments = raw_signal.get("arguments", {})
        arguments_text = str(arguments).lower()
        # ^ Cheap "vectorization" of arbitrary argument structures:
        #   stringify then lowercase, so keyword matching works regardless
        #   of whether `arguments` is a dict, list, or plain string. Replace
        #   with a real text embedding/classifier later if keyword matching
        #   proves too coarse.

        features: dict[str, float] = {
            "tool_risk": self.tool_risk_weights.get(tool_name, self.default_tool_risk),
        }

        for keyword in self.keyword_risk_weights:
            features[f"keyword:{keyword}"] = (
                1.0 if keyword.lower() in arguments_text else 0.0
            )
            # ^ One binary feature per configured keyword: THIS loop is what
            #   turns free-text arguments into a fixed-length numeric
            #   feature set (the actual "vectorizing" step for this source).

        return features

    def __call__(self, raw_signal: Mapping[str, object]) -> float:
        if not isinstance(raw_signal, Mapping):
            raise TypeError(
                "ToolCallEvidenceExtractor expects raw_signal to be a mapping "
                "like {'tool': ..., 'arguments': ...}."
            )

        features = self.vectorize(raw_signal)
        # ^ Stage 1 (see above): structured action -> numeric feature dict.

        tool_contribution = features["tool_risk"]
        keyword_contribution = sum(
            features[f"keyword:{keyword}"] * weight
            for keyword, weight in self.keyword_risk_weights.items()
        )
        # ^ Stage 2: weighted sum over the vectorized features -- the
        #   dot-product equivalent for this small, named/sparse feature set
        #   (as opposed to a dense np.dot over a large activation vector).

        evidence = (tool_contribution + keyword_contribution + self.bias) * self.scale
        return float(evidence)


# ---------------------------------------------------------------------------
# Convenience factory: a placeholder configuration to get started quickly.
# ---------------------------------------------------------------------------
def build_default_tool_call_extractor() -> ToolCallEvidenceExtractor:
    """         
    Example/starting configuration for tool-call evidence, useful as a
    placeholder before real risk weights have been decided or fit from data.
    Freely edit, extend, or replace these dictionaries for your actual tools.
    """
    return ToolCallEvidenceExtractor(
        tool_risk_weights={
            "read_file": 0.0,
            "search_web": 0.0,
            "send_email": 0.2,
            "run_shell_command": 0.6,
            "delete_file": 0.8,
        },
        default_tool_risk=0.1,
        keyword_risk_weights={
            "rm -rf": 1.0,
            "sudo": 0.4,
            "password": 0.3,
            "api_key": 0.3,
        },
        bias=-0.3,
        # ^ Negative bias so a "typical"/benign-looking tool call nets out
        #   slightly negative (toward the "benign" boundary) by default.
    )
