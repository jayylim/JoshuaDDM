"""
src/ddm/drift.py

Skeleton implementation of a Drift Diffusion Model (DDM) intended to act as an
active monitoring / meta-controller "judge" for an LLM's behaviour trajectory.

This module lives in its own package (src/ddm/), separate from
src/model-testing/, because it is deployment-agnostic: it contains no LLM
loading, prompting, or generation code. src/model-testing/ is where an actual
model gets run; src/ddm/ is the judge that will eventually consume signals
produced there (or from any other LLM deployment).

This file intentionally does NOT decide:
    - what the evidence signal actually is (e.g. probe output on hidden states,
      a text classifier score, a tool-call risk score, etc.)
    - what LLM / tokenizer / model architecture is being monitored

Instead, it provides the DDM *machinery* (accumulator, drift, noise, boundary,
step timing) so that a specific evidence source can be plugged in later via
the `EvidenceSource` callable / `evidence_fn` argument without rewriting the
core accumulation logic.

Core DDM concept:
    accumulator(t) = accumulator(t-1) + drift_rate * dt + noise_term

The process repeats, once per "decision step" (see `TimeScale` below), until
the accumulator crosses one of two decision boundaries:
    +boundary  -> classify trajectory as the "positive" class (e.g. malicious)
    -boundary  -> classify trajectory as the "negative" class (e.g. benign)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Time scale: what counts as "one step" of evidence accumulation.
# ---------------------------------------------------------------------------
class TimeScale(str, Enum):
    """
    Defines what a single DDM decision step corresponds to in an LLM's
    computation. This is what lets the same DDM engine be reused across very
    different deployment contexts (chat, agents, multi-agent systems) --
    only the TimeScale (and the evidence source feeding it) needs to change.

    TOKEN     -> one step per generated token (simplest starting point,
                 see project_idea.md "Recommended Time Scale")
    LAYER     -> one step per transformer layer within a single forward pass
                 (for "at what depth does evidence become detectable?")
    TOOL_CALL -> one step per tool/function call issued by an agent
    MESSAGE   -> one step per message passed between agents (multi-agent)
    CUSTOM    -> caller-defined; use this if none of the above fit
    """

    TOKEN = "token"
    LAYER = "layer"
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Config: all tunable DDM parameters live here so they can be swept/fit later.
# ---------------------------------------------------------------------------
@dataclass
class DDMConfig:
    """
    Container for the DDM's free parameters. Keeping these in one dataclass
    (rather than scattered as magic numbers) makes it easy to later fit them
    to data, sweep them in experiments, or expose them via a config file.
    """

    # --- Core DDM parameters (see project_idea.md "Core DDM Parameters") ---
    drift_rate: float = 0.0
    # ^ Mean rate + direction at which evidence pushes the accumulator per
    #   unit step. Positive drift_rate -> accumulator trends toward
    #   +decision_boundary (e.g. "malicious"). This will typically be
    #   *replaced per-step* by evidence_fn's output rather than used as a
    #   constant; it is kept here as a default / fallback bias term.

    decision_boundary: float = 1.0
    # ^ Distance from the starting point to EACH decision boundary. The DDM
    #   is symmetric: decision is triggered when
    #   accumulator >= +decision_boundary  (positive-class decision)
    #   or
    #   accumulator <= -decision_boundary  (negative-class decision)
    #   Lower boundary -> faster, noisier decisions. Higher boundary ->
    #   slower, more accurate decisions (classic speed/accuracy trade-off).

    starting_point: float = 0.0
    # ^ Initial accumulator value before any evidence is observed. A nonzero
    #   value encodes a prior bias toward one decision before evidence starts
    #   (e.g. a system-prompt risk prior). Must lie strictly between
    #   -decision_boundary and +decision_boundary.

    noise_scale: float = 0.1
    # ^ Standard deviation of the zero-mean Gaussian noise added at each
    #   step, representing trial-to-trial / step-to-step variability in the
    #   evidence signal (this is the "diffusion" part of "drift diffusion").

    non_decision_time_steps: int = 0
    # ^ Number of leading steps that are consumed WITHOUT affecting the
    #   accumulator (e.g. tokens before the model has "seen enough" to say
    #   anything meaningful). Mirrors the DDM's non-decision time parameter
    #   (encoding/motor delay in human RT models); mainly relevant if this
    #   model is later compared against human reaction-time data.

    time_step_size: float = 1.0
    # ^ "dt": the size of one decision step in whatever unit `time_scale`
    #   represents (default 1.0 = one token, one layer, one tool call, ...).
    #   Exposed separately from drift_rate so the same drift_rate can be
    #   reused across time scales by only changing dt (e.g. if steps are
    #   later measured in wall-clock seconds instead of token counts).

    time_scale: TimeScale = TimeScale.TOKEN
    # ^ What a "step" means for this run. See `TimeScale` docstring above.
    #   This does not change the math, only documents/labels what unit of
    #   LLM computation is being fed into the accumulator.

    max_steps: Optional[int] = None
    # ^ Safety cap on the number of evidence steps before forcing a decision
    #   (e.g. cap generation length). If None, the DDM will run until a
    #   boundary is crossed or the caller stops feeding evidence.

    random_seed: Optional[int] = None
    # ^ Seed for the internal RNG so runs/experiments are reproducible.

    def __post_init__(self) -> None:
        if not (-self.decision_boundary < self.starting_point < self.decision_boundary):
            raise ValueError(
                "starting_point must lie strictly between "
                "-decision_boundary and +decision_boundary."
            )
        if self.decision_boundary <= 0:
            raise ValueError("decision_boundary must be positive.")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative.")


# ---------------------------------------------------------------------------
# Per-step record: what evidence looked like at a single decision step.
# ---------------------------------------------------------------------------
@dataclass
class EvidenceStep:
    """
    One unit of evidence fed into the DDM at a single decision step.

    `raw_signal` is intentionally typed loosely (float | np.ndarray | None)
    because the *specific* signal (probe probability, hidden-state vector,
    tool-call risk score, ...) is not yet decided. `evidence_value` is the
    scalar value that actually gets added to the accumulator after being
    produced by `evidence_fn` (see DriftDiffusionModel.step / .run).
    """

    step_index: int
    # ^ 0-indexed position of this step within the current trial.

    time_scale: TimeScale
    # ^ Copied from the config for convenient logging/plotting later.

    raw_signal: object = None
    # ^ Placeholder for whatever unprocessed input this step carries
    #   (e.g. a hidden-state tensor, generated token string, tool-call dict).
    #   Left untyped on purpose -- fill in / replace with a concrete type
    #   (e.g. torch.Tensor) once the evidence source is implemented.

    evidence_value: float = 0.0
    # ^ Scalar evidence contribution actually used by the DDM at this step
    #   (i.e. the output of evidence_fn(raw_signal), scaled by drift_rate).

    accumulator_value: float = 0.0
    # ^ Accumulator value AFTER this step's evidence has been integrated.
    #   Stored per-step so the full trajectory can be plotted/analyzed later.


# ---------------------------------------------------------------------------
# Evidence source type: how raw model signals become a scalar DDM input.
# ---------------------------------------------------------------------------
# An EvidenceSource takes whatever raw signal arrives at a given step (a
# token, a hidden-state vector, a tool-call record, ...) and returns a single
# float representing evidence strength/direction for this step. Positive
# values should push the accumulator toward +decision_boundary; negative
# values push toward -decision_boundary.
#
# This is a placeholder signature. Swap it out later for, e.g., a trained
# probe: `lambda hidden_state: probe_model.predict_proba(hidden_state)`.
EvidenceSource = Callable[[object], float]


def default_evidence_source(raw_signal: object) -> float:
    """
    Fallback evidence function used only when no real evidence source has
    been supplied yet. Replace this with, e.g., a probe on hidden states,
    a text classifier score, or a tool-call risk heuristic.

    Currently: assumes `raw_signal` is already a float/int scalar and passes
    it through unchanged, or returns 0.0 (no evidence) if it isn't.
    """
    if isinstance(raw_signal, (int, float)):
        return float(raw_signal)
    return 0.0


# ---------------------------------------------------------------------------
# The DDM engine itself.
# ---------------------------------------------------------------------------
class DriftDiffusionModel:
    """
    Sequential evidence accumulator ("judge") for monitoring an LLM
    trajectory. Call `.step(raw_signal)` once per decision step (e.g. once
    per generated token) or `.run(signals)` to consume a whole sequence at
    once. The model keeps accumulating until a decision boundary is crossed
    or `max_steps` is reached.
    """

    def __init__(
        self,
        config: DDMConfig,
        evidence_fn: EvidenceSource = default_evidence_source,
    ) -> None:
        self.config = config
        self.evidence_fn = evidence_fn
        # ^ Injected function converting raw per-step signals into scalar
        #   evidence. Kept as an attribute (rather than hardcoded) so the
        #   same DDM engine works for tokens, layers, tool calls, etc. --
        #   only evidence_fn needs to change per use case.

        self._rng = np.random.default_rng(config.random_seed)
        # ^ Dedicated RNG instance (not global np.random) so multiple DDM
        #   instances / experiment runs don't interfere with each other.

        self.reset()

    def reset(self) -> None:
        """Reset the accumulator and history for a new trial (new prompt/response)."""
        self.accumulator: float = self.config.starting_point
        # ^ THE core mutable state of the DDM: running evidence total.

        self.step_history: list[EvidenceStep] = []
        # ^ Full per-step log, kept for later plotting (accumulator vs. step)
        #   and for post-hoc analysis of detection latency / trajectory shape.

        self.decision: Optional[str] = None
        # ^ Set to "positive" / "negative" once a boundary is crossed;
        #   remains None while the trial is still undecided.

        self.decision_step: Optional[int] = None
        # ^ Step index at which the decision was made (useful for measuring
        #   "detection latency", one of the evaluation metrics in
        #   project_idea.md).

    def _is_within_non_decision_period(self, step_index: int) -> bool:
        # ^ Steps before non_decision_time_steps don't move the accumulator,
        #   mirroring the DDM's non-decision-time parameter.
        return step_index < self.config.non_decision_time_steps

    def step(self, raw_signal: object = None) -> EvidenceStep:
        """
        Advance the DDM by exactly one decision step (e.g. one token, one
        layer, one tool call -- whatever `config.time_scale` represents).

        Returns an EvidenceStep record; check `self.decision` afterward to
        see whether a boundary was crossed.
        """
        if self.decision is not None:
            raise RuntimeError(
                "Trial already reached a decision; call reset() to start a new trial."
            )

        step_index = len(self.step_history)

        if self._is_within_non_decision_period(step_index):
            evidence_value = 0.0
            # ^ During non-decision time, evidence is recorded but ignored
            #   for accumulation purposes (accumulator stays unchanged).
        else:
            raw_evidence = self.evidence_fn(raw_signal)
            # ^ THIS is the line to replace/extend later: swap
            #   `self.evidence_fn` for a real probe / classifier / heuristic.

            drift_term = self.config.drift_rate * self.config.time_step_size
            # ^ Deterministic component of the update: constant drift scaled
            #   by the step size (dt). Represents the systematic tendency of
            #   evidence to point in one direction over this time scale.

            noise_term = (
                self._rng.normal(loc=0.0, scale=self.config.noise_scale)
                * np.sqrt(self.config.time_step_size)
            )
            # ^ Stochastic component: Gaussian noise scaled by sqrt(dt),
            #   the standard scaling for discretized diffusion processes
            #   (keeps noise variance proportional to elapsed "time").

            evidence_value = raw_evidence + drift_term + noise_term
            # ^ Combine externally-supplied evidence (raw_evidence, e.g. a
            #   probe score) with the model's own drift + noise terms. If
            #   raw_evidence alone should drive the process, set
            #   config.drift_rate = 0.0 and treat noise as measurement noise.

            self.accumulator += evidence_value
            # ^ THE update rule: accumulate evidence into the running total.

        record = EvidenceStep(
            step_index=step_index,
            time_scale=self.config.time_scale,
            raw_signal=raw_signal,
            evidence_value=evidence_value,
            accumulator_value=self.accumulator,
        )
        self.step_history.append(record)

        self._check_boundary_crossing(step_index)
        # ^ After updating, check whether we've crossed +/- decision_boundary.

        if (
            self.decision is None
            and self.config.max_steps is not None
            and step_index + 1 >= self.config.max_steps
        ):
            self._force_decision(step_index)
            # ^ Safety valve: if evidence never resolves within max_steps,
            #   force a decision (e.g. by sign of accumulator) so the judge
            #   always terminates.

        return record

    def _check_boundary_crossing(self, step_index: int) -> None:
        if self.accumulator >= self.config.decision_boundary:
            self.decision = "positive"
            # ^ Label convention: "positive" = the class associated with the
            #   +decision_boundary (e.g. "malicious"). Rename/remap this at
            #   the call site once the concrete labels are finalized.
            self.decision_step = step_index
        elif self.accumulator <= -self.config.decision_boundary:
            self.decision = "negative"
            # ^ "negative" = the class associated with -decision_boundary
            #   (e.g. "benign").
            self.decision_step = step_index

    def _force_decision(self, step_index: int) -> None:
        self.decision = "positive" if self.accumulator >= 0 else "negative"
        self.decision_step = step_index

    def run(self, raw_signals: Iterable[object]) -> Optional[str]:
        """
        Convenience wrapper to feed an entire sequence of raw signals
        (e.g. a list of per-token hidden states) through the DDM in order,
        stopping early if a decision boundary is crossed before the
        sequence ends.

        Returns the final decision ("positive" / "negative" / None if the
        sequence ended without a decision and max_steps was not set).
        """
        for raw_signal in raw_signals:
            self.step(raw_signal)
            if self.decision is not None:
                break
        return self.decision

    def get_trajectory(self) -> np.ndarray:
        """Return the accumulator value at every step so far, as an array (for plotting)."""
        return np.array([s.accumulator_value for s in self.step_history])

    def summary(self) -> dict:
        """Compact dict summary of the trial, useful for logging to outputs.jsonl-style files."""
        return {
            "time_scale": self.config.time_scale.value,
            "n_steps": len(self.step_history),
            "final_accumulator": self.accumulator,
            "decision": self.decision,
            "decision_step": self.decision_step,
            "config": dataclasses.asdict(self.config) | {"time_scale": self.config.time_scale.value},
        }


# ---------------------------------------------------------------------------
# Runnable smoke-test / demo / plotting script.
# ---------------------------------------------------------------------------
# Deliberately NOT included in this module: this file stays a pure DDM
# math/engine library (no plotting, no demo-config, no matplotlib
# dependency). The runnable demo that wires this engine up to the real
# evidence pipeline (`src/ddm/evidence.py`) and renders diagnostic plots
# lives in `src/ddm/runner.py` -- run it with:
#
#     python -m src.ddm.runner
