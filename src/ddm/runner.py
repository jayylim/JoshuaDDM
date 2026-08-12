"""
src/ddm/runner.py

Runnable smoke-test / demo script for the DDM judge. Wires together the
full real pipeline:

    src/ddm/activations.py  -- loads a real model, captures real per-token
                                hidden-state activations for a prompt
            |
            v
    src/ddm/evidence.py     -- `real_activation_stream` turns those
                                activations into a raw-signal stream;
                                `ActivationEvidenceExtractor` turns each raw
                                activation vector into one scalar (still
                                `probe_direction=None` -- mean-pooling --
                                since no trained probe exists yet)
            |
            v
    src/ddm/drift.py        -- `DriftDiffusionModel` accumulates those
                                per-step scalars until a decision boundary
                                is crossed

This script was split out of `src/ddm/drift.py`'s old `__main__` block so
that `drift.py` stays a pure DDM math/engine library (no plotting, no
demo-config, no matplotlib/transformers dependency) -- see the note at the
bottom of `drift.py`. This module is the "batteries included" demo: it runs
the DDM, prints a summary, and renders diagnostic plots of the accumulation
process.

Run directly:
    python -m src.ddm.runner
or:
    python src/ddm/runner.py

Output: two PNGs written to `src/ddm/figures/`:
    1. single_trial_trajectory.png -- one trial's accumulator trajectory,
       with the +/- decision boundaries and the boundary-crossing decision
       point marked. This is the core "does the evidence cross a boundary?"
       visualization of how the DDM behaves as a sequential judge.
    2. multi_trial_summary.png -- many independent trials run under the
       SAME config and the SAME real activation stream (only the DDM's own
       drift/noise RNG differs between trials), showing (a) all
       trajectories overlaid/color-coded by final decision, and (b) a
       histogram of decision step ("detection latency") split by decision
       outcome. This is the classic DDM reaction-time-distribution view,
       and is essential for a meta-controller: a single trajectory can't
       show whether the judge is fast-but-noisy or slow-but-reliable, or
       whether its verdicts are dominated by noise rather than by real
       evidence (see project_idea.md's accuracy / false-positive-rate /
       detection-latency evaluation metrics).
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Iterable, List

import numpy as np

if __package__ in (None, ""):
    # Allow running this file directly (e.g. `python src/ddm/runner.py`),
    # not just as a module (`python -m src.ddm.runner`). Without this,
    # `from src.ddm... import ...` fails with "No module named 'src'"
    # because the repo root isn't on sys.path in that case.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
# ^ Non-interactive backend so this script works headlessly (CI, servers,
#   no display) and always succeeds at saving PNGs via fig.savefig().

import matplotlib.pyplot as plt

from src.ddm.drift import DDMConfig, DriftDiffusionModel, TimeScale
from src.ddm.evidence import ActivationEvidenceExtractor, DEFAULT_PROMPT, real_activation_stream

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
# ^ Generated artifacts, not source -- see .gitignore.

N_MULTI_TRIALS = 200
# ^ How many independent trials to simulate for the "detection latency
#   distribution" plot. Higher = smoother histogram, slower to run.


def build_demo_config() -> DDMConfig:
    """
    THE single source of truth for demo DDM parameters. Kept in one place
    (rather than inlined per-trial) so single-trial and multi-trial runs
    below are guaranteed to use identical parameters, and to preserve the
    exact config that was previously inlined in `drift.py`'s `__main__`.
    """
    return DDMConfig(
        drift_rate=0.01,
        decision_boundary=1.0,  # 0.5-3.0 usually
        starting_point=0,  # 0.3-0.7 usually
        noise_scale=0.1,  # fixed noise scale for now
        time_step_size=1.0,
        time_scale=TimeScale.TOKEN,
        max_steps=200,
        random_seed=0,
    )


def load_activation_vectors(prompt_text: str = DEFAULT_PROMPT) -> List[np.ndarray]:
    """
    Materialize the REAL per-token activation stream for `prompt_text` (see
    `src/ddm/evidence.py`'s `real_activation_stream`, which sources from
    `src/ddm/activations.py`) into a list, once. Loaded eagerly (rather than
    left as a lazy generator) so it can be reused/cycled across every trial
    below without re-running the model's forward pass each time.
    """
    return list(real_activation_stream(prompt_text=prompt_text))


def _looped_activation_stream(
    activation_vectors: List[np.ndarray], n_steps: int
) -> Iterable[np.ndarray]:
    """
    Cycle through a short REAL per-token activation sequence to fill up to
    `n_steps` DDM steps. Needed because one prompt's real token count
    (typically far fewer than `max_steps`) is usually much shorter than the
    number of steps the DDM demo runs for -- this repeats the one real
    forward pass's activations so the accumulator has enough real signal to
    potentially reach a decision boundary. Replace with a genuine per-token
    streaming signal (e.g. one real forward pass per generated token) once
    the DDM is wired into an actual generation loop.
    """
    if not activation_vectors:
        return
    yield from itertools.islice(itertools.cycle(activation_vectors), n_steps)


def run_trial(config: DDMConfig, activation_vectors: List[np.ndarray]) -> DriftDiffusionModel:
    """Run exactly one DDM trial to completion (decision or max_steps) and return it."""
    evidence_source = ActivationEvidenceExtractor()
    # ^ probe_direction=None (default): no trained probe exists yet, so this
    #   falls back to a simple mean-of-activation placeholder. Replace with
    #   ActivationEvidenceExtractor(probe_direction=<fitted weights>,
    #   probe_bias=<fitted intercept>) once a real probe has been trained.

    ddm = DriftDiffusionModel(config=config, evidence_fn=evidence_source)
    ddm.run(_looped_activation_stream(activation_vectors, config.max_steps))
    return ddm


def plot_single_trial(ddm: DriftDiffusionModel, config: DDMConfig, save_path: Path) -> None:
    """
    Plot one trial's accumulator trajectory against the +/- decision
    boundaries, marking the starting point and (if reached) the
    boundary-crossing decision -- the core visualization of "how does
    accumulated evidence turn into a decision?"
    """
    trajectory = ddm.get_trajectory()
    steps = np.arange(len(trajectory))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(
        config.decision_boundary,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"+boundary ({config.decision_boundary:g}, 'positive')",
    )
    ax.axhline(
        -config.decision_boundary,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"-boundary (-{config.decision_boundary:g}, 'negative')",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)

    if config.non_decision_time_steps > 0:
        ax.axvspan(
            0,
            config.non_decision_time_steps - 1,
            color="lightgray",
            alpha=0.4,
            label="non-decision time",
        )

    ax.plot(steps, trajectory, color="black", linewidth=1.3, label="accumulator")
    ax.scatter(
        [0],
        [config.starting_point],
        color="black",
        zorder=5,
        s=30,
        label="starting point",
    )

    if ddm.decision is not None:
        decision_color = "crimson" if ddm.decision == "positive" else "steelblue"
        ax.scatter(
            [ddm.decision_step],
            [trajectory[ddm.decision_step]],
            color=decision_color,
            zorder=6,
            s=110,
            marker="*",
            label=f"decision: {ddm.decision} @ step {ddm.decision_step}",
        )

    ax.set_xlabel(f"decision step ({config.time_scale.value})")
    ax.set_ylabel("accumulator value")
    ax.set_title("DDM evidence accumulation: single trial")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_multi_trial(
    config: DDMConfig, activation_vectors: List[np.ndarray], n_trials: int
) -> List[DriftDiffusionModel]:
    """
    Run `n_trials` independent trials under the SAME config and the SAME
    real activation vectors.
    Vary DDM's internal drift/noise randomly (by generated RNG seed) per trial. 
    
    Reusing one real forward pass's activations across
    all trials (rather than re-running the model `n_trials` times) keeps
    this fast; trial-to-trial variability is then entirely due to the DDM's
    own stochastic accumulation (not a changing evidence source or config)
    -- exactly what the multi-trial plot below needs to show a meaningful
    decision-time distribution.
    """
    trials = []
    base_seed = config.random_seed or 0
    for i in range(n_trials):
        trial_config = dataclasses.replace(config, random_seed=base_seed + i)
        trials.append(run_trial(trial_config, activation_vectors))
    return trials


def plot_multi_trial(
    trials: List[DriftDiffusionModel], config: DDMConfig, save_path: Path
) -> None:
    """
    Two-panel figure summarizing many independent trials of the SAME config:

    Left panel:  every trial's trajectory overlaid (thin, semi-transparent,
                 color-coded by final decision) against the boundaries --
                 shows how much the accumulation process actually varies
                 run-to-run under identical parameters (pure noise effect).

    Right panel: histogram of decision step ("detection latency") split by
                 decision outcome -- the classic DDM reaction-time
                 distribution. This is the figure that actually matters for
                 judging a *meta-controller*: it shows the speed/accuracy
                 trade-off (how fast decisions happen, and how consistent
                 the outcome is) that a single trajectory plot cannot.
    """
    fig, (ax_traj, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    ax_traj.axhline(config.decision_boundary, color="crimson", linestyle="--", linewidth=1.5)
    ax_traj.axhline(-config.decision_boundary, color="steelblue", linestyle="--", linewidth=1.5)
    ax_traj.axhline(0.0, color="gray", linewidth=0.8)

    positive_latencies: list[int] = []
    negative_latencies: list[int] = []
    forced_count = 0

    for ddm in trials:
        trajectory = ddm.get_trajectory()
        steps = np.arange(len(trajectory))
        color = "crimson" if ddm.decision == "positive" else "steelblue"
        ax_traj.plot(steps, trajectory, color=color, linewidth=0.6, alpha=0.25)

        was_forced = (
            ddm.decision_step is not None
            and config.max_steps is not None
            and ddm.decision_step == config.max_steps - 1
            and abs(ddm.accumulator) < config.decision_boundary
        )
        # ^ A "forced" decision (see DriftDiffusionModel._force_decision) is
        #   one where max_steps ran out before a real boundary crossing --
        #   worth tracking separately since it reflects an inconclusive
        #   trial rather than a confident judgment.
        if was_forced:
            forced_count += 1

        if ddm.decision == "positive":
            positive_latencies.append(ddm.decision_step)
        elif ddm.decision == "negative":
            negative_latencies.append(ddm.decision_step)

    ax_traj.set_xlabel(f"decision step ({config.time_scale.value})")
    ax_traj.set_ylabel("accumulator value")
    ax_traj.set_title(f"{len(trials)} independent trials (same config, different noise)")

    max_step_seen = max((len(t.get_trajectory()) for t in trials), default=1)
    bins = np.linspace(0, config.max_steps or max_step_seen, 30)
    ax_hist.hist(
        positive_latencies,
        bins=bins,
        color="crimson",
        alpha=0.6,
        label=f"positive (n={len(positive_latencies)})",
    )
    ax_hist.hist(
        negative_latencies,
        bins=bins,
        color="steelblue",
        alpha=0.6,
        label=f"negative (n={len(negative_latencies)})",
    )
    ax_hist.set_xlabel("decision step (detection latency)")
    ax_hist.set_ylabel("trial count")
    ax_hist.set_title("Decision-time distribution by outcome")
    ax_hist.legend(loc="best", fontsize=8)

    fig.suptitle(
        "DDM as meta-controller: accumulation variability + speed/accuracy trade-off"
        + (f"  ({forced_count} forced decisions)" if forced_count else "")
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = build_demo_config()

    print(f"Loading real activations for prompt: {DEFAULT_PROMPT!r}")
    activation_vectors = load_activation_vectors()
    print(f"Got {len(activation_vectors)} real per-token activation vectors "
          f"(hidden_size={activation_vectors[0].shape[0]})")

    ddm = run_trial(config, activation_vectors)

    print("Decision:", ddm.decision)
    print("Decision step:", ddm.decision_step)
    print("Final accumulator:", round(ddm.accumulator, 4))
    print("Trajectory (first 10 steps):", np.round(ddm.get_trajectory()[:10], 4))

    single_trial_path = FIGURES_DIR / "single_trial_trajectory.png"
    plot_single_trial(ddm, config, single_trial_path)
    print(f"Saved single-trial trajectory plot to {single_trial_path}")

    trials = run_multi_trial(config, activation_vectors, n_trials=N_MULTI_TRIALS)
    multi_trial_path = FIGURES_DIR / "multi_trial_summary.png"
    plot_multi_trial(trials, config, multi_trial_path)
    print(f"Saved multi-trial summary plot to {multi_trial_path}")


if __name__ == "__main__":
    main()
