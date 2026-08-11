"""
src/ddm - Drift Diffusion Model (DDM) judge package.

Public API re-exported here so callers can do:

    from src.ddm import DriftDiffusionModel, DDMConfig, TimeScale

instead of reaching into `src.ddm.drift` directly. Add new modules
(e.g. `evidence.py` for evidence-source implementations, `metrics.py` for
evaluation) to this package as the project grows, and re-export their public
symbols here.
"""

from src.ddm.drift import (
    DDMConfig,
    DriftDiffusionModel,
    EvidenceSource,
    EvidenceStep,
    TimeScale,
    default_evidence_source,
)

__all__ = [
    "DDMConfig",
    "DriftDiffusionModel",
    "EvidenceSource",
    "EvidenceStep",
    "TimeScale",
    "default_evidence_source",
]
