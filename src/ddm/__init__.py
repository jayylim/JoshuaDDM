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
from src.ddm.evidence import (
    ActivationEvidenceExtractor,
    ToolCallEvidenceExtractor,
    build_default_tool_call_extractor,
    real_activation_stream,
)

__all__ = [
    "DDMConfig",
    "DriftDiffusionModel",
    "EvidenceSource",
    "EvidenceStep",
    "TimeScale",
    "default_evidence_source",
    "ActivationEvidenceExtractor",
    "ToolCallEvidenceExtractor",
    "build_default_tool_call_extractor",
    "real_activation_stream",
]

# Note: src.ddm.activations IS now transitively imported here (via
# src.ddm.evidence, which sources real activations/model defaults from it --
# see `real_activation_stream` above). This is safe: activations.py only
# *defines* functions at import time (model loading is deferred to
# `load_model_and_tokenizer()` / `get_hidden_states()`, both lru_cached), so
# `import src.ddm` no longer triggers a model load/download as a side
# effect. Actually loading the model happens the first time
# `real_activation_stream(...)` (or `src.ddm.activations` functions
# directly) is called -- e.g. in `src/ddm/runner.py`.
