"""TRACE v2 biaffine-MaxSim student.

Public surface re-exports the core training / inference primitives
from :mod:`architecture`. The five-head student, per-unit feature
computation, joint loss, and inference pooling variants are all
accessible from this package root.
"""

from __future__ import annotations

from research.triangular_maxsim.student_v2.architecture import (
    BiaffineStudent,
    ClassFiLM,
    CoverageHead,
    DeadWeightHead,
    GatedPhiFusion,
    LowRankBiaffine,
    PhiFeatureMLP,
    StudentConfig,
    compute_unit_features,
    loss_multi_task,
    soft_topk_logsumexp,
    top_k_mean_max,
)

__all__ = [
    "BiaffineStudent",
    "ClassFiLM",
    "CoverageHead",
    "DeadWeightHead",
    "GatedPhiFusion",
    "LowRankBiaffine",
    "PhiFeatureMLP",
    "StudentConfig",
    "compute_unit_features",
    "loss_multi_task",
    "soft_topk_logsumexp",
    "top_k_mean_max",
]
