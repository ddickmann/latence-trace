"""Training-time helpers for the TRACE v2 biaffine student.

Split out of ``train.py`` so the orchestrator stays short and each
primitive is unit-testable in isolation.
"""

from __future__ import annotations

from research.triangular_maxsim.student_v2.training.collate import (
    CLASS_TO_IDX,
    align_labels_to_tokens,
    build_batch,
)
from research.triangular_maxsim.student_v2.training.sampler import (
    ClassWeightedSampler,
    PairAwareBatchSampler,
    per_class_weights,
)
from research.triangular_maxsim.student_v2.training.schedule import (
    StageConfig,
    stage_configs,
)

__all__ = [
    "CLASS_TO_IDX",
    "ClassWeightedSampler",
    "PairAwareBatchSampler",
    "StageConfig",
    "align_labels_to_tokens",
    "build_batch",
    "per_class_weights",
    "stage_configs",
]
