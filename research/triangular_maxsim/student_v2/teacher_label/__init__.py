"""Teacher-labeler helpers for the v2 student distillation pipeline.

The orchestrator CLI is ``teacher_label.py`` (one level up). This
package is its engine room:

* :mod:`loaders` reads raw benchmark files (RAGTruth, Veracier) and
  normalizes each row into the common schema the student consumes.
* :mod:`extract` walks a :class:`GroundednessResponse` and pulls the
  five axis labels (turn_score, turn_band, token_support,
  dead_weight_unit, coverage_unit).
* :mod:`service` wraps :class:`GroundednessService` with a
  deterministic-config QUALITY profile ready for batched labeling.
"""

from __future__ import annotations

from research.triangular_maxsim.student_v2.teacher_label.extract import (
    extract_labels,
)
from research.triangular_maxsim.student_v2.teacher_label.loaders import (
    iter_ragtruth_train,
    iter_veracier_curated,
)
from research.triangular_maxsim.student_v2.teacher_label.service import (
    TeacherConfig,
    build_teacher,
)

__all__ = [
    "TeacherConfig",
    "build_teacher",
    "extract_labels",
    "iter_ragtruth_train",
    "iter_veracier_curated",
]
