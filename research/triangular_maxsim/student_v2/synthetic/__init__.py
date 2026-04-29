"""Synthetic-lane generators for the TRACE v2 student distillation dataset.

The enterprise lane is the primary generalization driver (6 training
industries + 2 OOD held-out). Smaller ``structured`` / ``code_rag`` /
``agentic_trace`` lanes follow the same template + adversarial +
paraphrase recipe and share the :mod:`quality_gate` primitives.
"""

from __future__ import annotations

from research.triangular_maxsim.student_v2.synthetic.adversarials import (
    ADVERSARIAL_KINDS,
    AdversarialOutput,
    apply_adversarial,
    entity_swap,
    numeric_flip,
    support_drop,
)
from research.triangular_maxsim.student_v2.synthetic.enterprise_generator import (
    CLASS_KEY,
    SyntheticRow,
    generate_enterprise_rows,
)
from research.triangular_maxsim.student_v2.synthetic.industry_banks import (
    INDUSTRIES,
    IndustryBank,
    get_industry,
    ood_industries,
    training_industries,
)
from research.triangular_maxsim.student_v2.synthetic.labeler import (
    EvidenceUnit,
    LabelledRow,
    chunk_evidence_into_units,
    label_synthetic_row,
)
from research.triangular_maxsim.student_v2.synthetic.paraphrase import (
    PARAPHRASE_VARIANTS,
    paraphrase,
)
from research.triangular_maxsim.student_v2.synthetic.quality_gate import (
    JaccardGateResult,
    jaccard,
    jaccard_gate,
    teacher_verdict_consistent,
)
from research.triangular_maxsim.student_v2.synthetic.templates import (
    SHAPES,
    TEMPLATES,
    Template,
    list_shapes,
    templates_for,
)

__all__ = [
    "ADVERSARIAL_KINDS",
    "AdversarialOutput",
    "CLASS_KEY",
    "EvidenceUnit",
    "INDUSTRIES",
    "IndustryBank",
    "JaccardGateResult",
    "LabelledRow",
    "PARAPHRASE_VARIANTS",
    "SHAPES",
    "SyntheticRow",
    "TEMPLATES",
    "Template",
    "apply_adversarial",
    "chunk_evidence_into_units",
    "entity_swap",
    "generate_enterprise_rows",
    "get_industry",
    "jaccard",
    "jaccard_gate",
    "label_synthetic_row",
    "list_shapes",
    "numeric_flip",
    "ood_industries",
    "paraphrase",
    "support_drop",
    "teacher_verdict_consistent",
    "templates_for",
    "training_industries",
]
