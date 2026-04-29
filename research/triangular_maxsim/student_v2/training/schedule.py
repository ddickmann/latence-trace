"""Three-stage training schedule config.

Stage 1 - Pure distillation (all three axes). Teacher targets only.
Stage 2 - Gold + pairs. Adds gold-band CE and margin ranking.
Stage 3 - Hard-case curriculum. Heavier pair margin + gold-band CE +
          dead-weight BCE on the ambiguous subset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    name: str
    epochs: int
    lambda_support: float
    lambda_band: float
    lambda_score: float
    lambda_pair: float
    lambda_dead: float
    lambda_cov: float
    lambda_gold: float = 0.0
    use_pair_sampler: bool = False
    pair_margin: float = 0.2


def stage_configs(
    *,
    stage1_epochs: int = 2,
    stage2_epochs: int = 3,
    stage3_epochs: int = 1,
) -> list[StageConfig]:
    """Return the three-stage schedule documented in the plan.

    Args:
        stage1_epochs / stage2_epochs / stage3_epochs: tunable epoch
            counts. Defaults match the plan (2 + 3 + 1).
    """
    return [
        StageConfig(
            name="stage1_distillation",
            epochs=stage1_epochs,
            lambda_support=0.2,
            lambda_band=1.0,
            lambda_score=0.5,
            lambda_pair=0.0,
            lambda_dead=0.7,
            lambda_cov=0.5,
            lambda_gold=0.0,
            use_pair_sampler=False,
            pair_margin=0.2,
        ),
        StageConfig(
            name="stage2_gold_and_pairs",
            epochs=stage2_epochs,
            lambda_support=1.0,
            lambda_band=1.0,
            lambda_score=0.5,
            lambda_pair=1.0,
            lambda_dead=0.7,
            lambda_cov=0.5,
            lambda_gold=0.5,
            use_pair_sampler=True,
            pair_margin=0.2,
        ),
        StageConfig(
            name="stage3_hard_cases",
            epochs=stage3_epochs,
            lambda_support=0.5,
            lambda_band=0.5,
            lambda_score=0.3,
            lambda_pair=1.5,
            lambda_dead=1.2,
            lambda_cov=0.3,
            lambda_gold=1.0,
            use_pair_sampler=True,
            pair_margin=0.4,
        ),
    ]


__all__ = ["StageConfig", "stage_configs"]
