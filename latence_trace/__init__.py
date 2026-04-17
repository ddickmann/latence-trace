"""latence-trace: Groundedness Tracker (Beta).

Public API surface re-exports the core scoring entry points and Pydantic
models for callers embedding the service in another FastAPI process.
"""

from latence_trace.api.models import (
    GroundednessEligibility,
    GroundednessRequest,
    GroundednessResponse,
    GroundednessResponseToken,
    GroundednessScores,
)
from latence_trace.core.groundedness import (
    score_groundedness,
    score_groundedness_chunked,
)

__all__ = [
    "GroundednessEligibility",
    "GroundednessRequest",
    "GroundednessResponse",
    "GroundednessResponseToken",
    "GroundednessScores",
    "score_groundedness",
    "score_groundedness_chunked",
]

__version__ = "0.1.0"
