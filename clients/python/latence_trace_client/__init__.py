"""Official Python SDK for the latence-trace Groundedness Tracker.

Two top-level clients:

- :class:`LatenceTraceClient` -- synchronous, suitable for scripts,
  Jupyter, CLI tools, and request-scoped server frameworks.
- :class:`AsyncLatenceTraceClient` -- asyncio version with the same
  surface, suitable for FastAPI / aiohttp servers and pipelines.

Both implementations share retry, backoff, OTel propagation, and
typed Pydantic models, so calling code stays identical regardless of
the runtime.
"""

from latence_trace_client.async_client import AsyncLatenceTraceClient, AsyncTraceSession
from latence_trace_client.client import LatenceTraceClient, TraceSession
from latence_trace_client.errors import (
    LatenceTraceAPIError,
    LatenceTraceAuthError,
    LatenceTraceRateLimited,
    LatenceTraceServerError,
    LatenceTraceTimeout,
    LatenceTraceValidationError,
)
from latence_trace_client.models import (
    AttributionMode,
    ComplianceCustomLabel,
    ComplianceEntity,
    ComplianceLabelMode,
    ComplianceRedactionMode,
    ComplianceRedactionRequest,
    ComplianceRedactionResponse,
    ComplianceUsage,
    CompressionResponse,
    GroundednessRequest,
    GroundednessResponse,
    MemoryUpdateResponse,
    NLIVerdict,
    RiskBand,
    RuntimeDecision,
    SupportUnit,
    TokenScore,
)
from latence_trace_client.sessions import (
    FileSessionStorage,
    InMemorySessionStorage,
    SessionStorage,
    TraceEvent,
    TraceSessionSnapshot,
)

__version__ = "1.0.0"

__all__ = [
    "AsyncLatenceTraceClient",
    "AsyncTraceSession",
    "AttributionMode",
    "ComplianceCustomLabel",
    "ComplianceEntity",
    "ComplianceLabelMode",
    "ComplianceRedactionMode",
    "ComplianceRedactionRequest",
    "ComplianceRedactionResponse",
    "ComplianceUsage",
    "CompressionResponse",
    "FileSessionStorage",
    "GroundednessRequest",
    "GroundednessResponse",
    "InMemorySessionStorage",
    "LatenceTraceAPIError",
    "LatenceTraceAuthError",
    "LatenceTraceClient",
    "LatenceTraceRateLimited",
    "LatenceTraceServerError",
    "LatenceTraceTimeout",
    "LatenceTraceValidationError",
    "MemoryUpdateResponse",
    "NLIVerdict",
    "RiskBand",
    "RuntimeDecision",
    "SessionStorage",
    "SupportUnit",
    "TokenScore",
    "TraceEvent",
    "TraceSession",
    "TraceSessionSnapshot",
    "__version__",
]
