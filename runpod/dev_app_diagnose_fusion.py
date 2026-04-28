"""Dev app variant that pins the `quality` preset fusion weights to the
`diagnose_halueval.py` reference (calibrated=0.5, literal=0.2, nli=0.3).

Used exclusively for the Phase 3.5 follow-up experiment: verify how much
of the 6-8pp HaluEval drift is attributable to the L7 sweep weights
(calibrated=0.0, literal=0.2, nli=0.8) vs the historical diagnose fusion
that produced the 0.78 paired-accuracy number.

Everything else (NLI model, atomic claims, reranker, profile dispatch)
is inherited unchanged from `dev_app.py`.
"""
from __future__ import annotations

from latence_trace.api import service as _service_module

_diagnose_overrides = {
    "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED": "0.5",
    "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL": "0.2",
    "VOYAGER_GROUNDEDNESS_FUSION_W_NLI": "0.3",
}
_quality_preset = _service_module.PROFILE_ENV_PRESETS.get("quality", {})
_quality_preset.update(_diagnose_overrides)

from dev_app import app  # noqa: E402,F401  (re-export FastAPI app for uvicorn)

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(
        "dev_app_diagnose_fusion:app",
        host="127.0.0.1",
        port=8091,
        log_level="warning",
    )
