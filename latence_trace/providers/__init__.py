"""Provider implementations (NLI, reranker, encoders, compression, GLiNER).

Each module here exposes a thin client for one external runtime
(transformers, vLLM, etc.) and stays import-light: heavy imports
(``transformers``, ``torch``, ``httpx``) live behind lazy methods so
``import latence_trace`` stays fast and side-effect-free.

Sub-modules:

* :mod:`.nli` — :class:`VllmFactoryNLIProvider` (HTTP /pooling client
  for the legacy ``nli_mdeberta`` plugin and the future
  ``minicheck_t5`` BYOP plugin).
* :mod:`.nli_classify` — :class:`VllmClassifyNLIProvider` (HTTP
  /v1/classify client for vLLM-native ``--task classify`` servers,
  e.g. ``bge-m3-zeroshot-v2.0``).
* :mod:`.nli_transformers` — in-process transformers fallbacks
  (:class:`MiniCheckNLIProvider`, :class:`BgeM3ZeroShotNLIProvider`)
  that the registry falls back to when no vLLM endpoint is configured.
* :mod:`.nli_registry` — :func:`resolve_nli_provider` (language-aware
  factory over the above).
* :mod:`.reranker` — :class:`VllmRerankerProvider` and
  :func:`resolve_reranker` (HTTP /v1/score client for vLLM-served
  cross-encoders, falls back to the in-process
  :class:`latence_trace.core.nli.CrossEncoderPremiseReranker`).
"""

from __future__ import annotations

__all__: list[str] = []
