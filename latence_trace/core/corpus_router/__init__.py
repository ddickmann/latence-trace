"""Corpus-type router: infers which of the six corpus classes a request
belongs to and loads the matching calibration bundle.

Runtime surface (see :mod:`latence_trace.middleware.corpus_router`):

* :func:`latence_trace.core.corpus_router.features.featurize`
* :func:`latence_trace.core.corpus_router.classifier.classify`
* :func:`latence_trace.core.corpus_router.bundles.load_bundle`
"""
