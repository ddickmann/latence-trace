"""Unit tests for the corpus-router rule overlay.

The rule overlay is what gives the router its out-of-distribution
robustness. These tests lock in the structural rules so a refactor
cannot silently regress the generalization guarantees.

Every rule assertion also covers the opposite direction: an example
that matches superficially but *should not* fire the rule, so we avoid
false positives.
"""

from __future__ import annotations

import json

import pytest

from latence_trace.core.corpus_router.rules import (
    MIN_RULE_CONFIDENCE,
    RuleDecision,
    apply_rules,
)


# --------------------------------------------------------------------
# Structured (JSON / markdown table)
# --------------------------------------------------------------------


def test_json_rooted_context_routes_to_structured() -> None:
    ctx = json.dumps(
        {"product": "Kettle", "sku": "KT-0421", "units": 143}, indent=2
    )
    dec = apply_rules(query="q", response="The kettle SKU is KT-0421.", raw_context=ctx)
    assert dec.corpus_type == "rag.structured"
    assert dec.confidence >= MIN_RULE_CONFIDENCE
    assert dec.reason == "rule:json_rooted_context"


def test_json_array_root_routes_to_structured() -> None:
    ctx = json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    dec = apply_rules(query="q", response="there are two rows.", raw_context=ctx)
    assert dec.corpus_type == "rag.structured"


def test_veracier_style_bracket_prefix_is_not_structured() -> None:
    # Veracier metadata header starts with ``[`` but is prose, not JSON.
    # It must not match the structured rule - LR / explicit override
    # handles Veracier-style inputs.
    ctx = (
        "[US-01:DOC-abc | RELEVANT | us_01/doc_001.pdf] --- Page 1 --- "
        "# SPEC TECH - Section 1 - Object and Scope"
    )
    resp = (
        "The contract requires the supplier to maintain traceability "
        "of materials and quality controls across all production runs."
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type != "rag.structured", (
        "bracketed metadata header should NOT match the JSON rule"
    )


def test_markdown_table_routes_to_structured() -> None:
    ctx = (
        "| ticket_id | status      | sla_tier |\n"
        "|-----------|-------------|----------|\n"
        "| INC-9988  | in_progress | Gold     |\n"
        "| INC-9989  | new         | Bronze   |\n"
    )
    dec = apply_rules(query="q", response="Ticket INC-9988 is in_progress.", raw_context=ctx)
    assert dec.corpus_type == "rag.structured"
    assert dec.reason == "rule:markdown_table_context"


def test_single_pipe_line_is_not_markdown_table() -> None:
    ctx = (
        "prose content with | a pipe | in the middle but no separator "
        "row, just ordinary punctuation in a sentence."
    )
    resp = (
        "The answer is based on the evidence above and includes some "
        "additional context for completeness and clarity."
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type != "rag.structured"


# --------------------------------------------------------------------
# Agentic code trace (multi-file)
# --------------------------------------------------------------------


def test_multi_file_bundle_plus_long_fenced_code_routes_to_agentic() -> None:
    ctx = (
        "# File: billing/core.py\n"
        "def cents_for(x): return int(x * 100)\n\n"
        "# File: billing/tests/test_core.py\n"
        "def test_cents(): assert cents_for(1) == 100\n\n"
        "# File: BUILD_LOG.txt\nfail\n"
    )
    resp = (
        "Here is the fix - apply this diff and rerun the failing test:\n\n"
        "```python\n"
        "def cents_for(x: float) -> int:\n"
        "    if x < 0:\n"
        "        raise ValueError\n"
        "    return int(round(x * 100))\n"
        "\n"
        "def test_cents_for_basic():\n"
        "    assert cents_for(1.0) == 100\n"
        "```"
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "code.agentic_trace"
    assert "headers=3" in (dec.reason or "")


def test_multi_file_bundle_with_short_fence_routes_to_code_in_context() -> None:
    # Mirrors the training-time splitter: multi-file bundle + < 5 line
    # fenced response = ``rag.code_in_context``, not agentic trace.
    ctx = (
        "# File: a.py\nprint('a')\n\n"
        "# File: b.py\nprint('b')\n\n"
        "# File: c.py\nprint('c')\n"
    )
    resp = "Quick answer:\n\n```python\nprint('done')\n```"
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "rag.code_in_context"
    assert "code_in_context_short" in (dec.reason or "")


def test_triple_equals_file_header_format_is_recognised() -> None:
    ctx = "=== plans/rollout.md ===\nplan body\n\n=== src/app.py ===\ncode\n\n=== README.md ===\ndocs"
    resp = (
        "Looking at this plan, the rollout step needs the following:\n\n"
        "```python\n"
        "from app import service\n"
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def rollout():\n"
        "    logger.info('starting')\n"
        "    service.start()\n"
        "    logger.info('done')\n"
        "```"
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "code.agentic_trace"


def test_bracketed_filename_header_is_recognised() -> None:
    ctx = (
        "[app/service.py]\n"
        "code a\n\n"
        "[app/tests/test_service.py]\n"
        "code b\n\n"
        "[README.md]\n"
        "docs\n"
    )
    resp = (
        "Here is the fix to apply - paste this into service.py and rerun:\n\n"
        "```python\n"
        "class Service:\n"
        "    def __init__(self) -> None:\n"
        "        self._started = False\n"
        "    def start(self) -> None:\n"
        "        if self._started:\n"
        "            return\n"
        "        self._started = True\n"
        "```"
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "code.agentic_trace"


# --------------------------------------------------------------------
# Code in context (single fence, RAG-style)
# --------------------------------------------------------------------


def test_single_fence_short_context_routes_to_code_in_context() -> None:
    ctx = "Python's sorted() accepts a key parameter for comparison."
    resp = (
        "Use sorted() with a key function:\n\n"
        "```python\nsorted(items, key=lambda x: x[1])\n```"
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "rag.code_in_context"


def test_fence_with_file_header_is_not_code_in_context() -> None:
    # One file header + fenced response is ambiguous - rule chooses to
    # stay silent rather than mis-fire; classifier handles it.
    ctx = "# File: util.py\ncode"
    resp = "```python\nreturn 1\n```"
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    # Either defers (None) or picks code_in_context (1 file header is
    # still <= the rule's threshold). Not agentic_trace at least.
    assert dec.corpus_type != "code.agentic_trace"


def test_no_fence_does_not_route_to_code() -> None:
    dec = apply_rules(
        query="q",
        response="This is a plain prose answer.",
        raw_context="Plain prose evidence.",
    )
    assert dec.corpus_type not in {"rag.code_in_context", "code.agentic_trace"}


# --------------------------------------------------------------------
# Prose length heuristics
# --------------------------------------------------------------------


def test_single_short_sentence_routes_to_short_factoid() -> None:
    dec = apply_rules(
        query="Who designed the Sagrada Familia?",
        response="Antoni Gaudi designed the Sagrada Familia.",
        raw_context="Antoni Gaudi was a Catalan architect.",
    )
    assert dec.corpus_type == "rag.prose.short_factoid"


def test_long_paragraph_with_many_sentences_routes_to_multi_claim() -> None:
    resp = (
        "The 2024 EU AI Act introduces tiered obligations for general "
        "purpose AI providers active across the entire European union. "
        "Providers must publish a model card summarising training data "
        "and capabilities, and maintain a technical documentation "
        "package ready for inspection by the EU competent authorities. "
        "A copyright opt-out mechanism for text and data mining is "
        "mandatory for all covered models. Models above the compute "
        "threshold face additional red-teaming obligations and serious "
        "incident reporting duties within fifteen days of detection."
    )
    ctx = "EU AI Act text and supporting analysis from the official journal."
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type == "rag.prose.multi_claim"


def test_prose_rule_defers_on_medium_length() -> None:
    # Medium-length enterprise-ish prose: one sentence but >25 tokens -
    # should not hit either of the length-based prose rules; defers to
    # LR (or explicit override in production).
    resp = (
        "As of the March 2026 steering review, Annex A.5 information "
        "security policies have been signed off by the CISO and "
        "cascaded to most EMEA business units, with two subsidiaries "
        "still pending localised translation work."
    )
    ctx = "status report body with enough detail to avoid the short-context guard."
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type is None


def test_short_factoid_rule_rejects_condensed_enterprise_summary() -> None:
    # Single-sentence response that cites revenue + margin + EPS in
    # one go is NOT a factoid — it's a dense enterprise summary.
    # Rule must defer to the LR classifier so it can route to
    # ``rag.prose.enterprise`` instead of the NLI-heavy factoid
    # bundle (which false-reds enterprise payloads).
    resp = "Q3 revenue rose 11% to $48.2B, margin expanded to 29.4%, EPS was $2.10."
    ctx = "Apple Q3 earnings release. Revenue 48.2B. Operating margin 29.4%. EPS 2.10."
    dec = apply_rules(query="Summarise Apple Q3.", response=resp, raw_context=ctx)
    assert dec.corpus_type is None, (
        "dense enterprise summaries must not hit the short_factoid rule"
    )


def test_short_factoid_rule_rejects_long_context() -> None:
    # HaluEval-style one-liner answer but with a production-sized
    # vector-DB context (>= 60 tokens). Real RAG users chunk at
    # 256-512 tokens so the rule must defer here too.
    resp = "The capital of France is Paris."
    ctx = " ".join(
        [
            "France is a country in Western Europe with an extensive",
            "history of political, cultural, and scientific influence",
            "across the continent. Its capital city, Paris, is located",
            "on the Seine river in the north-central part of the country",
            "and is widely regarded as a global centre of art, fashion,",
            "gastronomy and diplomacy; numerous international summits",
            "have been hosted there since the post-war period.",
        ]
    )
    dec = apply_rules(query="q", response=resp, raw_context=ctx)
    assert dec.corpus_type is None


def test_short_factoid_rule_still_fires_for_true_halueval_shape() -> None:
    # Classic HaluEval QA shape: one-liner answer, short single-
    # sentence context, minimal numeric density. Rule must still
    # fire so the NLI-heavy fusion bundle is selected.
    resp = "Antoni Gaudi designed the Sagrada Familia."
    ctx = "Antoni Gaudi was a Catalan architect active in Barcelona."
    dec = apply_rules(query="Who designed the Sagrada Familia?", response=resp, raw_context=ctx)
    assert dec.corpus_type == "rag.prose.short_factoid"


def test_apply_rules_is_deterministic() -> None:
    ctx = json.dumps({"a": 1, "b": 2})
    d1 = apply_rules(query="q", response="r", raw_context=ctx)
    d2 = apply_rules(query="q", response="r", raw_context=ctx)
    assert d1 == d2


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "response": "", "raw_context": ""},
        {"query": "q", "response": "r", "raw_context": ""},
        {"query": "", "response": "", "raw_context": "ctx"},
    ],
)
def test_empty_payloads_do_not_crash(payload: dict) -> None:
    dec = apply_rules(**payload)
    assert isinstance(dec, RuleDecision)
