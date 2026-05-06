"""Tests for the heatmap convenience renderer.

Locks in:

- Band thresholds are deterministic (green/amber/red bucketing).
- Token + file payload shapes are populated correctly.
- HTML fragment is self-contained (single ``<div>`` with inline
  ``<style>``, no external resources), HTML-parseable, and includes
  the expected band-chip classes.
"""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

from latence_trace.api.heatmap import (
    DEFAULT_THRESHOLDS,
    build_heatmap,
    render_heatmap_html,
)
from latence_trace.api.models import (
    CodeLanePerFileUsage,
    FileAttributionDiagnostics,
)


def _fa(per_file_args) -> FileAttributionDiagnostics:
    per_file = [
        CodeLanePerFileUsage(
            path=p,
            n_units=1,
            used=1,
            uncertain=0,
            unused=0,
            coverage=1.0,
            mean_score=cos,
            max_evidence=cos,
            owner_tokens=owner,
            owner_share=share,
            query_owner_tokens=0,
            query_owner_share=0.0,
            dead_weight=bool(share < 0.01),
            reason_codes=reasons,
            dominating_peer=None,
        )
        for (p, cos, owner, share, reasons) in per_file_args
    ]
    return FileAttributionDiagnostics(
        per_file=per_file,
        per_unit=[],
        dead_weight_files=[p for (p, _c, _o, s, _r) in per_file_args if s < 0.01],
        dead_weight_ratio=sum(1 for (_p, _c, _o, s, _r) in per_file_args if s < 0.01)
        / max(len(per_file_args), 1),
        n_files=len(per_file_args),
        n_response_tokens=10,
        n_query_tokens=0,
        reason_code_histogram={
            "never_won_argmax": sum(
                1 for (_p, _c, _o, s, _r) in per_file_args if s < 0.01
            ),
        },
    )


def test_token_band_thresholds_match_documented_cutoffs() -> None:
    tokens = [
        {"token": "Berlin", "reverse_context": 0.85},  # green
        {"token": "is", "reverse_context": 0.40},  # amber
        {"token": "flightless", "reverse_context": 0.05},  # red
    ]
    payload = build_heatmap(
        scores_dict={"groundedness_v2": 0.8, "risk_band": "green"},
        file_attribution=None,
        response_tokens=tokens,
    )
    assert [tok.band for tok in payload.tokens] == ["green", "amber", "red"]
    assert payload.thresholds == DEFAULT_THRESHOLDS


def test_file_band_thresholds_match_owner_share() -> None:
    fa = _fa(
        [
            ("owner.py", 0.9, 8, 0.8, []),
            ("medium.py", 0.5, 1, 0.10, []),
            ("dead.py", 0.1, 0, 0.0, ["never_won_argmax"]),
        ]
    )
    payload = build_heatmap(
        scores_dict={"dead_weight_ratio": 1 / 3, "groundedness_v2": 0.6},
        file_attribution=fa,
        response_tokens=[],
    )
    by_path = {f.path: f for f in payload.files}
    assert by_path["owner.py"].band == "green"
    assert by_path["medium.py"].band == "amber"
    assert by_path["dead.py"].band == "red"
    assert by_path["dead.py"].dead_weight is True
    assert by_path["dead.py"].reason_codes == ["never_won_argmax"]


def test_summary_prefers_composite_then_v2_then_primary() -> None:
    payload = build_heatmap(
        scores_dict={
            "composite_phantom_score": 0.42,
            "groundedness_v2": 0.77,
            "primary_score": 0.11,
            "dead_weight_ratio": 0.25,
            "risk_band": "amber",
        },
        file_attribution=None,
        response_tokens=[],
    )
    assert payload.summary.headline_score == pytest.approx(0.42)
    assert payload.summary.dead_weight_pct == pytest.approx(0.25)
    assert payload.summary.risk_band == "amber"


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tags: list[str] = []
        self.classes: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            if name == "class" and value:
                self.classes.extend(value.split())


def test_render_heatmap_html_is_selfcontained_and_parses() -> None:
    fa = _fa(
        [
            ("a.py", 0.9, 8, 0.8, []),
            ("b.py", 0.1, 0, 0.0, ["never_won_argmax"]),
        ]
    )
    payload = build_heatmap(
        scores_dict={
            "groundedness_v2": 0.75,
            "dead_weight_ratio": 0.5,
            "risk_band": "amber",
        },
        file_attribution=fa,
        response_tokens=[{"token": "hello", "reverse_context": 0.8}],
    )
    html_fragment = render_heatmap_html(payload)
    # Self-contained: single top-level div, inline style only, no
    # external <link>, no <script>, no <img> requests.
    assert html_fragment.startswith('<div class="lt-heatmap">')
    assert html_fragment.endswith("</div>")
    assert "<style>" in html_fragment
    for forbidden in ("<link", "<script", "<img", "url(http"):
        assert forbidden not in html_fragment, forbidden

    parser = _TagCollector()
    parser.feed(html_fragment)
    assert "div" in parser.tags
    assert "style" in parser.tags
    # Band classes are emitted on chips and token spans.
    assert "lt-heatmap" in parser.classes
    assert any(cls.startswith("lt-band-") for cls in parser.classes)


def test_build_heatmap_default_renders_every_token_with_no_cap() -> None:
    """Heatmap must cover the entire response by default.

    The previous default (``max_tokens=512``) silently truncated
    long-response heatmaps. Production responses regularly exceed 512
    tokens; the default is now ``0`` (unlimited) and the explicit
    cap path is preserved for operators who really want one.
    """
    tokens = [
        {"token": f"tok{idx}", "reverse_context": 0.7}
        for idx in range(2000)
    ]
    payload = build_heatmap(
        scores_dict={"groundedness_v2": 0.7, "risk_band": "green"},
        file_attribution=None,
        response_tokens=tokens,
    )
    assert len(payload.tokens) == 2000
    assert payload.tokens[0].index == 0
    assert payload.tokens[-1].index == 1999


def test_build_heatmap_explicit_max_tokens_still_caps() -> None:
    """A positive ``max_tokens`` argument keeps its hard-cap behaviour."""
    tokens = [
        {"token": f"tok{idx}", "reverse_context": 0.7}
        for idx in range(1000)
    ]
    payload = build_heatmap(
        scores_dict={"groundedness_v2": 0.7, "risk_band": "green"},
        file_attribution=None,
        response_tokens=tokens,
        max_tokens=128,
    )
    assert len(payload.tokens) == 128


def test_build_heatmap_default_renders_every_file_with_no_cap() -> None:
    """File rollup must cover every owner file by default (no max_files)."""
    fa = _fa(
        [(f"src/file_{idx}.py", 0.5, 1, 0.05, []) for idx in range(75)]
    )
    payload = build_heatmap(
        scores_dict={"groundedness_v2": 0.7, "risk_band": "green"},
        file_attribution=fa,
        response_tokens=[],
    )
    assert len(payload.files) == 75
