"""Heatmap convenience renderer.

Produces the compact ``HeatmapPayload`` structure surfaced on every
scored response (and, on request, a self-contained HTML fragment).

Design notes
------------

- **Data first.** The primary contract is a JSON structure any HTML
  template can bind to. The HTML fragment is a convenience for callers
  who want zero integration work; it contains inline CSS only and
  renders as a single ``<div>`` that can be dropped into any page,
  email, or Markdown preview.
- **Fixed thresholds** for the colour bands so the same bucket shows up
  green/amber/red regardless of lane. The exact cut-offs are echoed on
  ``HeatmapPayload.thresholds`` so callers can reproduce the bucketing
  server-side if they want to.
- **Pure Python, no HTTP, no I/O.** Safe to call on the request path.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from latence_trace.api.models import (
    FileAttributionDiagnostics,
    HeatmapFile,
    HeatmapPayload,
    HeatmapSummary,
    HeatmapThresholds,
    HeatmapToken,
    SessionSignals,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "build_heatmap",
    "render_heatmap_html",
]


#: Default band cut-offs. The token cut-offs track the ModernColBERT
#: reverse-context scale (typical well-grounded tokens land above 0.6);
#: the file cut-offs track ``owner_share`` (a grounded file usually
#: owns >= 20 % of response tokens in the code lane). Tune via the
#: ``HeatmapThresholds`` wire model if downstream calibration shifts.
DEFAULT_THRESHOLDS = HeatmapThresholds(
    token_green_min=0.60,
    token_amber_min=0.35,
    file_green_min=0.20,
    file_amber_min=0.05,
)


def _band(value: float, green_min: float, amber_min: float) -> str:
    if value >= green_min:
        return "green"
    if value >= amber_min:
        return "amber"
    return "red"


def _score_for_token(row: Any) -> float:
    """Pick the best-available groundedness score for a response token.

    Prefers the already-computed ``heatmap_score`` when present, falls
    back to the most representative alternative. Both row shapes
    (pydantic model and plain dict, as produced by the RAG scorer) are
    accepted.
    """

    def _get(key: str) -> Optional[float]:
        if isinstance(row, Mapping):
            value = row.get(key)
        else:
            value = getattr(row, key, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for key in (
        "heatmap_score",
        "reverse_context_calibrated",
        "reverse_context",
        "triangular",
    ):
        val = _get(key)
        if val is not None:
            return val
    return 0.0


def _token_text(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("token") or "")
    return str(getattr(row, "token", "") or "")


def _summary_from(
    scores_dict: Mapping[str, Any],
    file_attribution: Optional[FileAttributionDiagnostics],
    session_signals: Optional[SessionSignals],
) -> HeatmapSummary:
    def _get_float(key: str) -> Optional[float]:
        value = scores_dict.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    headline = _get_float("composite_phantom_score")
    if headline is None:
        headline = _get_float("groundedness_v2")
    if headline is None:
        headline = _get_float("primary_score")

    risk_band = scores_dict.get("risk_band")
    if hasattr(risk_band, "value"):
        risk_band = risk_band.value

    recommendation: Optional[str] = None
    if session_signals is not None:
        recommendation = getattr(session_signals, "recommendation", None)

    dead_weight_pct = _get_float("dead_weight_ratio")
    if dead_weight_pct is None and file_attribution is not None:
        dead_weight_pct = float(file_attribution.dead_weight_ratio)

    groundedness_pct = headline

    histogram: Dict[str, int] = {}
    if file_attribution is not None and file_attribution.reason_code_histogram:
        histogram = dict(file_attribution.reason_code_histogram)

    return HeatmapSummary(
        headline_score=headline,
        risk_band=risk_band if risk_band is None or isinstance(risk_band, str) else str(risk_band),
        recommendation=recommendation,
        groundedness_pct=groundedness_pct,
        dead_weight_pct=dead_weight_pct,
        reason_code_histogram=histogram,
    )


def build_heatmap(
    *,
    scores_dict: Mapping[str, Any],
    file_attribution: Optional[FileAttributionDiagnostics],
    response_tokens: Sequence[Any],
    session_signals: Optional[SessionSignals] = None,
    thresholds: HeatmapThresholds = DEFAULT_THRESHOLDS,
    max_tokens: int = 0,
    max_files: int = 0,
) -> HeatmapPayload:
    """Assemble the :class:`HeatmapPayload` for one response.

    ``response_tokens`` accepts either the pydantic ``GroundednessResponseToken``
    objects the RAG lane emits or plain dicts (the scorer's raw rows).
    The code lane ships with an empty list — the heatmap is still
    useful there because the file/summary sections carry the main
    story.

    ``max_tokens`` and ``max_files`` default to ``0`` (unlimited).
    Heatmap coverage is a production correctness contract: a long
    response or a many-file code change must be coloured end-to-end so
    the user can see *every* unsupported token / dead-weight file. The
    ``0`` sentinel preserves the historical hard-cap behaviour for
    operators who explicitly need it (set to a positive integer to
    cap), but the in-tree default never silently truncates.
    """
    tokens: List[HeatmapToken] = []
    for idx, row in enumerate(response_tokens or []):
        if max_tokens > 0 and idx >= max_tokens:
            break
        score = _score_for_token(row)
        band = _band(score, thresholds.token_green_min, thresholds.token_amber_min)
        tokens.append(
            HeatmapToken(
                index=int(idx),
                token=_token_text(row),
                score=float(score),
                band=band,
            )
        )

    files: List[HeatmapFile] = []
    if file_attribution is not None:
        # ``per_file`` comes already sorted by ascending owner_share
        # (worst first) from ``attribute_files`` — worst-first is the
        # most actionable order for dashboards.
        per_file_list = (
            file_attribution.per_file[:max_files]
            if max_files > 0
            else file_attribution.per_file
        )
        for per_file in per_file_list:
            owner_share = float(per_file.owner_share)
            band = _band(
                owner_share,
                thresholds.file_green_min,
                thresholds.file_amber_min,
            )
            files.append(
                HeatmapFile(
                    path=str(per_file.path),
                    owner_share=owner_share,
                    band=band,
                    reason_codes=list(per_file.reason_codes or []),
                    dead_weight=bool(per_file.dead_weight),
                )
            )

    summary = _summary_from(scores_dict, file_attribution, session_signals)

    return HeatmapPayload(
        summary=summary,
        tokens=tokens,
        files=files,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# HTML fragment
# ---------------------------------------------------------------------------

_CSS = """
.lt-heatmap {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.55;
  max-width: 720px;
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #fafafa;
  color: #111827;
}
.lt-heatmap h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
}
.lt-heatmap .lt-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 14px;
}
.lt-heatmap .lt-chip {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 12px;
}
.lt-heatmap .lt-chip b { font-weight: 600; }
.lt-heatmap .lt-band-green  { background: #d1fae5; color: #065f46; }
.lt-heatmap .lt-band-amber  { background: #fef3c7; color: #92400e; }
.lt-heatmap .lt-band-red    { background: #fee2e2; color: #991b1b; }
.lt-heatmap .lt-tokens { margin-bottom: 14px; word-break: break-word; }
.lt-heatmap .lt-tok {
  display: inline-block;
  padding: 1px 4px;
  margin: 1px;
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  font-size: 12px;
}
.lt-heatmap .lt-file-row {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 0; border-top: 1px solid #eef2f7;
}
.lt-heatmap .lt-file-row:first-child { border-top: none; }
.lt-heatmap .lt-file-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis;
}
.lt-heatmap .lt-file-share { font-variant-numeric: tabular-nums; font-size: 12px; }
.lt-heatmap .lt-reason {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: #fff;
  border: 1px solid #e5e7eb;
  margin-left: 4px;
  color: #4b5563;
}
""".strip()


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "&mdash;"
    return f"{float(value) * 100.0:.1f}%"


def _float_fmt(value: Optional[float]) -> str:
    if value is None:
        return "&mdash;"
    return f"{float(value):.2f}"


def render_heatmap_html(payload: HeatmapPayload, *, title: str = "latence-trace") -> str:
    """Render a ``HeatmapPayload`` to a self-contained HTML fragment.

    The fragment is a single ``<div>`` with inline ``<style>``; no
    external stylesheets, no JavaScript. Safe to drop into any HTML
    page or email client that supports inline styles.
    """
    summary = payload.summary
    risk_band = summary.risk_band or "unknown"
    recommendation = summary.recommendation or "continue"

    chips: List[str] = []
    chips.append(
        f'<span class="lt-chip lt-band-{_safe_band(risk_band)}">'
        f'<b>Risk:</b> {html.escape(str(risk_band))}</span>'
    )
    chips.append(
        f'<span class="lt-chip"><b>Groundedness:</b> {_pct(summary.groundedness_pct)}</span>'
    )
    chips.append(
        f'<span class="lt-chip"><b>Dead weight:</b> {_pct(summary.dead_weight_pct)}</span>'
    )
    chips.append(
        f'<span class="lt-chip"><b>Recommendation:</b> {html.escape(recommendation)}</span>'
    )
    for reason, count in sorted(summary.reason_code_histogram.items()):
        chips.append(
            f'<span class="lt-chip"><b>{html.escape(reason)}:</b> {int(count)}</span>'
        )

    token_spans: List[str] = []
    for tok in payload.tokens:
        band = _safe_band(tok.band)
        token_text = html.escape(tok.token or "&nbsp;")
        score_attr = f"{tok.score:.3f}"
        token_spans.append(
            f'<span class="lt-tok lt-band-{band}" title="score={score_attr}">'
            f'{token_text}</span>'
        )

    file_rows: List[str] = []
    for f in payload.files:
        band = _safe_band(f.band)
        reasons = "".join(
            f'<span class="lt-reason">{html.escape(r)}</span>'
            for r in f.reason_codes
        )
        file_rows.append(
            '<div class="lt-file-row">'
            f'<span class="lt-chip lt-band-{band}">{html.escape(band)}</span>'
            f'<span class="lt-file-path">{html.escape(f.path)}</span>'
            f'<span class="lt-file-share">{_pct(f.owner_share)}</span>'
            f'{reasons}'
            '</div>'
        )

    tokens_html = (
        '<div class="lt-tokens">' + "".join(token_spans) + "</div>" if token_spans else ""
    )
    files_html = (
        '<div class="lt-files">' + "".join(file_rows) + "</div>" if file_rows else ""
    )

    return (
        '<div class="lt-heatmap">'
        f'<style>{_CSS}</style>'
        f'<h4>{html.escape(title)}</h4>'
        f'<div class="lt-summary">{"".join(chips)}</div>'
        f'{tokens_html}'
        f'{files_html}'
        '</div>'
    )


def _safe_band(value: Optional[str]) -> str:
    band = (value or "").lower()
    if band not in {"green", "amber", "red"}:
        return "amber"
    return band
