"""Parity bench: legacy mDeBERTa vs SOTA dual-NLI (MiniCheck + bge-m3-zs).

Compares the OLD single-NLI lane (``resolve_default_provider`` —
mDeBERTa via /pooling, or its HF fallback) against the NEW
language-aware registry (``resolve_provider_for_language`` —
MiniCheck-Flan-T5-Large for English, bge-m3-zeroshot-v2.0 for German)
on the same labelled fixtures Phase 0 used.

This script is intentionally provider-level. The
acceptance gate in the plan also asks for full-service p50/p99 latency
which is best measured against ``dev_app`` (separate harness in
``scripts/bench_runpod_local.py``); here we want a clean signal on
*model* quality so a regression on the upgrade is unambiguous.

Usage::

    python scripts/parity_nli_upgrade.py \\
        --fixtures /tmp/proof/fixtures \\
        --output /tmp/proof/parity_nli_upgrade.md

The script honours the same env vars the registry consumes — set
``LATENCE_TRACE_NLI_EN_ENDPOINT`` / ``LATENCE_TRACE_NLI_MULTI_ENDPOINT``
to bench against vLLM-served versions instead of the in-process
transformers fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: str
    dataset: str
    language: str
    premise: str
    claim: str
    label: str  # ``supported`` | ``hallucinated``
    meta: dict = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return self.label.lower() == "supported"


def load_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                Row(
                    id=str(data["id"]),
                    dataset=str(data.get("dataset", path.stem)),
                    language=str(data.get("language", "en")).lower(),
                    premise=str(data.get("premise", "")),
                    claim=str(data.get("claim", "")),
                    label=str(data.get("label", "")).lower(),
                    meta=dict(data.get("meta") or {}),
                )
            )
    return rows


def load_all_fixtures(fixtures_dir: Path) -> dict[str, List[Row]]:
    out: dict[str, List[Row]] = {}
    for jsonl in sorted(fixtures_dir.glob("*.jsonl")):
        rows = load_jsonl(jsonl)
        if rows:
            out[jsonl.stem] = rows
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _confusion(rows: Sequence[Row], scores: Sequence[float], threshold: float):
    tp = fp = tn = fn = 0
    for row, score in zip(rows, scores):
        predicted_supported = score >= threshold
        if row.is_supported and predicted_supported:
            tp += 1
        elif row.is_supported and not predicted_supported:
            fn += 1
        elif (not row.is_supported) and predicted_supported:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _balanced_acc(tp: int, fp: int, tn: int, fn: int) -> float:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (sens + spec) / 2.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _best_threshold(rows: Sequence[Row], scores: Sequence[float]) -> Tuple[float, float, float]:
    """Sweep thresholds in 0.05 steps; return (threshold, balanced_acc, f1).

    Sweeping here is cheap (O(20 * N)) and matches how the production
    fusion stage picks per-class operating points. We report the
    *best* balanced accuracy so the table is comparable across models
    even when they emit very different output distributions.
    """

    best_thr = 0.5
    best_bal = -1.0
    best_f1 = -1.0
    for i in range(1, 20):
        thr = i / 20.0
        tp, fp, tn, fn = _confusion(rows, scores, thr)
        bal = _balanced_acc(tp, fp, tn, fn)
        if bal > best_bal:
            best_bal = bal
            best_thr = thr
            best_f1 = _f1(tp, fp, fn)
    return best_thr, best_bal, best_f1


def _roc_auc(rows: Sequence[Row], scores: Sequence[float]) -> float:
    """Mann-Whitney U-form AUC. Returns 0.5 on degenerate input."""

    pos = [s for r, s in zip(rows, scores) if r.is_supported]
    neg = [s for r, s in zip(rows, scores) if not r.is_supported]
    if not pos or not neg:
        return 0.5
    rank_sum = 0
    pairs = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    pairs.sort(key=lambda p: p[0])
    rank = 1
    for _, label in pairs:
        if label == 1:
            rank_sum += rank
        rank += 1
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Provider drivers (OLD vs NEW)
# ---------------------------------------------------------------------------


def _ensure_nli_enabled() -> None:
    """``resolve_default_provider`` early-exits when the NLI feature flag is off.

    The bench needs the legacy provider regardless, so flip the flag
    in-process before the first call. We only mutate it locally — the
    parent process env is untouched if the bench was launched via
    ``python -m`` from a shell that doesn't have NLI enabled.
    """

    os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "1")


@dataclass
class ProviderRun:
    """Per-row scoring result for one provider on one fixture."""

    rows: List[Row]
    scores: List[float]
    latencies_ms: List[float]


def _entail_in_chunks(provider, rows: Sequence[Row], batch: int) -> Tuple[List[float], List[float]]:
    """Score rows in batches; return (entail_probs, per_batch_latencies_ms).

    Per-row latency is the per-batch latency divided across the batch.
    For the latency p50/p99 we report we use the per-batch wall clock
    instead — that's the number that maps onto a real request's
    ``verify_claims`` call.
    """

    out_scores: List[float] = []
    out_latencies: List[float] = []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        premises = [r.premise for r in chunk]
        claims = [r.claim for r in chunk]
        t0 = time.perf_counter()
        results = provider.entail(premises, claims)
        dt = (time.perf_counter() - t0) * 1000.0
        out_scores.extend(float(triple[0]) for triple in results)
        out_latencies.append(dt)
    return out_scores, out_latencies


def run_provider_on_dataset(
    provider,
    rows: Sequence[Row],
    *,
    batch: int = 8,
) -> ProviderRun:
    scores, batch_latencies = _entail_in_chunks(provider, rows, batch=batch)
    return ProviderRun(
        rows=list(rows),
        scores=scores,
        latencies_ms=list(batch_latencies),
    )


def run_per_language_dispatch(
    rows: Sequence[Row],
    new_providers: dict[str, object],
    *,
    batch: int = 8,
) -> ProviderRun:
    """Dispatch each row to the matching language's NEW provider.

    Re-orders the per-row scores back into the original input order so
    the diff table lines up with ``run_provider_on_dataset(old_provider,
    rows)`` exactly. When no provider exists for a row's language we
    skip that row (and warn); this is rare in practice because the two
    languages we support both have providers in the registry.
    """

    by_language: dict[str, list[Tuple[int, Row]]] = {}
    for idx, row in enumerate(rows):
        lang = row.language if row.language in new_providers else "en"
        by_language.setdefault(lang, []).append((idx, row))

    scores: List[Optional[float]] = [None] * len(rows)
    latencies: List[float] = []
    for language, lang_rows in by_language.items():
        provider = new_providers.get(language)
        if provider is None:
            print(
                f"  warning: no NEW provider for language={language!r}; skipping {len(lang_rows)} rows",
                file=sys.stderr,
            )
            for idx, _ in lang_rows:
                scores[idx] = 0.0
            continue
        chunk_rows = [r for _, r in lang_rows]
        chunk_scores, chunk_latencies = _entail_in_chunks(provider, chunk_rows, batch=batch)
        for (idx, _), score in zip(lang_rows, chunk_scores):
            scores[idx] = float(score)
        latencies.extend(chunk_latencies)
    return ProviderRun(
        rows=list(rows),
        scores=[s if s is not None else 0.0 for s in scores],
        latencies_ms=latencies,
    )


def build_old_provider():
    from latence_trace.core.nli import resolve_default_provider

    _ensure_nli_enabled()
    provider = resolve_default_provider()
    if provider is None:
        raise RuntimeError(
            "Legacy provider unavailable. Install transformers and the mDeBERTa "
            "weights, or point LATENCE_TRACE_NLI_VLLM_ENDPOINT at a healthy "
            "vLLM /pooling server."
        )
    return provider


def build_new_provider(language: str):
    """Get the right new-stack provider for ``language`` via the registry."""

    from latence_trace.core.nli import resolve_provider_for_language

    _ensure_nli_enabled()
    # Force the per-language lane on so the registry doesn't fall
    # through to the legacy provider when the operator hasn't set the
    # explicit opt-in env flag.
    upper = language.upper()
    os.environ.setdefault(f"LATENCE_TRACE_NLI_{upper}_ENABLED", "1")
    provider = resolve_provider_for_language(language)
    if provider is None:
        raise RuntimeError(
            f"New-stack provider for language={language!r} unavailable. "
            "Check the language registry env vars and confirm the "
            "transformers fallback model can load."
        )
    return provider


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def summarise_run(name: str, run: ProviderRun) -> dict:
    thr, bal, f1 = _best_threshold(run.rows, run.scores)
    auc = _roc_auc(run.rows, run.scores)
    return {
        "name": name,
        "n": len(run.rows),
        "best_threshold": thr,
        "balanced_acc": bal,
        "f1": f1,
        "roc_auc": auc,
        "latency_p50_ms": _percentile(run.latencies_ms, 0.5),
        "latency_p99_ms": _percentile(run.latencies_ms, 0.99),
        "latency_mean_ms": (
            statistics.mean(run.latencies_ms) if run.latencies_ms else 0.0
        ),
    }


def headline_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        return "_(no datasets)_"
    headers = [
        "dataset",
        "n",
        "model",
        "thr",
        "bal_acc",
        "F1",
        "AUC",
        "p50 ms",
        "p99 ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append(
            "| {dataset} | {n} | {model} | {thr:.2f} | {bal:.3f} | {f1:.3f} | "
            "{auc:.3f} | {p50:.1f} | {p99:.1f} |".format(
                dataset=row["dataset"],
                n=row["n"],
                model=row["model"],
                thr=row["best_threshold"],
                bal=row["balanced_acc"],
                f1=row["f1"],
                auc=row["roc_auc"],
                p50=row["latency_p50_ms"],
                p99=row["latency_p99_ms"],
            )
        )
    return "\n".join(lines)


def diff_table(old: dict, new: dict) -> str:
    fmt = "| {metric} | {old:.3f} | {new:.3f} | {delta:+.3f} |"
    rows = [
        "| metric | OLD | NEW | Δ |",
        "|---|---|---|---|",
        fmt.format(
            metric="balanced_acc",
            old=old["balanced_acc"],
            new=new["balanced_acc"],
            delta=new["balanced_acc"] - old["balanced_acc"],
        ),
        fmt.format(
            metric="F1",
            old=old["f1"],
            new=new["f1"],
            delta=new["f1"] - old["f1"],
        ),
        fmt.format(
            metric="ROC-AUC",
            old=old["roc_auc"],
            new=new["roc_auc"],
            delta=new["roc_auc"] - old["roc_auc"],
        ),
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def detect_dataset_language(name: str, rows: Sequence[Row]) -> str:
    by_language = {row.language for row in rows if row.language}
    if len(by_language) == 1:
        return next(iter(by_language))
    if "de" in name.lower():
        return "de"
    return "en"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("/tmp/proof/fixtures"),
        help="Directory containing *.jsonl fixture files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/proof/parity_nli_upgrade.md"),
        help="Markdown verdict output path",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Per-call batch size for provider.entail",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows per dataset (0 = all)",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only run datasets whose stem matches any of these substrings",
    )
    args = parser.parse_args(argv)

    fixtures_dir = args.fixtures.resolve()
    if not fixtures_dir.is_dir():
        print(f"fixtures dir not found: {fixtures_dir}", file=sys.stderr)
        return 2
    datasets = load_all_fixtures(fixtures_dir)
    if not datasets:
        print(f"no fixtures found under {fixtures_dir}", file=sys.stderr)
        return 2

    if args.include:
        keep = {key: rows for key, rows in datasets.items() if any(token in key for token in args.include)}
        if not keep:
            print(f"--include filter {args.include!r} matched no datasets", file=sys.stderr)
            return 2
        datasets = keep

    if args.max_rows and args.max_rows > 0:
        datasets = {k: rows[: args.max_rows] for k, rows in datasets.items()}

    print(f"loaded {len(datasets)} datasets:", file=sys.stderr)
    for name, rows in datasets.items():
        lang = detect_dataset_language(name, rows)
        print(f"  - {name}: {len(rows)} rows ({lang})", file=sys.stderr)

    print("building OLD provider (mDeBERTa)...", file=sys.stderr)
    old_provider = build_old_provider()

    new_providers: dict[str, object] = {}
    for lang in {"en", "de"}:
        try:
            print(f"building NEW provider for language={lang}...", file=sys.stderr)
            new_providers[lang] = build_new_provider(lang)
        except Exception as exc:
            print(f"  warning: failed to build NEW provider for {lang}: {exc}", file=sys.stderr)

    headline_rows: List[dict] = []
    detail_blocks: List[str] = []
    for dataset_name in sorted(datasets):
        rows = datasets[dataset_name]
        # Per-row language so a mixed-language dataset (e.g. catalogue_5
        # which has both DE Kafka and EN AI Act rows) gets dispatched
        # to MiniCheck for the EN rows and bge-m3-zs for the DE rows.
        languages = sorted({row.language for row in rows if row.language})
        language_label = "+".join(languages) or detect_dataset_language(dataset_name, rows)

        print(f"running OLD on {dataset_name} ({len(rows)} rows)...", file=sys.stderr)
        old_run = run_provider_on_dataset(old_provider, rows, batch=args.batch)
        old_summary = summarise_run("mDeBERTa-XNLI", old_run)
        old_summary.update(dataset=dataset_name, model="OLD: mDeBERTa-XNLI")
        headline_rows.append(old_summary)

        new_summary = None
        if new_providers:
            print(
                f"running NEW on {dataset_name} ({language_label}, {len(rows)} rows)...",
                file=sys.stderr,
            )
            new_run = run_per_language_dispatch(rows, new_providers, batch=args.batch)
            new_summary = summarise_run("dual-NLI", new_run)
            new_summary.update(
                dataset=dataset_name,
                model="NEW: MiniCheck (en) + bge-m3-zs (de)",
            )
            headline_rows.append(new_summary)

        if new_summary is not None:
            detail_blocks.append(
                f"### {dataset_name} ({language_label}, n={len(rows)})\n\n"
                + diff_table(old_summary, new_summary)
            )
        else:
            detail_blocks.append(
                f"### {dataset_name} ({language_label}, n={len(rows)})\n\n"
                f"_(NEW providers unavailable; skipping diff)_"
            )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Parity bench: legacy mDeBERTa vs SOTA dual-NLI\n\n")
        fh.write(
            "Drives `latence_trace.core.nli.resolve_default_provider` "
            "(OLD) against `latence_trace.core.nli.resolve_provider_for_language` "
            "(NEW) on the same fixtures Phase 0 used. Latencies are per-batch wall "
            "clock; thresholds are swept on the same run so the OLD vs NEW table "
            "is apples-to-apples on operating points.\n\n"
        )
        fh.write(f"- fixtures: `{fixtures_dir}`\n")
        fh.write(f"- batch size: {args.batch}\n")
        if args.include:
            fh.write(f"- include filter: `{args.include}`\n")
        if args.max_rows:
            fh.write(f"- max rows / dataset: {args.max_rows}\n")
        fh.write("\n## Headline\n\n")
        fh.write(headline_table(headline_rows))
        fh.write("\n\n## Per-dataset diff\n\n")
        for block in detail_blocks:
            fh.write(block)
            fh.write("\n\n")

    print(f"wrote verdict -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
