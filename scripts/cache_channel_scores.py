"""Phase B.1 - Score every labelled training row once, cache all channels.

Per row we capture the full :class:`GroundednessScores` object (via
``verbose=true``) so the downstream calibration sweep can fuse the five
channels (``calibrated``, ``literal``, ``nli``, ``semantic_entropy``,
``structured``) offline without re-hitting the GPU for every fusion
config. The composite code-lane score is also captured for
``code.agentic_trace`` rows.

Output layout:

    data/corpus_classifier/channel_scores/
        <class_key>.<split>.jsonl

Each line carries ``{row_id, class_key, is_grounded, channels: {...},
composite_score, band, primary_score, scoring_mode, latency_ms}``.

The script is resumable: on re-run it skips rows already present in the
output file (matched by ``row_id``). A per-class cap keeps the scoring
budget bounded for classes with >> 300 train rows (e.g. HaluEval QA).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/corpus_classifier"
OUT_DIR = DATA_DIR / "channel_scores"
GERMAN_TRANSLATION_DIR = DATA_DIR / "german_translation"

# Supported source languages. ``en`` reads the historical ``<split>.parquet``
# corpus; ``de`` reads the per-class JSONL produced by
# ``scripts/translate_bench_to_german.py``.
SUPPORTED_LANGUAGES = ("en", "de")
DEFAULT_LANGUAGE = "en"

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

# Default per-class cap (train rows scored). Ample for F1/ROC on binary
# classes and matches Veracier / transcripts scale on the rest.
DEFAULT_CAP = 300

# Channels surfaced from GroundednessScores. See
# :class:`latence_trace.api.models.GroundednessScores`.
CHANNEL_KEYS = (
    "reverse_context_calibrated",  # calibrated
    "literal_guarded",              # literal
    "nli_aggregate",                # nli
    "semantic_entropy_aggregate",   # semantic_entropy
    "structured_source_guarded",    # structured
)

CHANNEL_ALIAS = {
    "reverse_context_calibrated": "calibrated",
    "literal_guarded": "literal",
    "nli_aggregate": "nli",
    "semantic_entropy_aggregate": "semantic_entropy",
    "structured_source_guarded": "structured",
}


def _score_one(
    url: str,
    *,
    row: Dict[str, Any],
    scoring_mode: str,
    timeout: float,
    response_language_hint: Optional[str],
    language: str = DEFAULT_LANGUAGE,
    auth_bearer: Optional[str] = None,
) -> Tuple[Dict[str, Any], float]:
    payload_input: Dict[str, Any] = {
        "action": "score",
        "query_text": row.get("query", ""),
        "response_text": row.get("response", ""),
        "raw_context": row.get("raw_context", ""),
        "profile": "quality",
        "scoring_mode": scoring_mode,
        "verbose": True,
    }
    # When scoring the German bench, force ``language="de"`` so the
    # service applies the balanced German NLI defaults (top_k=2,
    # premise_concat=False, max-aggregate) and -- once Phase C bundles
    # ship -- loads the matching ``calibration.<class>.de.json``.
    # English keeps the historical "no language field" behaviour so the
    # cache is bit-for-bit compatible with the existing English caches.
    if language == "de":
        payload_input["language"] = "de"
    if scoring_mode == "code" and response_language_hint:
        payload_input["response_language_hint"] = response_language_hint
    body = {"input": payload_input}
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "x-latence-tenant-id": "corpus-classifier-cache",
    }
    if auth_bearer:
        headers["authorization"] = f"Bearer {auth_bearer}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    dt = (time.perf_counter() - t0) * 1000.0
    inner = out.get("output") or out
    # Surface service-level errors so the caller can count the row as
    # failed rather than silently recording all-None channels.
    if isinstance(inner, dict) and inner.get("success") is False:
        raise RuntimeError(
            f"service_error: {inner.get('error_code')}: {inner.get('error')}"
        )
    return inner, dt


def _extract_channels(out: Dict[str, Any]) -> Dict[str, Any]:
    full = out.get("full") or {}
    scores = full.get("scores") or {}
    channels = {
        CHANNEL_ALIAS[k]: scores.get(k) for k in CHANNEL_KEYS
    }
    return {
        "channels": channels,
        "primary_score": scores.get("primary_score"),
        "primary_name": scores.get("primary_name"),
        "groundedness_v2": scores.get("groundedness_v2"),
        "structured_source_detected": scores.get("structured_source_detected"),
        "composite_phantom_score": scores.get("composite_phantom_score"),
        "band": scores.get("risk_band") or out.get("band"),
        "reverse_context": scores.get("reverse_context"),
    }


def _scoring_mode_for_class(class_key: str) -> str:
    return "code" if class_key == "code.agentic_trace" else "rag"


def _response_lang_hint(row: Dict[str, Any], class_key: str) -> Optional[str]:
    if class_key != "code.agentic_trace":
        return None
    # Fallback heuristic; the code-lane AST extractor auto-detects when
    # omitted, but we hint python since the transcripts_v2 bank is mostly
    # Python-heavy (~85% of fenced blocks).
    return "python"


def _load_rows(split: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / f"{split}.parquet"
    t = pq.read_table(path)
    cols = {name: t.column(name).to_pylist() for name in t.schema.names}
    n = t.num_rows
    return [
        {k: cols[k][i] for k in cols}
        for i in range(n)
    ]


def _load_rows_de(split: str, classes: List[str]) -> List[Dict[str, Any]]:
    """Load the GPT-4.1 German translations produced by
    ``scripts/translate_bench_to_german.py``.

    The translator writes one JSONL per class (rows from all splits
    interleaved, each row tagged with ``split``). We re-key the row to
    the same shape ``cache_split`` expects from parquet (``row_id,
    class_key, query, response, raw_context, is_grounded``) and filter
    by the requested split. Missing class files are simply skipped --
    the operator may translate one class at a time.
    """

    rows: List[Dict[str, Any]] = []
    for class_key in classes:
        path = GERMAN_TRANSLATION_DIR / f"{class_key}.jsonl"
        if not path.exists():
            logging.warning("german_translation missing for class=%s at %s", class_key, path)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("split") != split:
                    continue
                rows.append(
                    {
                        "row_id": rec["row_id"],
                        "class_key": rec.get("class_key", class_key),
                        "is_grounded": rec.get("is_grounded"),
                        "query": rec.get("query", ""),
                        "response": rec.get("response", ""),
                        "raw_context": rec.get("raw_context", ""),
                    }
                )
    return rows


def _sample_per_class(
    rows: List[Dict[str, Any]], cap: int, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CLASS_KEYS}
    for r in rows:
        by_class.setdefault(r["class_key"], []).append(r)
    for k in by_class:
        bucket = by_class[k]
        # Prefer labelled rows for calibration metrics; fall back to
        # unlabelled if needed to reach the cap.
        labelled = [r for r in bucket if r.get("is_grounded") is not None]
        unlabelled = [r for r in bucket if r.get("is_grounded") is None]
        labelled.sort(key=lambda r: r["row_id"])
        unlabelled.sort(key=lambda r: r["row_id"])
        rng.shuffle(labelled)
        rng.shuffle(unlabelled)
        merged = labelled + unlabelled
        if cap is not None and cap > 0 and len(merged) > cap:
            merged = merged[:cap]
        by_class[k] = merged
    return by_class


def _load_existing(path: Path) -> set:
    present = set()
    if not path.exists():
        return present
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                present.add(rec["row_id"])
            except Exception:
                continue
    return present


def cache_split(
    split: str,
    *,
    url: str,
    cap: int,
    classes: List[str],
    concurrency: int,
    timeout: float,
    seed: int,
    force: bool,
    language: str = DEFAULT_LANGUAGE,
    auth_bearer: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    logging.info(
        "caching channel scores for split=%s cap=%s language=%s",
        split,
        cap,
        language,
    )
    if language == "de":
        all_rows = _load_rows_de(split, classes)
    else:
        all_rows = _load_rows(split)
    sampled = _sample_per_class(all_rows, cap=cap, seed=seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}
    for class_key in classes:
        rows = sampled.get(class_key, [])
        if not rows:
            summary[class_key] = {"scored": 0, "skipped": 0, "failed": 0, "cap": cap}
            continue
        suffix = f".{language}" if language != DEFAULT_LANGUAGE else ""
        out_path = OUT_DIR / f"{class_key}{suffix}.{split}.jsonl"
        if force and out_path.exists():
            out_path.unlink()
        already = _load_existing(out_path)
        pending = [r for r in rows if r["row_id"] not in already]
        logging.info("class=%s total=%d already=%d pending=%d", class_key, len(rows), len(already), len(pending))
        scoring_mode = _scoring_mode_for_class(class_key)
        failures = 0
        t0 = time.perf_counter()
        with out_path.open("a", encoding="utf-8") as fh:

            def _task(row):
                try:
                    out, dt = _score_one(
                        url,
                        row=row,
                        scoring_mode=scoring_mode,
                        timeout=timeout,
                        response_language_hint=_response_lang_hint(row, class_key),
                        language=language,
                        auth_bearer=auth_bearer,
                    )
                    info = _extract_channels(out)
                    return row, info, dt, None
                except Exception as exc:  # pragma: no cover - network
                    return row, None, 0.0, str(exc)

            # as_completed avoids the head-of-line blocking in ex.map
            # (one slow row would otherwise stall every subsequent
            # result). Flush after each write so we always have a
            # usable cache even if the process is killed mid-run.
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [ex.submit(_task, row) for row in pending]
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row, info, dt, err = fut.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        failures += 1
                        logging.warning("task failed: %s", exc)
                        continue
                    if err:
                        failures += 1
                        continue
                    rec = {
                        "row_id": row["row_id"],
                        "class_key": row["class_key"],
                        "is_grounded": row.get("is_grounded"),
                        "scoring_mode": scoring_mode,
                        "latency_ms": round(dt, 2),
                        **info,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    if done % 50 == 0:
                        logging.info(
                            "class=%s progress=%d/%d failures=%d",
                            class_key, done, len(pending), failures,
                        )
        elapsed = time.perf_counter() - t0
        summary[class_key] = {
            "scored": len(pending) - failures,
            "skipped": len(already),
            "failed": failures,
            "elapsed_sec": round(elapsed, 2),
            "cap": cap,
        }
        logging.info("done class=%s scored=%d failed=%d in %.1fs", class_key, summary[class_key]["scored"], failures, elapsed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/runsync")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--classes", nargs="+", default=list(CLASS_KEYS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="Per-class per-split row cap. 0 = no cap.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Delete existing class.split jsonl before scoring.")
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help=(
            "Source language to score. ``en`` reads ``<split>.parquet`` (the "
            "historical English bench); ``de`` reads the GPT-4.1 "
            "translations under ``german_translation/<class>.jsonl`` and "
            "writes the cache under ``<class>.de.<split>.jsonl`` so the "
            "calibration sweep can target the German bundle without "
            "stomping the English one."
        ),
    )
    parser.add_argument(
        "--auth-bearer-env",
        default="LATENCE_TRACE_API_KEY",
        help=(
            "Environment variable to read a bearer token from. When set, "
            "every request gets ``Authorization: Bearer <value>``. Required "
            "when ``--url`` points at RunPod (``api.runpod.ai/...``)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    auth_bearer = os.environ.get(args.auth_bearer_env) if args.auth_bearer_env else None
    if "api.runpod.ai" in args.url and not auth_bearer:
        parser.error(
            f"--url targets RunPod but env var {args.auth_bearer_env} is empty. "
            "Export the RunPod API key before running."
        )
    total_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split in args.splits:
        total_summary[split] = cache_split(
            split,
            url=args.url,
            cap=args.cap,
            classes=args.classes,
            concurrency=args.concurrency,
            timeout=args.timeout,
            seed=args.seed,
            force=args.force,
            language=args.language,
            auth_bearer=auth_bearer,
        )
    summary_suffix = f".{args.language}" if args.language != DEFAULT_LANGUAGE else ""
    (OUT_DIR / f"cache_summary{summary_suffix}.json").write_text(
        json.dumps(total_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(total_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
