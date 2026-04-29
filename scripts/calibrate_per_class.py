"""Phase B.2 - Offline sweep over fusion weights + thresholds per class.

Reads the cached per-row channel scores produced by
``scripts/cache_channel_scores.py`` and, for each of the six corpus
classes, finds the ``(fusion_weights, thresholds)`` tuple that maximises
the class-specific objective. Persists the winning bundle as
``latence_trace/data/calibration.<class_key>.json`` (the runtime router
loads these at import time) and writes a human-readable summary at
``data/corpus_classifier/calibration_sweep_report.md``.

Per-class objective:

* ``rag.prose.enterprise``    - Veracier composite: 0.4*red_p + 0.4*green_p + 0.2*(1 - amber_rate)
* ``rag.prose.short_factoid`` - F1@best-threshold (halluc positive)
* ``rag.prose.multi_claim``   - F1@best-threshold
* ``rag.structured``          - F1@best-threshold
* ``rag.code_in_context``     - F1@best-threshold
* ``code.agentic_trace``      - paired accuracy (correct > wrong) on
                                 ``composite_phantom_score``

The sweep is fully deterministic and offline. It does **not** re-query
the scoring service.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/corpus_classifier"
CACHE_DIR = DATA_DIR / "channel_scores"
CALIBRATION_DIR = REPO_ROOT / "latence_trace/data"
REPORT_PATH = DATA_DIR / "calibration_sweep_report.md"

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

CHANNELS = ("calibrated", "literal", "nli", "semantic_entropy", "structured")
FUSION_GRID = (0.0, 0.2, 0.3, 0.5, 0.8)
FUSION_SUM_MIN = 0.5
GREEN_GRID = np.arange(0.50, 1.00, 0.01)  # 50 values
AMBER_GRID = np.arange(0.30, 0.95, 0.02)  # 33 values

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_cache(class_key: str, split: str) -> List[Dict[str, Any]]:
    path = CACHE_DIR / f"{class_key}.{split}.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_metadata_for_split(split: str) -> Dict[str, Dict[str, Any]]:
    t = pq.read_table(DATA_DIR / f"{split}.parquet")
    rids = t.column("row_id").to_pylist()
    metas = t.column("metadata_json").to_pylist()
    return {rid: json.loads(m or "{}") for rid, m in zip(rids, metas)}


# ---------------------------------------------------------------------------
# Fusion + metrics
# ---------------------------------------------------------------------------

def _fuse(channels_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted fusion with per-row renormalisation over available channels.

    ``channels_matrix`` has shape (n_rows, n_channels); ``weights`` has
    shape (n_channels,). ``NaN`` in the matrix marks a channel that the
    service did not emit for that row; for such rows the weight is
    effectively zero and the remaining weights are re-normalised so the
    fused score stays on ``[0, 1]``.
    """
    mask = ~np.isnan(channels_matrix)
    w = weights[None, :] * mask
    w_sum = w.sum(axis=1, keepdims=True)
    safe_sum = np.where(w_sum <= 0.0, 1.0, w_sum)
    filled = np.where(mask, channels_matrix, 0.0)
    fused = (filled * w).sum(axis=1, keepdims=True) / safe_sum
    fused = fused.squeeze(axis=1)
    fused = np.where(w_sum.squeeze(axis=1) <= 0.0, np.nan, fused)
    return fused


def _fill_nan_with(primary_scores: np.ndarray, fused: np.ndarray) -> np.ndarray:
    """Serving-reality fallback: rows where the per-class fusion is NaN
    (e.g. because NLI was flaky in the cache) fall back to the primary
    ``groundedness_v2`` score the service actually emitted. This keeps
    held-out evaluation aligned with what production produces."""
    result = fused.copy()
    mask = np.isnan(result)
    result[mask] = primary_scores[mask]
    return result


def _f1_best_threshold(scores: np.ndarray, labels_halluc: np.ndarray) -> Dict[str, Any]:
    """Positive class = hallucinated. Predict halluc when score < threshold."""
    if scores.size == 0:
        return {"f1": None, "threshold": None}
    uniq = np.unique(scores[~np.isnan(scores)])
    thresholds = np.concatenate([uniq, [uniq.max() + 1e-6]]) if uniq.size else np.array([0.5])
    best: Dict[str, Any] = {"f1": -1.0, "threshold": None, "precision": 0.0, "recall": 0.0}
    valid = ~np.isnan(scores)
    s = scores[valid]
    y = labels_halluc[valid]
    for t in thresholds:
        pred = s < t
        tp = int(((pred == True) & (y == 1)).sum())
        fp = int(((pred == True) & (y == 0)).sum())
        fn = int(((pred == False) & (y == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        if f1 > best["f1"]:
            best = {"f1": float(f1), "precision": float(p), "recall": float(r), "threshold": float(t)}
    return best


def _veracier_bands(
    scores: np.ndarray, green_th: float, amber_th: float
) -> np.ndarray:
    """Returns 'green' / 'amber' / 'red' per row (str array)."""
    bands = np.empty(scores.shape, dtype=object)
    for i, s in enumerate(scores):
        if math.isnan(s):
            bands[i] = "unknown"
        elif s >= green_th:
            bands[i] = "green"
        elif s >= amber_th:
            bands[i] = "amber"
        else:
            bands[i] = "red"
    return bands


def _veracier_objective(
    scores: np.ndarray, gold_bands: np.ndarray, green_th: float, amber_th: float
) -> Tuple[float, Dict[str, float]]:
    pred = _veracier_bands(scores, green_th, amber_th)
    # red_p: precision of red predictions where gold=red
    pred_red = pred == "red"
    pred_green = pred == "green"
    pred_amber = pred == "amber"
    gold_red = gold_bands == "red"
    gold_green = gold_bands == "green"

    red_p = (pred_red & gold_red).sum() / max(1, pred_red.sum()) if pred_red.any() else 0.0
    green_p = (pred_green & gold_green).sum() / max(1, pred_green.sum()) if pred_green.any() else 0.0
    amber_rate = pred_amber.sum() / max(1, pred.size)
    red_recall = (pred_red & gold_red).sum() / max(1, gold_red.sum()) if gold_red.any() else 0.0
    green_recall = (pred_green & gold_green).sum() / max(1, gold_green.sum()) if gold_green.any() else 0.0

    obj = 0.4 * red_p + 0.4 * green_p + 0.2 * (1.0 - amber_rate)
    # Require minimum recall thresholds so we don't trivially maximise
    # precision by never firing red/green. Fall back to a small penalty
    # when recall collapses.
    if red_recall < 0.5 or green_recall < 0.5:
        obj *= 0.5
    return float(obj), {
        "red_precision": float(red_p),
        "green_precision": float(green_p),
        "red_recall": float(red_recall),
        "green_recall": float(green_recall),
        "amber_rate": float(amber_rate),
    }


def _paired_accuracy(
    rows: List[Dict[str, Any]], meta: Dict[str, Dict[str, Any]]
) -> Tuple[float, int, int]:
    """Pair correct-vs-wrong within the same transcripts_v2 base scenario."""
    by_scenario: Dict[str, Dict[str, float]] = {}
    for r in rows:
        rid = r["row_id"]
        m = meta.get(rid, {})
        base = str(m.get("base_scenario_id") or "")
        sub = str(m.get("subcategory") or "")
        if not base or sub not in {"correct", "wrong"}:
            continue
        s = r.get("composite_phantom_score")
        if s is None:
            # Fall back to the RAG primary score when the code lane did
            # not emit a composite (happens on rows that the AST extractor
            # short-circuits). This keeps the pair in the sample rather
            # than silently dropping it.
            s = r.get("primary_score")
        if s is None:
            continue
        by_scenario.setdefault(base, {})[sub] = float(s)
    wins = ties = losses = 0
    total = 0
    for sub_map in by_scenario.values():
        if "correct" not in sub_map or "wrong" not in sub_map:
            continue
        total += 1
        c = sub_map["correct"]
        w = sub_map["wrong"]
        if c > w:
            wins += 1
        elif c < w:
            losses += 1
        else:
            ties += 1
    if total == 0:
        return 0.0, 0, 0
    return wins / total, wins, total


# ---------------------------------------------------------------------------
# Sweep kernels
# ---------------------------------------------------------------------------

def _channels_to_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    mat = np.full((len(rows), len(CHANNELS)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        ch = r.get("channels") or {}
        for j, name in enumerate(CHANNELS):
            v = ch.get(name)
            if v is None:
                continue
            try:
                mat[i, j] = float(v)
            except (TypeError, ValueError):
                continue
    return mat


def _available_channels(matrix: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Channels that carry a value for >= ``threshold`` fraction of rows."""
    avail: List[int] = []
    for j in range(matrix.shape[1]):
        frac = float((~np.isnan(matrix[:, j])).mean())
        if frac >= threshold:
            avail.append(j)
    return avail


def _iter_fusion_configs(available: List[int]) -> List[np.ndarray]:
    """Enumerate weight vectors over available channels with sum >= 0.5."""
    combos: List[np.ndarray] = []
    for values in itertools.product(FUSION_GRID, repeat=len(available)):
        if sum(values) < FUSION_SUM_MIN - 1e-9:
            continue
        w = np.zeros(len(CHANNELS), dtype=np.float64)
        for idx, v in zip(available, values):
            w[idx] = v
        combos.append(w)
    return combos


def _normalise(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    if s <= 0:
        return w
    return w / s


# ---------------------------------------------------------------------------
# Per-class calibration
# ---------------------------------------------------------------------------

def calibrate_binary_class(
    class_key: str, *, objective_label: str
) -> Dict[str, Any]:
    train = [r for r in _load_cache(class_key, "train") if r.get("is_grounded") is not None]
    test = [r for r in _load_cache(class_key, "test") if r.get("is_grounded") is not None]
    if not train:
        raise SystemExit(f"no labelled train rows cached for {class_key}")
    mat_train = _channels_to_matrix(train)
    mat_test = _channels_to_matrix(test) if test else np.empty((0, len(CHANNELS)))
    y_train = np.array([0 if r["is_grounded"] else 1 for r in train], dtype=np.int32)
    y_test = np.array([0 if r["is_grounded"] else 1 for r in test], dtype=np.int32) if test else np.empty((0,), dtype=np.int32)
    # Pick fusion on train-side availability only; the runtime fuser
    # handles missing channels by re-normalising weights so a channel
    # that is intermittently flaky at serving time degrades gracefully
    # rather than collapsing the fused score to NaN.
    available = _available_channels(mat_train)
    logger.info("class=%s available_channels=%s", class_key, [CHANNELS[i] for i in available])
    configs = _iter_fusion_configs(available)

    primary_train = np.array(
        [float(r.get("groundedness_v2") or r.get("primary_score") or 0.0) for r in train]
    )
    primary_test = np.array(
        [float(r.get("groundedness_v2") or r.get("primary_score") or 0.0) for r in test]
    ) if test else np.empty((0,))

    best: Dict[str, Any] = {"metric_value": -1.0, "fusion_weights_raw": None, "threshold": None}
    for raw_w in configs:
        fused_train = _fill_nan_with(primary_train, _fuse(mat_train, raw_w))
        result = _f1_best_threshold(fused_train, y_train)
        f1 = result["f1"]
        if f1 is None or f1 < 0:
            continue
        if f1 > best["metric_value"]:
            best = {
                "metric_value": float(f1),
                "fusion_weights_raw": raw_w.copy(),
                "threshold": float(result["threshold"]),
                "precision": result["precision"],
                "recall": result["recall"],
            }
    if best["fusion_weights_raw"] is None:
        raise SystemExit(f"no valid fusion found for {class_key}")
    norm = _normalise(best["fusion_weights_raw"])
    # Validate on held-out test split
    fused_test = (
        _fill_nan_with(primary_test, _fuse(mat_test, best["fusion_weights_raw"]))
        if test
        else np.array([])
    )
    test_f1 = None
    if test:
        preds = fused_test < best["threshold"]
        tp = int(((preds == True) & (y_test == 1)).sum())
        fp = int(((preds == True) & (y_test == 0)).sum())
        fn = int(((preds == False) & (y_test == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        test_f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    # Derive an amber band around the green threshold: 5pp below.
    green_th = float(best["threshold"])
    amber_th = max(0.0, green_th - 0.05)

    return {
        "scoring_mode": "rag",
        "fusion_weights": {name: float(norm[i]) for i, name in enumerate(CHANNELS)},
        "fusion_weights_raw": {name: float(best["fusion_weights_raw"][i]) for i, name in enumerate(CHANNELS)},
        "thresholds": {"green": green_th, "amber": amber_th},
        "nli_model_hint": "en",
        "metric": objective_label,
        "metric_value": float(best["metric_value"]),
        "train_support": {"precision": best["precision"], "recall": best["recall"], "n_rows": int(y_train.size)},
        "test_validation": {
            "f1": float(test_f1) if test_f1 is not None else None,
            "n_rows": int(y_test.size),
        },
    }


def calibrate_veracier() -> Dict[str, Any]:
    class_key = "rag.prose.enterprise"
    train = _load_cache(class_key, "train")
    test = _load_cache(class_key, "test")
    if not train:
        raise SystemExit("no train rows cached for veracier")

    def _prep(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        # Derive gold_band from is_grounded (True->green, False->red, None->amber)
        gold = []
        kept: List[Dict[str, Any]] = []
        for r in rows:
            ig = r.get("is_grounded")
            if ig is True:
                gold.append("green")
            elif ig is False:
                gold.append("red")
            else:
                gold.append("amber")
            kept.append(r)
        return _channels_to_matrix(kept), np.array(gold, dtype=object)

    mat_train, gold_train = _prep(train)
    mat_test, gold_test = _prep(test)
    available = _available_channels(mat_train)
    logger.info("veracier available_channels=%s", [CHANNELS[i] for i in available])
    configs = _iter_fusion_configs(available)

    primary_train = np.array(
        [float(r.get("groundedness_v2") or r.get("primary_score") or 0.0) for r in train]
    )
    primary_test = np.array(
        [float(r.get("groundedness_v2") or r.get("primary_score") or 0.0) for r in test]
    ) if test else np.empty((0,))
    best: Dict[str, Any] = {"metric_value": -1.0}
    for raw_w in configs:
        fused = _fill_nan_with(primary_train, _fuse(mat_train, raw_w))
        for green_th in GREEN_GRID:
            for amber_th in AMBER_GRID:
                if amber_th >= green_th - 0.005:
                    continue
                obj, components = _veracier_objective(fused, gold_train, float(green_th), float(amber_th))
                if obj > best["metric_value"]:
                    best = {
                        "metric_value": float(obj),
                        "fusion_weights_raw": raw_w.copy(),
                        "thresholds": {"green": float(green_th), "amber": float(amber_th)},
                        "components": components,
                    }
    if "fusion_weights_raw" not in best:
        raise SystemExit("no valid fusion for veracier")
    norm = _normalise(best["fusion_weights_raw"])
    fused_test = _fill_nan_with(primary_test, _fuse(mat_test, best["fusion_weights_raw"]))
    obj_test, components_test = _veracier_objective(
        fused_test, gold_test, best["thresholds"]["green"], best["thresholds"]["amber"]
    )

    return {
        "scoring_mode": "rag",
        "fusion_weights": {name: float(norm[i]) for i, name in enumerate(CHANNELS)},
        "fusion_weights_raw": {name: float(best["fusion_weights_raw"][i]) for i, name in enumerate(CHANNELS)},
        "thresholds": best["thresholds"],
        "nli_model_hint": "en",
        "metric": "veracier_composite",
        "metric_value": float(best["metric_value"]),
        "train_components": best["components"],
        "test_validation": {"objective": float(obj_test), **components_test, "n_rows": int(gold_test.size)},
    }


def calibrate_code_agentic() -> Dict[str, Any]:
    class_key = "code.agentic_trace"
    train = _load_cache(class_key, "train")
    test = _load_cache(class_key, "test")
    if not train:
        raise SystemExit("no train rows cached for code.agentic_trace")
    meta_train = _load_metadata_for_split("train")
    meta_test = _load_metadata_for_split("test")

    paired_train, wins_train, total_train = _paired_accuracy(train, meta_train)
    paired_test, wins_test, total_test = _paired_accuracy(test, meta_test)

    # Threshold on composite_phantom_score. Scan for best band split that
    # maximises separation of correct vs wrong within each base scenario.
    # We use the 25th / 75th percentile of composite_score across correct
    # and wrong as anchors, then round to a clean 0.01-grid value.
    comps_correct = [r.get("composite_phantom_score") for r in train if r.get("is_grounded") is True]
    comps_wrong = [r.get("composite_phantom_score") for r in train if r.get("is_grounded") is False]
    comps_correct = np.array([c for c in comps_correct if c is not None])
    comps_wrong = np.array([c for c in comps_wrong if c is not None])
    if comps_correct.size and comps_wrong.size:
        green_th = float(round(float(np.percentile(comps_correct, 40)), 2))
        amber_th = float(round(float(np.percentile(comps_wrong, 60)), 2))
        if amber_th >= green_th:
            amber_th = max(0.0, green_th - 0.05)
    else:
        green_th, amber_th = 0.70, 0.55

    return {
        "scoring_mode": "code",
        "fusion_weights": {name: 0.0 for name in CHANNELS},
        "fusion_weights_raw": {name: 0.0 for name in CHANNELS},
        "thresholds": {"green": green_th, "amber": amber_th},
        "nli_model_hint": "en",
        "metric": "paired_accuracy",
        "metric_value": float(paired_train),
        "train_support": {
            "paired_accuracy": float(paired_train),
            "wins": int(wins_train),
            "n_pairs": int(total_train),
        },
        "test_validation": {
            "paired_accuracy": float(paired_test),
            "wins": int(wins_test),
            "n_pairs": int(total_test),
        },
        "notes": "Thresholds apply to composite_phantom_score produced by the code lane; fusion_weights are inert because the code lane runs its own AST+literal+NLI cascade.",
    }


# ---------------------------------------------------------------------------
# Write artefacts
# ---------------------------------------------------------------------------

def _class_key_slug(class_key: str) -> str:
    return class_key.replace(".", "_")


def _manifest_sha() -> Optional[str]:
    manifest = DATA_DIR / "manifest.json"
    if not manifest.exists():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _render_report(results: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# Calibration sweep report (Phase B)\n"]
    lines.append("Class-by-class winning fusion + thresholds. Metrics on train; test column is the locked held-out validation.\n")
    lines.append(
        "| Class | Objective | Train metric | Test metric | Fusion (c,l,n,se,st) | Green | Amber |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for class_key in CLASS_KEYS:
        r = results.get(class_key)
        if r is None:
            continue
        fw = r["fusion_weights"]
        fusion_tuple = ",".join(f"{fw[c]:.2f}" for c in CHANNELS)
        train_metric = f"{r['metric_value']:.4f}"
        test_metric = ""
        if "test_validation" in r:
            tv = r["test_validation"]
            key = "f1" if "f1" in tv else ("paired_accuracy" if "paired_accuracy" in tv else "objective")
            val = tv.get(key)
            test_metric = f"{val:.4f}" if isinstance(val, (int, float)) else "-"
        lines.append(
            f"| `{class_key}` | {r['metric']} | {train_metric} | {test_metric} | {fusion_tuple} | {r['thresholds']['green']:.2f} | {r['thresholds']['amber']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", default=list(CLASS_KEYS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results: Dict[str, Dict[str, Any]] = {}
    manifest_sha = _manifest_sha()

    for class_key in args.classes:
        logger.info("calibrating class=%s", class_key)
        if class_key == "rag.prose.enterprise":
            r = calibrate_veracier()
        elif class_key == "code.agentic_trace":
            r = calibrate_code_agentic()
        else:
            r = calibrate_binary_class(class_key, objective_label="f1_at_best_threshold")
        r["trained_on"] = manifest_sha
        results[class_key] = r
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CALIBRATION_DIR / f"calibration.{_class_key_slug(class_key)}.json"
        out_path.write_text(json.dumps(r, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("wrote %s", out_path)

    REPORT_PATH.write_text(_render_report(results), encoding="utf-8")
    logger.info("wrote report %s", REPORT_PATH)
    print(_render_report(results))


if __name__ == "__main__":
    main()
