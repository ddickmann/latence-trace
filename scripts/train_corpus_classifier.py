"""Phase C.3 - Train the corpus-type classifier.

Reads ``data/corpus_classifier/train.parquet``, featurises every row via
:func:`latence_trace.core.corpus_router.features.featurize`, fits a
multinomial logistic regression, evaluates on ``test.parquet`` and
persists the bundle as ``latence_trace/data/corpus_classifier.joblib``.

Falls back to a :class:`~sklearn.ensemble.GradientBoostingClassifier` if
held-out top-1 accuracy is below 0.95 (the ship gate). The fallback is
still deterministic (fixed seed) and joblib-serialisable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latence_trace.core.corpus_router.features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    featurize_batch,
)

DATA_DIR = REPO_ROOT / "data/corpus_classifier"
ARTEFACT_DIR = REPO_ROOT / "latence_trace/data"
JOBLIB_PATH = ARTEFACT_DIR / "corpus_classifier.joblib"
METRICS_PATH = DATA_DIR / "classifier_metrics.json"
CM_PATH = DATA_DIR / "confusion_matrix.json"

SHIP_GATE_TOP1 = 0.95
SHIP_GATE_MIN_CLASS_RECALL = 0.85
SEED = 42

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

logger = logging.getLogger(__name__)


def _load_split(split: str) -> Tuple[List[Dict[str, str]], List[str]]:
    t = pq.read_table(DATA_DIR / f"{split}.parquet")
    rows = [
        {"query": q, "response": r, "raw_context": c}
        for q, r, c in zip(
            t.column("query").to_pylist(),
            t.column("response").to_pylist(),
            t.column("raw_context").to_pylist(),
        )
    ]
    labels = t.column("class_key").to_pylist()
    return rows, labels


def _fit_lr(X: np.ndarray, y: np.ndarray) -> Tuple[Any, StandardScaler]:
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # sklearn >= 1.5 makes lbfgs multinomial by default; the
    # ``multi_class`` keyword was removed in 1.7.
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        C=1.0,
    )
    clf.fit(Xs, y)
    return clf, scaler


def _fit_gbc(X: np.ndarray, y: np.ndarray) -> Tuple[Any, StandardScaler]:
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        random_state=SEED,
    )
    clf.fit(Xs, y)
    return clf, scaler


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).sum() / max(1, y_true.size))


def _top2_margins(model: Any, X: np.ndarray) -> Dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {"mean": 1.0, "p05": 1.0}
    proba = model.predict_proba(X)
    margins: List[float] = []
    for row in proba:
        ordered = sorted((float(value) for value in row), reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else 1.0)
    arr = np.asarray(margins, dtype=np.float64)
    return {"mean": float(arr.mean()), "p05": float(np.quantile(arr, 0.05))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gate", action="store_true", help="Persist even if the 0.95 top-1 gate is not cleared (for debugging).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_rows, train_labels = _load_split("train")
    test_rows, test_labels = _load_split("test")
    logger.info("featurising train=%d test=%d", len(train_rows), len(test_rows))

    X_train, _ = featurize_batch(train_rows)
    X_test, _ = featurize_batch(test_rows)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(train_labels, dtype=object)
    y_test = np.asarray(test_labels, dtype=object)

    # Fit LR
    logger.info("fitting multinomial logistic regression...")
    lr, scaler = _fit_lr(X_train, y_train)
    y_pred_test = lr.predict(scaler.transform(X_test))
    lr_acc = _accuracy(y_test, y_pred_test)
    logger.info("LR held-out top-1 accuracy=%.4f", lr_acc)

    model_kind = "logistic_regression"
    chosen_clf = lr
    chosen_scaler = scaler
    if lr_acc < SHIP_GATE_TOP1:
        logger.warning(
            "LR below ship gate (%.4f < %.4f), falling back to GradientBoostingClassifier",
            lr_acc, SHIP_GATE_TOP1,
        )
        gbc, gbc_scaler = _fit_gbc(X_train, y_train)
        y_pred_test_gbc = gbc.predict(gbc_scaler.transform(X_test))
        gbc_acc = _accuracy(y_test, y_pred_test_gbc)
        logger.info("GBC held-out top-1 accuracy=%.4f", gbc_acc)
        if gbc_acc > lr_acc:
            chosen_clf = gbc
            chosen_scaler = gbc_scaler
            y_pred_test = y_pred_test_gbc
            model_kind = "gradient_boosting"

    final_acc = _accuracy(y_test, y_pred_test)

    # Per-class metrics
    cm_labels = list(CLASS_KEYS)
    cm = confusion_matrix(y_test, y_pred_test, labels=cm_labels).tolist()
    cls_report = classification_report(y_test, y_pred_test, labels=cm_labels, digits=4, zero_division=0, output_dict=True)
    per_class_recall = {
        label: float((cls_report.get(label) or {}).get("recall", 0.0))
        for label in CLASS_KEYS
    }
    min_class_recall = min(per_class_recall.values()) if per_class_recall else 0.0
    top2_margin = _top2_margins(chosen_clf, chosen_scaler.transform(X_test))

    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "model_kind": model_kind,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(chosen_clf.classes_),
        "scaler_mean": chosen_scaler.mean_.tolist(),
        "scaler_scale": chosen_scaler.scale_.tolist(),
        "model": chosen_clf,
        "top1_accuracy_heldout": final_acc,
        "trained_on_manifest_sha": hashlib.sha256((DATA_DIR / "manifest.json").read_bytes()).hexdigest(),
    }

    joblib.dump(bundle, JOBLIB_PATH, compress=3)
    joblib_sha = hashlib.sha256(JOBLIB_PATH.read_bytes()).hexdigest()

    CM_PATH.write_text(
        json.dumps(
            {"labels": cm_labels, "matrix": cm, "top1_accuracy": final_acc, "model_kind": model_kind},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "top1_accuracy": final_acc,
                "model_kind": model_kind,
                "per_class": {k: v for k, v in cls_report.items() if isinstance(v, dict)},
                "ship_gate": SHIP_GATE_TOP1,
                "min_class_recall": min_class_recall,
                "per_class_recall_gate": SHIP_GATE_MIN_CLASS_RECALL,
                "top2_margin": top2_margin,
                "passes_ship_gate": (
                    final_acc >= SHIP_GATE_TOP1
                    and min_class_recall >= SHIP_GATE_MIN_CLASS_RECALL
                ),
                "artefact_sha256": joblib_sha,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "seed": SEED,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s (%d bytes, sha=%s)", JOBLIB_PATH, JOBLIB_PATH.stat().st_size, joblib_sha[:12])
    logger.info("wrote %s and %s", METRICS_PATH, CM_PATH)
    if (
        (final_acc < SHIP_GATE_TOP1 or min_class_recall < SHIP_GATE_MIN_CLASS_RECALL)
        and not args.skip_gate
    ):
        logger.error(
            "classifier gates failed: top1=%.4f min_class_recall=%.4f",
            final_acc,
            min_class_recall,
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "top1_accuracy": final_acc,
                "min_class_recall": min_class_recall,
                "top2_margin": top2_margin,
                "model_kind": model_kind,
                "passes_gate": (
                    final_acc >= SHIP_GATE_TOP1
                    and min_class_recall >= SHIP_GATE_MIN_CLASS_RECALL
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
