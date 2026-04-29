"""Raw-benchmark loaders that emit rows in the student's common schema.

The goal is to produce a uniform dict for every source so the teacher
wrapper and label extractor do not need to care where a row came from.

Common schema (strings for text fields, ints for counts):

    {
        "pair_id":        str,   # groups variants of the same prompt+evidence
        "pair_row_id":    str,   # globally unique row id (pair_id::role::seq)
        "class_key":      str,   # FiLM class bucket ("ragtruth::qa", ...)
        "split_hint":     str,   # "train" / "test"; respected by caller
        "role":           str,   # "grounded" / "hallucinated" / "partial"
        "response_text":  str,
        "evidence_text":  str,   # raw_context (teacher will sentence-pack)
        "query_text":     str,
        "gold_band":      str,   # "green" / "amber" / "red"
        "source":         str,   # "ragtruth" / "veracier"
        "meta":           dict,  # free-form diagnostics
    }

Only fields the student needs are filled in; the five teacher-derived
label fields are left for :mod:`extract` to attach later.

Hygiene note
~~~~~~~~~~~~
HaluEval is deliberately **not** loaded here. HaluEval is held out for
reporting-day evaluation and must not touch the training corpus or we
lose our right to publish on it.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Iterable, Iterator, Mapping

logger = logging.getLogger("trace.v2.teacher_label.loaders")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_RAGTRUTH_DIR = _REPO_ROOT / "research" / "triangular_maxsim" / "external_data" / "RAGTruth" / "dataset"
_VERACIER_VARIANTS = (
    _REPO_ROOT
    / "data"
    / "veracier-industries"
    / "proof_bundle_v1"
    / "variants.curated.jsonl"
)


def _band_from_ragtruth(row: Mapping[str, object]) -> str:
    """RAGTruth gold = red iff any hallucination span is present, else green.

    RAGTruth does not publish an explicit 'amber' / uncertain band; the
    dataset is binary (faithful vs hallucinated) at the span level, so we
    map {no spans -> green, >=1 evident span -> red, only subtle/uninformative
    spans -> amber} to preserve three-way supervision for the student.
    """

    labels = row.get("labels") or []
    if not labels:
        return "green"
    evident = 0
    subtle = 0
    for span in labels:
        lt = str(span.get("label_type", "")).lower()
        if "evident" in lt:
            evident += 1
        else:
            subtle += 1
    if evident >= 1:
        return "red"
    if subtle >= 1:
        return "amber"
    return "green"


def _role_from_band(band: str) -> str:
    return {
        "green": "grounded",
        "amber": "partial_support",
        "red": "hallucinated",
    }.get(band, "grounded")


def iter_ragtruth_train(
    *,
    root: pathlib.Path | None = None,
    tasks: Iterable[str] = ("QA", "Summary", "Data2txt"),
) -> Iterator[dict]:
    """Yield normalized RAGTruth **train-split** rows.

    RAGTruth groups responses by ``source_id`` (prompt) with 6 different
    models generating on each prompt. We preserve that structure: every
    response-row carries the same ``pair_id = f"ragtruth::{source_id}"``
    so the pair-aware sampler later sees natural 6-way clusters with a
    mix of faithful and hallucinated siblings (89% of source clusters
    have both).
    """

    base = pathlib.Path(root) if root else _RAGTRUTH_DIR
    response_path = base / "response.jsonl"
    source_path = base / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError(
            f"expected RAGTruth dataset at {base} (response.jsonl + source_info.jsonl)"
        )

    sources: dict[str, dict] = {}
    with source_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sources[str(r["source_id"])] = r
    logger.info("loaded %d RAGTruth source_info rows", len(sources))

    want_tasks = {t.lower() for t in tasks}
    kept = 0
    skipped_missing_source = 0
    with response_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            sid = str(row["source_id"])
            src = sources.get(sid)
            if src is None:
                skipped_missing_source += 1
                continue
            task = str(src.get("task_type", "")).lower()
            if task not in want_tasks:
                continue
            evidence_text = str(src.get("source_info") or "").strip()
            query_text = str(src.get("prompt") or "").strip()
            response_text = str(row.get("response") or "").strip()
            if not evidence_text or not response_text:
                continue
            band = _band_from_ragtruth(row)
            role = _role_from_band(band)
            pair_id = f"ragtruth::{sid}"
            pair_row_id = f"{pair_id}::{task}::{row['id']}::{row['model']}"
            class_key = f"ragtruth::{task}"
            yield {
                "pair_id": pair_id,
                "pair_row_id": pair_row_id,
                "class_key": class_key,
                "split_hint": "train",
                "role": role,
                "response_text": response_text,
                "evidence_text": evidence_text,
                "query_text": query_text or "Summarize / answer the passage.",
                "gold_band": band,
                "source": "ragtruth",
                "meta": {
                    "ragtruth_id": row["id"],
                    "source_id": sid,
                    "model": row.get("model"),
                    "task_type": src.get("task_type"),
                    "n_hallucination_spans": len(row.get("labels") or []),
                },
            }
            kept += 1

    logger.info(
        "ragtruth train: yielded=%d skipped_missing_source=%d",
        kept,
        skipped_missing_source,
    )


def _veracier_band(row: Mapping[str, object]) -> str:
    """Veracier curated carries ``expected_band`` directly."""

    band = str(row.get("expected_band", "")).strip().lower()
    if band in {"green", "amber", "red"}:
        return band
    return "amber"


def iter_veracier_curated(
    *,
    path: pathlib.Path | None = None,
) -> Iterator[dict]:
    """Yield normalized rows from the curated Veracier variants file.

    Veracier groups variants by ``question_id`` — perfect / ambiguous /
    contradictory / off-topic mutations share a prompt + evidence set.
    We reuse that structure as ``pair_id`` so the pair-aware sampler
    sees each contrast set together, which is exactly what we need for
    the margin-ranking head.
    """

    p = pathlib.Path(path) if path else _VERACIER_VARIANTS
    if not p.exists():
        raise FileNotFoundError(f"expected Veracier variants at {p}")

    kept = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            evidence_text = str(row.get("raw_context") or "").strip()
            response_text = str(row.get("response_text") or "").strip()
            if not evidence_text or not response_text:
                continue
            qid = str(row.get("question_id") or "").strip()
            example_id = str(row.get("example_id") or "").strip()
            mutation = str(row.get("mutation_type") or "").strip().lower()
            band = _veracier_band(row)
            role_map = {
                "perfect": "grounded",
                "ambiguous": "partial_support",
                "contradiction": "hallucinated",
                "unsupported": "hallucinated",
                "off_topic": "hallucinated",
                "refusal": "refusal",
            }
            role = role_map.get(mutation, _role_from_band(band))
            pair_id = f"veracier::{qid}" if qid else f"veracier::{example_id}"
            pair_row_id = f"veracier::{example_id or mutation}"
            verticals = row.get("verticals") or []
            vertical_key = "::".join(sorted(str(v) for v in verticals)) or "mixed"
            class_key = f"veracier::{vertical_key}"
            yield {
                "pair_id": pair_id,
                "pair_row_id": pair_row_id,
                "class_key": class_key,
                "split_hint": "train",
                "role": role,
                "response_text": response_text,
                "evidence_text": evidence_text,
                "query_text": str(row.get("query_text") or "").strip()
                or "Answer the question using the provided context.",
                "gold_band": band,
                "source": "veracier",
                "meta": {
                    "question_id": qid,
                    "example_id": example_id,
                    "mutation_type": mutation,
                    "verticals": list(verticals),
                    "expected_groundedness_range": row.get(
                        "expected_groundedness_range"
                    ),
                },
            }
            kept += 1
    logger.info("veracier curated: yielded=%d rows", kept)
