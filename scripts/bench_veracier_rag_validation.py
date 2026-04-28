"""Build and run a Veracier Industries RAG validation bench for Latence Trace.

The harness is intentionally staged:

1. select-docs: deterministic pilot/capped/full document manifests from
   MASTER_INDEX.csv and ANSWER_KEY.json.
2. process-docs: run selected PDFs through the latence-python pipeline.
3. generate-responses: use OpenAI to create labelled Trace variants.
4. trace: score each generated response through the SDK Trace RAG lane.
5. report: summarize document processing, generation labels, and Trace metrics.

The output is a self-contained run directory with JSONL artifacts that can be
inspected before scaling. The default scope is a three-use-case pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

_REPO = Path(__file__).resolve().parent.parent
_SDK_SRC = _REPO.parent / "latence-python" / "src"
if _SDK_SRC.exists() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

DATA_DIR = _REPO / "data" / "veracier-industries"
DEFAULT_OUT = DATA_DIR / "trace_bench_runs" / "pilot"
PILOT_USE_CASES = ("CEO-01", "LEGAL-01", "CISO-02")

PRIMARY_CLASSES = {
    "CEO-01": {"SANCTIONS_RISK", "COMPLIANCE_RISK", "SUMMARY"},
    "LEGAL-01": {"YES"},
    "CISO-02": {"NIS2_SCOPE", "GAP", "GAP_ANALYSIS"},
}

DISTRACTOR_CLASSES = {
    "CEO-01": {"REFERENCE", "REVIEW"},
    "LEGAL-01": {"NO"},
    "CISO-02": {"REFERENCE", "POLICY", "AUDIT"},
}
NEGATIVE_GT_KEYS = {"no", "not_relevant", "irrelevant", "trap", "noise", "negative"}

TRACE_SCORE_FIELDS = (
    "score",
    "band",
    "primary_metric",
    "nli_aggregate",
    "context_coverage_ratio",
    "context_usage_ratio",
    "context_unused_ratio",
    "context_uncertain_ratio",
    "latency_ms",
    "profile",
    "effective_profile",
    "reason",
    "warnings",
    "profile_diagnostics",
)


@dataclass(frozen=True)
class DatasetRow:
    doc_id: str
    question_id: str
    role: str
    entity: str
    filename: str
    classification: str
    language: str
    fmt: str
    pages: int | None
    description: str

    @property
    def source_id(self) -> str:
        return f"{self.question_id}:{self.doc_id}"

    @property
    def path(self) -> Path:
        return DATA_DIR / "by_entity" / self.entity / self.filename

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "question_id": self.question_id,
            "role": self.role,
            "entity": self.entity,
            "filename": self.filename,
            "classification": self.classification,
            "language": self.language,
            "format": self.fmt,
            "pages": self.pages,
            "description": self.description,
            "path": str(self.path),
            "exists": self.path.exists(),
        }


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_write(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonl_append(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress_log(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"{_now()} event={event}"
    if payload:
        line += f" {payload}"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _load_rows() -> list[DatasetRow]:
    with (DATA_DIR / "MASTER_INDEX.csv").open(encoding="utf-8") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            pages_raw = (raw.get("pages") or "").strip()
            rows.append(
                DatasetRow(
                    doc_id=raw["doc_id"],
                    question_id=raw["question_id"],
                    role=raw["role"],
                    entity=raw["entity"],
                    filename=raw["filename"],
                    classification=raw["classification"],
                    language=raw["language"],
                    fmt=raw["format"],
                    pages=int(pages_raw) if pages_raw.isdigit() else None,
                    description=raw.get("description") or "",
                )
            )
    return rows


def _load_answer_key() -> dict[str, Any]:
    return json.loads((DATA_DIR / "ANSWER_KEY.json").read_text(encoding="utf-8"))


def _flatten_ground_truth(answer: dict[str, Any], *, include_negative: bool) -> set[str]:
    flattened: set[str] = set()
    for key, values in (answer.get("ground_truth") or {}).items():
        is_negative = str(key).lower() in NEGATIVE_GT_KEYS
        if is_negative != include_negative:
            continue
        if isinstance(values, list):
            flattened.update(str(value) for value in values)
            flattened.update(Path(str(value)).name for value in values)
    return flattened


def _round_robin_by_class(rows: list[DatasetRow], limit: int) -> list[DatasetRow]:
    by_class: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        by_class[row.classification].append(row)

    # Prefer format diversity when documents share a class.
    fmt_rank = {"scanned": 0, "mixed": 1, "searchable": 2}
    for values in by_class.values():
        values.sort(key=lambda r: (fmt_rank.get(r.fmt, 99), r.filename))

    selected: list[DatasetRow] = []
    class_names = sorted(by_class)
    while len(selected) < limit and class_names:
        remaining: list[str] = []
        for class_name in class_names:
            if by_class[class_name] and len(selected) < limit:
                selected.append(by_class[class_name].pop(0))
            if by_class[class_name]:
                remaining.append(class_name)
        class_names = remaining
    return selected


def _select_for_use_case(
    qid: str,
    rows: list[DatasetRow],
    answers: dict[str, Any],
    *,
    positives_per_case: int,
    distractors_per_case: int,
) -> dict[str, Any]:
    answer = answers[qid]
    case_rows = [row for row in rows if row.question_id == qid]
    positive_gt_files = _flatten_ground_truth(answer, include_negative=False)
    negative_gt_files = _flatten_ground_truth(answer, include_negative=True)

    primary_classes = PRIMARY_CLASSES.get(qid)
    if positive_gt_files:
        positives = [
            row
            for row in case_rows
            if row.filename in positive_gt_files
            or Path(row.filename).name in positive_gt_files
            or row.classification in (primary_classes or set())
        ]
    else:
        positives = [row for row in case_rows if row.classification in (primary_classes or set())]

    distractor_classes = DISTRACTOR_CLASSES.get(qid, {"NO", "TRAP", "REFERENCE"})
    positive_source_ids = {row.source_id for row in positives}
    distractors = [
        row
        for row in case_rows
        if row.source_id not in positive_source_ids
        and (
            row.classification in distractor_classes
            or row.filename in negative_gt_files
            or Path(row.filename).name in negative_gt_files
        )
    ]

    selected_positives = _round_robin_by_class(positives, positives_per_case)
    selected_distractors = _round_robin_by_class(distractors, distractors_per_case)
    selected_positive_ids = {row.source_id for row in selected_positives}
    selected = selected_positives + [
        row for row in selected_distractors if row.source_id not in selected_positive_ids
    ]

    return {
        "question_id": qid,
        "question": answer["question"],
        "role": answer.get("role"),
        "asker": answer.get("asker"),
        "entity": answer.get("entity"),
        "difficulty_factors": answer.get("difficulty_factors", []),
        "ground_truth": answer.get("ground_truth", {}),
        "documents": [row.to_manifest() for row in selected],
        "selection": {
            "positive_classes": sorted(primary_classes or []),
            "distractor_classes": sorted(distractor_classes),
            "positive_count": len(selected_positives),
            "distractor_count": len(selected) - len(selected_positives),
        },
    }


def _select_full_corpus(rows: list[DatasetRow]) -> dict[str, Any]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.entity, row.filename)
        doc_hash = hashlib.sha1(f"{row.entity}/{row.filename}".encode("utf-8")).hexdigest()[:12]
        entry = unique.setdefault(
            key,
            {
                **row.to_manifest(),
                "source_id": f"FULL:{doc_hash}",
                "doc_id": f"DOC-{doc_hash}",
                "question_id": "FULL",
                "role": "all",
                "classification": row.classification,
                "description": row.description,
                "question_ids": [],
                "classifications": [],
                "descriptions": [],
            },
        )
        entry["question_ids"].append(row.question_id)
        entry["classifications"].append(row.classification)
        if row.description and row.description not in entry["descriptions"]:
            entry["descriptions"].append(row.description)

    docs = []
    for doc in unique.values():
        doc["question_ids"] = sorted(set(doc["question_ids"]))
        doc["classifications"] = sorted(set(doc["classifications"]))
        doc["classification"] = ",".join(doc["classifications"])
        docs.append(doc)

    docs.sort(key=lambda d: (d["entity"], d["filename"]))
    return {
        "question_id": "FULL-CORPUS",
        "question": "Process every unique Veracier Industries PDF for RAG validation.",
        "role": "all",
        "asker": "benchmark",
        "entity": "all",
        "difficulty_factors": ["full_corpus", "multilingual", "searchable_scanned_mixed"],
        "ground_truth": {},
        "documents": docs,
        "selection": {
            "mode": "all_unique_pdfs",
            "unique_document_count": len(docs),
            "master_index_rows": len(rows),
        },
    }


def select_docs(args: argparse.Namespace) -> Path:
    rows = _load_rows()
    answers = _load_answer_key()
    if args.scope == "full":
        qids = ["FULL-CORPUS"]
        cases = [_select_full_corpus(rows)]
    else:
        qids = list(args.use_cases or PILOT_USE_CASES)
        cases = [
            _select_for_use_case(
                qid,
                rows,
                answers,
                positives_per_case=args.positives_per_case,
                distractors_per_case=args.distractors_per_case,
            )
            for qid in qids
        ]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(DATA_DIR),
        "scope": args.scope,
        "use_cases": qids,
        "cases": cases,
    }
    path = args.output_dir / "manifest.json"
    _json_dump(manifest, path)
    print(f"wrote manifest: {path}")
    for case in cases:
        docs = case["documents"]
        missing = [doc["path"] for doc in docs if not doc["exists"]]
        if case["question_id"] == "FULL-CORPUS":
            print(f"{case['question_id']}: {len(docs)} unique docs")
        else:
            print(
                f"{case['question_id']}: {len(docs)} docs "
                f"({case['selection']['positive_count']} positive, "
                f"{case['selection']['distractor_count']} distractor)"
            )
        if missing:
            raise FileNotFoundError(f"missing selected PDFs: {missing[:3]}")
    return path


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest at {path}; run --stage select-docs first")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_manifest_docs(manifest: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for case in manifest["cases"]:
        for doc in case["documents"]:
            yield case, doc


def _local_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - optional offline debug helper
        raise RuntimeError("install pypdf or use SDK document processing") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _document_row_for_skip(
    case: dict[str, Any],
    doc: dict[str, Any],
    *,
    error: str,
) -> dict[str, Any]:
    return {
        **doc,
        "question": case["question"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processor": "skipped",
        "job_id": None,
        "status": "SKIPPED",
        "elapsed_ms": 0.0,
        "char_count": 0,
        "quality": {},
        "markdown": "",
        "error": error,
    }


def _wait_for_pipeline_package(job: Any, args: argparse.Namespace, *, save_to_disk: Path | None) -> Any:
    """Wait for one submitted pipeline job, retrying transient status/result errors."""
    last_error: str | None = None
    for wait_attempt in range(1, args.wait_max_attempts + 1):
        try:
            return job.wait_for_completion(
                poll_interval=args.poll_interval,
                timeout=args.pipeline_timeout,
                save_to_disk=save_to_disk,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if wait_attempt >= args.wait_max_attempts:
                break
            time.sleep(args.wait_retry_backoff_seconds * wait_attempt)
    raise RuntimeError(
        f"job {job.id} did not return a package after "
        f"{args.wait_max_attempts} wait attempts: {last_error}"
    )


def _process_one_document(
    case: dict[str, Any],
    doc: dict[str, Any],
    args: argparse.Namespace,
    *,
    api_key: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(doc["path"])
    file_size_mb = path.stat().st_size / 1024 / 1024

    if args.max_file_mb is not None and file_size_mb > args.max_file_mb:
        return _document_row_for_skip(
            case,
            doc,
            error=(
                f"file_size_mb={file_size_mb:.2f} exceeds "
                f"--max-file-mb={args.max_file_mb:.2f}"
            ),
        )

    last_error: str | None = None
    last_phase = "unknown"
    last_job_id: str | None = None
    for attempt in range(1, args.max_attempts + 1):
        if args.local_extract:
            try:
                last_phase = "local_extract"
                markdown = _local_pdf_text(path)
                job_id = "local-pypdf"
                status = "LOCAL"
                quality = {}
                processor = "local_pypdf"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
        else:
            if not api_key:
                raise RuntimeError("set LATENCE_API_KEY before running SDK document processing")
            from latence import Latence

            client = Latence(
                api_key=api_key,
                base_url=os.environ.get("LATENCE_BASE_URL"),
                timeout=args.sdk_timeout,
            )
            try:
                last_phase = "submit"
                job = client.pipeline.run(
                    files=[path],
                    steps={"ocr": {"mode": args.ocr_mode}},
                    name=f"veracier-{doc['question_id']}-{doc['doc_id']}",
                    request_id=f"veracier-{doc['question_id']}-{doc['doc_id']}",
                )
                last_job_id = job.id
                last_phase = "wait"
                archive_path = (
                    args.output_dir / "archives" / f"{doc['source_id'].replace(':', '_')}.zip"
                ) if args.save_archives else None
                pkg = _wait_for_pipeline_package(job, args, save_to_disk=archive_path)
                markdown = pkg.document.markdown if pkg.document else ""
                job_id = pkg.id
                status = pkg.status
                quality = pkg.quality.model_dump(mode="python") if pkg.quality else {}
                processor = "latence_sdk_pipeline"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Once a job exists, do not submit duplicate jobs for the same
                # document. Surface the job id so it can be recovered/polled.
                if last_job_id:
                    break
                if attempt < args.max_attempts:
                    _progress_log(
                        args.output_dir / "progress.log",
                        "submit_retry",
                        source_id=doc.get("source_id"),
                        attempt=attempt,
                        max_attempts=args.max_attempts,
                        sleep_seconds=args.retry_backoff_seconds * attempt,
                        error=last_error[:120].replace(" ", "_"),
                    )
                    time.sleep(args.retry_backoff_seconds * attempt)
                    continue
                break
            finally:
                client.close()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            **doc,
            "question": case["question"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "processor": processor,
            "job_id": job_id,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 2),
            "char_count": len(markdown),
            "quality": quality,
            "markdown": markdown,
            "attempts": attempt,
            "phase": last_phase,
        }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        **doc,
        "question": case["question"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processor": "latence_sdk_pipeline" if not args.local_extract else "local_pypdf",
        "job_id": last_job_id,
        "status": "FAILED",
        "elapsed_ms": round(elapsed_ms, 2),
        "char_count": 0,
        "quality": {},
        "markdown": "",
        "attempts": attempt,
        "phase": last_phase,
        "error": last_error or "unknown processing failure",
    }


def process_docs(args: argparse.Namespace) -> Path:
    manifest = _load_manifest(args.output_dir)
    output_path = args.output_dir / "documents.jsonl"
    progress_path = args.output_dir / "progress.log"

    if args.force:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    existing_rows = _jsonl_read(output_path)
    checkpoint_statuses = {"COMPLETED", "LOCAL", "SKIPPED"}
    existing = {
        row["source_id"]: row
        for row in existing_rows
        if row.get("status") in checkpoint_statuses
    }
    dropped_rows = len(existing_rows) - len(existing)
    if dropped_rows:
        _jsonl_write(existing.values(), output_path)
    all_items = list(_iter_manifest_docs(manifest))
    pending = [
        (case, doc)
        for case, doc in all_items
        if args.force or doc["source_id"] not in existing
    ]

    api_key = None
    if not args.local_extract:
        api_key = os.environ.get("LATENCE_API_KEY") or os.environ.get("LATENCE_GATEWAY_KEY")
        if not api_key:
            raise RuntimeError("set LATENCE_API_KEY before running SDK document processing")

    total = len(all_items)
    already_done = len(all_items) - len(pending)
    counts = defaultdict(int)
    for row in existing.values():
        counts[row.get("status", "UNKNOWN")] += 1
    write_lock = Lock()

    _progress_log(
        progress_path,
        "start",
        total=total,
        already_done=already_done,
        pending=len(pending),
        dropped_failed_checkpoints=dropped_rows,
        concurrency=args.concurrency,
        output=output_path,
    )

    def record(row: dict[str, Any]) -> None:
        with write_lock:
            _jsonl_append(row, output_path)
            counts[row.get("status", "UNKNOWN")] += 1
            processed = sum(counts.values())
            remaining = max(total - processed, 0)
            _progress_log(
                progress_path,
                "document_done",
                processed=processed,
                total=total,
                remaining=remaining,
                completed=counts.get("COMPLETED", 0),
                failed=counts.get("FAILED", 0),
                skipped=counts.get("SKIPPED", 0),
                status=row.get("status"),
                source_id=row.get("source_id"),
                job_id=row.get("job_id"),
                chars=row.get("char_count", 0),
                elapsed_ms=row.get("elapsed_ms", 0),
                attempts=row.get("attempts", 1),
                phase=row.get("phase", "unknown"),
                error=(row.get("error") or "")[:120].replace(" ", "_"),
            )

    if not pending:
        _progress_log(progress_path, "complete", total=total, output=output_path)
        return output_path

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_process_one_document, case, doc, args, api_key=api_key): doc
            for case, doc in pending
        }
        for future in as_completed(futures):
            row = future.result()
            record(row)

    _progress_log(
        progress_path,
        "complete",
        total=total,
        completed=counts.get("COMPLETED", 0),
        failed=counts.get("FAILED", 0),
        skipped=counts.get("SKIPPED", 0),
        output=output_path,
    )
    return output_path


def _trim_text(text: str, max_chars: int) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return cleaned[:half] + "\n\n[...middle truncated...]\n\n" + cleaned[-half:]


def _build_generation_schema() -> dict[str, Any]:
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_text": {"type": "string"},
            "expected_label": {
                "type": "string",
                "enum": ["supported", "partially_supported", "ambiguous", "unsupported"],
            },
            "source_doc_ids": {"type": "array", "items": {"type": "string"}},
            "source_filenames": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "claim_text",
            "expected_label",
            "source_doc_ids",
            "source_filenames",
            "rationale",
        ],
    }
    variant_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mutation_type": {"type": "string", "enum": ["perfect", "ambiguous", "wrong"]},
            "expected_band": {"type": "string", "enum": ["green", "amber", "red"]},
            "expected_groundedness_range": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["min", "max"],
            },
            "response_text": {"type": "string"},
            "claims": {"type": "array", "items": claim_schema, "minItems": 3},
            "coverage_notes": {"type": "string"},
            "utilization_notes": {"type": "string"},
        },
        "required": [
            "mutation_type",
            "expected_band",
            "expected_groundedness_range",
            "response_text",
            "claims",
            "coverage_notes",
            "utilization_notes",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "use_case_id": {"type": "string"},
            "variants": {
                "type": "array",
                "items": variant_schema,
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["use_case_id", "variants"],
    }


def _evidence_pack(case: dict[str, Any], docs: list[dict[str, Any]], max_chars: int) -> dict[str, Any]:
    per_doc_chars = max(1_500, max_chars // max(1, len(docs)))
    return {
        "use_case_id": case["question_id"],
        "question": case["question"],
        "role": case.get("role"),
        "asker": case.get("asker"),
        "ground_truth": case.get("ground_truth", {}),
        "difficulty_factors": case.get("difficulty_factors", []),
        "selected_documents": [
            {
                "source_id": doc["source_id"],
                "filename": doc["filename"],
                "classification": doc["classification"],
                "language": doc["language"],
                "format": doc["format"],
                "description": doc["description"],
                "text": _trim_text(doc.get("markdown") or "", per_doc_chars),
            }
            for doc in docs
        ],
    }


def _generation_messages(pack: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You create labelled RAG answer variants for evaluating groundedness scoring. "
        "Use only the supplied synthetic enterprise documents. Do not use outside facts. "
        "Return strict JSON matching the schema. Every material factual sentence in each "
        "response must be represented as a claim with an expected label."
    )
    user = f"""
Create exactly three answer variants for this enterprise RAG question:

1. perfect: fully grounded in the provided documents, precise, no unsupported claims.
2. ambiguous: partly useful but incomplete or vague; include supported and ambiguous/partial claims.
3. wrong: plausible but materially unsupported or contradicted by the provided documents.

Rules:
- Preserve the language expected by the user question when practical.
- Prefer concrete document-specific claims over generic summaries.
- The perfect answer must not invent missing facts; say when evidence is absent.
- The wrong answer must still sound realistic, but its unsupported claims must be clearly labelled.
- Include source_doc_ids and source_filenames only where the claim is actually supported.
- Use expected_band green for perfect, amber for ambiguous, red for wrong.

Evidence pack:
{json.dumps(pack, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_generated(use_case_id: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("use_case_id") != use_case_id:
        errors.append(f"use_case_id mismatch: {payload.get('use_case_id')} != {use_case_id}")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        errors.append("expected exactly three variants")
        return errors
    seen = {variant.get("mutation_type") for variant in variants if isinstance(variant, dict)}
    if seen != {"perfect", "ambiguous", "wrong"}:
        errors.append(f"expected variant types perfect/ambiguous/wrong, got {sorted(seen)}")
    for variant in variants:
        if not isinstance(variant.get("response_text"), str) or len(variant["response_text"]) < 80:
            errors.append(f"{variant.get('mutation_type')}: response_text is too short")
        labels = {
            claim.get("expected_label")
            for claim in variant.get("claims", [])
            if isinstance(claim, dict)
        }
        if variant.get("mutation_type") == "perfect" and not labels <= {"supported"}:
            errors.append("perfect variant contains non-supported claim labels")
        if variant.get("mutation_type") == "wrong" and "unsupported" not in labels:
            errors.append("wrong variant lacks an unsupported claim")
    return errors


def generate_responses(args: argparse.Namespace) -> Path:
    manifest = _load_manifest(args.output_dir)
    docs = _jsonl_read(args.output_dir / "documents.jsonl")
    if not docs:
        raise FileNotFoundError("missing documents.jsonl; run --stage process-docs first")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY before generating labelled responses")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    docs_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        if doc.get("markdown"):
            docs_by_qid[doc["question_id"]].append(doc)

    rows_out: list[dict[str, Any]] = []
    schema = _build_generation_schema()
    for case in manifest["cases"]:
        qid = case["question_id"]
        if not docs_by_qid[qid]:
            raise ValueError(f"no processed markdown available for {qid}")
        pack = _evidence_pack(case, docs_by_qid[qid], args.max_prompt_context_chars)
        messages = _generation_messages(pack)
        prompt_path = args.output_dir / "prompts" / f"{qid}.json"
        _json_dump({"messages": messages, "schema": schema}, prompt_path)

        print(f"generating variants for {qid} with {args.openai_model}")
        completion = client.chat.completions.create(
            model=args.openai_model,
            messages=messages,
            temperature=args.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "veracier_trace_variants",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = completion.choices[0].message.content or "{}"
        payload = json.loads(content)
        errors = _validate_generated(qid, payload)
        if errors:
            raise ValueError(f"generated payload failed validation for {qid}: {errors}")

        raw_path = args.output_dir / "openai_raw" / f"{qid}.json"
        _json_dump(payload, raw_path)
        for variant in payload["variants"]:
            rows_out.append(
                {
                    "example_id": f"{qid}:{variant['mutation_type']}",
                    "question_id": qid,
                    "query_text": case["question"],
                    "mutation_type": variant["mutation_type"],
                    "expected_band": variant["expected_band"],
                    "expected_groundedness_range": variant["expected_groundedness_range"],
                    "response_text": variant["response_text"],
                    "claims": variant["claims"],
                    "coverage_notes": variant["coverage_notes"],
                    "utilization_notes": variant["utilization_notes"],
                    "context_doc_ids": [doc["source_id"] for doc in docs_by_qid[qid]],
                    "context_filenames": [doc["filename"] for doc in docs_by_qid[qid]],
                    "raw_context": "\n\n".join(
                        f"[{doc['source_id']} | {doc['classification']} | {doc['filename']}]\n"
                        f"{doc.get('markdown') or ''}"
                        for doc in docs_by_qid[qid]
                    ),
                }
            )

    output_path = args.output_dir / "variants.jsonl"
    _jsonl_write(rows_out, output_path)
    print(f"wrote variants: {output_path}")
    return output_path


def _model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python", exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}


def run_trace(args: argparse.Namespace) -> Path:
    variants = _jsonl_read(args.output_dir / "variants.jsonl")
    if not variants:
        raise FileNotFoundError("missing variants.jsonl; run --stage generate-responses first")

    api_key = os.environ.get("LATENCE_API_KEY") or os.environ.get("LATENCE_GATEWAY_KEY")
    if not api_key:
        raise RuntimeError("set LATENCE_API_KEY before running SDK Trace validation")

    from latence import Latence

    profiles = ["standard", "quality"] if args.trace_profile == "both" else [args.trace_profile]
    client = Latence(
        api_key=api_key,
        base_url=os.environ.get("LATENCE_BASE_URL"),
        timeout=args.sdk_timeout,
    )

    rows_out: list[dict[str, Any]] = []
    try:
        for variant in variants:
            for profile in profiles:
                print(f"trace {variant['example_id']} profile={profile}")
                started = time.perf_counter()
                response = client.experimental.trace.rag(
                    response_text=variant["response_text"],
                    query_text=variant["query_text"],
                    raw_context=variant["raw_context"],
                    primary_metric="triangular",
                    segmentation_mode="sentence_packed",
                    heatmap_format="none",
                    profile=profile,  # type: ignore[arg-type]
                    verbose=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                dumped = _model_dump(response)
                rows_out.append(
                    {
                        "example_id": variant["example_id"],
                        "question_id": variant["question_id"],
                        "mutation_type": variant["mutation_type"],
                        "expected_band": variant["expected_band"],
                        "expected_groundedness_range": variant["expected_groundedness_range"],
                        "profile": profile,
                        "wall_ms": round(elapsed_ms, 2),
                        "trace": {key: dumped.get(key) for key in TRACE_SCORE_FIELDS if key in dumped},
                    }
                )
    finally:
        client.close()

    output_path = args.output_dir / "trace_results.jsonl"
    _jsonl_write(rows_out, output_path)
    print(f"wrote trace results: {output_path}")
    return output_path


def _fmt_float(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else ""


def write_report(args: argparse.Namespace) -> Path:
    manifest = _load_manifest(args.output_dir)
    docs = _jsonl_read(args.output_dir / "documents.jsonl")
    variants = _jsonl_read(args.output_dir / "variants.jsonl")
    traces = _jsonl_read(args.output_dir / "trace_results.jsonl")

    lines = [
        "# Veracier Trace RAG Validation Report",
        "",
        f"- Created: {datetime.now(timezone.utc).isoformat()}",
        f"- Run directory: `{args.output_dir}`",
        f"- Scope: `{manifest.get('scope')}`",
        f"- Use cases: {', '.join(manifest.get('use_cases', []))}",
        "",
        "## Document Processing",
        "",
        f"- Documents processed: {len(docs)}",
    ]
    if docs:
        chars = [doc.get("char_count", 0) for doc in docs]
        elapsed = [doc.get("elapsed_ms", 0) for doc in docs]
        processors = sorted({doc.get("processor", "unknown") for doc in docs})
        statuses = defaultdict(int)
        for doc in docs:
            statuses[doc.get("status", "unknown")] += 1
        lines.extend(
            [
                f"- Processor(s): {', '.join(processors)}",
                "- Statuses: "
                + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())),
                f"- Mean chars/doc: {statistics.mean(chars):.0f}",
                f"- Mean processing wall ms/doc: {statistics.mean(elapsed):.0f}",
            ]
        )

    lines.extend(["", "## Generated Variants", "", f"- Variants: {len(variants)}"])
    if variants:
        by_type = defaultdict(int)
        for variant in variants:
            by_type[variant["mutation_type"]] += 1
        lines.append(
            "- By type: "
            + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items()))
        )

    lines.extend(["", "## Trace Results", ""])
    if traces:
        lines.append("| Example | Profile | Expected | Band | Score | NLI | Coverage | Usage | Wall ms |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in traces:
            trace = row.get("trace", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["example_id"],
                        row["profile"],
                        row["expected_band"],
                        str(trace.get("band", "")),
                        _fmt_float(trace.get("score")),
                        _fmt_float(trace.get("nli_aggregate")),
                        _fmt_float(trace.get("context_coverage_ratio")),
                        _fmt_float(trace.get("context_usage_ratio")),
                        _fmt_float(row.get("wall_ms")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- Trace not run yet.")

    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["select-docs", "process-docs", "generate-responses", "trace", "report", "all"],
        default="all",
    )
    parser.add_argument("--scope", choices=["pilot", "capped", "full"], default="pilot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--use-cases", nargs="*", default=list(PILOT_USE_CASES))
    parser.add_argument("--positives-per-case", type=int, default=4)
    parser.add_argument("--distractors-per-case", type=int, default=2)
    parser.add_argument("--ocr-mode", default="performance")
    parser.add_argument("--pipeline-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--sdk-timeout", type=float, default=180.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent document-processing requests.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum processing attempts per document before marking it failed.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Linear retry backoff base for transient document-processing failures.",
    )
    parser.add_argument(
        "--wait-max-attempts",
        type=int,
        default=6,
        help="Retries for status/result polling after a pipeline job has been submitted.",
    )
    parser.add_argument(
        "--wait-retry-backoff-seconds",
        type=float,
        default=10.0,
        help="Linear retry backoff base for status/result polling failures.",
    )
    parser.add_argument("--save-archives", action="store_true")
    parser.add_argument("--local-extract", action="store_true")
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=None,
        help="Skip files larger than this size; useful when presigned uploads are unavailable.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-prompt-context-chars", type=int, default=36_000)
    parser.add_argument("--trace-profile", choices=["standard", "quality", "both"], default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages = (
        ["select-docs", "process-docs", "generate-responses", "trace", "report"]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "select-docs":
            select_docs(args)
        elif stage == "process-docs":
            process_docs(args)
        elif stage == "generate-responses":
            generate_responses(args)
        elif stage == "trace":
            run_trace(args)
        elif stage == "report":
            write_report(args)
        else:  # pragma: no cover
            raise AssertionError(stage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
