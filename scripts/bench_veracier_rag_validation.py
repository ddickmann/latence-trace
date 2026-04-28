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
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_SDK_SRC = _REPO.parent / "latence-python" / "src"
if _SDK_SRC.exists() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

DATA_DIR = _REPO / "data" / "veracier-industries"
DEFAULT_PROCESSED_DOCUMENTS = DATA_DIR / "trace_bench_runs" / "full" / "documents.jsonl"
DEFAULT_OUT = DATA_DIR / "trace_bench_runs" / "veracier_trace_validation_proof"
PILOT_USE_CASES = ("CEO-01", "LEGAL-01", "CISO-02")
SUCCESS_DOCUMENT_STATUS = "COMPLETED"

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
LOW_PRIORITY_GT_KEYS = {"all_review", "review", "all", "background", "reference"}
EVIDENCE_KEYWORDS = {
    "CEO-01": [
        "sanctions",
        "Severneft",
        "RosNuclear",
        "Sapin",
        "KYC",
        "change of control",
        "changement de contrôle",
    ],
    "LEGAL-01": [
        "force majeure",
        "chaîne d'approvisionnement",
        "chaine d'approvisionnement",
        "supply chain",
        "approvisionnement",
        "interruption",
    ],
    "CISO-02": [
        "NIS2",
        "systèmes essentiels",
        "systemes essentiels",
        "lacunes",
        "gap",
        "segmentation",
        "incident",
    ],
}

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


@dataclass(frozen=True)
class CaseArchetype:
    archetype_id: str
    label: str
    focus: str
    perfect_guidance: tuple[str, ...]
    ambiguous_guidance: tuple[str, ...]
    wrong_guidance: tuple[str, ...]
    keywords: tuple[str, ...]

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "id": self.archetype_id,
            "label": self.label,
            "focus": self.focus,
            "perfect_guidance": list(self.perfect_guidance),
            "ambiguous_guidance": list(self.ambiguous_guidance),
            "wrong_guidance": list(self.wrong_guidance),
        }


ARCHETYPES: dict[str, CaseArchetype] = {
    "finance_tax": CaseArchetype(
        archetype_id="finance_tax",
        label="Finance, tax, transfer pricing, accounting",
        focus="amounts, periods, entities, filings, assumptions, percentages, and provisions",
        perfect_guidance=(
            "Use affirmative fiscal or financial facts that are explicitly present.",
            "Numeric values, years, percentages, and currency amounts are allowed only when copied verbatim.",
            "Do not answer with an evidence-absence sentence such as missing years or missing files.",
        ),
        ambiguous_guidance=(
            "Use one supported metric, transaction type, filing, or accounting fact.",
            "Pair it with one unresolved limitation about period coverage, entity coverage, filing scope, or supporting schedule completeness.",
        ),
        wrong_guidance=(
            "Mutate an amount, rate, year, entity, filing status, or tax/accounting conclusion.",
            "Do not keep any claim partially grounded.",
        ),
        keywords=(
            "finance",
            "fiscal",
            "tax",
            "prix de transfert",
            "transfer pricing",
            "provision",
            "audit",
            "cash",
            "budget",
            "redevance",
            "royalty",
        ),
    ),
    "legal_litigation": CaseArchetype(
        archetype_id="legal_litigation",
        label="Litigation and legal proceedings",
        focus="claims, case status, parties, exposure, provisions, deadlines, and procedural posture",
        perfect_guidance=(
            "Use concrete case facts that are directly stated: party, dispute, status, amount, deadline, or risk.",
            "Avoid saying that a consolidated view is missing; perfect must be an affirmative supported answer.",
        ),
        ambiguous_guidance=(
            "Use one supported case fact.",
            "Pair it with one unresolved limitation about group-wide completeness, threatened claims, contagion, or amount coverage.",
        ),
        wrong_guidance=(
            "Invent or invert case status, court outcome, exposure amount, settlement, or affected entity.",
            "Unsupported litigation conclusions should not cite source documents.",
        ),
        keywords=(
            "litigation",
            "contentieux",
            "claim",
            "dispute",
            "provision",
            "court",
            "tribunal",
            "settlement",
            "registre",
        ),
    ),
    "legal_contracts": CaseArchetype(
        archetype_id="legal_contracts",
        label="Contracts, clauses, obligations, and regulatory legal review",
        focus="parties, clauses, obligations, rights, scope, exceptions, dates, and governing restrictions",
        perfect_guidance=(
            "Use clause-level obligations or restrictions explicitly stated in the contract evidence.",
            "Keep the answer about what the clause says, not a broader legal conclusion unless the text states it.",
        ),
        ambiguous_guidance=(
            "Use one supported clause or obligation.",
            "Pair it with one unresolved limitation about applicability, exception coverage, counterparty scope, or missing schedule.",
        ),
        wrong_guidance=(
            "Invert a clause, assign it to the wrong party, change the scope, or invent consent/termination rights.",
            "Do not include any source ids for unsupported legal claims.",
        ),
        keywords=(
            "contract",
            "contrat",
            "clause",
            "obligation",
            "termination",
            "force majeure",
            "indemnisation",
            "licence",
            "bail",
        ),
    ),
    "cyber_security": CaseArchetype(
        archetype_id="cyber_security",
        label="Cybersecurity, NIS2, systems, and controls",
        focus="system scope, controls, gaps, incidents, remediation, audits, RTO/RPO, and data protection",
        perfect_guidance=(
            "Use stated controls, systems, gaps, audit findings, or remediation facts.",
            "Do not promote a gap into full compliance or a plan into completed remediation.",
        ),
        ambiguous_guidance=(
            "Use one supported system, control, or gap fact.",
            "Pair it with one unresolved limitation about asset coverage, remediation completion, control effectiveness, or NIS2 scope.",
        ),
        wrong_guidance=(
            "Invent compliance, remediation completion, incident closure, or system scope.",
            "Unsupported control conclusions should have no source ids.",
        ),
        keywords=(
            "nis2",
            "cyber",
            "security",
            "rto",
            "rpo",
            "incident",
            "segmentation",
            "audit",
            "rgpd",
            "gdpr",
            "pca",
        ),
    ),
    "technical_architecture": CaseArchetype(
        archetype_id="technical_architecture",
        label="Technology architecture, migration, and dependencies",
        focus="platforms, dependencies, versions, migration status, capacity, incidents, and technical constraints",
        perfect_guidance=(
            "Use concrete technical facts explicitly present in the evidence.",
            "Separate planned work from completed work.",
        ),
        ambiguous_guidance=(
            "Use one supported architecture or migration fact.",
            "Pair it with one unresolved limitation about rollout scope, dependency status, capacity, or completion.",
        ),
        wrong_guidance=(
            "Mutate version, system, completion status, dependency, outage, or capacity.",
            "Do not cite source documents for invented technical facts.",
        ),
        keywords=(
            "architecture",
            "migration",
            "system",
            "plateforme",
            "platform",
            "version",
            "dependency",
            "iiot",
            "cloud",
        ),
    ),
    "quality_audit": CaseArchetype(
        archetype_id="quality_audit",
        label="Quality, batch record, audit, and certification",
        focus="lot IDs, certificates, checklist status, test results, specifications, non-conformities, and missing artifacts",
        perfect_guidance=(
            "Use exact lot, certificate, checklist, conformance, or missing-artifact facts.",
            "A missing or incomplete status is valid only when the evidence explicitly says ABSENT, INCOMPLET, missing, or equivalent.",
        ),
        ambiguous_guidance=(
            "Use one supported artifact, test, checklist, or lot fact.",
            "Pair it with one unresolved limitation about audit completeness, missing attachments, traceability, or certification coverage.",
        ),
        wrong_guidance=(
            "Mutate lot number, conformance status, missing artifact, test result, supplier, or certificate.",
            "Do not leave any unsupported quality claim partially grounded.",
        ),
        keywords=(
            "quality",
            "qualite",
            "lot",
            "batch",
            "checklist",
            "certificat",
            "certificate",
            "audit",
            "conforme",
            "incomplet",
            "absent",
            "easa",
        ),
    ),
    "procurement_supplier": CaseArchetype(
        archetype_id="procurement_supplier",
        label="Procurement, supplier risk, delivery, and solvency",
        focus="supplier status, delivery risk, credits, receivables, contracts, guarantees, and exposure",
        perfect_guidance=(
            "Use supplier, contract, delivery, exposure, or credit facts that are explicitly stated.",
            "Do not infer liquidation, full recovery, or offsets unless the evidence states them.",
        ),
        ambiguous_guidance=(
            "Use one supported supplier or exposure fact.",
            "Pair it with one unresolved limitation about recovery, delivery continuity, guarantee coverage, or insolvency status.",
        ),
        wrong_guidance=(
            "Invent liquidation, guarantees, automatic offsets, delivery continuity, or total irrecoverability.",
            "Unsupported supplier-risk claims should have no source ids.",
        ),
        keywords=(
            "procurement",
            "fournisseur",
            "supplier",
            "delivery",
            "livraison",
            "creance",
            "receivable",
            "solvency",
            "garantie",
        ),
    ),
    "sales_export": CaseArchetype(
        archetype_id="sales_export",
        label="Sales, export control, sanctions, and customer clearance",
        focus="customers, countries, end-use, shipment status, sanctions screening, licenses, and clearance",
        perfect_guidance=(
            "Use explicit customer, destination, end-use, screening, license, or shipment facts.",
            "Do not state clearance, shipment, or approval unless the evidence states it.",
        ),
        ambiguous_guidance=(
            "Use one supported customer, order, destination, or screening fact.",
            "Pair it with one unresolved limitation about end-use, license status, sanctions clearance, or shipment authorization.",
        ),
        wrong_guidance=(
            "Invent clearance, approval, shipment, customer identity, destination, or end-use.",
            "Do not attach source ids to unsupported export-control claims.",
        ),
        keywords=(
            "sales",
            "vente",
            "export",
            "sanctions",
            "customer",
            "client",
            "shipment",
            "license",
            "licence",
            "end-use",
        ),
    ),
    "operations_capacity": CaseArchetype(
        archetype_id="operations_capacity",
        label="Operations, production, maintenance, capacity, and planning",
        focus="sites, capacity, maintenance, production constraints, capex, schedules, quantities, and bottlenecks",
        perfect_guidance=(
            "Use explicit operational facts: site, capacity, budget, schedule, maintenance, or production constraint.",
            "Do not add missing site-capacity conclusions unless those are directly stated.",
        ),
        ambiguous_guidance=(
            "Use one supported operations or capex fact.",
            "Pair it with one unresolved limitation about site split, timing, realized capacity, or production coverage.",
        ),
        wrong_guidance=(
            "Mutate site, capacity, amount, schedule, maintenance status, or production conclusion.",
            "Unsupported operations claims should not cite source documents.",
        ),
        keywords=(
            "operations",
            "production",
            "maintenance",
            "capacity",
            "capacite",
            "capex",
            "budget",
            "site",
            "toulouse",
            "casablanca",
        ),
    ),
    "hr_employment": CaseArchetype(
        archetype_id="hr_employment",
        label="HR, employment, workforce, and labor compliance",
        focus="employees, contracts, consultation, dates, populations, obligations, disputes, and compliance status",
        perfect_guidance=(
            "Use explicit employment, consultation, workforce, date, or obligation facts.",
            "Do not infer affected population or compliance completion unless the text states it.",
        ),
        ambiguous_guidance=(
            "Use one supported HR or employment fact.",
            "Pair it with one unresolved limitation about population scope, consultation status, date coverage, or missing file.",
        ),
        wrong_guidance=(
            "Invent employee counts, termination status, consultation completion, dispute outcome, or compliance status.",
            "Unsupported HR claims should have no source ids.",
        ),
        keywords=(
            "hr",
            "employment",
            "employee",
            "salaries",
            "travail",
            "consultation",
            "workforce",
            "labor",
        ),
    ),
    "executive_contract_risk": CaseArchetype(
        archetype_id="executive_contract_risk",
        label="Executive contract, sanctions, and compliance risk map",
        focus="inherited contracts, counterparties, sanctions, anti-corruption, change of control, and integration risk",
        perfect_guidance=(
            "Use explicit counterparty, contract, sanctions, compliance, or change-of-control facts.",
            "Do not turn a partial risk map into a complete inherited-contract assessment.",
        ),
        ambiguous_guidance=(
            "Use one supported counterparty or contract-risk fact.",
            "Pair it with one unresolved limitation about full perimeter, KYC, anti-corruption review, or inherited contract coverage.",
        ),
        wrong_guidance=(
            "Invent sanctions status, counterparty identity, KYC completion, approval, or contract rights.",
            "Unsupported executive risk claims should have no source ids.",
        ),
        keywords=(
            "sanctions",
            "contract",
            "contrat",
            "counterparty",
            "client",
            "fournisseur",
            "kyc",
            "sapin",
            "change of control",
            "anti-corruption",
        ),
    ),
    "default_enterprise": CaseArchetype(
        archetype_id="default_enterprise",
        label="General enterprise evidence review",
        focus="explicit document facts, stated limitations, and unsupported business conclusions",
        perfect_guidance=(
            "Use only affirmative facts directly stated in evidence.",
            "Do not answer perfect with a generic missing-evidence caveat.",
        ),
        ambiguous_guidance=(
            "Use one supported fact and one unresolved limitation tied to the user question.",
            "The unresolved limitation must not contradict the evidence.",
        ),
        wrong_guidance=(
            "Make both claims materially unsupported.",
            "Do not include source ids for unsupported claims.",
        ),
        keywords=(),
    ),
}


ARCHETYPE_BY_PREFIX = {
    "CISO": "cyber_security",
    "CTO": "technical_architecture",
    "FIN": "finance_tax",
    "HR": "hr_employment",
    "LEGAL": "legal_contracts",
    "OPS": "operations_capacity",
    "PROC": "procurement_supplier",
    "QUAL": "quality_audit",
    "SALES": "sales_export",
}

ARCHETYPE_BY_QID = {
    "CEO-01": "executive_contract_risk",
    "CEO-02": "legal_litigation",
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


def _doc_key(entity: str, filename: str) -> str:
    return f"{entity}/{filename}"


def _row_key(row: DatasetRow | dict[str, Any]) -> str:
    if isinstance(row, dict):
        return _doc_key(str(row["entity"]), str(row["filename"]))
    return _doc_key(row.entity, row.filename)


def _filename_aliases(filename: str) -> set[str]:
    return {filename, Path(filename).name}


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


def _ground_truth_files(
    answer: dict[str, Any],
    *,
    include_negative: bool,
    include_low_priority: bool,
) -> set[str]:
    flattened: set[str] = set()
    for key, values in (answer.get("ground_truth") or {}).items():
        normalized_key = str(key).lower()
        is_negative = normalized_key in NEGATIVE_GT_KEYS
        if is_negative != include_negative:
            continue
        if not include_low_priority and normalized_key in LOW_PRIORITY_GT_KEYS:
            continue
        if isinstance(values, list):
            for value in values:
                flattened.add(str(value))
                flattened.add(Path(str(value)).name)
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
        doc_hash = hashlib.sha1(f"{row.entity}/{row.filename}".encode()).hexdigest()[:12]
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


def _master_docs(rows: list[DatasetRow]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        entry = docs.setdefault(
            key,
            {
                "key": key,
                "entity": row.entity,
                "filename": row.filename,
                "path": str(row.path),
                "exists": row.path.exists(),
                "question_ids": [],
                "roles": [],
                "classifications": [],
                "languages": [],
                "formats": [],
                "descriptions": [],
            },
        )
        entry["question_ids"].append(row.question_id)
        entry["roles"].append(row.role)
        entry["classifications"].append(row.classification)
        entry["languages"].append(row.language)
        entry["formats"].append(row.fmt)
        if row.description and row.description not in entry["descriptions"]:
            entry["descriptions"].append(row.description)

    for entry in docs.values():
        for field in ("question_ids", "roles", "classifications", "languages", "formats"):
            entry[field] = sorted(set(entry[field]))
    return docs


def _filtered_corpus_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    master = _master_docs(_load_rows())
    processed_rows = _jsonl_read(args.processed_documents)
    processed_keys = {_row_key(row) for row in processed_rows if row.get("entity") and row.get("filename")}

    corpus_by_key: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)

    for row in processed_rows:
        key = _row_key(row)
        status = row.get("status") or "UNKNOWN"
        status_counts[status] += 1
        markdown = row.get("markdown") or ""
        if status == SUCCESS_DOCUMENT_STATUS and markdown:
            corpus_by_key[key] = {
                **row,
                "key": key,
                "master": master.get(key, {}),
            }
            continue
        excluded.append(
            {
                "key": key,
                "entity": row.get("entity"),
                "filename": row.get("filename"),
                "source_id": row.get("source_id"),
                "status": status,
                "reason": "not_completed" if status != SUCCESS_DOCUMENT_STATUS else "missing_markdown",
                "error": row.get("error"),
                "char_count": row.get("char_count", 0),
            }
        )

    for key in sorted(set(master) - processed_keys):
        entry = master[key]
        excluded.append(
            {
                "key": key,
                "entity": entry["entity"],
                "filename": entry["filename"],
                "source_id": None,
                "status": "MISSING",
                "reason": "missing_from_processed_documents",
                "error": None,
                "char_count": 0,
            }
        )

    corpus = [corpus_by_key[key] for key in sorted(corpus_by_key)]
    char_counts = [row.get("char_count", 0) for row in corpus]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processed_documents": str(args.processed_documents),
        "master_unique_documents": len(master),
        "processed_rows": len(processed_rows),
        "completed_with_markdown": len(corpus),
        "excluded_documents": len(excluded),
        "status_counts": dict(sorted(status_counts.items())),
        "total_completed_chars": sum(char_counts),
        "mean_completed_chars": round(statistics.mean(char_counts), 2) if char_counts else 0,
        "min_completed_chars": min(char_counts) if char_counts else 0,
        "max_completed_chars": max(char_counts) if char_counts else 0,
    }
    return corpus, excluded, summary


def filter_corpus(args: argparse.Namespace) -> Path:
    corpus, excluded, summary = _filtered_corpus_rows(args)
    corpus_path = args.output_dir / "corpus.jsonl"
    excluded_path = args.output_dir / "excluded_documents.jsonl"
    summary_path = args.output_dir / "corpus_summary.json"
    _jsonl_write(corpus, corpus_path)
    _jsonl_write(excluded, excluded_path)
    _json_dump(summary, summary_path)
    print(
        "filtered corpus: "
        f"{summary['completed_with_markdown']} completed, "
        f"{summary['excluded_documents']} excluded"
    )
    return corpus_path


def _load_filtered_corpus(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    corpus_path = args.output_dir / "corpus.jsonl"
    if not corpus_path.exists():
        filter_corpus(args)
    return {_row_key(row): row for row in _jsonl_read(corpus_path)}


def _select_evidence_rows(
    qid: str,
    rows: list[DatasetRow],
    answers: dict[str, Any],
    *,
    positives_per_case: int,
    distractors_per_case: int,
) -> list[DatasetRow]:
    answer = answers[qid]
    case_rows = [row for row in rows if row.question_id == qid]
    high_priority_gt_files = _ground_truth_files(
        answer,
        include_negative=False,
        include_low_priority=False,
    )
    positive_gt_files = _ground_truth_files(
        answer,
        include_negative=False,
        include_low_priority=True,
    )
    negative_gt_files = _ground_truth_files(
        answer,
        include_negative=True,
        include_low_priority=True,
    )
    primary_classes = PRIMARY_CLASSES.get(qid)
    distractor_classes = DISTRACTOR_CLASSES.get(qid, {"NO", "REFERENCE", "TRAP"})

    def matches_any(filename: str, candidates: set[str]) -> bool:
        return bool(_filename_aliases(filename) & candidates)

    exact_positives = [row for row in case_rows if matches_any(row.filename, high_priority_gt_files)]
    if positive_gt_files:
        positives = [
            row
            for row in case_rows
            if matches_any(row.filename, positive_gt_files)
            or row.classification in (primary_classes or set())
        ]
    elif primary_classes:
        positives = [row for row in case_rows if row.classification in primary_classes]
    else:
        generic_distractors = distractor_classes | {"NO", "REFERENCE", "TRAP", "NOT_RELEVANT"}
        positives = [row for row in case_rows if row.classification not in generic_distractors]
        if not positives:
            positives = case_rows

    exact_ids = {row.source_id for row in exact_positives}
    positives = exact_positives + [row for row in positives if row.source_id not in exact_ids]
    positive_ids = {row.source_id for row in positives}
    distractors = [
        row
        for row in case_rows
        if row.source_id not in positive_ids
        and (
            row.classification in distractor_classes
            or matches_any(row.filename, negative_gt_files)
        )
    ]
    selected = positives[:positives_per_case]
    if len(selected) < positives_per_case:
        selected_ids = {row.source_id for row in selected}
        selected.extend(
            row
            for row in _round_robin_by_class(
                [row for row in positives if row.source_id not in selected_ids],
                positives_per_case - len(selected),
            )
        )
    selected_ids = {row.source_id for row in selected}
    selected.extend(
        row
        for row in _round_robin_by_class(distractors, distractors_per_case)
        if row.source_id not in selected_ids
    )

    if len(selected) < positives_per_case + distractors_per_case:
        selected_ids = {row.source_id for row in selected}
        fill = [row for row in case_rows if row.source_id not in selected_ids]
        selected.extend(_round_robin_by_class(fill, positives_per_case + distractors_per_case - len(selected)))
    return selected


def _pack_context_documents(
    selected_rows: list[DatasetRow],
    corpus_by_key: dict[str, dict[str, Any]],
    max_chars: int,
    keywords: list[str],
) -> list[dict[str, Any]]:
    per_doc_chars = max(1_500, max_chars // max(1, len(selected_rows)))
    docs: list[dict[str, Any]] = []
    for row in selected_rows:
        corpus_doc = corpus_by_key[_row_key(row)]
        doc_keywords = [
            *keywords,
            row.classification.replace("_", " "),
            row.description,
            Path(row.filename).stem.replace("_", " "),
        ]
        docs.append(
            {
                "source_id": row.source_id,
                "corpus_source_id": corpus_doc.get("source_id"),
                "doc_id": row.doc_id,
                "entity": row.entity,
                "filename": row.filename,
                "classification": row.classification,
                "language": row.language,
                "format": row.fmt,
                "description": row.description,
                "char_count": corpus_doc.get("char_count", 0),
                "text": _keyword_excerpt(corpus_doc.get("markdown") or "", per_doc_chars, doc_keywords),
            }
        )
    return docs


def _raw_context_from_pack(pack: dict[str, Any]) -> str:
    return "\n\n".join(
        f"[{doc['source_id']} | {doc['classification']} | {doc['filename']}]\n{doc['text']}"
        for doc in pack["selected_documents"]
    )


def build_evidence_packs(args: argparse.Namespace) -> Path:
    rows = _load_rows()
    answers = _load_answer_key()
    corpus_by_key = _load_filtered_corpus(args)
    available_rows = [row for row in rows if _row_key(row) in corpus_by_key]
    qids = sorted(answers) if args.scope == "full-use-cases" else list(args.use_cases or PILOT_USE_CASES)
    pack_dir = args.output_dir / "evidence_packs"
    pack_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []
    for qid in qids:
        selected_rows = _select_evidence_rows(
            qid,
            available_rows,
            answers,
            positives_per_case=args.positives_per_case,
            distractors_per_case=args.distractors_per_case,
        )
        if not selected_rows:
            raise ValueError(f"no available processed documents selected for {qid}")
        answer = answers[qid]
        archetype_payload = _infer_case_archetype(
            {
                "use_case_id": qid,
                "question": answer["question"],
                "role": answer.get("role"),
                "entity": answer.get("entity"),
                "difficulty_factors": answer.get("difficulty_factors", []),
                "selected_documents": [row.to_manifest() for row in selected_rows],
            }
        )
        archetype = ARCHETYPES[str(archetype_payload["id"])]
        keywords = [
            *EVIDENCE_KEYWORDS.get(qid, []),
            *archetype.keywords,
            *[
                token
                for token in re.split(r"\W+", answer["question"])
                if len(token) >= 5
            ],
            *[key.replace("_", " ") for key in (answer.get("ground_truth") or {})],
        ]
        docs = _pack_context_documents(
            selected_rows,
            corpus_by_key,
            args.max_prompt_context_chars,
            keywords,
        )
        pack = {
            "use_case_id": qid,
            "question": answer["question"],
            "role": answer.get("role"),
            "asker": answer.get("asker"),
            "entity": answer.get("entity"),
            "difficulty_factors": answer.get("difficulty_factors", []),
            "ground_truth": answer.get("ground_truth", {}),
            "ground_truth_proxy": {
                "source": "MASTER_INDEX.classification",
                "positive_classes": sorted(PRIMARY_CLASSES.get(qid, set())),
                "distractor_classes": sorted(DISTRACTOR_CLASSES.get(qid, set())),
            },
            "selected_documents": docs,
            "raw_context": _raw_context_from_pack({"selected_documents": docs}),
            "context_char_count": sum(len(doc["text"]) for doc in docs),
        }
        pack["generation_archetype"] = archetype_payload
        _json_dump(pack, pack_dir / f"{qid}.json")
        manifest_cases.append(
            {
                "use_case_id": qid,
                "question": answer["question"],
                "generation_archetype": pack["generation_archetype"],
                "document_count": len(docs),
                "context_char_count": pack["context_char_count"],
                "documents": [
                    {
                        "source_id": doc["source_id"],
                        "filename": doc["filename"],
                        "classification": doc["classification"],
                        "char_count": doc["char_count"],
                    }
                    for doc in docs
                ],
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "use_cases": qids,
        "evidence_pack_dir": str(pack_dir),
        "cases": manifest_cases,
    }
    manifest_path = args.output_dir / "evidence_manifest.json"
    _json_dump(manifest, manifest_path)
    print(f"wrote evidence packs: {pack_dir} ({len(manifest_cases)} use cases)")
    return manifest_path


def _load_evidence_packs(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = args.output_dir / "evidence_manifest.json"
    if not manifest_path.exists():
        build_evidence_packs(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        json.loads((args.output_dir / "evidence_packs" / f"{qid}.json").read_text(encoding="utf-8"))
        for qid in manifest["use_cases"]
    ]


def _trim_text(text: str, max_chars: int) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    if len(cleaned) <= max_chars:
        return cleaned
    half = max_chars // 2
    return cleaned[:half] + "\n\n[...middle truncated...]\n\n" + cleaned[-half:]


def _keyword_excerpt(text: str, max_chars: int, keywords: list[str]) -> str:
    # The previous implementation stripped numbers and markdown to avoid
    # tripping the Trace structured-source detector on prose. That
    # workaround is no longer necessary: the scorer now treats RAG prose
    # as narrative (prose-safety guard) so we pass raw markdown through
    # and let the real service see the real evidence.
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    lowered = cleaned.lower()
    positions: list[int] = []
    for keyword in keywords:
        needle = keyword.strip().lower()
        if not needle:
            continue
        start = 0
        while True:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + max(1, len(needle))
            if len(positions) >= 12:
                break

    if not positions:
        return _trim_text(cleaned, max_chars)

    positions = sorted(set(positions))
    window_count = min(4, len(positions))
    window_chars = max(900, max_chars // window_count)
    excerpts: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for idx in positions[:window_count]:
        start = max(0, idx - window_chars // 2)
        end = min(len(cleaned), idx + window_chars // 2)
        if any(not (end < used_start or start > used_end) for used_start, used_end in used_ranges):
            continue
        used_ranges.append((start, end))
        excerpts.append(cleaned[start:end].strip())

    if not excerpts:
        return _trim_text(cleaned, max_chars)
    return "\n\n[...evidence excerpt...]\n\n".join(excerpts)[:max_chars]


def _archetype_signal(pack_or_docs: dict[str, Any] | list[dict[str, Any]]) -> str:
    if isinstance(pack_or_docs, dict):
        docs = pack_or_docs.get("selected_documents") or []
        fields = [
            pack_or_docs.get("use_case_id"),
            pack_or_docs.get("question"),
            pack_or_docs.get("role"),
            pack_or_docs.get("entity"),
            " ".join(str(value) for value in pack_or_docs.get("difficulty_factors", [])),
        ]
    else:
        docs = pack_or_docs
        fields = []

    for doc in docs:
        fields.extend(
            [
                doc.get("filename"),
                doc.get("classification"),
                doc.get("description"),
                doc.get("language"),
                doc.get("format"),
            ]
        )
    return " ".join(str(field or "") for field in fields).lower()


def _infer_case_archetype(pack: dict[str, Any]) -> dict[str, Any]:
    qid = str(pack.get("use_case_id") or "")
    if qid in ARCHETYPE_BY_QID:
        return ARCHETYPES[ARCHETYPE_BY_QID[qid]].to_prompt_payload()

    prefix = qid.split("-", 1)[0]
    if prefix in ARCHETYPE_BY_PREFIX:
        return ARCHETYPES[ARCHETYPE_BY_PREFIX[prefix]].to_prompt_payload()

    signal = _archetype_signal(pack)
    if "contentieux" in signal or "litigation" in signal or "dispute" in signal:
        return ARCHETYPES["legal_litigation"].to_prompt_payload()
    if "qualite" in signal or "quality" in signal or "batch" in signal or "lot" in signal:
        return ARCHETYPES["quality_audit"].to_prompt_payload()
    if "prix_transfert" in signal or "transfer pricing" in signal or "fiscal" in signal:
        return ARCHETYPES["finance_tax"].to_prompt_payload()
    if "nis2" in signal or "cyber" in signal or "incident" in signal:
        return ARCHETYPES["cyber_security"].to_prompt_payload()
    if "contrat" in signal or "contract" in signal or "clause" in signal:
        return ARCHETYPES["legal_contracts"].to_prompt_payload()

    return ARCHETYPES["default_enterprise"].to_prompt_payload()


def _archetype_from_pack(pack: dict[str, Any]) -> CaseArchetype:
    payload = pack.get("generation_archetype")
    archetype_id = payload.get("id") if isinstance(payload, dict) else None
    if isinstance(archetype_id, str) and archetype_id in ARCHETYPES:
        return ARCHETYPES[archetype_id]
    inferred = _infer_case_archetype(pack)
    return ARCHETYPES[str(inferred["id"])]


def _archetype_prompt_block(archetype: CaseArchetype) -> str:
    def bullets(values: tuple[str, ...]) -> str:
        return "\n".join(f"  - {value}" for value in values)

    return f"""
Generation archetype: {archetype.label}
Focus: {archetype.focus}
Perfect guidance:
{bullets(archetype.perfect_guidance)}
Ambiguous guidance:
{bullets(archetype.ambiguous_guidance)}
Wrong guidance:
{bullets(archetype.wrong_guidance)}
""".strip()


_QUOTE_RE = re.compile(r'"([^"]{8,260})"')
_UNRESOLVED_ABSENCE_CUES = (
    "documents do not",
    "document does not",
    "does not specify",
    "do not specify",
    "does not show",
    "do not show",
    "no information",
    "not established",
    "not clear",
    "visible records",
    "records do not",
    "ne precise pas",
    "ne précisent pas",
    "ne precisent pas",
    "ne permet pas",
    "ne permettent pas",
    "aucune information",
    "pas clair",
    "pas établi",
    "pas etabli",
    "nicht klar",
    "nicht belegt",
    "nicht ersichtlich",
)
_EXPLICIT_MISSING_MARKERS = (
    "absent",
    "missing",
    "manquant",
    "incomplet",
    "incomplete",
    "nicht vorhanden",
)
# The core scorer now owns the canonical hedge-cue list so the benchmark
# and the epistemic-hedge gate stay in sync. The benchmark keeps a thin
# superset for French accent/apostrophe variants that the OpenAI generator
# sometimes emits but that do not need to reach the scorer.
try:
    from latence_trace.core.groundedness import (
        EPISTEMIC_HEDGE_CUES as _CORE_HEDGE_CUES,
    )
except Exception:  # pragma: no cover - benchmark is a CLI, core may be absent
    _CORE_HEDGE_CUES = ()

_AMBIGUITY_CUES = tuple(
    sorted(
        {
            *_CORE_HEDGE_CUES,
            "unclear",
            "not established",
            "does not show",
            "not clear",
            "insufficient",
            "missing",
            "ne permet pas",
            "pas clair",
            "n'établit pas",
            "n’etablit pas",
            "n'est pas établi",
            "n'est pas établie",
            "n'est pas etablie",
            "n’est pas établi",
            "n’est pas établie",
            "n’est pas etablie",
            "insuffisant",
            "incertain",
            "incertitude",
            "manquant",
            "non clarifié",
            "non clarifie",
            "nicht klar",
            "nicht belegt",
            "unvollständig",
        }
    )
)


def _normalize_validation_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quoted_phrases(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTE_RE.finditer(text or "")]


def _quote_in_context(quote: str, context: str) -> bool:
    normalized_quote = _normalize_validation_text(quote)
    normalized_context = _normalize_validation_text(context)
    return normalized_quote in normalized_context


def _has_unresolved_absence_cue(text: str) -> bool:
    lowered = _normalize_validation_text(text)
    return any(cue in lowered for cue in _UNRESOLVED_ABSENCE_CUES)


def _has_explicit_missing_quote(text: str) -> bool:
    for quote in _quoted_phrases(text):
        lowered = _normalize_validation_text(quote)
        if any(marker in lowered for marker in _EXPLICIT_MISSING_MARKERS):
            return True
    return False


def _validate_ambiguous_variant(
    variant: dict[str, Any],
    *,
    allowed_source_ids: set[str] | None = None,
) -> list[str]:
    """Per-variant ambiguous checks shared by initial generation and refinement."""

    errors: list[str] = []
    if not isinstance(variant.get("response_text"), str) or len(variant["response_text"]) < 80:
        errors.append("ambiguous: response_text is too short")
    if "DOC-" in variant.get("response_text", ""):
        errors.append("ambiguous: response_text contains inline document id")
    if variant.get("expected_band") != "amber":
        errors.append("ambiguous variant must use expected_band amber")
    range_payload = variant.get("expected_groundedness_range") or {}
    if range_payload.get("min") != 0.55 or range_payload.get("max") != 0.74:
        errors.append("ambiguous variant must use expected_groundedness_range min=0.55 max=0.74")
    claims = [claim for claim in variant.get("claims", []) if isinstance(claim, dict)]
    claim_labels = [claim.get("expected_label") for claim in claims]
    if claim_labels.count("supported") != 1 or (
        claim_labels.count("ambiguous") + claim_labels.count("partially_supported")
    ) != 1:
        errors.append(
            "ambiguous variant must contain exactly one supported claim and one ambiguous/partially_supported claim"
        )
    if "unsupported" in set(claim_labels):
        errors.append("ambiguous variant must not contain unsupported claims")
    response_lower = variant.get("response_text", "").lower()
    if not any(cue in response_lower for cue in _AMBIGUITY_CUES):
        errors.append("ambiguous variant lacks an explicit uncertainty cue")
    if variant.get("response_text", "").count('"') >= 2:
        errors.append("ambiguous variant should not quote evidence directly")
    for claim in claims:
        label = claim.get("expected_label")
        if label == "supported" and not claim.get("source_doc_ids"):
            errors.append("ambiguous supported claim must include source_doc_ids")
            break
        if label in {"ambiguous", "partially_supported"} and (
            claim.get("source_doc_ids") or claim.get("source_filenames")
        ):
            errors.append("ambiguous unresolved claim must not include source ids or filenames")
            break
    if allowed_source_ids is not None:
        for claim in claims:
            for source_id in claim.get("source_doc_ids", []):
                if source_id not in allowed_source_ids:
                    errors.append(f"ambiguous: invalid source_doc_id {source_id}")
    return errors


def _build_generation_schema() -> dict[str, Any]:
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_text": {"type": "string", "maxLength": 160},
            "expected_label": {
                "type": "string",
                "enum": ["supported", "partially_supported", "ambiguous", "unsupported"],
            },
            "source_doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "source_filenames": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "rationale": {"type": "string", "maxLength": 90},
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
            "response_text": {"type": "string", "maxLength": 520},
            "claims": {"type": "array", "items": claim_schema, "minItems": 2, "maxItems": 2},
            "coverage_notes": {"type": "string", "maxLength": 120},
            "utilization_notes": {"type": "string", "maxLength": 120},
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
    pack = {
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
    pack["raw_context"] = _raw_context_from_pack(pack)
    pack["generation_archetype"] = _infer_case_archetype(pack)
    return pack


def _generation_messages(pack: dict[str, Any]) -> list[dict[str, str]]:
    archetype = _archetype_from_pack(pack)
    system = (
        "You create compact labelled RAG answer variants for groundedness scoring. "
        "Use only supplied documents. Return strict JSON only. Keep every field terse."
    )
    user = f"""
Create exactly three answer variants for this enterprise RAG question:

1. perfect: short, extractive, affirmative, fully grounded in the provided document text, no unsupported or missing-evidence claims.
2. ambiguous: intentionally mid-band. Include exactly one modest supported claim and exactly one unresolved/insufficient-evidence limitation.
3. wrong: plausible but materially unsupported or contradicted by the provided documents; both claims must be unsupported.

{_archetype_prompt_block(archetype)}

Rules:
- Preserve the language expected by the user question when practical.
- Prefer concrete document-specific claims over generic summaries.
- The perfect answer must not invent missing facts and must not say the evidence is absent, incomplete, unclear, or insufficient unless that exact absence/incomplete status is explicitly quoted from selected_documents.text.
- The perfect answer must use only facts visible in selected_documents.text, not filenames, classifications, descriptions, or prior knowledge.
- The perfect answer must not include inline document IDs, parenthetical citations, tables, colon-led lists, or semicolon-heavy enumerations.
- The perfect answer must include two short ASCII double-quoted phrases copied verbatim from selected_documents.text.
- For the perfect answer, preserve the original language of the quoted evidence instead of translating it.
- The perfect answer may include numeric literals, percentages, years, or quantities only when they are copied verbatim from selected_documents.text.
- For perfect, set expected_groundedness_range to {{"min": 0.8, "max": 1.0}} or tighter.
- The wrong answer must still sound realistic, but both claims must be labelled unsupported and must have empty source_doc_ids/source_filenames.
- The ambiguous answer must NOT be a complete answer. It should be useful but inconclusive.
- The ambiguous answer must contain one uncertainty cue such as "unclear", "not established", "does not show",
  "insufficient", "missing", "ne permet pas", "pas clair", "n'établit pas", "manquant", "nicht klar", or "nicht belegt".
- The ambiguous answer must not include direct evidence quotes, DOC ids, citations, tables, or semicolon-heavy lists.
- The ambiguous answer must not contradict the documents and must not contain claims labelled unsupported.
- The ambiguous answer's unresolved/insufficient-evidence claim must have empty source_doc_ids/source_filenames.
- For ambiguous, set expected_groundedness_range to {{"min": 0.55, "max": 0.74}}; this variant should be amber, not near-green.
- Include exact source_id values from selected_documents.source_id in source_doc_ids only where the claim is actually supported.
- Include source_filenames only where the claim is actually supported.
- Use expected_band green for perfect, amber for ambiguous, red for wrong.
- Keep output compact: response_text should be 2-3 concise sentences per variant.
- Include exactly 2 high-signal claims per variant.

Evidence pack:
{json.dumps(pack, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_generated(
    use_case_id: str,
    payload: dict[str, Any],
    *,
    pack: dict[str, Any] | None = None,
    allowed_source_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    raw_context = str((pack or {}).get("raw_context") or "")
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
        if "DOC-" in variant.get("response_text", ""):
            errors.append(f"{variant.get('mutation_type')}: response_text contains inline document id")
        if variant.get("expected_band") not in {"green", "amber", "red"}:
            errors.append(f"{variant.get('mutation_type')}: invalid expected_band")
        labels = {
            claim.get("expected_label")
            for claim in variant.get("claims", [])
            if isinstance(claim, dict)
        }
        claims = [claim for claim in variant.get("claims", []) if isinstance(claim, dict)]
        if variant.get("mutation_type") == "perfect":
            if variant.get("expected_band") != "green":
                errors.append("perfect variant must use expected_band green")
            if not labels <= {"supported"}:
                errors.append("perfect variant contains non-supported claim labels")
            range_payload = variant.get("expected_groundedness_range") or {}
            if range_payload.get("min", 0) < 0.8 or range_payload.get("max") != 1.0:
                errors.append("perfect variant must use expected_groundedness_range min>=0.8 max=1.0")
            quotes = _quoted_phrases(variant.get("response_text", ""))
            if len(quotes) < 2:
                errors.append("perfect variant lacks two double-quoted evidence snippets")
            if raw_context:
                for quote in quotes[:3]:
                    if not _quote_in_context(quote, raw_context):
                        errors.append("perfect variant quote is not copied verbatim from raw_context")
                        break
            if _has_unresolved_absence_cue(variant.get("response_text", "")) and not _has_explicit_missing_quote(
                variant.get("response_text", "")
            ):
                errors.append("perfect variant contains an unresolved/missing-evidence caveat")
            for claim in claims:
                if not claim.get("source_doc_ids"):
                    errors.append("perfect supported claims must include source_doc_ids")
                    break
        if variant.get("mutation_type") == "ambiguous":
            errors.extend(_validate_ambiguous_variant(variant))
        if variant.get("mutation_type") == "wrong":
            if variant.get("expected_band") != "red":
                errors.append("wrong variant must use expected_band red")
            if labels != {"unsupported"}:
                errors.append("wrong variant must contain only unsupported claim labels")
            for claim in claims:
                if claim.get("source_doc_ids") or claim.get("source_filenames"):
                    errors.append("wrong unsupported claims must not include source ids or filenames")
                    break
        if allowed_source_ids is not None:
            for claim in claims:
                for source_id in claim.get("source_doc_ids", []):
                    if source_id not in allowed_source_ids:
                        errors.append(
                            f"{variant.get('mutation_type')}: invalid source_doc_id {source_id}"
                        )
    return errors


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]


def _clean_extractive_sentence(sentence: str) -> str:
    cleaned = sentence.strip().strip("# ").strip()
    cleaned = re.sub(
        r"^(section|article|clause)\s+\d+[A-Za-z.]?\s*[-–—:]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" -–—:")
    return cleaned.strip()


def _extractive_perfect_variant(pack: dict[str, Any]) -> dict[str, Any]:
    qid = pack["use_case_id"]
    archetype = _archetype_from_pack(pack)
    keywords = [
        keyword.lower()
        for keyword in [
            *EVIDENCE_KEYWORDS.get(qid, []),
            *archetype.keywords,
            *[
                token
                for token in re.split(r"\W+", pack["question"])
                if len(token) >= 5
            ],
        ]
        if keyword.strip()
    ]

    candidates: list[tuple[int, dict[str, Any], str]] = []
    for doc in pack["selected_documents"]:
        for sentence in _split_sentences(doc.get("text") or ""):
            sentence = _clean_extractive_sentence(sentence)
            if not (60 <= len(sentence) <= 260):
                continue
            if archetype.archetype_id not in {
                "finance_tax",
                "operations_capacity",
                "quality_audit",
                "procurement_supplier",
            } and re.search(r"%|(?<![A-Za-z])\d", sentence):
                continue
            lowered = sentence.lower()
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if hits <= 0:
                continue
            candidates.append((hits, doc, sentence))

    if len(candidates) < 2:
        for doc in pack["selected_documents"]:
            for sentence in _split_sentences(doc.get("text") or ""):
                sentence = _clean_extractive_sentence(sentence)
                if 60 <= len(sentence) <= 260:
                    lowered = sentence.lower()
                    hits = sum(1 for keyword in keywords if keyword in lowered)
                    if hits > 0:
                        candidates.append((hits, doc, sentence))

    if len(candidates) < 2:
        raise ValueError(f"could not build extractive perfect variant for {qid}")

    candidates.sort(key=lambda item: (-item[0], item[1]["source_id"], len(item[2])))
    selected: list[tuple[dict[str, Any], str]] = []
    seen_docs: set[str] = set()
    for _hits, doc, sentence in candidates:
        normalized = sentence.lower()
        if any(normalized == existing.lower() for _doc, existing in selected):
            continue
        if doc["source_id"] in seen_docs and len(seen_docs) < 2:
            continue
        selected.append((doc, sentence))
        seen_docs.add(doc["source_id"])
        if len(selected) == 2:
            break
    if len(selected) < 2:
        for _hits, doc, sentence in candidates:
            if all(sentence != existing for _doc, existing in selected):
                selected.append((doc, sentence))
            if len(selected) == 2:
                break

    response_text = " ".join(f'"{sentence}"' for _doc, sentence in selected)
    return {
        "mutation_type": "perfect",
        "expected_band": "green",
        "expected_groundedness_range": {"min": 0.8, "max": 1.0},
        "response_text": response_text,
        "claims": [
            {
                "claim_text": sentence,
                "expected_label": "supported",
                "source_doc_ids": [doc["source_id"]],
                "source_filenames": [doc["filename"]],
                "rationale": "Exact sentence copied from evidence.",
            }
            for doc, sentence in selected
        ],
        "coverage_notes": "Perfect variant is exact evidence text.",
        "utilization_notes": "Uses two directly relevant evidence sentences.",
    }


def _force_extractive_perfect(pack: dict[str, Any], payload: dict[str, Any]) -> None:
    perfect = _extractive_perfect_variant(pack)
    variants = payload.get("variants") or []
    for idx, variant in enumerate(variants):
        if variant.get("mutation_type") == "perfect":
            variants[idx] = perfect
            return
    variants.append(perfect)
    payload["variants"] = variants


def _responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "input_text", "text": message["content"]}],
        }
        for message in messages
    ]


def _responses_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    dumped = _model_dump(response)
    for item in dumped.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise RuntimeError("OpenAI Responses API returned no text output")


AMBER_ACCEPT_MIN = 0.55
AMBER_ACCEPT_MAX = 0.74


def _build_ambiguous_regen_schema() -> dict[str, Any]:
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_text": {"type": "string", "maxLength": 160},
            "expected_label": {
                "type": "string",
                "enum": ["supported", "partially_supported", "ambiguous"],
            },
            "source_doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "source_filenames": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "rationale": {"type": "string", "maxLength": 90},
        },
        "required": [
            "claim_text",
            "expected_label",
            "source_doc_ids",
            "source_filenames",
            "rationale",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "response_text": {"type": "string", "maxLength": 520},
            "claims": {
                "type": "array",
                "items": claim_schema,
                "minItems": 2,
                "maxItems": 2,
            },
            "coverage_notes": {"type": "string", "maxLength": 120},
            "utilization_notes": {"type": "string", "maxLength": 120},
        },
        "required": ["response_text", "claims", "coverage_notes", "utilization_notes"],
    }


def _target_reason_from_feedback(feedback: dict[str, Any]) -> str:
    band = str(feedback.get("band") or "").lower()
    score = float(feedback.get("score") or 0.0)
    if band == "green" or score > AMBER_ACCEPT_MAX:
        return (
            "The previous answer was scored too high for amber. Make the unresolved limitation "
            "more central and use a clearer epistemic cue (for example 'unclear', 'not established', "
            "'ne permet pas d'établir', 'nicht belegt'). Keep the supported claim briefer and avoid "
            "any phrasing that sounds like a full, confident answer."
        )
    if band == "red" or score < AMBER_ACCEPT_MIN:
        return (
            "The previous answer was scored below amber. The unresolved claim likely read as a "
            "contradiction. Rewrite it as strictly epistemic ('we cannot establish X', 'the documents "
            "do not provide Y') rather than an accusation or a negated documented fact. Keep the "
            "supported claim fully aligned with the evidence."
        )
    return (
        "The previous answer scored amber but outside the 0.55-0.74 target band. Rebalance the two "
        "claims so the answer is clearly partially helpful and clearly incomplete."
    )


def _ambiguous_regen_messages(
    pack: dict[str, Any],
    current: dict[str, Any],
    feedback: dict[str, Any],
    allowed_source_ids: set[str],
) -> list[dict[str, str]]:
    archetype = _archetype_from_pack(pack)
    bullets = "\n".join(f"  - {item}" for item in archetype.ambiguous_guidance)
    system = (
        "You create mid-band ambiguous RAG answers for groundedness validation. "
        "Return strict JSON only. Keep every field terse."
    )
    current_snippet = (current.get("response_text") or "")[:400]
    user = f"""
Rewrite ONLY the ambiguous variant for this use case so the Latence Trace RAG scorer
classifies it as amber (groundedness score between {AMBER_ACCEPT_MIN} and {AMBER_ACCEPT_MAX}).

Use case: {pack.get("use_case_id")}
Question: {pack.get("question")}
Archetype: {archetype.label}
Archetype ambiguous guidance:
{bullets}

Scorer feedback on the previous attempt:
- band: {feedback.get("band")}
- score: {feedback.get("score")}
- notes: {_target_reason_from_feedback(feedback)}

Previous ambiguous response (do not copy):
{current_snippet}

Hard contract:
- Exactly two claims: one clearly supported (present in selected_documents.text) and one unresolved
  limitation labelled "ambiguous" or "partially_supported".
- Include exactly one explicit uncertainty cue such as "unclear", "not established",
  "does not show", "insufficient", "missing", "ne permet pas", "pas clair", "n'établit pas",
  "manquant", "non clarifié", "nicht klar", "nicht belegt".
- response_text must be 2-3 concise sentences, between 80 and 520 characters.
- response_text must not contradict the documents.
- response_text must not contain direct evidence quotes (no double-quoted phrases) and must not
  include DOC ids, citations, tables, colon-led lists, or semicolon-heavy enumerations.
- The supported claim must include source_doc_ids using only values from:
  {sorted(allowed_source_ids)}.
- The ambiguous/partially_supported claim must have empty source_doc_ids and source_filenames.
- Preserve the language of the original user question when practical.

Evidence pack:
{json.dumps(pack, ensure_ascii=False, indent=2)}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _regenerate_ambiguous_variant(
    *,
    pack: dict[str, Any],
    current: dict[str, Any],
    feedback: dict[str, Any],
    allowed_source_ids: set[str],
    oai_client: Any,
    args: argparse.Namespace,
    attempt_number: int,
    qid: str,
) -> tuple[dict[str, Any] | None, list[str], Any]:
    schema = _build_ambiguous_regen_schema()
    messages = _ambiguous_regen_messages(pack, current, feedback, allowed_source_ids)
    completion = oai_client.responses.create(
        model=args.openai_model,
        input=_responses_input(messages),
        text={
            "format": {
                "type": "json_schema",
                "name": "veracier_ambiguous_refine",
                "schema": schema,
                "strict": True,
            }
        },
        reasoning={},
        tools=[],
        temperature=args.temperature,
        max_output_tokens=args.openai_max_output_tokens,
        top_p=args.openai_top_p,
        store=False,
        include=["web_search_call.action.sources"],
    )
    content = _responses_output_text(completion)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON response: {exc}"], completion

    candidate = {
        **current,
        "response_text": payload.get("response_text"),
        "claims": payload.get("claims"),
        "coverage_notes": payload.get("coverage_notes"),
        "utilization_notes": payload.get("utilization_notes"),
        "mutation_type": "ambiguous",
        "expected_band": "amber",
        "expected_groundedness_range": {"min": 0.55, "max": 0.74},
    }
    errors = _validate_ambiguous_variant(candidate, allowed_source_ids=allowed_source_ids)
    raw_path = (
        args.output_dir
        / "openai_raw"
        / f"{qid}.refine_attempt_{attempt_number}.json"
    )
    _json_dump(
        {
            "attempt": attempt_number,
            "errors": errors,
            "payload": payload,
            "content": content,
            "feedback": feedback,
            "response": _model_dump(completion),
        },
        raw_path,
    )
    if errors:
        return None, errors, completion
    return candidate, [], completion


def _refine_ambiguous_loop(
    variants: list[dict[str, Any]],
    *,
    score_fn,
    regenerate_fn,
    max_attempts: int,
    amber_min: float = AMBER_ACCEPT_MIN,
    amber_max: float = AMBER_ACCEPT_MAX,
) -> dict[str, Any]:
    """Pure-logic refinement loop; callers supply score_fn and regenerate_fn.

    ``score_fn(variant) -> {"band": str, "score": float}`` returns the live sandbox
    band/score. ``regenerate_fn(variant, feedback, attempt)`` returns a new variant
    dict that replaces the mutable fields (``response_text``, ``claims``, etc.) or
    ``None`` if regeneration fails. The loop updates ``variants`` in-place and
    returns an aggregate summary.
    """

    stats = {
        "total_ambiguous": 0,
        "accepted_first_pass": 0,
        "regenerated_accepted": 0,
        "unstable_excluded": 0,
        "sandbox_calls": 0,
    }
    log: list[dict[str, Any]] = []
    for variant in variants:
        if variant.get("mutation_type") != "ambiguous":
            continue
        stats["total_ambiguous"] += 1
        example_log = {
            "example_id": variant.get("example_id"),
            "question_id": variant.get("question_id"),
            "attempts": [],
            "final_status": "unstable",
            "accepted_attempt": None,
        }
        accepted = False
        for attempt in range(max_attempts + 1):
            live = score_fn(variant)
            stats["sandbox_calls"] += 1
            band = str(live.get("band") or "").lower()
            score = float(live.get("score") or 0.0)
            example_log["attempts"].append(
                {"attempt": attempt, "band": band, "score": round(score, 4)}
            )
            if band == "amber" and amber_min <= score <= amber_max:
                accepted = True
                example_log["final_status"] = "accepted"
                example_log["accepted_attempt"] = attempt
                variant["refined_sandbox_band"] = band
                variant["refined_sandbox_score"] = round(score, 4)
                variant["ambiguous_unstable"] = False
                if attempt == 0:
                    stats["accepted_first_pass"] += 1
                else:
                    stats["regenerated_accepted"] += 1
                break
            if attempt == max_attempts:
                break
            try:
                candidate = regenerate_fn(variant, live, attempt + 1)
            except Exception as exc:  # noqa: BLE001 - surfaced in the log
                example_log["attempts"][-1]["regen_error"] = str(exc)
                candidate = None
            if candidate is None:
                example_log["attempts"][-1]["regen_error"] = example_log["attempts"][-1].get(
                    "regen_error", "regeneration failed"
                )
                break
            variant.update(
                {
                    key: candidate[key]
                    for key in (
                        "response_text",
                        "claims",
                        "coverage_notes",
                        "utilization_notes",
                    )
                    if key in candidate
                }
            )
        if not accepted:
            last_band = (
                example_log["attempts"][-1]["band"]
                if example_log["attempts"]
                else None
            )
            last_score = (
                example_log["attempts"][-1]["score"]
                if example_log["attempts"]
                else None
            )
            variant["refined_sandbox_band"] = last_band
            variant["refined_sandbox_score"] = last_score
            variant["ambiguous_unstable"] = True
            stats["unstable_excluded"] += 1
        log.append(example_log)
    return {"stats": stats, "log": log}


def generate_responses(args: argparse.Namespace) -> Path:
    packs = _load_evidence_packs(args)
    if not packs:
        raise FileNotFoundError("missing evidence packs; run --stage build-evidence-packs first")

    schema = _build_generation_schema()
    prompt_inputs: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
    for pack in packs:
        qid = pack["use_case_id"]
        messages = _generation_messages(pack)
        prompt_path = args.output_dir / "prompts" / f"{qid}.json"
        _json_dump({"messages": messages, "schema": schema}, prompt_path)
        prompt_inputs.append((pack, messages))

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY before generating labelled responses")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    rows_out: list[dict[str, Any]] = []
    for pack, messages in prompt_inputs:
        qid = pack["use_case_id"]

        payload: dict[str, Any] | None = None
        completion: Any = None
        attempt_messages = list(messages)
        errors: list[str] = []
        allowed_source_ids = {doc["source_id"] for doc in pack["selected_documents"]}
        for attempt in range(1, args.openai_max_attempts + 1):
            print(f"generating variants for {qid} with {args.openai_model} attempt={attempt}")
            completion = client.responses.create(
                model=args.openai_model,
                input=_responses_input(attempt_messages),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "veracier_trace_variants",
                        "schema": schema,
                        "strict": True,
                    }
                },
                reasoning={},
                tools=[],
                temperature=args.temperature,
                max_output_tokens=args.openai_max_output_tokens,
                top_p=args.openai_top_p,
                store=False,
                include=["web_search_call.action.sources"],
            )
            content = _responses_output_text(completion)
            try:
                payload = json.loads(content)
                # Keep "perfect" deterministic: it is the anchor used to
                # distinguish scorer calibration issues from generator drift.
                # The model still creates ambiguous/wrong variants, but exact
                # evidence quotes make the green label auditable per archetype.
                _force_extractive_perfect(pack, payload)
                errors = _validate_generated(
                    qid,
                    payload,
                    pack=pack,
                    allowed_source_ids=allowed_source_ids,
                )
            except json.JSONDecodeError as exc:
                payload = None
                errors = [f"invalid JSON response: {exc}"]
            except ValueError as exc:
                payload = None
                errors = [str(exc)]
            if not errors:
                break
            _json_dump(
                {
                    "attempt": attempt,
                    "errors": errors,
                    "payload": payload,
                    "content": content,
                    "response": _model_dump(completion),
                },
                args.output_dir / "openai_raw" / f"{qid}.invalid_attempt_{attempt}.json",
            )
            attempt_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "The previous JSON failed validation: "
                        f"{errors}. Regenerate the full JSON. The perfect variant "
                        "must be affirmative/extractive, contain only claims labelled supported, "
                        "use expected_groundedness_range min>=0.8 max=1.0, and must not "
                        "use missing-evidence caveats unless the missing/incomplete status is "
                        "explicitly quoted from evidence; ambiguous must "
                        "contain exactly one supported claim and exactly one ambiguous "
                        "or partially_supported claim, with expected_groundedness_range "
                        "min=0.55 max=0.74, an explicit uncertainty cue, and no source ids "
                        "on the unresolved claim; wrong "
                        "must contain only unsupported claims and no source ids. The response_text "
                        "must not include DOC ids or citations. The perfect response must include "
                        "two ASCII double-quoted phrases copied verbatim from evidence; numeric "
                        "literals are allowed only when copied from evidence. "
                        "source_doc_ids must use "
                        f"only these exact values: {sorted(allowed_source_ids)}."
                    ),
                },
            ]

        if payload is None or errors:
            raise ValueError(f"generated payload failed validation for {qid}: {errors}")

        raw_path = args.output_dir / "openai_raw" / f"{qid}.json"
        _json_dump({"payload": payload, "response": _model_dump(completion)}, raw_path)
        for variant in payload["variants"]:
            rows_out.append(
                {
                    "example_id": f"{qid}:{variant['mutation_type']}",
                    "question_id": qid,
                    "query_text": pack["question"],
                    "generation_archetype": pack.get("generation_archetype") or _infer_case_archetype(pack),
                    "mutation_type": variant["mutation_type"],
                    "expected_band": variant["expected_band"],
                    "expected_groundedness_range": variant["expected_groundedness_range"],
                    "response_text": variant["response_text"],
                    "claims": variant["claims"],
                    "coverage_notes": variant["coverage_notes"],
                    "utilization_notes": variant["utilization_notes"],
                    "context_doc_ids": [doc["source_id"] for doc in pack["selected_documents"]],
                    "context_filenames": [doc["filename"] for doc in pack["selected_documents"]],
                    "raw_context": pack["raw_context"],
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


def refine_ambiguous(args: argparse.Namespace) -> Path:
    """Live self-filter ambiguous variants against TRACE; regenerate when mislabeled."""

    variants_path = args.output_dir / "variants.jsonl"
    variants = _jsonl_read(variants_path)
    if not variants:
        raise FileNotFoundError("missing variants.jsonl; run --stage generate-responses first")

    ambiguous = [v for v in variants if v.get("mutation_type") == "ambiguous"]
    if not ambiguous:
        print("no ambiguous variants to refine")
        return variants_path

    latence_key = os.environ.get("LATENCE_API_KEY") or os.environ.get("LATENCE_GATEWAY_KEY")
    if not latence_key:
        raise RuntimeError("set LATENCE_API_KEY before running refine-ambiguous")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("set OPENAI_API_KEY before running refine-ambiguous")

    packs = {pack["use_case_id"]: pack for pack in _load_evidence_packs(args)}

    from latence import Latence
    from openai import OpenAI

    lat_client = Latence(
        api_key=latence_key,
        base_url=os.environ.get("LATENCE_BASE_URL"),
        timeout=args.sdk_timeout,
    )
    oai_client = OpenAI(api_key=openai_key)

    def score_fn(variant: dict[str, Any]) -> dict[str, Any]:
        response = lat_client.experimental.trace.rag(
            response_text=variant["response_text"],
            query_text=variant["query_text"],
            raw_context=variant["raw_context"],
            primary_metric="triangular",
            segmentation_mode="sentence_packed",
            heatmap_format="none",
            profile="standard",
            structured_verification=args.structured_verification,
            verbose=False,
        )
        dumped = _model_dump(response)
        return {"band": dumped.get("band"), "score": dumped.get("score")}

    def regenerate_fn(
        variant: dict[str, Any], feedback: dict[str, Any], attempt: int
    ) -> dict[str, Any] | None:
        pack = packs.get(variant["question_id"])
        if pack is None:
            print(f"refine: no pack for {variant['example_id']}")
            return None
        allowed_source_ids = {doc["source_id"] for doc in pack["selected_documents"]}
        candidate, errors, _completion = _regenerate_ambiguous_variant(
            pack=pack,
            current=variant,
            feedback=feedback,
            allowed_source_ids=allowed_source_ids,
            oai_client=oai_client,
            args=args,
            attempt_number=attempt,
            qid=variant["question_id"],
        )
        if errors:
            print(
                f"refine: regen invalid for {variant['example_id']} attempt={attempt} "
                f"errors={errors}"
            )
            return None
        return candidate

    print(f"refining {len(ambiguous)} ambiguous variants (max_attempts={args.refine_max_attempts})")
    try:
        summary = _refine_ambiguous_loop(
            variants,
            score_fn=score_fn,
            regenerate_fn=regenerate_fn,
            max_attempts=args.refine_max_attempts,
        )
    finally:
        lat_client.close()

    _jsonl_write(variants, variants_path)
    summary_path = args.output_dir / "refinement_summary.json"
    _json_dump(summary, summary_path)
    stats = summary["stats"]
    print(
        "refine-ambiguous stats: "
        f"total={stats['total_ambiguous']} "
        f"accepted_first_pass={stats['accepted_first_pass']} "
        f"regenerated_accepted={stats['regenerated_accepted']} "
        f"unstable_excluded={stats['unstable_excluded']} "
        f"sandbox_calls={stats['sandbox_calls']}"
    )
    print(f"wrote refinement summary: {summary_path}")
    return variants_path


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
                    structured_verification=args.structured_verification,
                    verbose=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                dumped = _model_dump(response)
                rows_out.append(
                    {
                        "example_id": variant["example_id"],
                        "question_id": variant["question_id"],
                        "generation_archetype": variant.get("generation_archetype"),
                        "mutation_type": variant["mutation_type"],
                        "expected_band": variant["expected_band"],
                        "expected_groundedness_range": variant["expected_groundedness_range"],
                        "ambiguous_unstable": bool(variant.get("ambiguous_unstable", False)),
                        "refined_sandbox_band": variant.get("refined_sandbox_band"),
                        "refined_sandbox_score": variant.get("refined_sandbox_score"),
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


def _row_archetype_id(row: dict[str, Any]) -> str:
    payload = row.get("generation_archetype")
    if isinstance(payload, dict):
        return str(payload.get("id") or "unknown")
    return "unknown"


def _row_archetype_label(row: dict[str, Any]) -> str:
    payload = row.get("generation_archetype")
    if isinstance(payload, dict):
        return str(payload.get("label") or payload.get("id") or "unknown")
    return "unknown"


GREEN_PRECISION_TARGET = 0.97
RED_PRECISION_TARGET = 0.95
AMBER_AGREEMENT_TARGET = 28
AMBER_AGREEMENT_DENOMINATOR = 34


def _classify_confusion(row: dict[str, Any]) -> tuple[str, str]:
    """Return ``(expected, observed)`` band labels for confusion-matrix math."""

    expected = str(row.get("expected_band") or "").lower()
    observed = str((row.get("trace") or {}).get("band") or "").lower()
    return expected, observed


def _compute_headline_metrics(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Customer-defensible headline metrics for the Veracier report.

    * Green precision: of the rows TRACE classified as green, how many had
      expected_band=green.
    * Red precision: of the rows TRACE classified as red, how many had
      expected_band=red.
    * Amber agreement: of the rows whose expected_band is amber AND that
      were NOT excluded as unstable, how many TRACE classified as amber.
    """

    green_tp = green_fp = 0
    red_tp = red_fp = 0
    amber_expected_total = 0
    amber_observed_match = 0
    amber_excluded = 0
    for row in traces:
        expected, observed = _classify_confusion(row)
        # Unstable-amber rows are excluded from every headline denominator
        # because the benchmark already labelled their ambiguous ground truth
        # as unreliable. They are still reported as context under amber.
        if row.get("ambiguous_unstable"):
            if expected == "amber":
                amber_excluded += 1
            continue
        if observed == "green":
            if expected == "green":
                green_tp += 1
            else:
                green_fp += 1
        if observed == "red":
            if expected == "red":
                red_tp += 1
            else:
                red_fp += 1
        if expected == "amber":
            amber_expected_total += 1
            if observed == "amber":
                amber_observed_match += 1
    green_denominator = green_tp + green_fp
    red_denominator = red_tp + red_fp
    return {
        "green": {
            "precision": (green_tp / float(green_denominator)) if green_denominator else 0.0,
            "true_positive": green_tp,
            "false_positive": green_fp,
            "denominator": green_denominator,
        },
        "red": {
            "precision": (red_tp / float(red_denominator)) if red_denominator else 0.0,
            "true_positive": red_tp,
            "false_positive": red_fp,
            "denominator": red_denominator,
        },
        "amber": {
            "agreement": (
                amber_observed_match / float(amber_expected_total)
                if amber_expected_total
                else 0.0
            ),
            "match": amber_observed_match,
            "expected": amber_expected_total,
            "excluded_unstable": amber_excluded,
        },
    }


def _compute_archetype_metrics(traces: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traces:
        buckets[_row_archetype_id(row)].append(row)
    out: list[dict[str, Any]] = []
    for archetype, rows in sorted(buckets.items()):
        metrics = _compute_headline_metrics(rows)
        sample_label = _row_archetype_label(rows[0]) if rows else archetype
        exact = sum(
            1
            for row in rows
            if (row.get("trace") or {}).get("band") == row.get("expected_band")
            and not row.get("ambiguous_unstable")
        )
        denom = sum(1 for row in rows if not row.get("ambiguous_unstable"))
        out.append(
            {
                "id": archetype,
                "label": sample_label,
                "rows": len(rows),
                "headline_accuracy": exact / float(denom) if denom else 0.0,
                "headline_denominator": denom,
                "green_precision": metrics["green"]["precision"],
                "green_denominator": metrics["green"]["denominator"],
                "red_precision": metrics["red"]["precision"],
                "red_denominator": metrics["red"]["denominator"],
                "amber_agreement": metrics["amber"]["agreement"],
                "amber_expected": metrics["amber"]["expected"],
                "amber_unstable_excluded": metrics["amber"]["excluded_unstable"],
            }
        )
    return out


def _compute_latency_summary(
    traces: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in traces:
        wall = row.get("wall_ms")
        profile = str(row.get("profile") or "")
        if isinstance(wall, (int, float)):
            buckets[profile].append(float(wall))
    out: dict[str, dict[str, float]] = {}
    for profile, values in buckets.items():
        if not values:
            continue
        values = sorted(values)
        n = len(values)
        out[profile] = {
            "count": float(n),
            "p50_ms": float(values[n // 2]),
            "p95_ms": float(values[min(n - 1, int(round(n * 0.95)))]),
            "mean_ms": float(sum(values) / n),
            "max_ms": float(values[-1]),
        }
    return out


def _pick_annotated_examples(
    traces: Sequence[dict[str, Any]],
    *,
    max_examples: int = 8,
) -> list[dict[str, Any]]:
    """Pick one illustrative row per archetype for the annotated examples block."""

    seen: dict[str, dict[str, Any]] = {}
    # Prefer rows where the expected band matches observed so examples are
    # unambiguous, but fall back to mismatched rows so reviewers still see
    # how TRACE responds to borderline cases.
    for row in traces:
        archetype = _row_archetype_id(row)
        if archetype in seen:
            continue
        if row.get("ambiguous_unstable"):
            continue
        expected, observed = _classify_confusion(row)
        if expected == observed:
            seen[archetype] = row
    for row in traces:
        archetype = _row_archetype_id(row)
        if archetype in seen:
            continue
        if row.get("ambiguous_unstable"):
            continue
        seen[archetype] = row
    return list(seen.values())[:max_examples]


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _target_badge(actual: float, target: float, unit: str = "percent") -> str:
    if unit == "percent":
        status = "PASS" if actual >= target else "FAIL"
        return f"{_format_pct(actual)} (target {_format_pct(target)}) - {status}"
    status = "PASS" if actual >= target else "FAIL"
    return f"{actual:.0f} (target {target:.0f}) - {status}"


def write_report(args: argparse.Namespace) -> Path:
    corpus_summary_path = args.output_dir / "corpus_summary.json"
    evidence_manifest_path = args.output_dir / "evidence_manifest.json"
    corpus_summary = (
        json.loads(corpus_summary_path.read_text(encoding="utf-8"))
        if corpus_summary_path.exists()
        else {}
    )
    evidence_manifest = (
        json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
        if evidence_manifest_path.exists()
        else {}
    )
    excluded = _jsonl_read(args.output_dir / "excluded_documents.jsonl")
    variants = _jsonl_read(args.output_dir / "variants.jsonl")
    traces = _jsonl_read(args.output_dir / "trace_results.jsonl")
    refinement_path = args.output_dir / "refinement_summary.json"
    refinement_summary = (
        json.loads(refinement_path.read_text(encoding="utf-8"))
        if refinement_path.exists()
        else {}
    )

    headline = _compute_headline_metrics(traces) if traces else None
    archetype_rows = _compute_archetype_metrics(traces) if traces else []
    latency_summary = _compute_latency_summary(traces) if traces else {}
    annotated_examples = _pick_annotated_examples(traces) if traces else []
    refinement_stats = refinement_summary.get("stats") if refinement_summary else {}
    amber_accepted_total = 0
    if refinement_stats:
        amber_accepted_total = int(
            refinement_stats.get("accepted_first_pass", 0)
        ) + int(refinement_stats.get("regenerated_accepted", 0))

    lines: list[str] = []

    # --- Section 1: Scope and dataset -------------------------------------
    lines.extend(
        [
            "# Veracier Industries - TRACE RAG Validation Report",
            "",
            f"- Created: {datetime.now(timezone.utc).isoformat()}",
            f"- Run directory: `{args.output_dir}`",
            f"- Scope: `{evidence_manifest.get('scope', args.scope)}`",
            "",
            "## 1. Scope and dataset",
            "",
            f"- Dataset: `lightonai/veracier-industries`",
            f"- Use cases included: {len(evidence_manifest.get('use_cases', []) or [])}",
            f"- Variants generated: {len(variants)}",
            f"- TRACE calls executed: {len(traces)}",
            f"- Archetypes represented: "
            + ", ".join(sorted({_row_archetype_id(row) for row in variants}) or ["-"]),
        ]
    )
    if corpus_summary:
        lines.append(
            f"- Processed source PDFs: {corpus_summary.get('completed_with_markdown', '-')}"
            f" / master unique {corpus_summary.get('master_unique_documents', '-')}"
        )
        lines.append(
            f"- Total completed chars: {corpus_summary.get('total_completed_chars', '-')}"
            f" (mean {corpus_summary.get('mean_completed_chars', '-')})"
        )
    if excluded:
        lines.append(f"- Documents excluded during corpus filtering: {len(excluded)}")

    # --- Section 2: Headline metrics --------------------------------------
    lines.extend(["", "## 2. Headline metrics", ""])
    if headline:
        green = headline["green"]
        red = headline["red"]
        amber = headline["amber"]
        lines.extend(
            [
                "| Metric | Value | Target |",
                "| --- | --- | --- |",
                (
                    "| Green precision (TRACE green => expected green) "
                    f"| {green['true_positive']}/{green['denominator']} "
                    f"({_format_pct(green['precision'])}) "
                    f"| >= {_format_pct(GREEN_PRECISION_TARGET)} |"
                ),
                (
                    "| Red precision (TRACE red => expected red) "
                    f"| {red['true_positive']}/{red['denominator']} "
                    f"({_format_pct(red['precision'])}) "
                    f"| >= {_format_pct(RED_PRECISION_TARGET)} |"
                ),
                (
                    "| Amber agreement (expected amber => TRACE amber) "
                    f"| {amber['match']}/{amber['expected']} "
                    f"({_format_pct(amber['agreement'])}) "
                    f"| >= {AMBER_AGREEMENT_TARGET}/{AMBER_AGREEMENT_DENOMINATOR} on pilot |"
                ),
                "",
                "- Amber unstable rows excluded from headline: "
                f"{amber['excluded_unstable']}",
                "- Ambiguous variants accepted by refinement: "
                f"{amber_accepted_total}",
                "",
                "Amber is reported as a reviewer-queue band; accuracy targets above do"
                " not assume amber is a final decision (see section 6).",
            ]
        )
    else:
        lines.append("- TRACE stage has not produced results yet.")

    # --- Section 3: Archetype breakdown -----------------------------------
    lines.extend(["", "## 3. Archetype breakdown", ""])
    if archetype_rows:
        lines.append(
            "| Archetype | Rows | Accuracy | Green precision | Red precision | Amber agreement |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for entry in archetype_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        entry["label"],
                        str(entry["rows"]),
                        f"{entry['headline_accuracy'] * 100:.1f}% ({entry['headline_denominator']})",
                        (
                            f"{entry['green_precision'] * 100:.1f}% ({entry['green_denominator']})"
                            if entry["green_denominator"]
                            else "-"
                        ),
                        (
                            f"{entry['red_precision'] * 100:.1f}% ({entry['red_denominator']})"
                            if entry["red_denominator"]
                            else "-"
                        ),
                        (
                            f"{entry['amber_agreement'] * 100:.1f}% ({entry['amber_expected']})"
                            if entry["amber_expected"]
                            else "-"
                        ),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- No trace results available for archetype breakdown.")

    # --- Section 4: Latency and cost --------------------------------------
    lines.extend(["", "## 4. Latency summary by profile", ""])
    if latency_summary:
        lines.append("| Profile | Count | P50 ms | P95 ms | Mean ms | Max ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for profile, stats in sorted(latency_summary.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        profile,
                        str(int(stats["count"])),
                        f"{stats['p50_ms']:.0f}",
                        f"{stats['p95_ms']:.0f}",
                        f"{stats['mean_ms']:.0f}",
                        f"{stats['max_ms']:.0f}",
                    ]
                )
                + " |"
            )
    else:
        lines.append("- Latency data not available.")

    # --- Section 5: Annotated examples ------------------------------------
    lines.extend(["", "## 5. Annotated examples", ""])
    if annotated_examples:
        variants_by_id = {variant["example_id"]: variant for variant in variants}
        for row in annotated_examples:
            variant_row = variants_by_id.get(row["example_id"], {})
            trace = row.get("trace") or {}
            hedge = None
            hedge_gate = trace.get("epistemic_hedge_gate")
            if isinstance(hedge_gate, dict):
                hedge = hedge_gate
            lines.extend(
                [
                    f"### {row['example_id']} ({_row_archetype_label(row)})",
                    "",
                    f"- Profile: `{row.get('profile')}`",
                    f"- Expected band: `{row.get('expected_band')}` - "
                    f"TRACE band: `{trace.get('band', '-')}` - "
                    f"score: {_fmt_float(trace.get('score'))}",
                    f"- NLI aggregate: {_fmt_float(trace.get('nli_aggregate'))},"
                    f" coverage: {_fmt_float(trace.get('context_coverage_ratio'))},"
                    f" usage: {_fmt_float(trace.get('context_usage_ratio'))}",
                ]
            )
            if hedge:
                lines.append(
                    "- Epistemic hedge gate: "
                    f"fired={hedge.get('fired')}, applied={hedge.get('applied')}, "
                    f"cap={hedge.get('cap'):.3f}, floor={hedge.get('floor'):.3f}, "
                    f"claim_indices={hedge.get('claim_indices')}"
                )
            query_text = str(variant_row.get("query_text") or "")[:220]
            response_text = str(variant_row.get("response_text") or "")[:320]
            if query_text:
                lines.append(f"- Question: {query_text}")
            if response_text:
                lines.append(f"- Response: {response_text}")
            expected, observed = _classify_confusion(row)
            rationale = (
                "TRACE agrees with the expected band."
                if expected == observed
                else (
                    "TRACE disagrees with the expected band; this row is a candidate"
                    " for reviewer attention."
                )
            )
            lines.append(f"- Rationale: {rationale}")
            lines.append("")
    else:
        lines.append("- No trace rows to annotate yet.")

    # --- Section 6: Amber as reviewer queue -------------------------------
    lines.extend(
        [
            "",
            "## 6. Amber is a reviewer queue, not an auto-decision",
            "",
            "- TRACE green means the answer is grounded. Paying users can auto-approve"
            " green outputs to their downstream workflow.",
            "- TRACE red means the answer is not grounded. Paying users can auto-block"
            " red outputs.",
            "- TRACE amber means route to a human reviewer. Amber-band volume is"
            " expected and beneficial; it is where TRACE creates value for ambiguous"
            " prose by flagging epistemic uncertainty instead of guessing.",
            "- Example reviewer workflow:",
            "  1. Amber output lands in the reviewer inbox with score, band, and"
            " hedge-gate diagnostics.",
            "  2. Reviewer inspects the evidence snippet flagged by TRACE.",
            "  3. Reviewer confirms, edits, or rejects the answer in under a minute.",
            "",
            "See `latence-trace/docs/amber_reviewer_queue.md` for the full contract.",
        ]
    )

    # --- Refinement + evidence pack tables (diagnostic appendix) ----------
    lines.extend(["", "## Appendix A - Ambiguous refinement", ""])
    if refinement_summary:
        stats = refinement_summary.get("stats") or {}
        lines.extend(
            [
                f"- Total ambiguous variants: {stats.get('total_ambiguous', 0)}",
                f"- Accepted first pass: {stats.get('accepted_first_pass', 0)}",
                f"- Accepted after regeneration: {stats.get('regenerated_accepted', 0)}",
                f"- Excluded as unstable: {stats.get('unstable_excluded', 0)}",
                f"- Sandbox trace calls: {stats.get('sandbox_calls', 0)}",
            ]
        )
    else:
        lines.append("- Refinement stage not run for this report.")

    lines.extend(["", "## Appendix B - Evidence packs", ""])
    cases = evidence_manifest.get("cases") or []
    if cases:
        lines.append("| Use case | Archetype | Docs | Context chars |")
        lines.append("| --- | --- | ---: | ---: |")
        for case in cases:
            lines.append(
                "| "
                + " | ".join(
                    [
                        case.get("use_case_id", ""),
                        _row_archetype_id(case),
                        str(case.get("document_count", "")),
                        str(case.get("context_char_count", "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- Evidence packs not available.")

    lines.extend(["", "## Appendix C - Per-row trace results", ""])
    if traces:
        lines.append(
            "| Example | Profile | Expected | Band | Score | NLI | Coverage | Usage | Wall ms | Unstable |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |")
        for row in traces:
            trace = row.get("trace") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["example_id"],
                        row["profile"],
                        str(row.get("expected_band") or ""),
                        str(trace.get("band", "")),
                        _fmt_float(trace.get("score")),
                        _fmt_float(trace.get("nli_aggregate")),
                        _fmt_float(trace.get("context_coverage_ratio")),
                        _fmt_float(trace.get("context_usage_ratio")),
                        _fmt_float(row.get("wall_ms")),
                        "yes" if row.get("ambiguous_unstable") else "no",
                    ]
                )
                + " |"
            )
    else:
        lines.append("- TRACE stage not run.")

    report_path = args.output_dir / "proof_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": evidence_manifest.get("scope", args.scope),
        "run_dir": str(args.output_dir),
        "use_cases": evidence_manifest.get("use_cases", []),
        "headline_metrics": headline,
        "headline_targets": {
            "green_precision_target": GREEN_PRECISION_TARGET,
            "red_precision_target": RED_PRECISION_TARGET,
            "amber_agreement_target": AMBER_AGREEMENT_TARGET,
            "amber_agreement_denominator": AMBER_AGREEMENT_DENOMINATOR,
        },
        "archetype_metrics": archetype_rows,
        "latency_summary": latency_summary,
        "refinement": refinement_summary,
        "variant_counts": {
            "total": len(variants),
            "by_mutation_type": dict(
                Counter(variant.get("mutation_type") for variant in variants)
            ),
            "by_archetype": dict(
                Counter(_row_archetype_id(variant) for variant in variants)
            ),
        },
        "trace_counts": {
            "total": len(traces),
            "by_profile": dict(
                Counter(row.get("profile") for row in traces)
            ),
        },
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote report: {report_path}")
    print(f"wrote summary: {summary_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "filter-corpus",
            "build-evidence-packs",
            "select-docs",
            "process-docs",
            "generate-responses",
            "refine-ambiguous",
            "trace",
            "report",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--scope",
        choices=["pilot", "capped", "full", "full-use-cases"],
        default="pilot",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--processed-documents", type=Path, default=DEFAULT_PROCESSED_DOCUMENTS)
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
    parser.add_argument("--openai-model", default="gpt-4.1")
    parser.add_argument("--openai-max-attempts", type=int, default=3)
    parser.add_argument("--openai-max-output-tokens", type=int, default=2048)
    parser.add_argument("--openai-top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompt-context-chars", type=int, default=36_000)
    parser.add_argument("--trace-profile", choices=["standard", "quality", "both"], default="both")
    parser.add_argument(
        "--refine-max-attempts",
        type=int,
        default=4,
        help=(
            "Max live-regeneration attempts per ambiguous variant during --stage "
            "refine-ambiguous. Each attempt uses one sandbox trace call plus one "
            "OpenAI regeneration when out of amber."
        ),
    )
    parser.add_argument(
        "--structured-verification",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Forwarded to the Trace RAG call. 'auto' (default) uses the "
            "scorer's prose-safety guard; 'off' disables the typed "
            "structured lane entirely; 'on' forces it."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages = (
        [
            "filter-corpus",
            "build-evidence-packs",
            "generate-responses",
            "refine-ambiguous",
            "trace",
            "report",
        ]
        if args.stage == "all"
        else [args.stage]
    )

    for stage in stages:
        if stage == "filter-corpus":
            filter_corpus(args)
        elif stage == "build-evidence-packs":
            build_evidence_packs(args)
        elif stage == "select-docs":
            select_docs(args)
        elif stage == "process-docs":
            process_docs(args)
        elif stage == "generate-responses":
            generate_responses(args)
        elif stage == "refine-ambiguous":
            refine_ambiguous(args)
        elif stage == "trace":
            run_trace(args)
        elif stage == "report":
            write_report(args)
        else:  # pragma: no cover
            raise AssertionError(stage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
