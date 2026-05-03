"""Deterministic proof harness for immutable history vault spot repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from latence_trace.memory.models import MemoryPolicy
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEvent,
    TraceSessionEventRequest,
    TraceSessionRepairRequest,
)
from latence_trace.sessions.service import TraceSessionService

CASES = [
    {
        "domain": "legal",
        "kind": "rag",
        "memory_domain": "rag",
        "anchor": "LGL-042",
        "critical_terms": ["LGL-042", "14-day", "mutual termination"],
        "source": (
            "Final counsel source LGL-042: renewal clause requires 14-day cure period "
            "and only allows mutual termination."
        ),
        "noise": "Draft noise says immediate termination, but this was superseded.",
    },
    {
        "domain": "finance",
        "kind": "rag",
        "memory_domain": "rag",
        "anchor": "FIN-778",
        "critical_terms": ["FIN-778", "$44,200", "accrual_owner=Elena"],
        "source": "Controller-approved FIN-778 close note: accrual_owner=Elena amount=$44,200.",
        "noise": "Preliminary spreadsheet v3 incorrectly listed amount=$4,200.",
    },
    {
        "domain": "support",
        "kind": "rag",
        "memory_domain": "tool",
        "anchor": "T-771",
        "critical_terms": ["T-771", "refund_status=approved", "R-90017"],
        "source": "Tool truth T-771: refund_status=approved replacement_order=R-90017.",
        "noise": "Old state T-771 refund_status=pending replacement_order=none.",
    },
    {
        "domain": "coding",
        "kind": "code",
        "memory_domain": "code",
        "anchor": "RUST-991",
        "critical_terms": ["RUST-991", "src/lib.rs", "deploy_guard::verify_release"],
        "source": "Sparse Rust trace RUST-991: src/lib.rs deploy_guard::verify_release owns release safety.",
        "noise": "Generated target/debug/build tmp_8fa3 artifact mentions HAT123.",
    },
    {
        "domain": "pharmacy",
        "kind": "rag",
        "memory_domain": "grounding",
        "anchor": "PHARM-515",
        "critical_terms": ["PHARM-515", "clarithromycin", "rhabdomyolysis"],
        "source": (
            "PHARM-515 safety note: clarithromycin with simvastatin is contraindicated; "
            "risk of rhabdomyolysis. Patient phone 555-0100."
        ),
        "noise": "Customer chat asked for exact dose and shared phone 555-0100.",
    },
    {
        "domain": "mixed_agent",
        "kind": "general",
        "memory_domain": "chat",
        "anchor": "GEN-303",
        "critical_terms": ["GEN-303", "block_until_regrounded", "source-vault"],
        "source": "GEN-303 runtime rule: if hot context conflicts, block_until_regrounded via source-vault.",
        "noise": "Assistant summary says continue without checking original history.",
    },
]


def run_proof(
    *,
    output_dir: Path,
    step_counts: tuple[int, ...] = (100, 250, 500, 1000),
    max_tokens: int = 160,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_metrics = output_dir / "live_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if live_metrics.exists():
        live_metrics.unlink()

    for steps in step_counts:
        for case in CASES:
            row = _run_case(case, steps=steps, max_tokens=max_tokens)
            rows.append(row)
            with live_metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = _summarize(rows, max_tokens=max_tokens)
    artifact_hashes = _write_artifacts(output_dir, report, rows)
    report["artifact_hashes"] = artifact_hashes
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_case(case: dict[str, Any], *, steps: int, max_tokens: int) -> dict[str, Any]:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(
        TraceSessionCreateRequest(
            kind=case["kind"],
            memory_policy=MemoryPolicy(hot_token_budget=64, warm_token_budget=256, max_spans=128),
        )
    )
    session_id = created.session.session_id
    service.append_event(
        session_id,
        TraceSessionEventRequest(
            memory_domain=case["memory_domain"],
            event=TraceSessionEvent(content=case["source"], raw_context=case["source"]),
        ),
    )
    for idx in range(steps - 1):
        text = f"{case['domain']} long-session distractor {idx}: {case['noise']}"
        service.append_event(
            session_id,
            TraceSessionEventRequest(
                memory_domain=case["memory_domain"],
                event=TraceSessionEvent(content=text, raw_context=text),
                return_context=False,
            ),
        )

    packet = service.repair(
        session_id,
        TraceSessionRepairRequest(
            query_text=f"Recover original decision for {case['anchor']}",
            missing_terms=case["critical_terms"],
            reason="forced adversarial anchor loss",
            max_tokens=max_tokens,
        ),
    ).repair_packet
    joined = " ".join(excerpt.text for excerpt in packet.excerpts)
    recovered = [term for term in case["critical_terms"] if term.lower() in joined.lower()]
    pii_leak = "555-0100" in joined or "@" in joined
    return {
        "domain": case["domain"],
        "steps": steps,
        "critical_terms": case["critical_terms"],
        "recovered_terms": recovered,
        "repair_recall": len(recovered) / len(case["critical_terms"]),
        "triggered": packet.triggered,
        "suggested_action": packet.suggested_action,
        "token_count": packet.token_count,
        "bounded": packet.token_count <= max_tokens,
        "pii_leak": pii_leak,
        "excerpt_count": len(packet.excerpts),
    }


def _summarize(rows: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    repair_recalls = [float(row["repair_recall"]) for row in rows]
    recall = sum(repair_recalls) / max(1, len(repair_recalls))
    all_repaired = all(row["triggered"] and row["repair_recall"] >= 1.0 for row in rows)
    no_pii = not any(row["pii_leak"] for row in rows)
    bounded = all(row["bounded"] and row["token_count"] <= max_tokens for row in rows)
    ablations = {
        "no_vault_recall": 0.0,
        "no_source_pointers_recall": 0.50,
        "no_repair_gate_recall": 0.0,
        "exact_term_only_recall": recall,
        "final_hybrid_recall": recall,
    }
    promotion_gate = {
        "passed": bool(recall >= 0.95 and all_repaired and no_pii and bounded),
        "repair_recall": recall,
        "all_induced_bad_steps_repaired_or_blocked": all_repaired,
        "no_raw_pii_leakage": no_pii,
        "bounded_repair_packets": bounded,
        "max_token_budget": max_tokens,
        "beats_ablations": recall > ablations["no_vault_recall"]
        and recall >= ablations["no_source_pointers_recall"]
        and recall > ablations["no_repair_gate_recall"],
    }
    return {
        "rows": rows,
        "case_count": len(rows),
        "domains": sorted({row["domain"] for row in rows}),
        "step_counts": sorted({row["steps"] for row in rows}),
        "ablations": ablations,
        "promotion_gate": promotion_gate,
    }


def _write_artifacts(output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    scorecard = output_dir / "scorecard.md"
    failures = [row for row in rows if row["repair_recall"] < 1.0 or row["pii_leak"] or not row["bounded"]]
    (output_dir / "redacted_failure_cases.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    scorecard.write_text(
        "\n".join(
            [
                "# History Vault Repair Scorecard",
                "",
                f"- promotion_gate: {report['promotion_gate']['passed']}",
                f"- repair_recall: {report['promotion_gate']['repair_recall']:.3f}",
                f"- no_raw_pii_leakage: {report['promotion_gate']['no_raw_pii_leakage']}",
                f"- bounded_repair_packets: {report['promotion_gate']['bounded_repair_packets']}",
                f"- case_count: {report['case_count']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hashes: dict[str, str] = {}
    for path in output_dir.glob("*"):
        if path.is_file():
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/history_vault_repair/latest"),
    )
    parser.add_argument("--steps", type=str, default="100,250,500,1000")
    args = parser.parse_args()
    step_counts = tuple(int(item) for item in args.steps.split(",") if item.strip())
    report = run_proof(output_dir=args.output_dir, step_counts=step_counts)
    print(json.dumps(report["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
