"""Targeted non-public data generation for TRACE root-cause heads.

The generated rows are synthetic challenge cases for training and debugging
class-specific heads. They are not public benchmark rows and should not be used
as the sole source for customer-facing claims.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "research/triangular_maxsim/student_v2/root_cause_targeted/targeted_v1.jsonl"


def _row(
    *,
    row_id: str,
    class_key: str,
    gold_band: str,
    response: str,
    evidence: str,
    root_causes: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    grounded = gold_band == "green"
    numeric_match = _numeric_match(response, evidence)
    literal_coverage = _literal_coverage(response, evidence)
    identifier_coverage = _identifier_coverage(response, evidence)
    atom_match = 1.0 if grounded else min(numeric_match, literal_coverage, identifier_coverage)
    score = 0.92 if grounded else 0.35 + 0.20 * atom_match
    return {
        "schema": "trace_targeted_root_cause_row.v1",
        "row_id": row_id,
        "class_key": class_key,
        "split": _split(row_id),
        "gold_band": gold_band,
        "gold_binary": 1 if grounded else 0,
        "claim": response,
        "evidence": evidence,
        "root_causes": root_causes,
        "features": {
            "v1_score": score,
            "v1_nli_aggregate": 0.75 if grounded else -0.35,
            "primary_score": score,
            "reverse_context": score,
            "groundedness_v2": score,
            "literal_guarded": literal_coverage,
            "literal_mismatch_count": 0.0 if grounded else 1.0,
            "literal_match_count": float(round(literal_coverage * 10)),
            "literal_total_count": 10.0,
            "context_coverage_ratio": 1.0 if grounded else atom_match,
            "context_unused_ratio": 0.0 if grounded else 1.0 - atom_match,
            "context_uncertain_ratio": 0.05 if grounded else 0.35,
            "dead_weight_ratio": 0.05 if grounded else 0.45,
            "support_units_total": 1.0,
            "support_unit_label_mean": 1.0 if grounded else 0.0,
            "dead_weight_label_mean": 0.0 if grounded else 1.0,
            "coverage_label_mean": 1.0 if grounded else atom_match,
            "response_len_log": 1.0,
            "evidence_len_log": 1.0,
            "token_mean": score,
            "token_min": min(score, atom_match),
            "token_bottom10": min(score, atom_match),
            "token_saturation_rate": 1.0 if score >= 0.95 else 0.0,
            "calibrated_mean": score,
            "nli_token_mean": 0.75 if grounded else -0.35,
            "nli_token_available": 1.0,
            "literal_coverage": literal_coverage,
            "numeric_coverage": numeric_match,
            "identifier_coverage": identifier_coverage,
            "numeric_count": float(_numeric_count(response)),
            "identifier_count": float(_identifier_count(response)),
            "atom_match": atom_match,
            "claim_count": float(meta.get("claim_count", 1)),
            "unsupported_claim_fraction": float(meta.get("unsupported_claim_fraction", 0.0)),
            "schema_match": float(meta.get("schema_match", 1.0 if grounded else 0.0)),
            "api_symbol_match": float(meta.get("api_symbol_match", 1.0 if grounded else 0.0)),
            "trajectory_order_match": float(meta.get("trajectory_order_match", 1.0 if grounded else 0.0)),
            "test_outcome_match": float(meta.get("test_outcome_match", 1.0 if grounded else 0.0)),
        },
        "metadata": meta,
    }


def _split(row_id: str) -> str:
    # Deterministic disjoint split without hashing imports.
    bucket = sum(ord(ch) for ch in row_id) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "val"
    return "test"


def _terms(text: str) -> list[str]:
    return [part.strip(".,:;()[]{}'\"`").lower() for part in text.split() if part.strip()]


def _literal_coverage(response: str, evidence: str) -> float:
    terms = [term for term in _terms(response) if len(term) > 2]
    if not terms:
        return 0.0
    evidence_terms = set(_terms(evidence))
    return sum(1 for term in terms if term in evidence_terms) / len(terms)


def _numbers(text: str) -> list[str]:
    out: list[str] = []
    for token in _terms(text):
        if token.replace(".", "", 1).isdigit():
            out.append(token)
    return out


def _numeric_count(text: str) -> int:
    return len(_numbers(text))


def _numeric_match(response: str, evidence: str) -> float:
    nums = _numbers(response)
    if not nums:
        return 1.0
    evidence_nums = set(_numbers(evidence))
    return sum(1 for num in nums if num in evidence_nums) / len(nums)


def _identifiers(text: str) -> list[str]:
    out: list[str] = []
    for token in _terms(text):
        if "_" in token or "." in token or any(ch.isupper() for ch in token[1:]):
            out.append(token.lower())
    return out


def _identifier_count(text: str) -> int:
    return len(_identifiers(text))


def _identifier_coverage(response: str, evidence: str) -> float:
    ids = _identifiers(response)
    if not ids:
        return 1.0
    evidence_ids = set(_identifiers(evidence))
    return sum(1 for ident in ids if ident in evidence_ids) / len(ids)


def generate(seed: int = 17, per_family: int = 120) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    rows.extend(_factoid_rows(rng, per_family))
    rows.extend(_multi_claim_rows(rng, per_family))
    rows.extend(_structured_rows(rng, per_family))
    rows.extend(_code_context_rows(rng, per_family))
    rows.extend(_trajectory_rows(rng, per_family))
    return rows


def _factoid_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    entities = [
        ("AuroraDB", "2019", "Frankfurt", "42"),
        ("Helio CRM", "2021", "Toronto", "18"),
        ("Nimbus API", "2023", "Oslo", "64"),
        ("VectorPay", "2020", "Lisbon", "27"),
    ]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        name, year, city, count = rng.choice(entities)
        wrong_year = str(int(year) + rng.choice([1, 2, 3]))
        wrong_count = str(int(count) + rng.choice([5, 9, 11]))
        grounded = i % 2 == 0
        evidence = f"{name} launched in {year} in {city} with {count} enterprise connectors."
        response = evidence if grounded else f"{name} launched in {wrong_year} in {city} with {wrong_count} enterprise connectors."
        rows.append(
            _row(
                row_id=f"targeted_factoid_{i}",
                class_key="rag.prose.short_factoid",
                gold_band="green" if grounded else "red",
                response=response,
                evidence=evidence,
                root_causes=["short_answer_overlap", "domain_morphology_numeric", "false_allow"],
                meta={"family": "factoid_atom_swap"},
            )
        )
    return rows


def _multi_claim_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        product = rng.choice(["Atlas", "Beacon", "Cobalt", "Delta"])
        claims = [
            f"{product} supports SSO",
            f"{product} stores audit logs for 365 days",
            f"{product} encrypts data at rest",
        ]
        false_claim = f"{product} has SOC 2 Type II certification"
        partial = i % 3 == 0
        red = i % 3 == 1
        evidence = ". ".join(claims) + "."
        if partial:
            response = ". ".join(claims[:2] + [false_claim]) + "."
            band = "amber"
            unsupported = 1 / 3
        elif red:
            response = ". ".join([false_claim, f"{product} is HIPAA certified"]) + "."
            band = "red"
            unsupported = 1.0
        else:
            response = evidence
            band = "green"
            unsupported = 0.0
        rows.append(
            _row(
                row_id=f"targeted_multi_claim_{i}",
                class_key="rag.prose.multi_claim",
                gold_band=band,
                response=response,
                evidence=evidence,
                root_causes=["multi_claim_aggregation", "partial_support_collapse"],
                meta={"family": "multi_claim_partial", "claim_count": 3, "unsupported_claim_fraction": unsupported},
            )
        )
    return rows


def _structured_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        account = rng.choice(["A-104", "B-220", "C-731", "D-510"])
        amount = rng.choice(["1200", "2450", "3900", "5100"])
        status = rng.choice(["approved", "pending", "rejected"])
        wrong_amount = str(int(amount) + 100)
        grounded = i % 2 == 0
        evidence = f"row account_id={account}; amount_usd={amount}; status={status}; region=EU"
        response = (
            f"Account {account} has status {status} and amount USD {amount}."
            if grounded
            else f"Account {account} has status {status} and amount USD {wrong_amount}."
        )
        row = _row(
            row_id=f"targeted_structured_{i}",
            class_key="rag.structured",
            gold_band="green" if grounded else "red",
            response=response,
            evidence=evidence,
            root_causes=["structured_cell_alignment", "domain_morphology_numeric"],
            meta={"family": "structured_cell_swap", "schema_match": 1.0 if grounded else 0.5},
        )
        # Structured rows intentionally keep v1 similarity ambiguous: the
        # answer overlaps strongly with the row even when a single cell is
        # wrong, and exact cell features must carry the decision.
        ambiguous_v1 = 0.78 if grounded else 0.76
        for key in ("v1_score", "primary_score", "reverse_context", "groundedness_v2", "token_mean", "calibrated_mean"):
            row["features"][key] = ambiguous_v1
        row["features"]["token_bottom10"] = 0.72 if grounded else 0.68
        row["features"]["schema_match"] = 1.0 if grounded else 0.0
        row["features"]["atom_match"] = 1.0 if grounded else 0.0
        rows.append(row)
    return rows


def _code_context_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        fn = rng.choice(["fetch_user", "load_invoice", "sync_project", "parse_event"])
        real_api = rng.choice(["AsyncClient", "RetryPolicy", "ProjectStore", "EventParser"])
        fake_api = real_api + "V9"
        grounded = i % 2 == 0
        evidence = f"def {fn}(client: {real_api}) -> dict: return client.get('/v1/{fn}')"
        response = (
            f"Use `{real_api}` inside `{fn}` and call `client.get`."
            if grounded
            else f"Use `{fake_api}` inside `{fn}` and call `client.magic_get`."
        )
        rows.append(
            _row(
                row_id=f"targeted_code_context_{i}",
                class_key="rag.code_in_context",
                gold_band="green" if grounded else "red",
                response=response,
                evidence=evidence,
                root_causes=["code_identifier_morphology", "domain_morphology_identifier"],
                meta={"family": "code_identifier_api_drift", "api_symbol_match": 1.0 if grounded else 0.0},
            )
        )
    return rows


def _trajectory_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        file_name = rng.choice(["auth.py", "billing.py", "router.ts", "worker.go"])
        grounded = i % 2 == 0
        evidence = f"turn 1 edited {file_name}; tests passed; exported function validate_token exists"
        response = (
            f"The change in {file_name} uses validate_token and the tests passed."
            if grounded
            else f"The change in settings.py uses validate_session_v9 and the tests passed."
        )
        rows.append(
            _row(
                row_id=f"targeted_trajectory_{i}",
                class_key="code.agentic_trace",
                gold_band="green" if grounded else "red",
                response=response,
                evidence=evidence,
                root_causes=["trajectory_ranker_failure", "code_identifier_morphology"],
                meta={
                    "family": "agentic_trajectory_drift",
                    "trajectory_order_match": 1.0 if grounded else 0.0,
                    "test_outcome_match": 1.0 if grounded else 0.0,
                    "api_symbol_match": 1.0 if grounded else 0.0,
                },
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-family", type=int, default=120)
    args = parser.parse_args()
    rows = generate(seed=args.seed, per_family=args.per_family)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(out)
    print(len(rows))


if __name__ == "__main__":
    main()
