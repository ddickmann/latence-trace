from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_bench_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_veracier_rag_validation.py"
    spec = importlib.util.spec_from_file_location("bench_veracier_rag_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench_module()


def _base_pack() -> dict:
    return {
        "use_case_id": "FIN-01",
        "question": "L'administration fiscale demande la documentation prix de transfert.",
        "role": "Directeur Financier",
        "selected_documents": [
            {
                "source_id": "FIN-01:DOC-1",
                "filename": "fiscal/prix_transfert/master_file.pdf",
                "classification": "TP_DOC",
                "language": "fr",
                "format": "searchable",
                "description": "Group Master File per OECD guidelines",
                "text": (
                    "Les principales transactions intra-groupe comprennent les prestations "
                    "de services, les licences de PI et la fabrication sous contrat. "
                    "Le taux de redevance de 2% du chiffre d'affaires net est coherent "
                    "avec les comparables identifies."
                ),
            }
        ],
    }


def _valid_payload() -> dict:
    return {
        "use_case_id": "FIN-01",
        "variants": [
            {
                "mutation_type": "perfect",
                "expected_band": "green",
                "expected_groundedness_range": {"min": 0.8, "max": 1.0},
                "response_text": (
                    '"Les principales transactions intra-groupe comprennent les prestations '
                    'de services, les licences de PI et la fabrication sous contrat." '
                    '"Le taux de redevance de 2% du chiffre d\'affaires net est coherent '
                    'avec les comparables identifies."'
                ),
                "claims": [
                    {
                        "claim_text": "Les transactions incluent services, licences et fabrication.",
                        "expected_label": "supported",
                        "source_doc_ids": ["FIN-01:DOC-1"],
                        "source_filenames": ["fiscal/prix_transfert/master_file.pdf"],
                        "rationale": "Copied from evidence.",
                    },
                    {
                        "claim_text": "Le taux de redevance mentionne est 2%.",
                        "expected_label": "supported",
                        "source_doc_ids": ["FIN-01:DOC-1"],
                        "source_filenames": ["fiscal/prix_transfert/master_file.pdf"],
                        "rationale": "Copied from evidence.",
                    },
                ],
                "coverage_notes": "Two supported finance facts.",
                "utilization_notes": "Uses exact evidence quotes.",
            },
            {
                "mutation_type": "ambiguous",
                "expected_band": "amber",
                "expected_groundedness_range": {"min": 0.55, "max": 0.74},
                "response_text": (
                    "La documentation décrit des transactions intra-groupe, mais ne permet pas "
                    "d'établir la couverture complète par exercice."
                ),
                "claims": [
                    {
                        "claim_text": "La documentation décrit des transactions intra-groupe.",
                        "expected_label": "supported",
                        "source_doc_ids": ["FIN-01:DOC-1"],
                        "source_filenames": ["fiscal/prix_transfert/master_file.pdf"],
                        "rationale": "Visible in evidence.",
                    },
                    {
                        "claim_text": "La couverture complète par exercice reste incertaine.",
                        "expected_label": "ambiguous",
                        "source_doc_ids": [],
                        "source_filenames": [],
                        "rationale": "Scope is not established.",
                    },
                ],
                "coverage_notes": "One fact plus one limitation.",
                "utilization_notes": "Intentionally inconclusive.",
            },
            {
                "mutation_type": "wrong",
                "expected_band": "red",
                "expected_groundedness_range": {"min": 0.0, "max": 0.45},
                "response_text": (
                    "Le groupe applique une redevance de 9% et a depose une documentation "
                    "complete pour dix exercices."
                ),
                "claims": [
                    {
                        "claim_text": "Le taux de redevance est 9%.",
                        "expected_label": "unsupported",
                        "source_doc_ids": [],
                        "source_filenames": [],
                        "rationale": "Mutated value.",
                    },
                    {
                        "claim_text": "Dix exercices complets ont ete deposes.",
                        "expected_label": "unsupported",
                        "source_doc_ids": [],
                        "source_filenames": [],
                        "rationale": "Not in evidence.",
                    },
                ],
                "coverage_notes": "Unsupported finance claims.",
                "utilization_notes": "Value and period are mutated.",
            },
        ],
    }


def test_infers_finance_archetype_from_document_metadata() -> None:
    pack = _base_pack()
    assert bench._infer_case_archetype(pack)["id"] == "finance_tax"


def test_validation_accepts_domain_aware_payload() -> None:
    pack = _base_pack()
    pack["raw_context"] = bench._raw_context_from_pack(pack)
    errors = bench._validate_generated(
        "FIN-01",
        _valid_payload(),
        pack=pack,
        allowed_source_ids={"FIN-01:DOC-1"},
    )
    assert errors == []


def test_validation_rejects_perfect_missing_evidence_caveat() -> None:
    pack = _base_pack()
    pack["raw_context"] = bench._raw_context_from_pack(pack)
    payload = _valid_payload()
    payload["variants"][0]["response_text"] += (
        " Aucune information ne précise explicitement une exigence sur cinq exercices."
    )
    errors = bench._validate_generated(
        "FIN-01",
        payload,
        pack=pack,
        allowed_source_ids={"FIN-01:DOC-1"},
    )
    assert "perfect variant contains an unresolved/missing-evidence caveat" in errors


def _ambiguous_variant_template() -> dict:
    return {
        "example_id": "FIN-01:ambiguous",
        "question_id": "FIN-01",
        "mutation_type": "ambiguous",
        "expected_band": "amber",
        "expected_groundedness_range": {"min": 0.55, "max": 0.74},
        "response_text": (
            "La documentation décrit des transactions intra-groupe, mais ne permet pas "
            "d'établir la couverture complète par exercice."
        ),
        "claims": [
            {
                "claim_text": "La documentation décrit des transactions intra-groupe.",
                "expected_label": "supported",
                "source_doc_ids": ["FIN-01:DOC-1"],
                "source_filenames": ["fiscal/prix_transfert/master_file.pdf"],
                "rationale": "Visible in evidence.",
            },
            {
                "claim_text": "La couverture complète par exercice reste incertaine.",
                "expected_label": "ambiguous",
                "source_doc_ids": [],
                "source_filenames": [],
                "rationale": "Scope is not established.",
            },
        ],
        "coverage_notes": "One fact plus one limitation.",
        "utilization_notes": "Intentionally inconclusive.",
    }


def test_refine_loop_accepts_first_pass_amber() -> None:
    variants = [_ambiguous_variant_template()]

    def score_fn(_variant: dict) -> dict:
        return {"band": "amber", "score": 0.63}

    def regenerate_fn(_variant: dict, _feedback: dict, _attempt: int) -> dict | None:
        raise AssertionError("regenerate should not be called when already amber")

    summary = bench._refine_ambiguous_loop(
        variants,
        score_fn=score_fn,
        regenerate_fn=regenerate_fn,
        max_attempts=4,
    )
    stats = summary["stats"]
    assert stats == {
        "total_ambiguous": 1,
        "accepted_first_pass": 1,
        "regenerated_accepted": 0,
        "unstable_excluded": 0,
        "sandbox_calls": 1,
    }
    assert variants[0]["ambiguous_unstable"] is False
    assert variants[0]["refined_sandbox_band"] == "amber"
    assert variants[0]["refined_sandbox_score"] == 0.63


def test_refine_loop_regenerates_when_green_then_accepts() -> None:
    variants = [_ambiguous_variant_template()]
    scores = iter([
        {"band": "green", "score": 0.82},
        {"band": "amber", "score": 0.66},
    ])

    def score_fn(_variant: dict) -> dict:
        return next(scores)

    regen_calls: list[int] = []

    def regenerate_fn(_variant: dict, feedback: dict, attempt: int) -> dict | None:
        regen_calls.append(attempt)
        assert feedback["band"] == "green"
        return {
            "response_text": (
                "Les services intra-groupe sont décrits, mais la couverture globale "
                "n'est pas établie avec certitude."
            ),
            "claims": _ambiguous_variant_template()["claims"],
            "coverage_notes": "Refined hedge.",
            "utilization_notes": "Refined hedge.",
        }

    summary = bench._refine_ambiguous_loop(
        variants,
        score_fn=score_fn,
        regenerate_fn=regenerate_fn,
        max_attempts=4,
    )
    stats = summary["stats"]
    assert regen_calls == [1]
    assert stats["accepted_first_pass"] == 0
    assert stats["regenerated_accepted"] == 1
    assert stats["unstable_excluded"] == 0
    assert stats["sandbox_calls"] == 2
    assert variants[0]["ambiguous_unstable"] is False
    assert variants[0]["refined_sandbox_band"] == "amber"


def test_refine_loop_marks_unstable_after_max_attempts() -> None:
    variants = [_ambiguous_variant_template()]

    def score_fn(_variant: dict) -> dict:
        return {"band": "red", "score": 0.32}

    def regenerate_fn(_variant: dict, _feedback: dict, _attempt: int) -> dict | None:
        return {
            "response_text": variants[0]["response_text"],
            "claims": variants[0]["claims"],
            "coverage_notes": variants[0]["coverage_notes"],
            "utilization_notes": variants[0]["utilization_notes"],
        }

    summary = bench._refine_ambiguous_loop(
        variants,
        score_fn=score_fn,
        regenerate_fn=regenerate_fn,
        max_attempts=2,
    )
    stats = summary["stats"]
    assert stats["total_ambiguous"] == 1
    assert stats["accepted_first_pass"] == 0
    assert stats["regenerated_accepted"] == 0
    assert stats["unstable_excluded"] == 1
    assert stats["sandbox_calls"] == 3
    assert variants[0]["ambiguous_unstable"] is True
    assert variants[0]["refined_sandbox_band"] == "red"


def test_refine_loop_stops_when_regeneration_fails() -> None:
    variants = [_ambiguous_variant_template()]

    def score_fn(_variant: dict) -> dict:
        return {"band": "green", "score": 0.81}

    def regenerate_fn(_variant: dict, _feedback: dict, _attempt: int) -> dict | None:
        return None

    summary = bench._refine_ambiguous_loop(
        variants,
        score_fn=score_fn,
        regenerate_fn=regenerate_fn,
        max_attempts=3,
    )
    stats = summary["stats"]
    assert stats["sandbox_calls"] == 1
    assert stats["unstable_excluded"] == 1
    assert variants[0]["ambiguous_unstable"] is True


def test_validation_rejects_grounded_wrong_claims() -> None:
    pack = _base_pack()
    pack["raw_context"] = bench._raw_context_from_pack(pack)
    payload = _valid_payload()
    payload["variants"][2]["claims"][0]["source_doc_ids"] = ["FIN-01:DOC-1"]
    errors = bench._validate_generated(
        "FIN-01",
        payload,
        pack=pack,
        allowed_source_ids={"FIN-01:DOC-1"},
    )
    assert "wrong unsupported claims must not include source ids or filenames" in errors
