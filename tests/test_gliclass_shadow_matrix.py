from __future__ import annotations

from argparse import Namespace

from scripts.bench_gliclass_shadow_matrix import (
    FRESH_HARD_WAVE,
    LexicalSmokeScorer,
    build_diagnostics,
    candidate_action,
    reduce_shadow_features,
    run,
    score_case,
    split_chunks,
    sweep_thresholds,
    _repair_t5_encoder_embeddings,
)
from scripts.customer_breaker_smoke import VERTICAL_PILOT_CASES


def _case(case_id: str):
    return next(case for case in VERTICAL_PILOT_CASES + FRESH_HARD_WAVE if case.case_id == case_id)


def test_shadow_matrix_flags_structured_numeric_amount_swap() -> None:
    row = score_case(
        _case("pilot_finance_structured_amount_swap"),
        scorer=LexicalSmokeScorer(),
        matrix_modes=("pair-classification", "label-as-claim", "nli-single", "rerank-single"),
        max_chunks=8,
        max_sentences=8,
    )

    assert row["matrices"]["pair_classification"]
    assert row["matrices"]["nli_single_support"]
    assert row["matrices"]["rerank_single_support"]
    assert row["features"]["gliclass_numeric_unit_error_max"] == 1.0
    assert row["diagnostics"][0]["response_text"]
    assert candidate_action(
        row["features"],
        threshold=type(
            "Threshold",
            (),
            {"support_min": 0.55, "error_max": 0.65, "unsupported_rate": 0.75},
        )(),
    ) == "auto_repair"


def test_shadow_matrix_keeps_supported_pharmacy_answer_allow_capable() -> None:
    row = score_case(
        _case("fresh_pharmacy_supported_side_effect"),
        scorer=LexicalSmokeScorer(),
        matrix_modes=("nli-single", "rerank-single"),
        max_chunks=8,
        max_sentences=8,
    )

    assert row["features"]["gliclass_support_min"] >= 0.55
    assert row["features"]["gliclass_numeric_unit_error_max"] == 0.0
    assert candidate_action(
        row["features"],
        threshold=type(
            "Threshold",
            (),
            {"support_min": 0.55, "error_max": 0.65, "unsupported_rate": 0.75},
        )(),
    ) == "allow"


def test_reduce_shadow_features_combines_pair_and_label_as_claim_support() -> None:
    features = reduce_shadow_features(
        pair_matrix=[
            [
                {
                    "fully_supported": 0.2,
                    "partially_supported": 0.4,
                    "contradicts_evidence": 0.0,
                    "adds_unsupported_fact": 0.1,
                    "numeric_or_unit_error": 0.0,
                }
            ]
        ],
        label_as_claim_matrix=[[0.91]],
        rule_matrix=[],
    )

    assert features["gliclass_support_min"] == 0.91
    assert features["per_response"][0]["gliclass_top_context_index"] == 0


def test_threshold_sweep_reports_no_production_wiring() -> None:
    args = Namespace(
        scorer="lexical-smoke",
        model="unused",
        device="cpu",
        suite="fresh_hard_wave",
        matrix_mode="nli-single,rerank-single,pair-classification,label-as-claim,rule-following",
        max_chunks=8,
        max_sentences=8,
        batch_size=4,
        threshold_sweep=True,
        output=None,
    )

    report = run(args)

    assert report["purpose"] == "gliclass_shadow_matrix_no_production_wiring"
    assert report["threshold_sweep"]["best"] is not None
    assert report["summary"]["production_wiring_allowed"] is False
    assert report["summary"]["case_count"] == len(FRESH_HARD_WAVE)


def test_chunk_splitting_and_diagnostics_are_auditable() -> None:
    response_chunks = split_chunks("First claim. Second claim.", max_chunks=4, prefer_sentences=True)
    context_chunks = split_chunks("Evidence one.\n\nEvidence two.", max_chunks=4, prefer_sentences=False)
    features = {
        "per_response": [
            {
                "response_index": 0,
                "gliclass_support_max": 0.8,
                "gliclass_contradiction_max": 0.0,
                "gliclass_numeric_unit_error_max": 0.0,
                "gliclass_extra_claim_max": 0.0,
                "gliclass_top_context_index": 1,
            }
        ]
    }

    diagnostics = build_diagnostics(response_chunks, context_chunks, features)

    assert diagnostics[0]["response_text"] == "First claim."
    assert diagnostics[0]["top_context_text"] == "Evidence two."


def test_sweep_thresholds_counts_false_allows_and_blocks() -> None:
    rows = [
        {
            "case_id": "safe",
            "expected_actions": ["allow"],
            "features": {
                "gliclass_support_min": 0.9,
                "gliclass_unsupported_sentence_rate": 0.0,
                "gliclass_contradiction_max": 0.0,
                "gliclass_numeric_unit_error_max": 0.0,
                "gliclass_extra_claim_max": 0.0,
            },
        },
        {
            "case_id": "bad",
            "expected_actions": ["auto_repair", "block"],
            "features": {
                "gliclass_support_min": 0.9,
                "gliclass_unsupported_sentence_rate": 0.0,
                "gliclass_contradiction_max": 0.9,
                "gliclass_numeric_unit_error_max": 0.0,
                "gliclass_extra_claim_max": 0.0,
            },
        },
    ]

    sweep = sweep_thresholds(rows)

    assert sweep["best"]["passed"] == 2
    assert sweep["best"]["false_allows"] == 0
    assert sweep["best"]["false_blocks"] == 0


def test_repair_t5_encoder_embeddings_ties_shared_weight() -> None:
    import torch

    class _Embed:
        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.randn(2, 3))

    class _Encoder:
        def __init__(self) -> None:
            self.embed_tokens = _Embed()

    class _EncoderModel:
        def __init__(self) -> None:
            self.shared = _Embed()
            self.encoder = _Encoder()

    class _Inner:
        def __init__(self) -> None:
            self.encoder_model = _EncoderModel()

    class _Model:
        def __init__(self) -> None:
            self.model = _Inner()

    model = _Model()
    before = model.model.encoder_model.encoder.embed_tokens.weight.data_ptr()
    shared = model.model.encoder_model.shared.weight.data_ptr()

    assert before != shared
    assert _repair_t5_encoder_embeddings(model) is True
    assert model.model.encoder_model.encoder.embed_tokens.weight.data_ptr() == shared
    assert _repair_t5_encoder_embeddings(model) is False
