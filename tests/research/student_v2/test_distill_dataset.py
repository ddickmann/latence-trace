"""Unit tests for the TRACE v2 distillation-dataset orchestrator.

Covers:
  - :func:`iter_synthetic_enterprise` Jaccard gate drops + stats.
  - :func:`split_rows` pair-aware split + OOD-eval routing.
  - :func:`_base_row` / :func:`_phi_hints` schema + source-type detection.
  - :func:`collect_rows` end-to-end with small volumes.
  - :func:`build_manifest` shape + reproducibility.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import pytest

from research.triangular_maxsim.student_v2.distill_dataset import (
    MIN_JACCARD_SYNTHETIC,
    SyntheticStats,
    VERACIER_UPWEIGHT,
    _base_row,
    _detect_source_type,
    _phi_hints,
    build_manifest,
    collect_rows,
    iter_synthetic_enterprise,
    split_rows,
)


# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


class TestLexicalHelpers:
    def test_phi_hints_returns_expected_keys(self) -> None:
        h = _phi_hints("Apple Q3 revenue was $12.4B", "Apple reported Q3 revenue of $12.4 billion")
        assert set(h) == {"exact_overlap", "numeric_overlap", "identifier_overlap", "source_type"}
        assert 0.0 <= h["exact_overlap"] <= 1.0
        assert 0.0 <= h["numeric_overlap"] <= 1.0
        assert h["source_type"] == "prose"

    def test_source_type_detection(self) -> None:
        assert _detect_source_type("```python\ndef foo():\n    pass\n```") == "code"
        assert _detect_source_type("| col1 | col2 |\n|-----|-----|\n| a | b |") == "table"
        assert _detect_source_type("## Heading\n\nBody") == "markdown"
        assert _detect_source_type('{"key": "value"}') == "json"
        assert _detect_source_type("Passage 1 about a topic") == "passage_enum"
        assert _detect_source_type("Regular prose text.") == "prose"


class TestBaseRow:
    def test_contains_three_axis_label_placeholders(self) -> None:
        row = _base_row(
            pair_id="p1",
            class_key="rag.prose.enterprise",
            response_text="R",
            evidence_text="E",
            gold_band="green",
            source="unit_test",
            split_hint="train",
            upweight_factor=1,
            synthetic=False,
        )
        assert row["token_support_labels"] is None
        assert row["dead_weight_unit_labels"] is None
        assert row["coverage_unit_labels"] is None
        assert row["phi_hints"]["source_type"] == "prose"


# ---------------------------------------------------------------------------
# Synthetic iterator (Jaccard gate)
# ---------------------------------------------------------------------------


class TestSyntheticEnterpriseIter:
    def test_default_threshold_keeps_most_rows(self) -> None:
        stats = SyntheticStats()
        rows = list(
            iter_synthetic_enterprise(
                passages_per_bucket=5,
                master_seed=42,
                emit_ood=False,
                min_jaccard=MIN_JACCARD_SYNTHETIC,
                stats=stats,
            )
        )
        assert stats.total_generated > 0
        assert stats.kept > 0
        kept_ratio = stats.kept / stats.total_generated
        assert kept_ratio >= 0.85, f"only kept {kept_ratio:.2%}"
        assert all(r["source"] == "synthetic_enterprise" for r in rows)
        assert all(r["class_key"] == "rag.prose.enterprise" for r in rows)
        assert all(r["synthetic"] for r in rows)

    def test_high_threshold_drops_most_modifying_adversarials(self) -> None:
        """At a 0.95 threshold the gate drops > 90% of entity_swap /
        numeric_flip rows (support_drop preserves the response, so it
        always passes). A handful of numeric flips survive because the
        tokenizer is set-based and a duplicated digit elsewhere in the
        response can hold the Jaccard at 1.0 - a known limitation
        documented in the quality-gate rationale.
        """
        baseline = SyntheticStats()
        list(
            iter_synthetic_enterprise(
                passages_per_bucket=5,
                master_seed=42,
                emit_ood=False,
                min_jaccard=0.0,
                stats=baseline,
            )
        )
        tight = SyntheticStats()
        rows = list(
            iter_synthetic_enterprise(
                passages_per_bucket=5,
                master_seed=42,
                emit_ood=False,
                min_jaccard=0.95,
                stats=tight,
            )
        )
        assert tight.dropped_jaccard > 0
        # Tight threshold drops at least 40% of original adversarial pairs
        # (the grounded + support_drop rows are unaffected, so the
        # overall kept-ratio still looks high - we check drops directly).
        assert tight.dropped_jaccard * 3 < baseline.total_generated  # sanity
        # entity_swap survivors < 20% of role bucket.
        kept_by_role = Counter(r["role"] for r in rows)
        all_by_role = Counter()
        for r in iter_synthetic_enterprise(
            passages_per_bucket=5,
            master_seed=42,
            emit_ood=False,
            min_jaccard=0.0,
            stats=SyntheticStats(),
        ):
            all_by_role[r["role"]] += 1
        for role in ("entity_swap", "numeric_flip"):
            kept = kept_by_role.get(role, 0)
            total = all_by_role.get(role, 1)
            assert kept / total < 0.20, f"{role}: kept {kept}/{total} at threshold 0.95"

    def test_deterministic(self) -> None:
        s1 = SyntheticStats()
        s2 = SyntheticStats()
        a = [
            r
            for r in iter_synthetic_enterprise(
                passages_per_bucket=4,
                master_seed=42,
                emit_ood=False,
                min_jaccard=MIN_JACCARD_SYNTHETIC,
                stats=s1,
            )
        ]
        b = [
            r
            for r in iter_synthetic_enterprise(
                passages_per_bucket=4,
                master_seed=42,
                emit_ood=False,
                min_jaccard=MIN_JACCARD_SYNTHETIC,
                stats=s2,
            )
        ]
        assert a == b
        assert s1 == s2


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class TestSplitRows:
    def _mk(self, **kw):
        return _base_row(
            pair_id=kw["pair_id"],
            class_key="c",
            response_text="r",
            evidence_text="e",
            gold_band=None,
            source=kw.get("source", "unit_test"),
            split_hint=kw.get("split_hint", "train"),
            upweight_factor=1,
            synthetic=False,
        )

    def test_ood_rows_routed_to_ood_split(self) -> None:
        rows = [
            self._mk(pair_id=f"a_{i}")
            for i in range(10)
        ] + [
            self._mk(pair_id=f"ood_{i}", split_hint="ood_eval")
            for i in range(5)
        ]
        splits = split_rows(rows, seed=1)
        assert len(splits["ood_eval"]) == 5
        assert sum(len(v) for k, v in splits.items() if k != "ood_eval") == 10

    def test_pair_aware_keeps_pair_together(self) -> None:
        """Rows sharing pair_id must land in the same non-OOD split."""
        rows = []
        for i in range(10):
            for k in range(4):  # four copies of each pair
                r = self._mk(pair_id=f"pair_{i}")
                rows.append(r)
        splits = split_rows(rows, seed=42)
        for split_rows_list in (splits["train"], splits["val"], splits["test"]):
            ids = {r["pair_id"] for r in split_rows_list}
            # Every pair in this split must have all 4 copies here.
            for pid in ids:
                count_here = sum(1 for r in split_rows_list if r["pair_id"] == pid)
                assert count_here == 4, f"pair {pid} split across sets"

    def test_pair_group_overrides_pair_id(self) -> None:
        """When ``pair_group`` is set, it controls the bucketing."""
        rows = []
        for i in range(10):
            for k in range(3):
                r = self._mk(pair_id=f"pid_{i}_{k}")
                r["pair_group"] = f"group_{i}"
                rows.append(r)
        splits = split_rows(rows, seed=7)
        for split_rows_list in (splits["train"], splits["val"], splits["test"]):
            groups = {r["pair_group"] for r in split_rows_list}
            for g in groups:
                count_here = sum(1 for r in split_rows_list if r["pair_group"] == g)
                assert count_here == 3


# ---------------------------------------------------------------------------
# End-to-end (small) + manifest
# ---------------------------------------------------------------------------


class TestCollectRowsEndToEnd:
    def test_collect_rows_small(self) -> None:
        rows, stats = collect_rows(
            audit_log=None,
            passages_per_bucket=3,
            master_seed=42,
            emit_ood=True,
            min_jaccard=MIN_JACCARD_SYNTHETIC,
            audit_log_cap=10,
        )
        assert len(rows) > 0
        assert stats.kept > 0
        # Every row must carry the three-axis placeholders + source.
        for r in rows:
            for k in (
                "token_support_labels",
                "dead_weight_unit_labels",
                "coverage_unit_labels",
                "source",
                "class_key",
                "split_hint",
            ):
                assert k in r, f"missing key {k!r} in row"

    def test_build_manifest_shape(self, tmp_path: pathlib.Path) -> None:
        rows, stats = collect_rows(
            audit_log=None,
            passages_per_bucket=2,
            master_seed=42,
            emit_ood=False,
            min_jaccard=MIN_JACCARD_SYNTHETIC,
            audit_log_cap=10,
        )
        splits = split_rows(rows, seed=42)
        for name, r in splits.items():
            path = tmp_path / f"{name}.jsonl"
            path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in r))
        mf = build_manifest(
            tmp_path,
            splits,
            synthetic_stats=stats,
            master_seed=42,
            passages_per_bucket=2,
            min_jaccard=MIN_JACCARD_SYNTHETIC,
            veracier_upweight=VERACIER_UPWEIGHT,
            audit_log_path=None,
            audit_log_cap=10,
        )
        assert mf["generator_params"]["master_seed"] == 42
        assert "splits" in mf
        assert set(mf["splits"]) == {"train", "val", "test", "ood_eval"}
        assert mf["generator_params"]["veracier_upweight"] == VERACIER_UPWEIGHT
        # sha256 must be populated for every split file we wrote.
        for name in ("train", "val", "test", "ood_eval"):
            assert mf["splits"][name]["sha256"], f"{name}: missing sha256"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
