"""Unit tests for the class-weighted + pair-aware samplers."""

from __future__ import annotations

import random

from research.triangular_maxsim.student_v2.training import (
    ClassWeightedSampler,
    PairAwareBatchSampler,
    per_class_weights,
)


class TestPerClassWeights:
    def test_balanced_input_gives_equal_weights(self) -> None:
        weights = per_class_weights(["a", "b", "c"] * 10)
        # All equal (and normalised so mean ~ 1).
        vals = list(weights.values())
        assert max(vals) - min(vals) < 1e-6
        assert abs(sum(vals) / len(vals) - 1.0) < 1e-6

    def test_rare_class_upweighted(self) -> None:
        class_keys = ["big"] * 100 + ["small"] * 5
        weights = per_class_weights(class_keys)
        assert weights["small"] > weights["big"]

    def test_mean_normalised_to_one(self) -> None:
        weights = per_class_weights(["a"] * 100 + ["b"] * 10 + ["c"])
        vals = list(weights.values())
        assert abs(sum(vals) / len(vals) - 1.0) < 1e-6


class TestClassWeightedSampler:
    def test_deterministic_with_fixed_rng(self) -> None:
        keys = ["a"] * 5 + ["b"] * 5
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        s1 = ClassWeightedSampler(keys, num_samples=20, rng=rng1)
        s2 = ClassWeightedSampler(keys, num_samples=20, rng=rng2)
        assert list(s1) == list(s2)

    def test_respects_num_samples(self) -> None:
        keys = ["a", "b", "c"] * 4
        sampler = ClassWeightedSampler(keys, num_samples=7, rng=random.Random(0))
        assert len(list(sampler)) == 7

    def test_balances_toward_underrepresented_class(self) -> None:
        keys = ["big"] * 100 + ["small"] * 5
        rng = random.Random(1)
        sampler = ClassWeightedSampler(keys, num_samples=1000, rng=rng)
        samples = list(sampler)
        small_count = sum(1 for i in samples if keys[i] == "small")
        # With inverse-frequency weighting, "small" class should
        # represent meaningfully more than 5% of samples.
        frac = small_count / len(samples)
        assert frac > 0.20


class TestPairAwareBatchSampler:
    def _mk_pair_ids(self, n_pairs: int, per_pair: int = 2) -> list[str]:
        out: list[str] = []
        for i in range(n_pairs):
            for _ in range(per_pair):
                out.append(f"pair_{i}")
        return out

    def test_batches_keep_pair_clusters_together(self) -> None:
        ids = self._mk_pair_ids(16, per_pair=2)  # 32 rows in 16 pairs
        sampler = PairAwareBatchSampler(
            ids, batch_size=8, min_pairs_per_batch=2, rng=random.Random(3),
        )
        batches = list(sampler)
        assert batches
        for batch in batches:
            # Each batch contains at least min_pairs_per_batch pair_ids
            # with at least 2 rows sharing that pair.
            from collections import Counter
            counts = Counter(ids[i] for i in batch)
            pair_heavy = sum(1 for v in counts.values() if v >= 2)
            assert pair_heavy >= 2

    def test_orphans_skipped_when_no_cluster(self) -> None:
        """Rows whose pair_id has only one representative are never
        yielded by the current policy (pair-heavy batches only).
        """
        ids = ["orphan_a", "orphan_b", "orphan_c"]
        sampler = PairAwareBatchSampler(
            ids, batch_size=4, min_pairs_per_batch=1, rng=random.Random(0),
        )
        batches = list(sampler)
        assert batches == []

    def test_deterministic_with_fixed_rng(self) -> None:
        ids = self._mk_pair_ids(8, per_pair=2)
        b1 = list(PairAwareBatchSampler(ids, batch_size=4, rng=random.Random(7)))
        b2 = list(PairAwareBatchSampler(ids, batch_size=4, rng=random.Random(7)))
        assert b1 == b2
