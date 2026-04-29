"""Class-weighted + pair-aware batch sampling.

Two primitives:

* :class:`ClassWeightedSampler` - draws row indices with per-row weight
  ``w_c`` where ``c`` is the row's class. Used in Stage 1 distillation
  when pairs are not required.

* :class:`PairAwareBatchSampler` - yields batches where every
  ``pair_id`` appears with BOTH its grounded and hallucinated row
  present, so the margin-ranking loss has signal. Used in Stages 2-3.

Neither sampler calls torch; both return plain iterables of indices
suitable for ``torch.utils.data.Sampler`` / ``BatchSampler`` protocols.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, Sequence


def per_class_weights(
    class_keys: Sequence[str],
    *,
    smoothing: float = 0.1,
) -> dict[str, float]:
    """Inverse-frequency weighting with Laplace smoothing.

    Returns a weight per distinct class such that small classes get
    boosted (Veracier doesn't drown under HaluEval). Smoothing prevents
    extreme up-weighting when a class has only a handful of rows.
    """
    counts: dict[str, int] = defaultdict(int)
    for c in class_keys:
        counts[c or ""] += 1
    N = sum(counts.values())
    K = max(1, len(counts))
    weights: dict[str, float] = {}
    for c, n in counts.items():
        weights[c] = (N / K) / (n + smoothing * N / K)
    # Normalise so weights average to 1.
    mean = sum(weights.values()) / K
    return {c: w / mean for c, w in weights.items()}


class ClassWeightedSampler:
    """Index sampler that draws rows with per-class weights.

    For deterministic behaviour, pass an ``rng`` (``random.Random``).
    The sampler yields ``num_samples`` indices in random order; call
    again for a new epoch.
    """

    def __init__(
        self,
        class_keys: Sequence[str],
        *,
        num_samples: int | None = None,
        rng: random.Random | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.class_keys = list(class_keys)
        self.num_samples = num_samples or len(self.class_keys)
        self.rng = rng or random.Random(42)
        self.weights = weights or per_class_weights(self.class_keys)
        self._per_row_weights = [
            self.weights.get(c or "", 1.0) for c in self.class_keys
        ]

    def __iter__(self) -> Iterator[int]:
        n = len(self.class_keys)
        if n == 0:
            return iter(())
        choices = self.rng.choices(
            range(n), weights=self._per_row_weights, k=self.num_samples,
        )
        return iter(choices)

    def __len__(self) -> int:
        return self.num_samples


class PairAwareBatchSampler:
    """Yield batches where grounded + hallucinated pairs are co-located.

    A ``pair_id`` groups a set of rows (typically one grounded and one
    or more adversarial variants). To supply signal to the margin
    ranking loss, each batch should contain a "cluster" of size >= 2
    for at least ``min_pairs_per_batch`` pair_ids.

    Algorithm:
      * Group row indices by ``pair_id``.
      * At the start of each epoch, shuffle the pair_ids.
      * Greedily fill batches by adding whole pair clusters until the
        batch reaches ``batch_size`` (or exceeds it; we cap by taking
        the first ``batch_size`` indices of the over-filled batch).
      * Ensure ``>= min_pairs_per_batch`` pair_ids have at least 2 rows
        in each batch; if not, skip undersized batches.

    This sampler deliberately ignores class weighting - it is used
    in Stages 2-3 where gold-label + pair-ranking dominate; class
    balance is assumed to have been baked in during Stage 1.
    """

    def __init__(
        self,
        pair_ids: Sequence[str],
        *,
        batch_size: int = 32,
        min_pairs_per_batch: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self.pair_ids = list(pair_ids)
        self.batch_size = batch_size
        self.min_pairs_per_batch = min_pairs_per_batch
        self.rng = rng or random.Random(42)
        self._pair_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, pid in enumerate(self.pair_ids):
            self._pair_to_indices[pid or ""].append(i)
        # Orphan rows (singleton pair_ids): stored separately, used to
        # fill the tail end of a batch.
        self._clusters: list[list[int]] = [
            v for v in self._pair_to_indices.values() if len(v) >= 2
        ]
        self._orphans: list[int] = [
            v[0] for v in self._pair_to_indices.values() if len(v) == 1
        ]

    def __iter__(self) -> Iterator[list[int]]:
        clusters = [list(c) for c in self._clusters]
        self.rng.shuffle(clusters)

        # Policy:
        #   1. Keep adding whole clusters until we've accumulated AT
        #      LEAST ``min_pairs_per_batch`` clusters. Only then do we
        #      enforce the ``batch_size`` cap.
        #   2. When we would exceed ``batch_size`` after adding another
        #      cluster AND ``pairs_in_batch >= min_pairs_per_batch``,
        #      yield the current batch (trimmed to batch_size) and
        #      start a fresh one with the new cluster.
        #   3. At stream end, yield the tail iff it meets the pair
        #      floor.
        #
        # This gives consistent batch sizes on average and never drops
        # a cluster just because it would push us over batch_size - we
        # simply trim to ``batch_size`` at yield time.
        batch: list[int] = []
        pairs_in_batch = 0
        for cluster in clusters:
            # If we haven't hit the pair floor yet, absorb the cluster
            # unconditionally.
            if pairs_in_batch < self.min_pairs_per_batch:
                batch.extend(cluster)
                pairs_in_batch += 1
                continue

            # We have enough pairs; check the size cap before adding.
            if len(batch) + len(cluster) > self.batch_size:
                yield batch[: self.batch_size]
                batch = list(cluster)
                pairs_in_batch = 1
            else:
                batch.extend(cluster)
                pairs_in_batch += 1

        if batch and pairs_in_batch >= self.min_pairs_per_batch:
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        # Upper bound on number of batches (over-approx).
        total_paired = sum(len(c) for c in self._clusters)
        if self.batch_size == 0:
            return 0
        return max(0, total_paired // self.batch_size)


__all__ = [
    "ClassWeightedSampler",
    "PairAwareBatchSampler",
    "per_class_weights",
]
