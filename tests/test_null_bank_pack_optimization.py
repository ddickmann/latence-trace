"""PA3 hot-path optimization regression tests.

Verifies that the cached, pre-stacked ``_NullBankPack`` fast path inside
``compute_null_distribution`` returns numerically identical results to
the legacy per-bank-entry loop. Drift here would silently corrupt the
calibrated groundedness score that downstream NLI / risk-band logic
relies on, so the equivalence guard is worth its own test file.
"""

from __future__ import annotations

import torch

from latence_trace.core.groundedness import (
    _build_null_bank_pack,
    compute_null_distribution,
)


def _seeded_bank(seed: int = 42, *, entries: int = 8, max_tokens: int = 24, hidden: int = 64):
    g = torch.Generator().manual_seed(seed)
    bank = []
    for i in range(entries):
        token_count = int(torch.randint(1, max_tokens + 1, (1,), generator=g).item())
        bank.append(torch.randn(token_count, hidden, generator=g))
    return bank


def _seeded_response(seed: int = 1234, *, tokens: int = 17, hidden: int = 64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, hidden, generator=g)


def test_pack_path_matches_legacy_loop_for_typical_bank():
    bank = _seeded_bank()
    response = _seeded_response()

    legacy_mean, legacy_std, legacy_size = compute_null_distribution(response, bank)
    pack = _build_null_bank_pack(bank)
    assert pack is not None
    fast_mean, fast_std, fast_size = compute_null_distribution(response, pack)

    assert legacy_size == fast_size == len(bank)
    assert torch.allclose(legacy_mean, fast_mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(legacy_std, fast_std, atol=1e-5, rtol=1e-5)


def test_pack_path_handles_empty_bank_entries():
    bank = _seeded_bank(seed=7, entries=4)
    bank[1] = torch.empty(0, bank[0].shape[1])
    bank[3] = torch.empty(0, bank[0].shape[1])
    response = _seeded_response(seed=99)

    legacy_mean, legacy_std, legacy_size = compute_null_distribution(response, bank)
    pack = _build_null_bank_pack(bank)
    fast_mean, fast_std, fast_size = compute_null_distribution(response, pack)

    assert legacy_size == fast_size == 2
    assert torch.allclose(legacy_mean, fast_mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(legacy_std, fast_std, atol=1e-5, rtol=1e-5)


def test_pack_path_handles_all_empty_bank():
    response = _seeded_response()
    empty_bank = [torch.empty(0, 64), torch.empty(0, 64)]

    legacy_mean, legacy_std, legacy_size = compute_null_distribution(response, empty_bank)
    pack = _build_null_bank_pack(empty_bank)
    assert pack is None

    fast_mean, fast_std, fast_size = compute_null_distribution(response, [])

    assert legacy_size == fast_size == 0
    assert torch.equal(legacy_mean, fast_mean)
    assert torch.equal(legacy_std, fast_std)


def test_pack_path_handles_single_bank_entry_std_floor():
    bank = [torch.randn(12, 32)]
    response = _seeded_response(tokens=9, hidden=32)

    legacy_mean, legacy_std, legacy_size = compute_null_distribution(response, bank)
    pack = _build_null_bank_pack(bank)
    fast_mean, fast_std, fast_size = compute_null_distribution(response, pack)

    assert legacy_size == fast_size == 1
    assert torch.allclose(legacy_mean, fast_mean, atol=1e-5, rtol=1e-5)
    # Single-entry bank cannot estimate variance, so both paths must
    # collapse to the calibration floor; we only require they agree.
    assert torch.allclose(legacy_std, fast_std, atol=1e-5, rtol=1e-5)
