"""Regression tests for the shared file-attribution module.

Locks in three things:

1. The module lives at ``latence_trace.core.attribution.file_attribution``.
2. The old import path ``latence_trace.core.code_lane.file_attribution``
   is a back-compat re-export pointing at the same functions / classes.
3. ``attribute_files`` populates a ``reason_code_histogram`` and
   ``resolve_attribution_key`` prefers ``path`` -> ``metadata.path``
   -> ``metadata.source`` -> ``source_id`` -> ``support_id``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from latence_trace.core.attribution import (
    attribute_files as shared_attribute_files,
)
from latence_trace.core.attribution.file_attribution import (
    FileAttributionResult,
    ReasonCode,
    attribute_files,
    resolve_attribution_key,
)


def test_codelane_import_is_a_reexport() -> None:
    """``code_lane.file_attribution`` must point at the shared kernel."""

    from latence_trace.core.code_lane.file_attribution import (
        FileAttributionResult as LegacyResult,
        ReasonCode as LegacyReasonCode,
        attribute_files as legacy_attribute_files,
        resolve_attribution_key as legacy_resolve,
    )

    assert LegacyResult is FileAttributionResult
    assert LegacyReasonCode is ReasonCode
    assert legacy_attribute_files is attribute_files
    assert legacy_attribute_files is shared_attribute_files
    assert legacy_resolve is resolve_attribution_key


def test_resolve_attribution_key_prefers_path_over_metadata() -> None:
    unit = SimpleNamespace(
        support_id="support-0",
        path="/src/app.py",
        metadata={"path": "/ignored.py", "source": "ignored", "source_id": "ignored"},
    )
    assert resolve_attribution_key(unit) == "/src/app.py"


def test_resolve_attribution_key_falls_back_through_metadata_chain() -> None:
    without_path = SimpleNamespace(
        support_id="support-1",
        metadata={"source": "docs/README.md"},
    )
    assert resolve_attribution_key(without_path) == "docs/README.md"

    only_source_id = SimpleNamespace(
        support_id="support-2",
        metadata={"source_id": "doc-42"},
    )
    assert resolve_attribution_key(only_source_id) == "doc-42"

    bare = SimpleNamespace(support_id="support-bare")
    assert resolve_attribution_key(bare) == "support-bare"


def _unit(idx: int, path: str) -> SimpleNamespace:
    return SimpleNamespace(
        support_id=f"support-{idx}",
        path=path,
        metadata={},
        offset_start=0,
        offset_end=100,
    )


def test_attribute_files_emits_reason_code_histogram() -> None:
    units = [
        _unit(0, "/src/a.py"),
        _unit(1, "/src/b.py"),
        _unit(2, "/src/c.py"),
    ]
    # All owner tokens sink into a.py. b.py / c.py are dead weight.
    result = attribute_files(
        units=units,
        per_unit_max=[0.9, 0.2, 0.1],
        per_unit_owner_count=[8, 0, 0],
        per_unit_query_owner_count=[0, 0, 0],
        usage_states=["used", "unused", "unused"],
        n_response_tokens=8,
        n_query_tokens=0,
    )
    assert isinstance(result, FileAttributionResult)
    assert result.n_files == 3
    assert result.dead_weight_ratio == pytest.approx(2 / 3)
    assert set(result.dead_weight_files) == {"/src/b.py", "/src/c.py"}
    # Reason code histogram is always populated (no flag) and the
    # counts are the dataclass field counts aggregated across files.
    histo = result.reason_code_histogram
    assert histo[ReasonCode.NEVER_WON_ARGMAX.value] == 2
    # Both dead files should also land in ALL_TOKENS_BELOW_0_40 (<0.4 max cosine).
    assert histo[ReasonCode.ALL_TOKENS_BELOW_0_40.value] == 2
    # dominated_by_single_file fires only when the top_owner_tokens
    # reaches >= resp_denom // 2 (= 4). With 8 tokens in a.py, both
    # dead peers should be marked as dominated.
    assert histo[ReasonCode.DOMINATED_BY_SINGLE_FILE.value] == 2
    # Neither peer has query ownership either -> not_query_relevant.
    assert histo[ReasonCode.NOT_QUERY_RELEVANT.value] == 2


def test_attribute_files_dict_roundtrip_includes_histogram() -> None:
    units = [_unit(0, "x.py"), _unit(1, "y.py")]
    result = attribute_files(
        units=units,
        per_unit_max=[0.8, 0.1],
        per_unit_owner_count=[5, 0],
        per_unit_query_owner_count=[0, 0],
        usage_states=["used", "unused"],
        n_response_tokens=5,
        n_query_tokens=0,
    )
    payload = result.as_dict()
    assert "reason_code_histogram" in payload
    assert payload["reason_code_histogram"][ReasonCode.NEVER_WON_ARGMAX.value] == 1


def test_attribute_files_mismatched_lengths_raises() -> None:
    units = [_unit(0, "x.py"), _unit(1, "y.py")]
    with pytest.raises(ValueError):
        attribute_files(
            units=units,
            per_unit_max=[0.8],
            per_unit_owner_count=[1],
            per_unit_query_owner_count=[0],
            usage_states=["used"],
            n_response_tokens=1,
            n_query_tokens=0,
        )
