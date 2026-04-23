"""Loader for the transcripts_v2 case bank.

The v2 bank is the scaled-out version of ``transcripts_v1``: 60 base
scenarios mined from real Cursor agent sessions, each expanded into three
variants (``correct`` auto-extracted from the transcript, ``wrong`` and
``ambiguous`` embedded in ``cases_transcripts_v2.yaml``).

This module is a thin wrapper around
``transcript_cases.load_transcript_cases`` that points at the v2 YAML and
re-labels the source/metadata as ``transcript_v2:``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.triangular_maxsim.coding.code_cases import CodingCase
from research.triangular_maxsim.coding.transcript_cases import (
    load_transcript_cases as _load_v1_style,
    summarize_loaded_cases as _summarize,
)

logger = logging.getLogger(__name__)

_DEFAULT_YAML_V2 = Path(__file__).resolve().parent / "cases_transcripts_v2.yaml"


def load_transcript_cases_v2(
    yaml_path: Optional[Path] = None,
    *,
    transcript_root: Optional[Path] = None,
) -> List[CodingCase]:
    """Load the transcripts_v2 bank. Returns a list of ``CodingCase``.

    Re-labels ``source`` and ``metadata['schema_version']`` so downstream
    benchmarking can easily distinguish v1 from v2 rows.
    """
    target = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML_V2
    cases = _load_v1_style(yaml_path=target, transcript_root=transcript_root)
    relabelled: List[CodingCase] = []
    for case in cases:
        base_id = str(case.metadata.get("base_scenario_id") or "")
        new_source = f"transcript_v2:{base_id}" if base_id else case.source.replace("transcript_v1", "transcript_v2")
        new_meta: Dict[str, Any] = dict(case.metadata)
        new_meta["schema_version"] = "transcripts_v2"
        relabelled.append(
            CodingCase(
                id=case.id,
                source=new_source,
                query=case.query,
                response=case.response,
                label=case.label,
                subcategory=case.subcategory,
                notes=case.notes,
                context_files=case.context_files,
                metadata=new_meta,
            )
        )
    return relabelled


def summarize_v2_cases(cases: List[CodingCase]) -> Dict[str, Any]:
    return _summarize(cases)


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cases = load_transcript_cases_v2()
    summary = summarize_v2_cases(cases)
    print(json.dumps(summary, indent=2))
