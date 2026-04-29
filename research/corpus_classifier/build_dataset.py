"""Build the corpus-type classifier train / test parquet and manifest.

Assembles one labelled row per source example across six classes:

* ``rag.prose.enterprise``   - Veracier curated variants
* ``rag.prose.short_factoid``- HaluEval QA
* ``rag.prose.multi_claim``  - RAGTruth QA + HaluEval Summ + RAGTruth Summ
* ``rag.structured``         - RAGTruth Data2Text
* ``rag.code_in_context``    - transcripts_v2 + transcripts_v1 turns whose
                                response has no fenced code block with >= 5
                                non-blank lines (prose-about-code), plus the
                                handcrafted ``code_cases`` prose fixtures
* ``code.agentic_trace``     - transcripts_v2 + transcripts_v1 turns whose
                                response has a fenced code block with >= 5
                                non-blank lines, plus handcrafted fenced
                                cases

Outputs:

* ``data/corpus_classifier/train.parquet``  (80/20 stratified, seed=42)
* ``data/corpus_classifier/test.parquet``
* ``data/corpus_classifier/manifest.json``  (row counts, SHAs, augmentation
  notes)

The script is deterministic: same inputs produce identical parquet shards
and identical SHA256 digests in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXTERNAL = REPO_ROOT / "research/triangular_maxsim/external_data"
VERACIER_CURATED = (
    REPO_ROOT
    / "data/veracier-industries/proof_bundle_v1/variants.curated.jsonl"
)
VERACIER_VARIANTS = (
    REPO_ROOT / "data/veracier-industries/proof_bundle_v1/variants.jsonl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data/corpus_classifier"

SEED = 42
TRAIN_FRACTION = 0.8
MIN_TRAIN_ROWS_PER_CLASS = 120
FENCED_CODE_LINE_THRESHOLD = 5
MAX_CAP_PER_CLASS = 2000

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fenced_code_nonblank_lines(response: str) -> int:
    """Return the max non-blank line count over all fenced code blocks."""
    best = 0
    for m in FENCE_RE.finditer(response or ""):
        body = m.group(1)
        nonblank = sum(1 for ln in body.splitlines() if ln.strip())
        if nonblank > best:
            best = nonblank
    return best


@dataclass
class Row:
    row_id: str
    class_key: str
    query: str
    response: str
    raw_context: str
    is_grounded: Optional[bool]
    metadata: Dict[str, Any]


def iter_veracier() -> Iterator[Row]:
    for pth in (VERACIER_CURATED, VERACIER_VARIANTS):
        if not pth.exists():
            continue
        with pth.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                expected_band = str(rec.get("expected_band", "")).lower()
                is_grounded = expected_band == "green" if expected_band in {"green", "red"} else None
                yield Row(
                    row_id=f"veracier:{pth.stem}:{i}:{rec.get('example_id','')}",
                    class_key="rag.prose.enterprise",
                    query=str(rec.get("query_text") or rec.get("question_id") or ""),
                    response=str(rec.get("response_text") or ""),
                    raw_context=str(rec.get("raw_context") or ""),
                    is_grounded=is_grounded,
                    metadata={
                        "expected_band": expected_band,
                        "mutation_type": rec.get("mutation_type"),
                        "generation_archetype": rec.get("generation_archetype"),
                        "verticals": rec.get("verticals"),
                        "source_file": pth.name,
                    },
                )


def iter_halueval_qa() -> Iterator[Row]:
    path = EXTERNAL / "HaluEval/data/qa_data.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = rec.get("question", "")
            ctx = rec.get("knowledge", "")
            right = rec.get("right_answer")
            wrong = rec.get("hallucinated_answer")
            if right:
                yield Row(
                    row_id=f"halueval_qa:{i}:right",
                    class_key="rag.prose.short_factoid",
                    query=q,
                    response=right,
                    raw_context=ctx,
                    is_grounded=True,
                    metadata={"source": "halueval_qa"},
                )
            if wrong:
                yield Row(
                    row_id=f"halueval_qa:{i}:halluc",
                    class_key="rag.prose.short_factoid",
                    query=q,
                    response=wrong,
                    raw_context=ctx,
                    is_grounded=False,
                    metadata={"source": "halueval_qa"},
                )


def iter_halueval_summ() -> Iterator[Row]:
    path = EXTERNAL / "HaluEval/data/summarization_data.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            document = rec.get("document", "")
            right = rec.get("right_summary") or rec.get("summary")
            wrong = rec.get("hallucinated_summary")
            q = "Summarise the article."
            if right:
                yield Row(
                    row_id=f"halueval_summ:{i}:right",
                    class_key="rag.prose.multi_claim",
                    query=q,
                    response=right,
                    raw_context=document,
                    is_grounded=True,
                    metadata={"source": "halueval_summ"},
                )
            if wrong:
                yield Row(
                    row_id=f"halueval_summ:{i}:halluc",
                    class_key="rag.prose.multi_claim",
                    query=q,
                    response=wrong,
                    raw_context=document,
                    is_grounded=False,
                    metadata={"source": "halueval_summ"},
                )


def _ragtruth_raw_sources() -> Dict[str, Dict[str, Any]]:
    """Load the canonical ``source_info.jsonl`` once, keyed by ``source_id``.

    The ``voyager_layout/data2text/test.jsonl`` shard ships with empty
    ``source_info`` strings (the tabular evidence lives in the raw
    dataset). We join back here so the data2text class has real context
    to score against.
    """
    cache_key = "_ragtruth_raw_source_info"
    cache = _ragtruth_raw_sources.__dict__.setdefault(cache_key, None)
    if cache is not None:
        return cache
    path = EXTERNAL / "RAGTruth/dataset/source_info.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("source_id")
                if sid is not None:
                    out[str(sid)] = rec
    _ragtruth_raw_sources.__dict__[cache_key] = out
    return out


def iter_ragtruth(task: str, class_key: str) -> Iterator[Row]:
    path = EXTERNAL / f"RAGTruth/voyager_layout/{task}/test.jsonl"
    if not path.exists():
        return
    raw_sources = _ragtruth_raw_sources()
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels = rec.get("labels") or []
            is_grounded = not bool(labels)
            source_info = rec.get("source_info")
            # Fall back to the raw dataset's source_info when the
            # voyager shard ships an empty string (known for data2text).
            if not source_info:
                sid = rec.get("source_id")
                if sid is not None:
                    raw = raw_sources.get(str(sid), {})
                    source_info = raw.get("source_info")
            if isinstance(source_info, dict):
                raw_context = json.dumps(source_info, ensure_ascii=False)
                query = str(rec.get("query") or source_info.get("question") or "")
            else:
                raw_context = str(source_info or "")
                query = str(rec.get("query") or "")
            yield Row(
                row_id=f"ragtruth_{task}:{i}",
                class_key=class_key,
                query=query,
                response=str(rec.get("response", "")),
                raw_context=raw_context,
                is_grounded=is_grounded,
                metadata={"source": f"ragtruth_{task}", "n_label_spans": len(labels)},
            )


def _pack_code_context(context_files: Iterable[Any], *, max_files: int = 32, max_chars: int = 60_000) -> str:
    """Match the convention used by bench_transcripts_v2 so classifier features
    see the same ``=== path ===`` delimiters the runtime will see in production."""
    chunks: List[str] = []
    used = 0
    for f in list(context_files or [])[:max_files]:
        header = f"=== {getattr(f, 'path', '')} ===\n"
        body = getattr(f, "content", "") or ""
        blob = header + body + "\n"
        if used + len(blob) > max_chars:
            remaining = max_chars - used - len(header) - 8
            if remaining > 256:
                blob = header + body[:remaining] + "\n...[truncated]...\n"
                chunks.append(blob)
            break
        chunks.append(blob)
        used += len(blob)
    return "".join(chunks)


def iter_transcripts(transcript_version: str) -> Iterator[Row]:
    from research.triangular_maxsim.coding.transcript_cases_v2 import (  # noqa: E402
        load_transcript_cases_v2,
    )
    from research.triangular_maxsim.coding.transcript_cases import (  # noqa: E402
        load_transcript_cases as _load_v1,
    )
    if transcript_version == "v2":
        cases = load_transcript_cases_v2()
        prefix = "transcripts_v2"
    elif transcript_version == "v1":
        cases = _load_v1()
        prefix = "transcripts_v1"
    else:
        raise ValueError(f"unknown transcript version: {transcript_version}")
    for c in cases:
        resp = c.response or ""
        n_code_lines = _fenced_code_nonblank_lines(resp)
        class_key = (
            "code.agentic_trace"
            if n_code_lines >= FENCED_CODE_LINE_THRESHOLD
            else "rag.code_in_context"
        )
        raw_context = _pack_code_context(c.context_files)
        label = str(c.label or "").lower()
        is_grounded: Optional[bool]
        if label in {"grounded", "correct"}:
            is_grounded = True
        elif label in {"ungrounded", "wrong", "hallucinated"}:
            is_grounded = False
        else:
            is_grounded = None
        yield Row(
            row_id=f"{prefix}:{c.id}",
            class_key=class_key,
            query=str(c.query or ""),
            response=resp,
            raw_context=raw_context,
            is_grounded=is_grounded,
            metadata={
                "source": prefix,
                "subcategory": c.subcategory,
                "base_scenario_id": c.metadata.get("base_scenario_id") if c.metadata else None,
                "fenced_code_nonblank_lines": n_code_lines,
            },
        )


def iter_handcrafted_code() -> Iterator[Row]:
    from research.triangular_maxsim.coding.code_cases import HANDCRAFTED_CASES  # noqa: E402
    for c in HANDCRAFTED_CASES:
        resp = c.response or ""
        n_code_lines = _fenced_code_nonblank_lines(resp)
        # Diff-style responses with no fence but many code-looking lines
        # would otherwise be misclassified. Treat any response containing
        # a ``diff --git`` header or >= 5 lines of indented code as
        # agentic_trace even without a fence.
        if n_code_lines < FENCED_CODE_LINE_THRESHOLD:
            code_hints = 0
            for ln in resp.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if s.startswith(("+ ", "- ", "+\t", "-\t", "@@", "diff --git")) or (
                    ln.startswith(("    ", "\t")) and any(t in s for t in ("def ", "class ", "return ", "=", "(", "{"))
                ):
                    code_hints += 1
            if code_hints >= FENCED_CODE_LINE_THRESHOLD:
                n_code_lines = code_hints
        class_key = (
            "code.agentic_trace"
            if n_code_lines >= FENCED_CODE_LINE_THRESHOLD
            else "rag.code_in_context"
        )
        raw_context = _pack_code_context(c.context_files)
        label = str(c.label or "").lower()
        is_grounded: Optional[bool]
        if label == "grounded":
            is_grounded = True
        elif label == "ungrounded":
            is_grounded = False
        else:
            is_grounded = None
        yield Row(
            row_id=f"handcrafted:{c.id}",
            class_key=class_key,
            query=str(c.query or ""),
            response=resp,
            raw_context=raw_context,
            is_grounded=is_grounded,
            metadata={
                "source": "handcrafted_code",
                "subcategory": c.subcategory,
                "fenced_code_nonblank_lines": n_code_lines,
            },
        )


def _collect_rows(max_halueval_qa: int, max_ragtruth_multi: int) -> Tuple[List[Row], Dict[str, Dict[str, Any]]]:
    rng = random.Random(SEED)
    per_source: Dict[str, List[Row]] = {}

    def _pack(name: str, rows: Iterable[Row]) -> None:
        per_source[name] = list(rows)

    _pack("veracier", iter_veracier())
    _pack("halueval_qa", iter_halueval_qa())
    _pack("halueval_summ", iter_halueval_summ())
    _pack("ragtruth_qa", iter_ragtruth("qa", "rag.prose.multi_claim"))
    _pack("ragtruth_summ", iter_ragtruth("summarization", "rag.prose.multi_claim"))
    _pack("ragtruth_data2text", iter_ragtruth("data2text", "rag.structured"))
    _pack("transcripts_v2", list(iter_transcripts("v2")))
    _pack("transcripts_v1", list(iter_transcripts("v1")))
    _pack("handcrafted_code", list(iter_handcrafted_code()))

    # Deterministically down-sample oversized sources to keep training
    # balanced; order-preserving shuffle keyed on row_id for reproducibility.
    def _cap(rows: List[Row], cap: int) -> List[Row]:
        if len(rows) <= cap:
            return rows
        rng.shuffle(rows)
        return rows[:cap]

    per_source["halueval_qa"] = _cap(per_source["halueval_qa"], max_halueval_qa)
    # Balance multi_claim: take full RAGTruth QA/Summ, then fill remainder
    # from HaluEval summ up to max_ragtruth_multi so all three sub-sources
    # are represented.
    rt_qa = per_source["ragtruth_qa"]
    rt_su = per_source["ragtruth_summ"]
    he_su = _cap(per_source.pop("halueval_summ"), max(0, max_ragtruth_multi - len(rt_qa) - len(rt_su)))
    per_source["halueval_summ_multi_claim"] = he_su

    # Source metadata (SHA256 of the on-disk file) for the manifest.
    source_meta: Dict[str, Dict[str, Any]] = {}
    manifest_paths: Dict[str, Optional[Path]] = {
        "veracier_curated": VERACIER_CURATED if VERACIER_CURATED.exists() else None,
        "veracier_variants": VERACIER_VARIANTS if VERACIER_VARIANTS.exists() else None,
        "halueval_qa": EXTERNAL / "HaluEval/data/qa_data.jsonl",
        "halueval_summ": EXTERNAL / "HaluEval/data/summarization_data.jsonl",
        "ragtruth_qa": EXTERNAL / "RAGTruth/voyager_layout/qa/test.jsonl",
        "ragtruth_summ": EXTERNAL / "RAGTruth/voyager_layout/summarization/test.jsonl",
        "ragtruth_data2text": EXTERNAL / "RAGTruth/voyager_layout/data2text/test.jsonl",
        "transcripts_v2_yaml": REPO_ROOT / "research/triangular_maxsim/coding/cases_transcripts_v2.yaml",
        "transcripts_v1_yaml": REPO_ROOT / "research/triangular_maxsim/coding/cases_transcripts_v1.yaml",
    }
    for name, pth in manifest_paths.items():
        if pth and pth.exists():
            source_meta[name] = {"path": str(pth.relative_to(REPO_ROOT)), "sha256": _sha256(pth), "size_bytes": pth.stat().st_size}
        else:
            source_meta[name] = {"path": None, "sha256": None, "size_bytes": None}

    all_rows: List[Row] = []
    for rows in per_source.values():
        all_rows.extend(rows)
    return all_rows, source_meta


def _stratified_split(rows: List[Row]) -> Tuple[List[Row], List[Row]]:
    """Deterministic 80/20 split within each class_key; seed=42."""
    rng = random.Random(SEED)
    by_class: Dict[str, List[Row]] = {}
    for r in rows:
        by_class.setdefault(r.class_key, []).append(r)
    train: List[Row] = []
    test: List[Row] = []
    for key in CLASS_KEYS:
        bucket = by_class.get(key, [])
        bucket.sort(key=lambda r: r.row_id)
        rng.shuffle(bucket)
        n_test = max(1, int(round(len(bucket) * (1.0 - TRAIN_FRACTION))))
        test.extend(bucket[:n_test])
        train.extend(bucket[n_test:])
    return train, test


def _apply_min_row_guard(
    train: List[Row], test: List[Row], source_meta: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    augmentations: Dict[str, Any] = {}
    by_class_train: Dict[str, List[Row]] = {}
    for r in train:
        by_class_train.setdefault(r.class_key, []).append(r)
    for key in CLASS_KEYS:
        present = len(by_class_train.get(key, []))
        if present >= MIN_TRAIN_ROWS_PER_CLASS:
            continue
        # Augmentation policy: up-sample the existing class rows with
        # deterministic jitter markers on row_id; train the LR with
        # ``class_weight="balanced"`` so upsampling mostly matters for
        # the featurizer's fit-time coverage of the shape space.
        shortfall = MIN_TRAIN_ROWS_PER_CLASS - present
        existing = list(by_class_train.get(key, []))
        if not existing:
            augmentations[key] = {"status": "no_rows_available", "shortfall": shortfall}
            continue
        copies: List[Row] = []
        i = 0
        while len(copies) < shortfall:
            r = existing[i % len(existing)]
            copies.append(
                Row(
                    row_id=f"{r.row_id}#dup{i // len(existing) + 1}",
                    class_key=r.class_key,
                    query=r.query,
                    response=r.response,
                    raw_context=r.raw_context,
                    is_grounded=r.is_grounded,
                    metadata={**r.metadata, "augmented": True, "augment_strategy": "deterministic_duplicate"},
                )
            )
            i += 1
        train.extend(copies)
        augmentations[key] = {
            "status": "augmented",
            "original_train_rows": present,
            "augmented_train_rows": present + shortfall,
            "strategy": "deterministic_duplicate",
            "reason": "transcripts_v2 shard smaller than 120-row guard; deterministic duplication keeps the featurizer exposed to the same shape distribution without introducing cross-class contamination.",
        }
    return augmentations


def _to_arrow(rows: List[Row]) -> pa.Table:
    return pa.table(
        {
            "row_id": [r.row_id for r in rows],
            "class_key": [r.class_key for r in rows],
            "query": [r.query for r in rows],
            "response": [r.response for r in rows],
            "raw_context": [r.raw_context for r in rows],
            "is_grounded": [r.is_grounded for r in rows],
            "metadata_json": [json.dumps(r.metadata, ensure_ascii=False, sort_keys=True) for r in rows],
        }
    )


def build(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_halueval_qa: int = MAX_CAP_PER_CLASS,
    max_ragtruth_multi: int = MAX_CAP_PER_CLASS,
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("collecting rows...")
    rows, source_meta = _collect_rows(max_halueval_qa=max_halueval_qa, max_ragtruth_multi=max_ragtruth_multi)
    logger.info("collected %d rows across classes", len(rows))
    train, test = _stratified_split(rows)
    augmentations = _apply_min_row_guard(train, test, source_meta)

    def _counts(rows_: List[Row]) -> Dict[str, int]:
        out: Dict[str, int] = {k: 0 for k in CLASS_KEYS}
        for r in rows_:
            out[r.class_key] = out.get(r.class_key, 0) + 1
        return out

    train_counts = _counts(train)
    test_counts = _counts(test)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    pq.write_table(_to_arrow(train), train_path)
    pq.write_table(_to_arrow(test), test_path)

    manifest = {
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "fenced_code_line_threshold": FENCED_CODE_LINE_THRESHOLD,
        "max_cap_per_class": MAX_CAP_PER_CLASS,
        "classes": list(CLASS_KEYS),
        "train_row_counts": train_counts,
        "test_row_counts": test_counts,
        "train_total": len(train),
        "test_total": len(test),
        "augmentations": augmentations,
        "sources": source_meta,
        "artefacts": {
            "train_parquet": {
                "path": str(train_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            "test_parquet": {
                "path": str(test_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(test_path),
                "size_bytes": test_path.stat().st_size,
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote %s (%d rows) and %s (%d rows)", train_path.name, len(train), test_path.name, len(test))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-halueval-qa", type=int, default=MAX_CAP_PER_CLASS)
    parser.add_argument("--max-ragtruth-multi", type=int, default=MAX_CAP_PER_CLASS)
    args = parser.parse_args()
    manifest = build(
        out_dir=Path(args.out_dir),
        max_halueval_qa=args.max_halueval_qa,
        max_ragtruth_multi=args.max_ragtruth_multi,
    )
    print(json.dumps({"train_counts": manifest["train_row_counts"], "test_counts": manifest["test_row_counts"], "augmentations": manifest["augmentations"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
