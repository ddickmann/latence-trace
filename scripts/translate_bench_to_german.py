"""Phase C.1 - Translate the English calibration bench to German.

Reads the existing per-class training rows under
``data/corpus_classifier/<split>.parquet``, takes a stratified balanced
sample per class (faithful + hallucinated, deterministic seed), and
translates ``query``, ``response`` and ``raw_context`` through GPT-4.1.
Output is written to::

    data/corpus_classifier/german_translation/<class_key>.jsonl

Each row keeps ``row_id``, ``class_key``, ``is_grounded`` and
``metadata`` exactly as in the source so that:

1. The cache script can later score the German rows with
   ``--language de`` and produce ``<class>.de.<split>.jsonl`` artefacts.
2. The calibration sweep maps perfectly back onto its objective
   (we never re-label a row -- a hallucinated English row stays a
   hallucinated German row).

Translation prompt is engineered to:

* Preserve identifiers, function names, code blocks, JSON keys, numbers,
  ISO dates, and named entities verbatim. GPT-4.1 is otherwise too
  helpful and "fixes" hallucinations on the way through.
* Translate context + answer in the same OpenAI call so the model sees
  both at once and any factual mismatch survives the trip.
* Refuse to silently invent text -- if a field comes back blank we drop
  the row rather than ship a degraded translation.

The script is resumable: it skips ``row_id``s already present in the
output file. ``--per-class N`` caps the rows scored per class (default
200, plan target 1200 across six classes).

Usage::

    OPENAI_API_KEY=... python scripts/translate_bench_to_german.py \\
        --per-class 200 --classes all --output-dir data/corpus_classifier/german_translation

The OpenAI client is intentionally a hard dependency: this script is
only run by an operator preparing the German release. Production
runtime never imports openai.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/corpus_classifier"
DEFAULT_OUTPUT_DIR = DATA_DIR / "german_translation"

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OpenAI / GPT-4.1 prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a high-precision German-language translator working on
a hallucination-detection benchmark. Translate the provided English
context, query, and answer into native, fluent German.

Hard rules (violations break the benchmark):

1. Translate ONLY natural-language prose. NEVER translate:
   - identifiers, function names, class names, variable names, file paths,
     URLs, email addresses, ASCII command-line examples;
   - JSON keys, struct field names, or anything inside backticks/triple
     backticks/<code> tags;
   - numbers, dates, currency amounts, units, and percent signs;
   - named entities (people, organisations, products, technical brand
     names) -- transliterate only when the German form is universally
     established (e.g. "United States" -> "Vereinigte Staaten").
2. Preserve any factual mismatches between context and answer. Your job
   is to render them in German, NOT to "correct" them. If the English
   answer says the wrong year or invents a fact, the German answer must
   say the same wrong year / invent the same fact.
3. The translated answer must be the same length category as the
   original (one sentence -> one sentence; multi-paragraph -> multi-
   paragraph). Do not add explanatory text, do not omit content.
4. Output STRICTLY valid JSON with the keys ``query_de``, ``response_de``,
   ``raw_context_de``. If a source field is empty, return an empty string
   for the matching ``_de`` field. NEVER return null.

Quality bar: an educated German speaker reading only your translation
must reach the same factual conclusion as a fluent English speaker
reading the original. Idiomatic but accurate."""

USER_TEMPLATE = """Source row to translate. Reproduce identifiers / code / numbers / \
dates / named entities verbatim.

QUERY (English):
{query}

CONTEXT (English):
{context}

ANSWER (English):
{answer}

Return ONLY a JSON object of the shape:
{{
  "query_de": "...",
  "response_de": "...",
  "raw_context_de": "..."
}}"""

DEFAULT_MODEL = "gpt-4.1"


# --------------------------------------------------------------------------- #
# Data loading + stratified sampling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceRow:
    row_id: str
    class_key: str
    split: str
    is_grounded: Optional[bool]
    query: str
    response: str
    raw_context: str
    metadata: Dict[str, Any]


def _load_parquet_rows(split: str) -> List[SourceRow]:
    import pyarrow.parquet as pq

    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise SystemExit(f"missing source parquet: {path}")
    table = pq.read_table(path)
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = table.num_rows
    out: List[SourceRow] = []
    for i in range(n):
        row = {k: cols[k][i] for k in cols}
        out.append(
            SourceRow(
                row_id=str(row["row_id"]),
                class_key=str(row["class_key"]),
                split=split,
                is_grounded=row.get("is_grounded"),
                query=str(row.get("query") or ""),
                response=str(row.get("response") or ""),
                raw_context=str(row.get("raw_context") or ""),
                metadata=json.loads(row.get("metadata_json") or "{}"),
            )
        )
    return out


def _stratified_sample(
    rows: Sequence[SourceRow], per_class: int, seed: int
) -> Dict[str, List[SourceRow]]:
    """Balanced grounded/hallucinated split, deterministic ordering.

    For binary classes (the prose + structured + code lanes) we pick
    50/50 from ``is_grounded=True`` and ``is_grounded=False`` so the
    German bundle's F1 sweep sees the same class balance the English
    sweep saw. For ``rag.prose.enterprise`` (Veracier) we keep all three
    bands proportional. Unlabelled rows are skipped.
    """

    rng = random.Random(seed)
    by_class: Dict[str, List[SourceRow]] = {k: [] for k in CLASS_KEYS}
    for row in rows:
        if row.class_key not in by_class:
            continue
        by_class[row.class_key].append(row)

    out: Dict[str, List[SourceRow]] = {}
    for class_key, bucket in by_class.items():
        labelled = [r for r in bucket if r.is_grounded is not None]
        unlabelled = [r for r in bucket if r.is_grounded is None]
        labelled.sort(key=lambda r: r.row_id)
        unlabelled.sort(key=lambda r: r.row_id)

        if class_key == "rag.prose.enterprise":
            # Veracier: keep grounded + amber + hallucinated proportions.
            # Implementation: shuffle deterministically and slice.
            rng.shuffle(labelled)
            rng.shuffle(unlabelled)
            merged = labelled + unlabelled
            picked = merged[:per_class]
        else:
            true_rows = [r for r in labelled if r.is_grounded is True]
            false_rows = [r for r in labelled if r.is_grounded is False]
            rng.shuffle(true_rows)
            rng.shuffle(false_rows)
            half = max(1, per_class // 2)
            picked = true_rows[:half] + false_rows[: per_class - half]
            # Top up with unlabelled rows if the per-class target is not
            # met (e.g. ``rag.code_in_context`` may have <50 hallucinated
            # rows). Order is deterministic per ``row_id``.
            if len(picked) < per_class and unlabelled:
                rng.shuffle(unlabelled)
                picked += unlabelled[: per_class - len(picked)]
        # Final deterministic shuffle so the output JSONL ordering is
        # not biased toward the grounded class.
        rng.shuffle(picked)
        out[class_key] = picked
    return out


# --------------------------------------------------------------------------- #
# Translation transport
# --------------------------------------------------------------------------- #


class _Translator:
    """Thin wrapper around the OpenAI Chat Completions client.

    Kept generic so a future swap to Anthropic / Mistral is one method
    rewrite. The shape of ``translate(row) -> dict`` is what callers
    rely on.
    """

    def __init__(self, *, model: str, api_key: Optional[str], timeout: float):
        self.model = model
        self.timeout = timeout
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - operator-only script
            raise SystemExit(
                "openai package is required for translation; "
                "install with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=api_key)

    def translate(self, row: SourceRow) -> Dict[str, str]:
        prompt = USER_TEMPLATE.format(
            query=row.query or "(none)",
            context=row.raw_context or "(none)",
            answer=row.response or "(none)",
        )
        # response_format=json_object forces strict JSON, so a
        # single ``json.loads`` round-trips even when the model wants to
        # add prose around the JSON.
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=self.timeout,
        )
        text = completion.choices[0].message.content or "{}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON from translator: {exc}") from exc
        # Defensive: only accept the three expected keys.
        return {
            "query_de": str(payload.get("query_de") or ""),
            "response_de": str(payload.get("response_de") or ""),
            "raw_context_de": str(payload.get("raw_context_de") or ""),
        }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def _load_existing(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add(rec["row_id"])
            except Exception:
                continue
    return seen


def _write_row(path: Path, row: SourceRow, translation: Dict[str, str]) -> None:
    rec = {
        "row_id": row.row_id,
        "class_key": row.class_key,
        "split": row.split,
        "is_grounded": row.is_grounded,
        "query": translation["query_de"],
        "response": translation["response_de"],
        "raw_context": translation["raw_context_de"],
        "metadata": row.metadata,
        # Keep the original English text under ``source_*`` so spot-check
        # reviewers can compare without re-loading the parquet.
        "source_query_en": row.query,
        "source_response_en": row.response,
        "source_raw_context_en": row.raw_context,
        "language": "de",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=200,
        help="Per-class row count (default 200; 6 classes -> 1200 rows).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["all"],
        help="Class keys to translate. 'all' expands to the six known classes.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Source splits to draw from.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the OpenAI API key.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; emit the row inventory only. Useful for budget review.",
    )
    parser.add_argument(
        "--max-failures-per-class",
        type=int,
        default=10,
        help="Abort the per-class loop after this many translation failures (network / API errors).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classes: List[str] = []
    for c in args.classes:
        if c == "all":
            classes.extend(CLASS_KEYS)
        else:
            classes.append(c)
    classes = list(dict.fromkeys(classes))  # de-dup, preserve order

    rows: List[SourceRow] = []
    for split in args.splits:
        rows.extend(_load_parquet_rows(split))
    sampled = _stratified_sample(rows, per_class=args.per_class, seed=args.seed)

    if args.dry_run:
        for class_key in classes:
            bucket = sampled.get(class_key, [])
            grounded = sum(1 for r in bucket if r.is_grounded is True)
            halluc = sum(1 for r in bucket if r.is_grounded is False)
            unlab = sum(1 for r in bucket if r.is_grounded is None)
            logger.info(
                "DRY RUN class=%s rows=%d (grounded=%d halluc=%d unlabelled=%d)",
                class_key,
                len(bucket),
                grounded,
                halluc,
                unlab,
            )
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"missing API key in env var {args.api_key_env}. "
            "Set OPENAI_API_KEY or pass --api-key-env=<NAME>."
        )
    translator = _Translator(model=args.model, api_key=api_key, timeout=args.timeout)

    grand_total = 0
    for class_key in classes:
        bucket = sampled.get(class_key, [])
        if not bucket:
            logger.warning("class=%s has zero source rows; skipping", class_key)
            continue
        out_path = output_dir / f"{class_key}.jsonl"
        already = _load_existing(out_path)
        pending = [r for r in bucket if r.row_id not in already]
        logger.info(
            "class=%s rows=%d already=%d pending=%d",
            class_key,
            len(bucket),
            len(already),
            len(pending),
        )
        failures = 0
        t0 = time.perf_counter()
        for idx, row in enumerate(pending, start=1):
            try:
                translation = translator.translate(row)
            except Exception as exc:  # noqa: BLE001 - operator-visible
                failures += 1
                logger.warning("class=%s row=%s failed: %s", class_key, row.row_id, exc)
                if failures >= args.max_failures_per_class:
                    logger.error(
                        "class=%s exceeded max failures (%d); aborting class",
                        class_key,
                        failures,
                    )
                    break
                continue
            # Defensive: drop rows where any required field came back
            # empty. We never want a degraded translation to silently
            # poison the calibration cache.
            if (
                (row.query and not translation["query_de"])
                or (row.response and not translation["response_de"])
                or (row.raw_context and not translation["raw_context_de"])
            ):
                failures += 1
                logger.warning(
                    "class=%s row=%s incomplete translation; dropping",
                    class_key,
                    row.row_id,
                )
                continue
            _write_row(out_path, row, translation)
            grand_total += 1
            if idx % 25 == 0:
                logger.info(
                    "class=%s progress=%d/%d (failures=%d)",
                    class_key,
                    idx,
                    len(pending),
                    failures,
                )
        logger.info(
            "class=%s done in %.1fs (failures=%d, written total=%d)",
            class_key,
            time.perf_counter() - t0,
            failures,
            grand_total,
        )

    logger.info("translation complete; total rows written this run: %d", grand_total)


if __name__ == "__main__":
    main()
