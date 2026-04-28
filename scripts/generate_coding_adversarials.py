"""Generate deterministic adversarial hallucinated coding variants.

Produces three variant types per passing canonical row:

* ``identifier_swap`` — rename a symbol to a plausible but wrong one.
* ``literal_swap`` — change a numeric or string literal that appears in the
  response.
* ``api_signature_swap`` — swap a well-known stdlib / built-in call for
  another call with similar shape but different semantics.

The mutations are deterministic and rule-based so the bench is reproducible
without an external LLM adversary. (A strong-model adversary is a follow-up
listed in the Phase 3 todo backlog.)

Output: ``data/veracier-industries/proof_bundle_v1/coding_bench/variants.jsonl``
Row shape mirrors ``bench_external.py`` plus ``variant_kind`` and ``variant_type``.
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "research/coding_bench/external_data"
OUT = REPO_ROOT / "data/veracier-industries/proof_bundle_v1/coding_bench"

IDENTIFIER_SWAPS = {
    "append": "extend",
    "extend": "append",
    "sort": "reverse",
    "reverse": "sort",
    "sum": "max",
    "max": "min",
    "min": "max",
    "len": "sum",
    "abs": "round",
    "sorted": "reversed",
    "enumerate": "reversed",
    "dict": "list",
    "set": "dict",
    "tuple": "list",
    "range": "reversed",
    "print": "format",
    "lower": "upper",
    "upper": "lower",
    "strip": "split",
    "split": "join",
    "join": "split",
    "replace": "strip",
    "startswith": "endswith",
    "endswith": "startswith",
    "int": "float",
    "float": "int",
    "str": "repr",
    "items": "keys",
    "keys": "values",
    "values": "keys",
    "count": "index",
    "index": "count",
    "add": "update",
    "update": "add",
    "pop": "push",
    "append": "pop",
    "insert": "remove",
    "remove": "insert",
}

API_SIGNATURE_SWAPS = [
    (r"\bsorted\(([^,)]+),\s*reverse=True\)", r"sorted(\1)"),
    (r"\bsorted\(([^,)]+)\)", r"sorted(\1, reverse=True)"),
    (r"\bmax\(([^)]+)\)", r"min(\1)"),
    (r"\bmin\(([^)]+)\)", r"max(\1)"),
    (r"\babs\(([^)]+)\)", r"-\1"),
    (r"\.strip\(\)", ".split()"),
    (r"\.upper\(\)", ".lower()"),
    (r"\.lower\(\)", ".upper()"),
    (r"\blen\(([^)]+)\)", r"\1"),
]


def _find_identifiers(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ids: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ids.append(node.id)
        elif isinstance(node, ast.Attribute):
            ids.append(node.attr)
    return ids


def _swap_identifier(source: str, rng: random.Random) -> Optional[tuple[str, str, str]]:
    ids = _find_identifiers(source)
    candidates = [i for i in ids if i in IDENTIFIER_SWAPS]
    if not candidates:
        return None
    ident = rng.choice(candidates)
    repl = IDENTIFIER_SWAPS[ident]
    # Use a word-boundary regex so we don't clobber substrings.
    pattern = re.compile(rf"\b{re.escape(ident)}\b")
    mutated = pattern.sub(repl, source, count=1)
    if mutated == source:
        return None
    return mutated, ident, repl


_NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?\d+(\.\d+)?(?![A-Za-z_])")
_STRING_RE = re.compile(r"'([^'\n]{1,32})'|\"([^\"\n]{1,32})\"")


def _swap_literal(source: str, rng: random.Random) -> Optional[tuple[str, str, str]]:
    matches = list(_NUMBER_RE.finditer(source))
    if matches:
        m = rng.choice(matches)
        original = m.group(0)
        try:
            num = float(original)
        except ValueError:
            return None
        if num == 0:
            new_num = 1
        else:
            new_num = num + max(1.0, abs(num) * 0.5)
            if float(int(num)) == num:
                new_num = int(new_num)
        mutated = source[: m.start()] + str(new_num) + source[m.end() :]
        return mutated, original, str(new_num)
    matches = list(_STRING_RE.finditer(source))
    if matches:
        m = rng.choice(matches)
        original = m.group(0)
        body = m.group(1) or m.group(2) or ""
        new_body = body[::-1] if body else "xyz"
        quote = original[0]
        mutated = source[: m.start()] + f"{quote}{new_body}{quote}" + source[m.end() :]
        return mutated, original, f"{quote}{new_body}{quote}"
    return None


def _swap_api_signature(source: str, rng: random.Random) -> Optional[tuple[str, str, str]]:
    rng_swaps = list(API_SIGNATURE_SWAPS)
    rng.shuffle(rng_swaps)
    for pat, repl in rng_swaps:
        m = re.search(pat, source)
        if m is None:
            continue
        try:
            mutated = re.sub(pat, repl, source, count=1)
        except re.error:
            continue
        if mutated != source:
            return mutated, m.group(0), pat
    return None


def _maybe_variant(
    source: str,
    rng: random.Random,
    kind: str,
) -> Optional[tuple[str, str, str]]:
    if kind == "identifier_swap":
        return _swap_identifier(source, rng)
    if kind == "literal_swap":
        return _swap_literal(source, rng)
    if kind == "api_signature_swap":
        return _swap_api_signature(source, rng)
    raise ValueError(f"unknown variant kind: {kind}")


def iter_grounded(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "variants.jsonl"
    paired_path = OUT / "paired.jsonl"
    rng = random.Random(42)
    pairs: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as out_fh:
        for grounded_file in (
            SRC / "humanevalplus_grounded.jsonl",
            SRC / "cruxeval_grounded.jsonl",
        ):
            for row in iter_grounded(grounded_file):
                # Emit the grounded row as-is for the paired bench.
                out_fh.write(json.dumps({**row, "variant_kind": "grounded"}, ensure_ascii=False) + "\n")
                pairs.append({"green": row})
                for kind in ("identifier_swap", "literal_swap", "api_signature_swap"):
                    source = row["response_text"]
                    result = _maybe_variant(source, rng, kind)
                    if result is None:
                        continue
                    mutated, before, after = result
                    variant_row = {
                        **row,
                        "row_id": f"{row['row_id']}__{kind}",
                        "response_text": mutated,
                        "expected_band": "red",
                        "expected_label": "hallucinated",
                        "variant_kind": kind,
                        "variant_type": "rule_based",
                        "variant_before": before[:120],
                        "variant_after": after[:120],
                    }
                    out_fh.write(json.dumps(variant_row, ensure_ascii=False) + "\n")
                    pairs[-1][kind] = variant_row
    with paired_path.open("w", encoding="utf-8") as pair_fh:
        for pair in pairs:
            pair_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    counts = {}
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            counts[obj.get("variant_kind") or "grounded"] = (
                counts.get(obj.get("variant_kind") or "grounded", 0) + 1
            )
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"wrote {paired_path.relative_to(REPO_ROOT)}")
    print(f"variant counts: {counts}")


if __name__ == "__main__":
    main()
