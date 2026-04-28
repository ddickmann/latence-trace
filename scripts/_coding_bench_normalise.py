"""Normalise HumanEval+ and CRUXEval into the bench_external row shape.

Writes ``research/coding_bench/external_data/{humanevalplus,cruxeval}_grounded.jsonl``
with the canonical columns: ``bench``, ``row_id``, ``question``,
``response_text``, ``raw_context``, ``expected_band``, ``expected_label``.

Only grounded (faithful) rows are produced here. Hallucinated variants are
generated in a separate adversarial step (see ``scripts/generate_coding_adversarials.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "research/coding_bench/external_data"


def normalise_humanevalplus() -> Path:
    src = BASE / "humanevalplus.jsonl"
    dst = BASE / "humanevalplus_grounded.jsonl"
    with src.open("r", encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            row = json.loads(line)
            task_id = row.get("task_id", "")
            prompt = row.get("prompt", "")
            canonical = row.get("canonical_solution", "")
            entry_point = row.get("entry_point", "")
            # Grounded: the response is the canonical solution; context is the
            # prompt (signature + docstring + examples).
            normalised = {
                "bench": "humanevalplus",
                "row_id": f"humanevalplus_{task_id.split('/')[-1]}",
                "question": (
                    f"Implement the function `{entry_point}` as described in the docstring."
                ),
                "response_text": prompt + canonical,
                "raw_context": prompt,
                "expected_band": "green",
                "expected_label": "faithful",
                "task_id": task_id,
                "entry_point": entry_point,
                "canonical_solution": canonical,
                "test": row.get("test", ""),
                "response_kind": "code",
            }
            out.write(json.dumps(normalised, ensure_ascii=False) + "\n")
    return dst


def normalise_cruxeval() -> Path:
    src = BASE / "cruxeval.jsonl"
    dst = BASE / "cruxeval_grounded.jsonl"
    with src.open("r", encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for line in fh:
            row = json.loads(line)
            code = row.get("code", "")
            inp = row.get("input", "")
            outp = row.get("output", "")
            sample_id = row.get("id", "")
            normalised = {
                "bench": "cruxeval",
                "row_id": f"cruxeval_{sample_id}",
                "question": f"What does the function return for input {inp}?",
                "response_text": outp,
                "raw_context": code + "\n\n# call: f" + (f"({inp})" if inp else "()"),
                "expected_band": "green",
                "expected_label": "faithful",
                "code": code,
                "input": inp,
                "output": outp,
                "response_kind": "code",
            }
            out.write(json.dumps(normalised, ensure_ascii=False) + "\n")
    return dst


def main() -> None:
    for fn, label in (
        (normalise_humanevalplus, "HumanEval+"),
        (normalise_cruxeval, "CRUXEval"),
    ):
        out = fn()
        count = sum(1 for _ in out.open("r", encoding="utf-8"))
        print(f"{label}: wrote {count} grounded rows -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
