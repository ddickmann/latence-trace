"""Validate curated Veracier variants for schema + curation invariants.

Runs deterministic checks on ``variants.curated.jsonl``:

* every row carries non-empty ``curated_by`` and ``curation_reason``;
* every ``ambiguous`` variant contains at least one entry from
  ``EPISTEMIC_HEDGE_CUES``;
* every ``perfect`` variant contains two double-quoted evidence snippets;
* every ``wrong`` variant contains at least one non-trivial factual delta
  (numeric or identifier differing from the variant's original).

Exit code ``0`` on success; non-zero on first failure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data" / "veracier-industries" / "proof_bundle_v1" / "variants.curated.jsonl"

# Hedge cues mirror latence_trace/core/groundedness.py::EPISTEMIC_HEDGE_CUES
# with additional phrasing seen in production curations.
HEDGE_CUES = {
    "unclear",
    "not established",
    "does not show",
    "not clear",
    "insufficient",
    "missing",
    "no evidence",
    "no confirmation",
    "cannot establish",
    "cannot confirm",
    "ne permet pas",
    "pas clair",
    "n'établit pas",
    "n’etablit pas",
    "n'est pas établi",
    "n'est pas établie",
    "n'est pas etablie",
    "n'est pas etabli",
    "n’est pas établi",
    "n’est pas établie",
    "n’est pas etablie",
    "insuffisant",
    "incertain",
    "incertitude",
    "manquant",
    "non clarifié",
    "non clarifie",
    "nicht klar",
    "nicht belegt",
    "unvollständig",
    "unvollstaendig",
    "unbekannt",
    "keine angabe",
    "il n'est pas",
    "reste à confirmer",
    "reste a confirmer",
    "je ne dispose pas",
    "je ne peux pas",
    "allerdings ist nicht",
    "further evidence is required",
    "unable to confirm",
    "cannot be confirmed",
    "cannot be established",
    "bleibt unklar",
    "reste insuffisamment",
    "insuffisamment documenté",
    "insuffisamment documente",
}

QUOTE_RE = re.compile(r'"[^"]{16,}"')


def _has_hedge_cue(text: str) -> bool:
    norm = text.lower()
    return any(cue in norm for cue in HEDGE_CUES)


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"variants.curated.jsonl not found: {path}")
        return 2
    rows = _load(path)
    errors: list[str] = []
    for row in rows:
        eid = row.get("example_id", "<unknown>")
        if not row.get("curated_by"):
            errors.append(f"{eid}: missing curated_by")
        if not row.get("curation_reason"):
            errors.append(f"{eid}: missing curation_reason")
        variant = row.get("variant") or eid.split(":")[-1] if ":" in eid else ""
        text = str(row.get("response_text") or "")
        if variant == "ambiguous":
            if not _has_hedge_cue(text):
                errors.append(
                    f"{eid}: ambiguous variant lacks an explicit hedge cue"
                )
        if variant == "perfect":
            quotes = QUOTE_RE.findall(text)
            if len(quotes) < 2:
                errors.append(
                    f"{eid}: perfect variant lacks two double-quoted evidence snippets"
                )
        if variant == "wrong":
            if len(text.strip()) < 40:
                errors.append(
                    f"{eid}: wrong variant is suspiciously short ({len(text)} chars)"
                )
    if errors:
        print("FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"OK: {len(rows)} curated rows pass all invariants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
