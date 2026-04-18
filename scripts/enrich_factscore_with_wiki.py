#!/usr/bin/env python3
"""Enrich the FActScore biography dataset with Wikipedia source context.

The upstream FActScore biographies file ships ``topic`` + model ``output`` +
per-claim ``annotations`` but no source text. The benchmark harness needs
context to score, otherwise every response is compared against ``""`` and
all metrics collapse to noise.

This script downloads the lead section + sections of the matching Wikipedia
page for each biography topic, trims the result to a configurable token
budget, and writes ``biographies_wiki.jsonl`` next to the original file.

The enriched record adds:
- ``context``: the trimmed Wikipedia article text (UTF-8)
- ``wiki_title``: the resolved Wikipedia article title
- ``wiki_chars``: number of characters in ``context``
- ``wiki_status``: ``ok`` | ``not_found`` | ``error``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

USER_AGENT = "latence-trace-bench/1.0 (https://latence.ai)"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _wiki_extract(session: requests.Session, topic: str, max_chars: int) -> Optional[Dict[str, Any]]:
    """Return ``{'title':..., 'extract':...}`` for ``topic`` or ``None``."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "format": "json",
        "titles": topic,
    }
    response = session.get(WIKI_API, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages") or {}
    for _pid, page in pages.items():
        if int(_pid) < 0:
            return None
        extract = page.get("extract") or ""
        if not extract.strip():
            return None
        return {"title": page.get("title") or topic, "extract": extract[:max_chars]}
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/factscore/biographies.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/datasets/factscore/biographies_wiki.jsonl"),
    )
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 means enrich all input rows.")
    parser.add_argument("--throttle-ms", type=int, default=120,
                        help="Sleep between requests to be polite to Wikipedia.")
    parser.add_argument("--cache", type=Path,
                        default=Path("/workspace/datasets/factscore/.wiki_cache.json"))
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    cache: Dict[str, Dict[str, Any]] = {}
    if args.cache.exists():
        try:
            cache = json.loads(args.cache.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    written = 0
    ok = 0
    miss = 0
    error = 0
    sleep_s = max(0, args.throttle_ms) / 1000.0
    with args.output.open("w", encoding="utf-8") as out:
        for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            topic = str(obj.get("topic") or "").strip()
            if not topic:
                continue
            if topic in cache:
                hit = cache[topic]
            else:
                try:
                    hit = _wiki_extract(session, topic, args.max_chars) or {"_miss": True}
                except Exception as exc:  # network errors are non-fatal
                    hit = {"_error": str(exc)}
                cache[topic] = hit
                if sleep_s > 0:
                    time.sleep(sleep_s)
            if hit.get("_miss"):
                miss += 1
                obj["context"] = ""
                obj["wiki_status"] = "not_found"
            elif hit.get("_error"):
                error += 1
                obj["context"] = ""
                obj["wiki_status"] = "error"
                obj["wiki_error"] = hit["_error"]
            else:
                ok += 1
                obj["context"] = hit["extract"]
                obj["wiki_title"] = hit["title"]
                obj["wiki_chars"] = len(hit["extract"])
                obj["wiki_status"] = "ok"
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and written >= args.max_samples:
                break
            if written % 25 == 0:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                args.cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"[{written}] ok={ok} miss={miss} err={error}", flush=True)

    args.cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "written": written,
        "ok": ok,
        "not_found": miss,
        "error": error,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
