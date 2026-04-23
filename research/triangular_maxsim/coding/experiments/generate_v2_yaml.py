"""Generate cases_transcripts_v2.yaml from the mined candidate shortlist.

For each candidate (transcript_id, response_turn, fence_idx):
  1. Extract the real assistant response around the fence using the v1
     extractor helper.
  2. Detect the fence language (``python`` / ``bash`` / ``json`` / ``text`` /
     ``mermaid``) and run a language-specific rewrite:
      - ``python``:   rewrite imports + class/function identifiers to
                     fabricated packages (banana_ml, photon_labs, ...).
      - ``bash``:    rewrite executable names to invented CLIs.
      - ``json``:    rewrite top-level/nested keys to phantom synonyms.
      - ``text``/``mermaid``: rewrite the dominant identifiers in the block.
  3. Emit a ``wrong`` variant (phantom) with a heavy rewrite and a
     ``ambiguous`` variant with a single-token kwarg/identifier drift.
  4. Write the combined YAML to
     ``research/triangular_maxsim/coding/cases_transcripts_v2.yaml``.

The generator is deterministic given the candidate JSON so the YAML is
reproducible.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.transcript_cases import (  # noqa: E402
    _CODE_FENCE_RE,
    _extract_correct_response,
    _load_turns,
    _locate_transcript,
)

_ROOT = Path("/workspace/latence-trace/research/triangular_maxsim/coding")
_CANDIDATES = _ROOT / "artifacts" / "v2_candidates.json"
_OUT_YAML = _ROOT / "cases_transcripts_v2.yaml"


# --- Fabricated namespaces ----------------------------------------------------

FAKE_PACKAGES: List[str] = [
    "banana_ml",
    "photon_labs",
    "skylab_metrics",
    "crystal_lattice",
    "cheetah_profiler",
    "mochi_router",
    "prismatic_bench",
    "cactus_codegen",
    "basilisk_schema",
    "skyjournal",
    "twilight_index",
    "opal_embed",
    "quartz_runtime",
    "tangerine_tasks",
    "meteor_graph",
    "indigo_warp",
    "cinnamon_pipeline",
    "lotus_stream",
    "bumblebee_shard",
    "marble_transport",
    "iridium_cache",
    "aurora_harness",
    "plum_consensus",
    "saffron_rerank",
    "coral_fabric",
    "velvet_ledger",
    "juniper_kernel",
    "onyx_loader",
    "gossamer_emit",
    "thistle_diag",
]

FAKE_CLASS_PREFIXES: List[str] = [
    "Noodle", "Bubble", "Taxon", "Axiom", "Prism", "Cactus",
    "Basilisk", "Sky", "Mosaic", "Glimmer", "Quartz", "Indigo",
    "Velvet", "Marble", "Onyx", "Saffron", "Coral", "Gossamer",
    "Thistle", "Tangerine",
]

FAKE_CLASS_SUFFIXES: List[str] = [
    "Warp", "Harness", "Ledger", "Probe", "Scope", "Router",
    "Engine", "Bridge", "Gate", "Pipeline", "Shard", "Emitter",
    "Resolver", "Adapter", "Composer", "Dispatcher",
]

FAKE_BASH_CMDS: List[str] = [
    "prismatic-bench run",
    "noodle-warp exec",
    "skyjournal emit",
    "cactus-gen render",
    "mochi-router dispatch",
    "aurora-harness bench",
    "lotus-stream probe",
    "crystal-lattice scan",
    "banana-ml compile",
    "photon-ctl run",
    "skylab-metrics probe",
    "plum-consensus audit",
    "meteor-graph traverse",
    "twilight-index build",
    "opal-embed encode",
]


def _pick(pool: List[str], seed: int) -> str:
    return pool[seed % len(pool)]


# --- Python rewriting ---------------------------------------------------------

_PY_IMPORT_RE = re.compile(r"^(\s*)(from\s+[\w\.]+\s+import\s+[\w,\s\*\(\)]+|import\s+[\w\.]+(?:\s+as\s+\w+)?)", re.MULTILINE)
_PY_DEF_RE = re.compile(r"^(\s*)def\s+(\w+)\s*\(", re.MULTILINE)
_PY_CLASS_RE = re.compile(r"^(\s*)class\s+(\w+)\b", re.MULTILINE)
_PY_CALL_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,})\(")
_PY_KW_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*=")

_SAFE_KW = {"self", "cls", "return", "def", "class", "True", "False", "None"}


def _rewrite_python_phantom(code: str, seed: int) -> str:
    """Rewrite imports + defined symbols to fabricated versions."""

    pkg_a = _pick(FAKE_PACKAGES, seed)
    pkg_b = _pick(FAKE_PACKAGES, seed + 7)
    pkg_c = _pick(FAKE_PACKAGES, seed + 13)
    cls_a = _pick(FAKE_CLASS_PREFIXES, seed) + _pick(FAKE_CLASS_SUFFIXES, seed)
    cls_b = _pick(FAKE_CLASS_PREFIXES, seed + 3) + _pick(FAKE_CLASS_SUFFIXES, seed + 5)
    cls_c = _pick(FAKE_CLASS_PREFIXES, seed + 9) + _pick(FAKE_CLASS_SUFFIXES, seed + 11)

    # Replace or prepend imports. If code has no imports, inject two phantoms.
    def _import_sub(m: re.Match) -> str:
        indent = m.group(1)
        return f"{indent}from {pkg_a}.runtime import {cls_a}, {cls_b}"

    rewritten = _PY_IMPORT_RE.sub(_import_sub, code, count=1)
    if rewritten == code:
        # Prepend phantom imports.
        rewritten = (
            f"from {pkg_a}.runtime import {cls_a}, {cls_b}\n"
            f"from {pkg_b}.helpers import {cls_c}\n\n"
            + rewritten
        )

    # Rename all top-level defs/classes to phantom names.
    def_seen: Dict[str, str] = {}
    def _def_sub(m: re.Match) -> str:
        indent, name = m.group(1), m.group(2)
        if name.startswith("_") or name in def_seen:
            return m.group(0)
        new_name = f"{pkg_c.split('_')[0]}_{name}"
        def_seen[name] = new_name
        return f"{indent}def {new_name}("

    def _class_sub(m: re.Match) -> str:
        indent, name = m.group(1), m.group(2)
        if name in def_seen:
            return m.group(0)
        new_name = _pick(FAKE_CLASS_PREFIXES, seed + len(name)) + _pick(FAKE_CLASS_SUFFIXES, seed + len(name) + 4)
        def_seen[name] = new_name
        return f"{indent}class {new_name}"

    rewritten = _PY_DEF_RE.sub(_def_sub, rewritten)
    rewritten = _PY_CLASS_RE.sub(_class_sub, rewritten)

    # Replace a handful of CamelCase calls with phantom class names.
    call_targets = sorted(set(m.group(1) for m in _PY_CALL_RE.finditer(rewritten)))
    for i, target in enumerate(call_targets[:8]):
        if target in def_seen:
            continue
        new_name = _pick(FAKE_CLASS_PREFIXES, seed + i + 1) + _pick(FAKE_CLASS_SUFFIXES, seed + i + 2)
        rewritten = re.sub(rf"\b{re.escape(target)}\b", new_name, rewritten)

    return rewritten


def _rewrite_python_drift(code: str, seed: int) -> str:
    """Single surgical swap: rename one kwarg or one identifier to a near-synonym."""

    kw_synonyms = {
        "size": "extent",
        "batch_size": "chunk_size",
        "limit": "threshold",
        "path": "target",
        "max_len": "max_span",
        "num_workers": "worker_pool",
        "device": "accelerator",
        "temperature": "sampling_temperature",
        "top_k": "top_n",
        "tokens": "fragments",
        "chunk_size": "window_size",
        "config": "settings",
        "model": "encoder",
        "query": "prompt",
        "prompt": "query",
        "text": "utterance",
        "name": "slug",
        "index": "position",
        "enabled": "active",
        "verbose": "trace",
    }
    # Find first matching kwarg and rename it consistently.
    for old, new in kw_synonyms.items():
        pat = rf"\b{old}\s*="
        if re.search(pat, code):
            return re.sub(pat, f"{new}=", code, count=3)

    # Fallback: swap one identifier.
    ident_syn = {
        "pipeline": "workflow",
        "result": "outcome",
        "response": "reply",
        "request": "query_bundle",
        "stats": "metrics_frame",
    }
    for old, new in ident_syn.items():
        if re.search(rf"\b{old}\b", code):
            return re.sub(rf"\b{old}\b", new, code, count=2)

    # Last resort: insert one phantom import.
    pkg = _pick(FAKE_PACKAGES, seed)
    return f"from {pkg}.ops import retype  # subtle drift\n" + code


# --- Bash rewriting -----------------------------------------------------------


def _rewrite_bash_phantom(code: str, seed: int) -> str:
    cmd = _pick(FAKE_BASH_CMDS, seed)
    lines = code.splitlines()
    out: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        # Replace the first token (executable) on each non-comment line.
        parts = stripped.split(None, 1)
        if len(parts) == 1:
            out.append(f"{leading}{cmd}")
        else:
            out.append(f"{leading}{cmd} --suite {parts[1].split()[0] if parts[1] else 'bench'} --profile phantom")
    return "\n".join(out)


def _rewrite_bash_drift(code: str, seed: int) -> str:
    flag_synonyms = {
        "--output": "--dest",
        "--input": "--src",
        "--threads": "--workers",
        "--workers": "--threads",
        "--batch": "--chunk",
        "--verbose": "--trace",
        "--config": "--settings",
        "--dry-run": "--plan-only",
        "--force": "--override",
    }
    for old, new in flag_synonyms.items():
        if old in code:
            return code.replace(old, new, 2)
    # Fallback 1: change one numeric arg
    swapped = re.sub(r"\b(\d{2,5})\b", lambda m: str(int(m.group(1)) + 7), code, count=1)
    if swapped != code:
        return swapped
    # Fallback 2: inject a phantom flag on the first non-comment line
    lines = code.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            phantom_flag = " --" + _pick(FAKE_PACKAGES, seed).split("_")[0] + "-mode"
            lines[i] = line + phantom_flag
            return "\n".join(lines)
    return code + "\n# note: phantom drift"


# --- JSON rewriting -----------------------------------------------------------

_JSON_KEY_RE = re.compile(r'"([A-Za-z_][\w\-]*)"\s*:')


def _rewrite_json_phantom(code: str, seed: int) -> str:
    keys = sorted(set(_JSON_KEY_RE.findall(code)))
    out = code
    for i, key in enumerate(keys[:10]):
        phantom_key = f"{_pick(FAKE_PACKAGES, seed + i).split('_')[0]}_{key}"
        out = re.sub(rf'"{re.escape(key)}"(\s*:)', f'"{phantom_key}"\\1', out)
    return out


def _rewrite_json_drift(code: str, seed: int) -> str:
    key_synonyms = {
        "name": "slug",
        "id": "uid",
        "value": "payload",
        "type": "kind",
        "path": "target",
        "status": "state",
        "count": "total",
        "message": "detail",
    }
    for old, new in key_synonyms.items():
        pat = rf'"{old}"(\s*:)'
        if re.search(pat, code):
            return re.sub(pat, f'"{new}"\\1', code, count=2)
    # Fallback: rename the first discovered key to a synonym-suffixed name.
    keys = _JSON_KEY_RE.findall(code)
    if keys:
        target = keys[0]
        new_key = f"{target}_metric"
        return re.sub(rf'"{re.escape(target)}"(\s*:)', f'"{new_key}"\\1', code, count=1)
    return code


# --- Text / mermaid rewriting --------------------------------------------------


def _rewrite_text_phantom(code: str, seed: int) -> str:
    """For free text / mermaid, swap the most-repeated capitalised identifiers."""

    idents = sorted(
        set(re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", code)),
        key=lambda s: -code.count(s),
    )
    out = code
    for i, ident in enumerate(idents[:6]):
        new_name = _pick(FAKE_CLASS_PREFIXES, seed + i) + _pick(FAKE_CLASS_SUFFIXES, seed + i + 1)
        out = re.sub(rf"\b{re.escape(ident)}\b", new_name, out)
    # Also rewrite backticked identifiers.
    tick_idents = sorted(set(re.findall(r"`([A-Za-z_][\w\-\.]{3,})`", out)), key=lambda s: -out.count(s))
    for i, ident in enumerate(tick_idents[:4]):
        new_name = _pick(FAKE_PACKAGES, seed + i + 3) + "." + _pick(FAKE_CLASS_PREFIXES, seed + i).lower()
        out = out.replace(f"`{ident}`", f"`{new_name}`")
    return out


def _rewrite_text_drift(code: str, seed: int) -> str:
    subs = {
        "shall": "must",
        "enabled": "active",
        "returned": "emitted",
        "logged": "recorded",
        "built": "composed",
        "computed": "resolved",
    }
    out = code
    replaced = 0
    for old, new in subs.items():
        pat = rf"\b{old}\b"
        if re.search(pat, out, re.IGNORECASE):
            out = re.sub(pat, new, out, count=1, flags=re.IGNORECASE)
            replaced += 1
            if replaced >= 2:
                break
    if replaced:
        return out
    # Fallback: rename the first backticked identifier to a near-synonym suffix.
    tick_idents = re.findall(r"`([A-Za-z_][\w\-\.]{3,})`", out)
    if tick_idents:
        target = tick_idents[0]
        replacement = f"{target}_alt"
        return out.replace(f"`{target}`", f"`{replacement}`", 1)
    # Fallback 2: swap first CamelCase identifier to a near-synonym.
    caps = re.findall(r"\b[A-Z][A-Za-z0-9_]{3,}\b", out)
    if caps:
        target = caps[0]
        replacement = f"{target}V2"
        return re.sub(rf"\b{re.escape(target)}\b", replacement, out, count=1)
    # Fallback 3: append a drift note.
    return out + " (note: reconsider the active tier)"


# --- Dispatch -----------------------------------------------------------------


def _rewrite_fence(body: str, lang: str, seed: int, mode: str) -> str:
    lang = (lang or "").lower()
    if lang == "python":
        return (_rewrite_python_phantom if mode == "phantom" else _rewrite_python_drift)(body, seed)
    if lang in {"bash", "sh", "shell"}:
        return (_rewrite_bash_phantom if mode == "phantom" else _rewrite_bash_drift)(body, seed)
    if lang == "json":
        return (_rewrite_json_phantom if mode == "phantom" else _rewrite_json_drift)(body, seed)
    return (_rewrite_text_phantom if mode == "phantom" else _rewrite_text_drift)(body, seed)


def _build_variant_response(correct_response: str, lang: str, seed: int, mode: str) -> str:
    """Re-render ``correct_response`` with the fenced body rewritten."""

    def _sub(m: re.Match) -> str:
        existing_lang = m.group(1) or lang
        body = m.group(2)
        new_body = _rewrite_fence(body, existing_lang, seed, mode)
        return f"```{existing_lang}\n{new_body}\n```"

    return _CODE_FENCE_RE.sub(_sub, correct_response, count=1)


# --- Main pipeline ------------------------------------------------------------


def _pretty_id(idx: int, transcript: str, turn: int, fence: int) -> str:
    return f"base_{idx:02d}_{transcript}_t{turn}_f{fence}"


def _describe(body: str, lang: str) -> str:
    first = body.strip().splitlines()[0] if body.strip() else ""
    flat = re.sub(r"\s+", " ", first)[:120]
    lang_label = lang or "text"
    return f"{lang_label}: {flat}"


def main() -> None:
    with _CANDIDATES.open() as handle:
        candidates = json.load(handle)

    # Filter / dedupe / cap per transcript (mirror the earlier analysis)
    from collections import Counter

    per_t: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    for c in candidates:
        tid = c["transcript_id"]
        if per_t[tid] >= 8:
            continue
        if c["context_files"] < 3 or c["fence_chars"] < 120 or c["context_chars"] < 15_000:
            continue
        per_t[tid] += 1
        selected.append(c)
        if len(selected) >= 60:
            break

    print(f"Selected {len(selected)} bases")

    # Cache of (transcript_id -> list of turns) to avoid re-parsing.
    turn_cache: Dict[str, List[Dict[str, Any]]] = {}

    scenarios: List[Dict[str, Any]] = []
    for idx, c in enumerate(selected, start=1):
        tid = c["transcript_id"]
        if tid not in turn_cache:
            turn_cache[tid] = _load_turns(_locate_transcript(tid))
        turns = turn_cache[tid]

        correct = _extract_correct_response(
            turns,
            response_turn=c["response_turn"],
            fence_idx=c["fence_idx"],
            before_chars=500,
            after_chars=250,
        )
        lang = c["fence_lang"] or "text"
        base_id = _pretty_id(idx, tid, c["response_turn"], c["fence_idx"])
        seed = idx * 17 + len(tid)

        phantom = _build_variant_response(correct, lang, seed, "phantom")
        drift = _build_variant_response(correct, lang, seed, "drift")

        # Sanity: variants must differ from correct.
        if phantom.strip() == correct.strip() or drift.strip() == correct.strip():
            print(f"[warn] {base_id} lang={lang}: variant is identical to correct; skipping")
            continue

        scenarios.append(
            {
                "id": base_id,
                "transcript_id": tid,
                "response_turn": c["response_turn"],
                "fence_idx": c["fence_idx"],
                "description": _describe(c["fence_head"], lang),
                "lang": lang,
                "variants": [
                    {
                        "tier": "wrong",
                        "id": f"{base_id}__wrong",
                        "notes": "Auto-generated phantom: imports+classes rewritten to fabricated packages (banana_ml, photon_labs, ...) or CLIs replaced.",
                        "response": phantom,
                    },
                    {
                        "tier": "ambiguous",
                        "id": f"{base_id}__ambiguous",
                        "notes": "Auto-generated drift: single kwarg/identifier rename to a near-synonym.",
                        "response": drift,
                    },
                ],
            }
        )

    # Emit YAML manually (block-scalar literals for clarity).
    lines: List[str] = [
        "schema: transcripts_v2",
        "version: 2",
        "",
        "context:",
        "  max_files: 40",
        "  context_token_cap_chars: 160000",
        "  scope_before_chars: 500",
        "  scope_after_chars: 250",
        "",
        "# Generated by experiments/generate_v2_yaml.py from v2_candidates.json.",
        "# Each scenario is a real assistant turn; the correct variant is auto-",
        "# extracted, the wrong+ambiguous variants are programmatically rewritten",
        "# using fabricated-namespace templates and embedded verbatim below.",
        "",
        "scenarios:",
    ]
    for s in scenarios:
        lines.append(f"  - id: {s['id']}")
        lines.append(f"    transcript_id: {s['transcript_id']}")
        lines.append(f"    response_turn: {s['response_turn']}")
        lines.append(f"    fence_idx: {s['fence_idx']}")
        lines.append(f"    lang: {s['lang']}")
        desc = s["description"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    description: "{desc}"')
        lines.append("    variants:")
        for v in s["variants"]:
            lines.append(f"      - tier: {v['tier']}")
            lines.append(f"        id: {v['id']}")
            notes = v["notes"].replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'        notes: "{notes}"')
            lines.append("        response: |")
            for ln in v["response"].splitlines():
                lines.append(f"          {ln}")
            lines.append("")

    _OUT_YAML.write_text("\n".join(lines))
    print(f"Wrote {_OUT_YAML} ({len(scenarios)} scenarios, {len(scenarios) * 3} cases)")


if __name__ == "__main__":
    main()
