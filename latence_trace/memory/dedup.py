"""Deduplication and conservative supersession for TRACE Memory."""

from __future__ import annotations

from latence_trace.memory.models import MemoryAction, SpanRecord
from latence_trace.memory.signature import normalize_text


def merge_spans(
    existing: list[SpanRecord],
    incoming: list[SpanRecord],
) -> tuple[list[SpanRecord], list[MemoryAction]]:
    actions: list[MemoryAction] = []
    by_hash = {span.signature.normalized_hash: span for span in existing}
    by_key = {span.signature.typed_key: span for span in existing}
    merged = list(existing)

    for span in incoming:
        exact = by_hash.get(span.signature.normalized_hash)
        if exact is not None:
            exact.last_seen_turn = max(exact.last_seen_turn, span.last_seen_turn)
            exact.scores.redundancy = max(exact.scores.redundancy, 0.8)
            actions.append(
                MemoryAction(action="deduped", span_id=exact.id, reason="normalized_hash_match")
            )
            continue

        superseded_tool = []
        if _can_supersede_tool_state(span):
            superseded_tool = [
                old
                for old in merged
                if old.compression_level != "tombstone" and _superseded_tool_state(old, span)
            ]
        if superseded_tool:
            for old in superseded_tool:
                old.layer = "cold"
                old.compression_level = "tombstone"
                old.supersedes.append(span.id)
                actions.append(
                    MemoryAction(
                        action="superseded",
                        span_id=old.id,
                        reason="newer_tool_state_signature",
                        from_layer="warm",
                        to_layer="cold",
                    )
                )
            span.provenance["supersedes"] = [old.id for old in superseded_tool]
            merged.append(span)
            by_hash[span.signature.normalized_hash] = span
            by_key[span.signature.typed_key] = span
            continue

        keyed = by_key.get(span.signature.typed_key)
        if keyed is not None and _can_supersede(keyed, span):
            keyed.layer = "cold"
            keyed.compression_level = "tombstone"
            keyed.supersedes.append(span.id)
            span.provenance["supersedes"] = keyed.id
            merged.append(span)
            by_hash[span.signature.normalized_hash] = span
            by_key[span.signature.typed_key] = span
            actions.append(
                MemoryAction(
                    action="superseded",
                    span_id=keyed.id,
                    reason="newer_same_typed_signature",
                    from_layer="warm",
                    to_layer="cold",
                )
            )
            continue

        if keyed is not None and _overlap(keyed, span) >= 0.85:
            keyed.last_seen_turn = max(keyed.last_seen_turn, span.last_seen_turn)
            keyed.scores.redundancy = max(keyed.scores.redundancy, 0.7)
            actions.append(
                MemoryAction(action="deduped", span_id=keyed.id, reason="typed_signature_overlap")
            )
            continue

        merged.append(span)
        by_hash[span.signature.normalized_hash] = span
        by_key[span.signature.typed_key] = span
        actions.append(MemoryAction(action="added", span_id=span.id, reason="new_span"))

    return merged, actions


def _superseded_tool_state(old: SpanRecord, new: SpanRecord) -> bool:
    if new.created_turn <= old.created_turn:
        return False
    old_prefix = old.text.split(maxsplit=1)[0] if old.text else ""
    new_prefix = new.text.split(maxsplit=1)[0] if new.text else ""
    if old_prefix == new_prefix == "tool_exact_index":
        return _overlap(old, new) >= 0.75 or bool(
            _tool_identifier_prefixes(old.text) & _tool_identifier_prefixes(new.text)
        )
    old_field = _tool_fact_field(old.text)
    new_field = _tool_fact_field(new.text)
    if old_field and old_field == new_field:
        return True
    shared_state_keys = _tool_state_keys(old.text) & _tool_state_keys(new.text)
    if shared_state_keys:
        return True
    tool_state_words = {"reservation", "booking", "order", "payment", "shipment"}
    old_words = set(normalize_text(old.text).split())
    new_words = set(normalize_text(new.text).split())
    return bool(tool_state_words & old_words & new_words) and _overlap(old, new) >= 0.75


def _can_supersede_tool_state(span: SpanRecord) -> bool:
    text = span.text
    if text.startswith("tool_fact "):
        return True
    if text.startswith("tool_exact_index"):
        return True
    normalized = normalize_text(text)
    return any(
        token in normalized
        for token in ("reservation", "booking", "order", "payment", "shipment", "refund", "status")
    )


def _tool_fact_field(text: str) -> str:
    if not text.startswith("tool_fact "):
        return ""
    fact = text.split(maxsplit=1)[1]
    if "=" not in fact:
        return ""
    return fact.split("=", 1)[0]


def _tool_state_keys(text: str) -> set[str]:
    keys = set()
    for token in normalize_text(text).replace(",", " ").replace(";", " ").split():
        if "=" not in token:
            continue
        key = token.split("=", 1)[0].strip()
        if key and any(fragment in key for fragment in ("status", "order", "reservation", "booking", "refund")):
            keys.add(key)
    return keys


def _tool_identifier_prefixes(text: str) -> set[str]:
    prefixes = set()
    for token in normalize_text(text).replace(",", " ").split():
        if "-" not in token:
            continue
        prefix = token.split("-", 1)[0].upper()
        if 2 <= len(prefix) <= 16 and any(fragment in prefix for fragment in ("RSV", "ORD", "PAY", "RES", "BOOK")):
            prefixes.add(prefix)
    return prefixes


def _can_supersede(old: SpanRecord, new: SpanRecord) -> bool:
    if old.signature.numbers and new.signature.numbers and old.signature.numbers != new.signature.numbers:
        return False
    if old.signature.file_paths and new.signature.file_paths != old.signature.file_paths:
        return False
    newer = new.created_turn >= old.created_turn
    update_words = {"now", "instead", "replace", "supersede", "resolved", "fixed"}
    return newer and bool(update_words.intersection(normalize_text(new.text).split()))


def _overlap(left: SpanRecord, right: SpanRecord) -> float:
    left_terms = set(left.signature.rare_terms) | set(left.signature.file_paths) | set(left.signature.symbols)
    right_terms = set(right.signature.rare_terms) | set(right.signature.file_paths) | set(right.signature.symbols)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))
