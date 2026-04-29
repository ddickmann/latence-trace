"""Tokenisation + label-alignment helpers for training batches.

Given a labeled row (from the synthetic labeler or distill_dataset), we
must:

1. Tokenise response + evidence with a real BPE/WordPiece tokeniser.
2. Align the **word-level** token_support_labels onto the subword
   positions the encoder actually sees.
3. Build a ``unit_assign`` tensor that maps each evidence **subtoken**
   to the unit (sentence chunk) its character offset falls into.
4. Emit a fixed-length phi tensor broadcasting the row's 12-dim phi
   vector across every (r, e) pair.
5. Return the batched tensors in the exact layout the student's
   ``forward(...)`` expects.

The class index is mapped via :data:`CLASS_TO_IDX` so FiLM class
conditioning fires consistently across training + inference.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger("trace.v2.train.collate")


# One authoritative mapping used by both training and inference wiring.
CLASS_TO_IDX: dict[str, int] = {
    "rag.prose.enterprise": 0,
    "rag.prose.short_factoid": 1,
    "rag.prose.multi_claim": 2,
    "rag.structured": 3,
    "rag.code_in_context": 4,
    "code.agentic_trace": 5,
}

BAND_TO_IDX: dict[str, int] = {"green": 0, "amber": 1, "red": 2}


# ---------------------------------------------------------------------------
# Offset-based label alignment
# ---------------------------------------------------------------------------


def _subtoken_to_word_idx(
    offsets: Sequence[tuple[int, int]],
    word_spans: Sequence[tuple[int, int]],
) -> list[int]:
    """For each subtoken offset ``(cs, ce)``, return the word index
    whose span covers it, or -1 if none.

    Padding subtokens (``(0, 0)``) map to -1.
    """
    out: list[int] = []
    for cs, ce in offsets:
        if cs == 0 and ce == 0:
            out.append(-1)
            continue
        # Find the first word span whose [ws, we) contains cs.
        idx = -1
        for wi, (ws, we) in enumerate(word_spans):
            if ws <= cs < we or (cs == ws and ce <= we):
                idx = wi
                break
        out.append(idx)
    return out


def _word_spans(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """Return character spans + tokens for whitespace-ish word tokens.

    Tokens are defined as maximal runs of ``\\S`` after stripping
    leading/trailing punctuation; spans refer to the *original* text.
    """
    spans: list[tuple[int, int]] = []
    tokens: list[str] = []
    i = 0
    while i < len(text):
        # Skip whitespace.
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        end = i
        spans.append((start, end))
        tokens.append(text[start:end])
    return spans, tokens


def _subtoken_to_unit_idx(
    offsets: Sequence[tuple[int, int]],
    unit_spans: Sequence[tuple[int, int]],
) -> list[int]:
    """Map each subtoken offset to the unit whose ``[start, end)`` covers it."""
    out: list[int] = []
    for cs, ce in offsets:
        if cs == 0 and ce == 0:
            out.append(-1)
            continue
        idx = -1
        for u, (us, ue) in enumerate(unit_spans):
            if us <= cs < ue or (us == cs and ce <= ue):
                idx = u
                break
        out.append(idx)
    return out


def align_labels_to_tokens(
    *,
    response_text: str,
    evidence_text: str,
    tokenizer: Any,
    max_resp: int,
    max_ev: int,
    token_support_word_labels: Sequence[int],
    unit_char_spans: Sequence[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Tokenise + align.

    Returns a dict with:

    * ``resp_ids``: ``(max_resp,)``
    * ``resp_mask``: ``(max_resp,)``
    * ``ev_ids``: ``(max_ev,)``
    * ``ev_mask``: ``(max_ev,)``
    * ``unit_assign``: ``(max_ev,)`` int64 with -1 for padding / unmapped
    * ``token_support_aligned``: ``(max_resp,)`` float, broadcast from
      word-level labels onto subtokens.
    """
    # --- response side ---
    resp_encoded = tokenizer(
        response_text or "",
        max_length=max_resp,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    resp_offsets = [tuple(int(x) for x in span) for span in resp_encoded["offset_mapping"][0].tolist()]
    resp_spans, _ = _word_spans(response_text or "")
    resp_word_idx = _subtoken_to_word_idx(resp_offsets, resp_spans)
    resp_support = torch.zeros(max_resp, dtype=torch.float32)
    n_resp = len(resp_word_idx)
    for i, wi in enumerate(resp_word_idx):
        if i >= max_resp:
            break
        if 0 <= wi < len(token_support_word_labels):
            resp_support[i] = float(token_support_word_labels[wi])

    # --- evidence side ---
    ev_encoded = tokenizer(
        evidence_text or "",
        max_length=max_ev,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    ev_offsets = [tuple(int(x) for x in span) for span in ev_encoded["offset_mapping"][0].tolist()]
    unit_assign_list = _subtoken_to_unit_idx(ev_offsets, unit_char_spans)
    unit_assign = torch.tensor(unit_assign_list, dtype=torch.long)
    # Pad / truncate to max_ev.
    if unit_assign.numel() < max_ev:
        pad = torch.full((max_ev - unit_assign.numel(),), -1, dtype=torch.long)
        unit_assign = torch.cat([unit_assign, pad])
    else:
        unit_assign = unit_assign[:max_ev]

    return {
        "resp_ids": resp_encoded["input_ids"][0],
        "resp_mask": resp_encoded["attention_mask"][0],
        "ev_ids": ev_encoded["input_ids"][0],
        "ev_mask": ev_encoded["attention_mask"][0],
        "unit_assign": unit_assign,
        "token_support_aligned": resp_support,
    }


# ---------------------------------------------------------------------------
# Phi channel
# ---------------------------------------------------------------------------


def _phi_vector_from_row(row: Mapping[str, Any]) -> list[float]:
    """Build the 12-dim phi vector for a row.

    12-dim layout:
      [0:4]  lexical: exact / numeric / identifier / lemma overlaps
      [4:6]  structural: same_file / same_symbol (1 bit each)
      [6:12] source-type one-hot: {code, table, markdown, json, passage_enum, prose}
    """
    phi = row.get("phi_channels") or {}
    source_type = (phi.get("source_type") if isinstance(phi, Mapping) else None) or "prose"
    type_ohe = [
        1.0 if source_type == "code" else 0.0,
        1.0 if source_type == "table" else 0.0,
        1.0 if source_type == "markdown" else 0.0,
        1.0 if source_type == "json" else 0.0,
        1.0 if source_type == "passage_enum" else 0.0,
        1.0 if source_type == "prose" else 0.0,
    ]
    if isinstance(phi, Mapping):
        lex = [
            float(phi.get("exact_overlap") or 0.0),
            float(phi.get("numeric_overlap") or 0.0),
            float(phi.get("identifier_overlap") or 0.0),
            float(phi.get("lemma_overlap") or 0.0),
        ]
        structural = [
            float(phi.get("same_file") or 0.0),
            float(phi.get("same_symbol") or 0.0),
        ]
    else:
        lex = [0.0] * 4
        structural = [0.0] * 2
    return [*lex, *structural, *type_ohe]


# ---------------------------------------------------------------------------
# Build a single example dict from a labeled row
# ---------------------------------------------------------------------------


def build_example(
    *,
    row: Mapping[str, Any],
    labelled: Mapping[str, Any],
    tokenizer: Any,
    max_resp: int,
    max_ev: int,
    num_units: int,
) -> dict[str, torch.Tensor | Any]:
    """Build one training example from a row + its labelled counterpart.

    Args:
        row: original dict (class_key, phi_channels, source, pair_id, ...).
        labelled: dict form of :class:`LabelledRow` (or shaped exactly
            like its ``to_dict()``).
        tokenizer: an HF tokenizer with ``return_offsets_mapping=True``.
        max_resp / max_ev: truncation lengths.
        num_units: max number of evidence units in the batch.

    Returns:
        A dict with tensors the collate can stack directly.
    """
    response_text = labelled.get("response_text") or row.get("response_text") or ""
    evidence_text = labelled.get("evidence_text") or row.get("evidence_text") or ""
    units = labelled.get("evidence_units") or []
    unit_spans = [(int(u["char_start"]), int(u["char_end"])) for u in units]
    token_support = labelled.get("token_support_labels") or []
    dead_labels_raw: list[int] = list(labelled.get("dead_weight_unit_labels") or [])
    cov_labels_raw: list[float] = list(labelled.get("coverage_unit_labels") or [])

    # Pad / truncate unit labels to num_units.
    if len(dead_labels_raw) < num_units:
        dead_labels_raw = dead_labels_raw + [0] * (num_units - len(dead_labels_raw))
    if len(cov_labels_raw) < num_units:
        cov_labels_raw = cov_labels_raw + [0.0] * (num_units - len(cov_labels_raw))
    dead_labels = torch.tensor(dead_labels_raw[:num_units], dtype=torch.float32)
    cov_labels = torch.tensor(cov_labels_raw[:num_units], dtype=torch.float32)

    tokenized = align_labels_to_tokens(
        response_text=response_text,
        evidence_text=evidence_text,
        tokenizer=tokenizer,
        max_resp=max_resp,
        max_ev=max_ev,
        token_support_word_labels=token_support,
        unit_char_spans=unit_spans,
    )

    # Clamp unit_assign to [0, num_units) with -1 preserved for padding.
    ua = tokenized["unit_assign"].clone()
    ua = torch.where((ua >= 0) & (ua < num_units), ua, torch.full_like(ua, -1))
    tokenized["unit_assign"] = ua

    phi_vec = torch.tensor(_phi_vector_from_row(row), dtype=torch.float32)

    class_key = row.get("class_key") or labelled.get("class_key") or "rag.prose.enterprise"
    class_idx = CLASS_TO_IDX.get(class_key, 0)

    turn_band = BAND_TO_IDX.get(
        (labelled.get("turn_band") or row.get("turn_band") or "amber").lower(), 1
    )
    turn_score = float(labelled.get("turn_score") or row.get("turn_score") or 0.5)
    gold_band = BAND_TO_IDX.get(
        (labelled.get("gold_band") or row.get("gold_band") or "amber").lower(), 1
    )

    return {
        **tokenized,
        "phi_vector": phi_vec,                 # (F,)
        "class_idx": torch.tensor(class_idx, dtype=torch.long),
        "turn_band": torch.tensor(turn_band, dtype=torch.long),
        "turn_score": torch.tensor(turn_score, dtype=torch.float32),
        "gold_band": torch.tensor(gold_band, dtype=torch.long),
        "dead_weight_unit_labels": dead_labels,
        "coverage_unit_labels": cov_labels,
        "pair_id": row.get("pair_id") or labelled.get("pair_id") or "",
        "role": labelled.get("role") or row.get("role") or "grounded",
    }


def build_batch(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack a list of examples produced by :func:`build_example`."""
    B = len(examples)
    out: dict[str, Any] = {}
    tensor_keys = (
        "resp_ids", "resp_mask", "ev_ids", "ev_mask",
        "unit_assign", "token_support_aligned",
        "class_idx", "turn_band", "turn_score",
        "gold_band", "dead_weight_unit_labels", "coverage_unit_labels",
    )
    for key in tensor_keys:
        out[key] = torch.stack([ex[key] for ex in examples])

    F = examples[0]["phi_vector"].shape[0]
    phi = torch.stack([ex["phi_vector"] for ex in examples])  # (B, F)
    Tr = out["resp_ids"].size(1)
    Te = out["ev_ids"].size(1)
    out["phi"] = phi[:, None, None, :].expand(B, Tr, Te, F).contiguous()
    out["pair_ids"] = [ex["pair_id"] for ex in examples]
    out["roles"] = [ex["role"] for ex in examples]
    return out


__all__ = [
    "CLASS_TO_IDX",
    "BAND_TO_IDX",
    "align_labels_to_tokens",
    "build_batch",
    "build_example",
]
