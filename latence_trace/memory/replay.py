"""TRACE replay helpers for canonical memory trajectories."""

from __future__ import annotations

from typing import Any

from latence_trace.api.models import GroundednessRequest, ScoringMode
from latence_trace.memory.trajectory import CanonicalTrajectory


def replay_trace_signals(
    trajectory: CanonicalTrajectory,
    *,
    service: Any | None = None,
) -> CanonicalTrajectory:
    """Attach TRACE-like per-turn signal bundles to a canonical trajectory.

    When a ``GroundednessService`` is supplied, this runs real TRACE scoring.
    Offline tests and dataset preparation can use the deterministic proxy path
    to validate schema, counterfactual labeling, and baseline plumbing without
    requiring model servers.
    """

    updates = []
    for turn in trajectory.turns:
        if service is not None and (turn.response_text or "").strip():
            request = GroundednessRequest(
                query_text=turn.query_text,
                response_text=turn.response_text or turn.turn_text or "empty",
                raw_context=_context_for_turn(turn),
                scoring_mode=ScoringMode.CODE if trajectory.domain == "code" else ScoringMode.RAG,
                emit_chunk_ownership=trajectory.domain == "code",
                enable_memory_shadow=False,
            )
            response = service.groundedness(request)
            trace_scores = response.model_dump(
                mode="json",
                include={
                    "scores",
                    "runtime_head_features",
                    "runtime_decision",
                    "session_signals",
                    "file_attribution",
                    "code_lane_diagnostics",
                    "corpus_route",
                },
            )
        else:
            trace_scores = _proxy_scores(trajectory, turn.turn_index)
        updates.append(turn.model_copy(update={"trace_scores": trace_scores}))
    return trajectory.model_copy(update={"turns": updates})


def _context_for_turn(turn: Any) -> str | None:
    parts = [turn.raw_context or ""]
    parts.extend(span.text for span in turn.retrieved_context)
    parts.extend(span.text for span in turn.tool_outputs)
    text = "\n".join(part for part in parts if part.strip())
    return text or None


def _proxy_scores(trajectory: CanonicalTrajectory, turn_index: int) -> dict[str, Any]:
    turn = trajectory.turns[turn_index - 1]
    context = _context_for_turn(turn) or ""
    response = turn.response_text or turn.turn_text
    overlap = _term_overlap(context, response)
    unused = 1.0 - overlap if context else 0.0
    code_alignment = 0.9 if trajectory.domain == "code" and _has_code_cue(response + context) else overlap
    tool_alignment = 0.85 if trajectory.domain == "tool" and ("tool" in response.lower() or "api" in response.lower()) else overlap
    return {
        "scores": {
            "groundedness_v2": overlap,
            "reverse_context": overlap,
            "context_usage_ratio": overlap,
            "context_unused_ratio": unused,
            "dead_weight_ratio": unused,
        },
        "runtime_head_features": {
            "file_alignment": code_alignment,
            "symbol_alignment": code_alignment,
            "context_attribution_ratio": overlap,
            "dead_weight_ratio": unused,
            "test_outcome_alignment": code_alignment,
            "tool_result_alignment": tool_alignment,
        },
        "runtime_decision": {
            "action": "allow" if overlap >= 0.5 or trajectory.domain in {"code", "tool"} else "auto_repair",
            "reason_codes": ["proxy_replay"],
        },
        "trajectory_domain": trajectory.domain,
    }


def _term_overlap(left: str, right: str) -> float:
    left_terms = {term.lower() for term in left.split() if len(term) > 3}
    right_terms = {term.lower() for term in right.split() if len(term) > 3}
    if not left_terms or not right_terms:
        return 0.45
    return min(1.0, len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms))) + 0.25)


def _has_code_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in ("src/", ".py", "pytest", "class ", "def ", "function "))
