# TRACE Memory / InfiniMem Architecture Notes

TRACE Memory builds on existing TRACE runtime surfaces rather than introducing
a separate memory stack.

## Current Coding-Agent Flow

1. `GroundednessRequest` enters `GroundednessService.groundedness`.
2. Corpus routing resolves `code.agentic_trace` or another class.
3. Code requests run through the code lane, which emits AST drift, literal
   novelty, NLI cascade, composite score, file attribution, and risk signals.
4. `runtime_feature_synthesizer.synthesize_runtime_features` derives trajectory
   features when callers do not provide `runtime_head_features`.
5. `runtime_decision.build_runtime_decision` applies the promoted head and policy
   thresholds.
6. Optional caller-carried `session_state` is updated through
   `code_lane.session.update_session_state`.

## Memory Hook

InfiniMem should sit beside the existing caller-carried `session_state`:

- `session_state` keeps compact scalar temporal statistics.
- `memory_state` keeps span-level hot/warm/cold memory.

The server stays stateless. Callers echo `next_memory_state` into the next
request, just like `next_session_state`.

## TRACE Signals Used

RAG memory survival can use support unit `usage_state`, coverage, attribution,
literal diagnostics, NLI claims, structured-source verification, semantic
entropy, and file/source attribution.

Coding memory survival can use code composite scores, AST phantom signals,
literal novelty, NLI cascade results, runtime trajectory features, file owner
share, query owner share, dead-file candidates, and session drift signals.

## Production Rule

Memory decisions start in shadow mode. The system may return a compact
`hot_context_preview` and diagnostics, but it must not rewrite scoring input by
default until offline and shadow gates prove no degradation against full context,
sliding window, plain compression, vector memory, and summary memory baselines.
