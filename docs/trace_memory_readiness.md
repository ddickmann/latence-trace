# TRACE Memory / InfiniMem Readiness Report

## Current Status

Implemented as a shadow-mode, caller-carried v1:

- Standalone `/v1/compression` endpoint and RunPod `action="compress"`.
- Standalone `/v1/memory/update` endpoint and RunPod `action="memory.update"`.
- Optional `memory_state`, `memory_policy`, `next_memory_state`,
  `hot_context_preview`, and `memory_diagnostics` fields on groundedness calls.
- Rule-based span extraction, typing, signatures, dedup, supersession, survival,
  and hot/warm/cold selection.
- Code-agent shadow integration that returns memory diagnostics without changing
  default TRACE scoring inputs.
- Offline evaluation and label-gated baseline ranker scripts.

## Novelty Claim To Test

TRACE Memory is only product-worthy if it adds value from TRACE-specific
downstream utility signals:

- attribution and support-unit usage,
- exact literal and structured-source protection,
- code-lane AST and file attribution,
- runtime-head trajectory alignment,
- dead-weight and drift/session signals.

If observed lift comes only from reducing tokens, ship it as compression, not
InfiniMem.

## Promotion Gates

Production path remains blocked until all gates pass:

- No material task-quality regression against full history.
- Near-perfect preservation of code symbols, file paths, stack traces, numbers,
  dates, legal/financial facts, and active user constraints.
- Meaningful hot-context token reduction versus sliding window and plain
  compression.
- Better or equal groundedness/drift metrics versus raw transcript, vector
  memory, summary memory, and plain compression baselines.
- Replayed counterfactual labels exist for public, synthetic, and internal
  trajectories.
- Diagnostics explain the top survival/removal causes per span.

## Production Coupling Decision

Not approved for production enforcement yet. The implementation is safe for
shadow use and standalone memory evaluation. Wiring memory into scoring input
or agent continuation should wait for the counterfactual replay scorecards and
explicit approval.
