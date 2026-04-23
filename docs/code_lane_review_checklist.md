# Code-lane review checklist

Every PR that touches `latence_trace/core/code_lane/`, `runpod/handler.py`,
or `latence_trace/api/service.py` must be reviewed against the items
below. Paste this checklist into the PR description and tick each
item before requesting merge.

## Correctness

- [ ] RAG parity: `tests/api/test_rag_lane_parity.py` passes without
      regenerating the golden fixture.
- [ ] New public functions have a docstring that explains *why* (not
      just *what*).
- [ ] New unit tests added; coverage for `core/code_lane/**` stays
      ≥ 85 %.
- [ ] No new `raise` in the hot path without a fallback (the service
      must never crash a turn on a missing signal).

## Performance

- [ ] Per-signal budgets in `docs/code_lane_performance.md` still
      hold (re-run the cascade benchmark if the signal landed in
      the cascade or the orchestrator).
- [ ] Hot-path allocations are bounded: no per-turn construction of
      tree-sitter parsers, logistic composites, or NLI providers.
- [ ] Any new CUDA tensor op uses the scorer's dedicated stream if
      it is not already on CPU.

## Concurrency / blocking

- [ ] No new synchronous HTTP calls. Use the async provider paths
      (`entail_async`, `healthcheck_async`).
- [ ] CPU-bound helpers that can exceed ~5 ms on p95 (tree-sitter
      parsing, sklearn fits, compression) are wrapped with
      `asyncio.to_thread`.
- [ ] Per-lane semaphores in `runpod/handler.py` are not bypassed.

## Observability

- [ ] New signals appear in the turn event
      (`docs/code_lane_turn_event.md`) and have a named Prometheus
      metric (see `latence_trace/observability/metrics.py`).
- [ ] Logs never contain `query_text`, `response_text`, or raw file
      contents. Hashed/opaque IDs only.

## API / contract

- [ ] New response fields default to safe values (`None`, `0`) so old
      clients still parse.
- [ ] `GroundednessRequest` / `GroundednessResponse` Pydantic models
      have `json_schema_extra` examples for both `rag` and `code`
      scoring modes.

## Docs

- [ ] `docs/code_lane_v3.md` is up to date with any new signal or
      behaviour change.
- [ ] If the public client contract changes, `plugin_client/` is
      synced.
