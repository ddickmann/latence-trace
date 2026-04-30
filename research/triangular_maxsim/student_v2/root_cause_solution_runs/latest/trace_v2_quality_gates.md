# TRACE V2 Production Readiness Quality Gates

## Executable Head Gate

Pass. `scripts/verify_trace_v2_readiness.py --skip-pytest` verifies:

- all six router classes are present and enabled in
  `latence_trace/data/runtime_head_registry.root_cause_solution_v1.json`
- artifact checksums match the registry
- no promoted head uses `descriptor_only_until_promoted`
- runtime strategies are one of `response_score_passthrough`,
  `linear_feature_model`, `tfidf_linear_model`, or `rule_feature_model`
- policy autonomous paths match artifact capabilities
- every class has `allow_disabled: false`, `block_disabled: false`,
  `allowed > 0`, and `blocked > 0`

The previous block-only `rag.structured` state is closed. The structured head now
uses row/column/value/unit/date/schema provenance features and records both safe
allows and safe blocks under the same strict readiness harness.

## Focused Reproducibility Gate

Pass. Focused local command:

```bash
python -m pytest tests/core/test_runtime_decision.py tests/research/test_root_cause_solution_tracks.py tests/research/test_coding_trajectory_head.py tests/test_runpod_handler.py -q
```

Result after autonomous-threshold closure: `18 passed`, followed by
`artifact/policy readiness checks passed`.

## OOD / Generalization Gate

Pass for the held-out synthetic/root-cause banks currently available in this
repo. The readiness harness replays the artifact checksum and policy contract so
the same gates can be run after GitHub push, CI rebuild, and RunPod image
promotion.

Not yet a live customer OOD claim: the report does not include private customer
or live RunPod traffic. Those checks should be recorded in a new dated report
after the deployed pod emits the same registry SHA and policy SHA.

## Runtime Fallback Gate

Pass. Feature-gated heads default to `auto_repair` when explicit feature maps
are absent. Missing/corrupt artifacts remain non-fatal and rollback-safe.
