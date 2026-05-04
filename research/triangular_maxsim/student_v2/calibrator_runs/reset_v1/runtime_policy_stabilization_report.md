# Runtime Policy Stabilization Report

This report promotes only the optimized no-regression policy into the
production decision surface. The unconstrained learned calibrator is not used
for automatic decisions because it regressed ungrounded recall/F1.

## Policy Source

- Production policy: `latence_trace/data/runtime_policy.optimized_v1_plus_calibrator.json`
- Research source: `research/triangular_maxsim/student_v2/calibrator_runs/reset_v1/policy_optimized_no_regress_200.json`
- Fusion proof: `research/triangular_maxsim/student_v2/calibrator_runs/reset_v1/fusion_optimized_no_regress_200.json`
- Runtime DoD proof: `research/triangular_maxsim/student_v2/calibrator_runs/reset_v1/runtime_dod_optimized_200/runtime_dod_report.md`

## Class Decisions

| class | action mode | allow threshold | block threshold | false allow | false block | support | production stance |
|---|---|---:|---:|---:|---:|---:|---|
| `rag.prose.enterprise` | allow/block/repair | 0.83 | 0.50 | 0.0106 | 0.0 | 288 | automatic decision enabled when runtime policy is enabled |
| `rag.structured` | repair-only | 1.00 | 0.00 | 0.0 | 0.0 | 243 | no autonomous allow/block yet |
| `rag.prose.multi_claim` | repair-only | 1.00 | 0.00 | 0.0 | 0.0 | 672 | no autonomous allow/block yet |
| `rag.prose.short_factoid` | repair-only | 1.00 | 0.00 | 0.0 | 0.0 | 401 | no autonomous allow/block yet |
| `rag.code_in_context` | repair-only | 1.00 | 0.00 | 0.0 | 0.0 | 1 | insufficient support |
| `code.agentic_trace` | repair-only | 1.00 | 0.00 | 0.0 | 0.0 | 1 | insufficient support; trajectory benchmark did not clear gate |

## Rollback

The runtime decision layer is disabled by default. Rollback options:

- unset `LATENCE_TRACE_RUNTIME_DECISION_ENABLED`;
- point `LATENCE_TRACE_RUNTIME_POLICY_PATH` at a stricter policy;
- set all class `allow_disabled` / `block_disabled` values to `true`.

This preserves the existing v1/router scoring path and makes the new decision
surface opt-in rather than a forced behavior change.
