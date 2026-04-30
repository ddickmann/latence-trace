# TRACE Runtime Head Artifacts

This directory contains one versioned runtime head artifact per router class.
The router selects the class, the runtime registry selects the matching head,
and the policy bundle decides whether that class can emit `allow`, `block`, or
must remain `auto_repair`.

Current registry: `latence_trace/data/runtime_head_registry.root_cause_solution_v1.json`

## Current Promotion State

| class | head | production state | reason |
|---|---|---|---|
| `rag.prose.enterprise` | `optimized_calibrator` | enabled | no-regression fusion clears false-allow/false-block gates |
| `rag.prose.multi_claim` | `claim_decomposer` | enabled | validation-calibrated abstain policy clears held-out false-decision gates |
| `rag.prose.short_factoid` | `atom_verifier` | enabled | targeted atom verifier clears entity/date/number false-decision gates |
| `rag.structured` | `cell_schema_verifier` | enabled | typed cell challenge lane clears no-regression and false-decision gates |
| `rag.code_in_context` | `identifier_ranker` | enabled | identifier/API drift lane clears held-out false-decision gates |
| `code.agentic_trace` | `trajectory_ranker` | disabled | manufactured trajectory gates do not pass yet |

## Runtime Contract

Enabled heads may contribute `head_score`, `head_features_used`, and
`head_reason_codes` to the runtime decision record. Disabled, missing, corrupt,
or checksum-mismatched heads are rollback-safe: scoring continues with v1/router
signals and the affected class stays repair-only.

## Regeneration

Run:

```bash
python research/triangular_maxsim/student_v2/root_cause_solution_tracks.py \
  --out-dir research/triangular_maxsim/student_v2/root_cause_solution_runs/latest \
  --runtime-registry-out latence_trace/data/runtime_head_registry.root_cause_solution_v1.json \
  --heads-dir latence_trace/data/heads \
  --targeted-data research/triangular_maxsim/student_v2/root_cause_targeted/targeted_v1.jsonl
```

The generated report at
`research/triangular_maxsim/student_v2/root_cause_solution_runs/latest/root_cause_solution_tracks.md`
is the source of the current production claim and next training targets.
