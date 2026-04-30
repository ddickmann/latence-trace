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
| `code.agentic_trace` | `trajectory_ranker` | enabled | trajectory-native held-out gate clears AUROC and false-decision thresholds |

## Runtime Contract

Enabled heads may contribute `head_score`, `head_features_used`, and
`head_reason_codes` to the runtime decision record. Head artifacts are
executable: enterprise uses response-score passthrough and the other promoted
root-cause lanes use serialized linear feature heads. Feature-gated heads require
`runtime_head_features` / `trajectory_features`; when those maps are absent the
affected request remains rollback-safe `auto_repair`. Promotion now requires
both safe autonomous allow coverage and safe autonomous block coverage.

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
