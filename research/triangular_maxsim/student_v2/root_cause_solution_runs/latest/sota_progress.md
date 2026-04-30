# TRACE SOTA Progress

## Current Gate Read

The root-cause head iteration promotes 5 of 6 routed use cases for
allow/block/repair operation under the internal no-regression,
false-decision, and latency gates.

| class | enabled | selected head | gate read |
|---|---:|---|---|
| `rag.prose.enterprise` | yes | `optimized_calibrator` | prior optimized fusion gate still passes |
| `rag.prose.short_factoid` | yes | `atom_verifier` | targeted atom verifier clears held-out false-decision gates |
| `rag.prose.multi_claim` | yes | `claim_decomposer` | validation-calibrated abstain band clears held-out false-decision gates |
| `rag.structured` | yes | `cell_schema_verifier` | adversarial typed-cell lane clears no-regression and false-decision gates |
| `rag.code_in_context` | yes | `identifier_ranker` | identifier/API drift lane clears held-out false-decision gates |
| `code.agentic_trace` | no | `trajectory_ranker` | manufactured trajectory banks still fail AUROC and false-decision gates |

## What Changed

- Added a non-public targeted data lane for root-cause failures:
  factoid swaps, partial multi-claim support, structured cell mismatches,
  code identifier/API drift, and trajectory drift.
- Added validation-calibrated allow/block thresholds with an abstain band, so
  heads do not have to force low-confidence cases into autonomous decisions.
- Added root-cause rule heads and a v1-abstain policy candidate to the bake-off.
- Regenerated runtime head artifacts and the runtime head registry proposal.

## Remaining Blocker

`code.agentic_trace` is not SOTA-ready. The dedicated manufactured trajectory
gate remains the blocker:

- `transcripts_v2`: AUROC 0.6931, false_allow 0.25, false_block 0.4833.
- `both`: AUROC 0.6061, false_allow 0.1724, false_block 0.5769.

The available trajectory score features are not separable enough for a
production autonomous decision head. The next iteration must add real
trajectory features: turn order, patch/test outcome alignment, file ownership,
symbol/API drift, and action-result causality.
