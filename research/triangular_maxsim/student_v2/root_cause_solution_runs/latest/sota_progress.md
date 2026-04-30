# TRACE SOTA Progress

## Current Gate Read

The root-cause head iteration promotes 6 of 6 routed use cases with executable
runtime-head artifacts under the stricter internal no-regression,
false-decision, latency, and per-action coverage gates. All six classes now have
nonzero safe autonomous allow and block coverage in the local held-out reports.

| class | enabled | selected head | gate read |
|---|---:|---|---|
| `rag.prose.enterprise` | yes | `optimized_calibrator` | prior optimized fusion gate still passes |
| `rag.prose.short_factoid` | yes | `atom_verifier` | targeted atom verifier clears held-out false-decision gates |
| `rag.prose.multi_claim` | yes | `claim_decomposer` | validation-calibrated abstain band clears held-out false-decision gates |
| `rag.structured` | yes | `cell_schema_verifier` | typed-cell provenance lane clears no-regression, false-decision, and two-sided autonomous gates |
| `rag.code_in_context` | yes | `identifier_ranker` | identifier/API drift lane clears held-out false-decision gates |
| `code.agentic_trace` | yes | `trajectory_ranker` | trajectory-native held-out gate clears AUROC and false-decision thresholds |

## What Changed

- Added a non-public targeted data lane for root-cause failures:
  factoid swaps, partial multi-claim support, structured cell mismatches,
  code identifier/API drift, and trajectory drift.
- Added validation-calibrated allow/block thresholds with an abstain band, so
  heads do not have to force low-confidence cases into autonomous decisions.
- Tightened promotion to require both safe allow coverage and safe block
  coverage, preventing block-only or allow-only heads from counting as
  autonomous-ready.
- Added root-cause rule heads and a v1-abstain policy candidate to the bake-off.
- Added trajectory-native banks with file, symbol/API, test outcome, patch
  result, temporal order, partial-support, and near-miss hard negatives.
- Promoted serialized runtime strategies for all six heads: response-score
  passthrough, explicit-feature logistic heads, and trajectory-native logistic
  scoring.
- Regenerated runtime head artifacts and the runtime head registry proposal, and
  aligned policy thresholds with each executable artifact.

## Agentic Trace Result

`code.agentic_trace` now passes the trajectory-native held-out gate:

- `native_test`: AUROC 1.0, false_allow 0.0079, false_block 0.0,
  decision_coverage 1.0.

The legacy similarity-only transcript banks remain diagnostic only because they
do not carry the file/test/order facts required by the promoted runtime head:

- `transcripts_v2`: AUROC 0.5 under the trajectory-native feature model.
- `both`: AUROC 0.5225 under the trajectory-native feature model.

Residual risk: live production requests must provide the explicit feature map
(`runtime_head_features` / `trajectory_features`) for feature-gated heads to make
autonomous decisions. Missing features remain rollback-safe and should be
treated as repair-only operationally. Live RunPod OOD/customer traffic should be
recorded after rebuild before making a broader customer-data SOTA claim.
