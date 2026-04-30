# TRACE Stabilization Coding-Trajectory Report

Generated from manufactured coding-agent trajectory banks using
`research/triangular_maxsim/coding/coding_agent_validation.py`.

## Verdict

`code.agentic_trace` should remain `auto_repair` / beta for automatic
decisioning. The manufactured trajectory benchmarks do not clear the
`reverse_context AUROC >= 0.90` gate.

## Results

| case bank | cases | best observed AUROC | flagship AUROC | partial monotonicity | p95 ms | verdict |
|---|---:|---:|---:|---:|---:|---|
| `transcripts_v1` | 20 | 0.7344 (`sentence_packed/gte_only`) | 0.6250 | 0.8750 | 2200.25 | no-go |
| `transcripts_v2` | 180 | 0.7917 (`sentence_packed/gte_only`) | 0.6969 | 0.9333 | 1993.57 | no-go |
| `both` | 62 | 0.6247 (`sentence_packed/gte_only`) | 0.5955 | 0.8750 | 886.84 | no-go |

## Interpretation

- The generic coding benchmark was not the right primary comparator for
  agentic coding trajectories; the manufactured transcript banks are.
- The current flagship cell, `colgrep/fuse_mean`, underperforms the simpler
  `gte_only` or `fuse_max` readouts on these trajectories.
- The score distributions remain too saturated for robust allow/block
  automation. Human review should not be forced for all use cases, but this
  class is not yet calibrated enough for autonomous high-risk actions.

## Artifacts

- `research/triangular_maxsim/coding/artifacts/trace_stabilization_transcripts_v1.json`
- `research/triangular_maxsim/coding/artifacts/trace_stabilization_transcripts_v1.md`
- `research/triangular_maxsim/coding/artifacts/trace_stabilization_transcripts_v2.json`
- `research/triangular_maxsim/coding/artifacts/trace_stabilization_transcripts_v2.md`
- `research/triangular_maxsim/coding/artifacts/trace_stabilization_both.json`
- `research/triangular_maxsim/coding/artifacts/trace_stabilization_both.md`
