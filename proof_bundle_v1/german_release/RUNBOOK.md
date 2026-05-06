# German release verification runbook

This is the operator-facing checklist for taking the per-language
calibration work over the finish line. The code (Phase A, B, and the
loader / sweep skeleton of Phase C) is shipped; the steps below are
the things that require external infrastructure (OpenAI API key, a
GPU-backed runtime, and the RunPod control plane) and therefore cannot
be executed from a sandbox.

## Where we stand

Code-complete (in this branch):

- Per-request inference knobs (`language`, `nli_top_k_premises`,
  `nli_premise_concat`, `nli_premise_aggregate`) on `GroundednessRequest`,
  RunPod handler, and the bridge.
- `latence_trace.core.language_detector` with `langdetect` (sub-ms
  is_german probe, deterministic seed).
- `_apply_language_defaults` in `latence_trace.api.service` applies the
  balanced German defaults (`top_k=2, concat=False, aggregate="max"`)
  and surfaces them in `runtime_decision.profile_diagnostics`.
- `latence_trace.core.corpus_router.bundles` is `(class_key, language)`-
  aware with a logged English fallback (`bundle_language_fallback`).
- `latence_trace.middleware.corpus_router` resolves request language
  once and threads it into every `load_bundle` call.
- Demo bridge wraps `raw_context` with `# file:` headers and forwards
  the bridge-side language hint.
- Frontend calibration chip + per-channel score row.
- Three new operator scripts:
  - `scripts/translate_bench_to_german.py` (GPT-4.1 translation,
    stratified, resumable, dry-runnable).
  - `scripts/cache_channel_scores.py --language de` (consumes the
    German JSONL, forwards `language="de"`, writes
    `<class>.de.<split>.jsonl`).
  - `scripts/calibrate_per_class.py --language de` (sweeps the German
    cache, writes `latence_trace/data/calibration.<class>.de.json`,
    sanity-gates the German metric vs. the English baseline).
- Local test suite for the touched modules: `pytest_summary.txt`
  alongside this runbook captures all 101 affected tests passing.

Operator-only (need OpenAI key / GPU runtime / RunPod):

| Step | Command | Notes |
|---|---|---|
| C.1 Translate bench | `OPENAI_API_KEY=… python scripts/translate_bench_to_german.py --per-class 200` | ~1083 rows on the current parquet (4 classes hit 200, two cap at the labelled-row count). Cost: ~1 row ≈ one GPT-4.1 call ≈ 2-4k tokens ≈ ~$0.02 → ~$25 total. |
| C.2 Cache de | `python scripts/cache_channel_scores.py --language de --url $RUNTIME_URL` | Needs the runtime running `latence-trace` with this branch (so the German NLI defaults apply to every cached row). |
| C.3 Sweep de | `python scripts/calibrate_per_class.py --language de` | Sanity-gates each class against the English baseline (--max-en-gap defaults to 0.05). Hold any class back if its German metric trails by more than that. |
| C.4 Parity | `python scripts/bench_external.py --language en` plus the Kafka A/B re-run | Asserts English F1 within ±0.01 of pre-sprint and German Kafka seed/adv lands per the plan acceptance criteria. |
| D.1 Full pytest | `pytest -q` | Targeted-suite already captured in `pytest_summary.txt`; the full sweep includes pre-existing failures unrelated to this PR (license + tool count) which the plan flags as out-of-scope. |
| D.2 EN regression | `python scripts/bench_external.py …` (HaluEval QA/Summ + RAGTruth QA/Summ) | Needs the runtime endpoint. |
| D.3 DE holdout | Build 120-row holdout (separate from calibration cache) and run via the runtime endpoint with `language=de`. |
| D.4 Kafka A/B sign-off | Re-run `de-kafka-rossmann` seed + adversarial through the bridge with `language=auto`. |
| D.5 Latency | German workload p50/p95/p99 within +20% of English baseline; verify auto-downgrade to top_k=1 when latency budget is breached. |
| D.6 Bridge + LibreChat E2E | Manual UI test: calibration chip switches en/de, per-channel scores render, claim-span heatmap is unchanged. |
| D.7 RunPod push | Bump `LATENCE_TRACE_RUNPOD_VERSION`, rebuild image, deploy to a NEW staging endpoint, re-run D.2 + D.3 + D.4 against staging, promote on parity, tag `v0.2.0-de-bundles`. |

## Acceptance gate

Promote to RunPod production only when **all** of the following hold:

- D.1 + D.2 green on staging + local.
- Per-class German F1 within 0.05 of the English equivalent (or the
  class is documented as "held back to v0.2.1" with the bundle file
  removed so the runtime falls back to English explicitly via
  `bundle_language_fallback`).
- `de-kafka-rossmann` seed lands `green` (`groundedness_v2 ≥ 0.96`,
  claim 0 ent ≥ 0.6) and adversarial lands `red` (Brüder-Karamasow
  ent ≤ 0.4).
- p99 latency on German workload below the runpod request_timeout.

## Rollback plan

- Keep the English bundle files untouched (they are bit-compatible with
  `main`). The runtime loader falls back to English on any unknown
  language, so deleting the German bundles re-ships the previous
  behaviour without a code revert.
- Pin the previous `LATENCE_TRACE_RUNPOD_VERSION` template id so
  promotion can be reverted in one click.
