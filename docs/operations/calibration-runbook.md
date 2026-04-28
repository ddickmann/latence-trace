# Per-Customer Calibration Runbook

> Audience: ops / ML engineer running `latence-trace` for a specific
> RAG product. Time budget: ~90 minutes the first time, ~15 minutes
> for subsequent re-fits.

`latence-trace` ships with **default risk-band thresholds**
(`green` / `amber` / `red`) that we pre-calibrated against RAGTruth
QA, HaluEval QA, and our internal sweep. Those defaults are excellent
**out-of-the-box for most workloads** -- production deployments still
benefit from a per-customer refit because:

1. Your domain has its own base rate of hallucination (legal vs
   support tickets vs medical).
2. Your retriever's chunk granularity is different from ours
   (the calibration assumes ~256-token chunks).
3. Your "amber == human review" cost is your decision, not ours.

This runbook walks you through the refit end-to-end. Run it once
during onboarding, then once a quarter to track drift.

---

## Prereqs

| Requirement | Why |
|---|---|
| `latence-trace` v1.0+ installed (or running pod) | the `latence-trace calibrate` CLI |
| 60-300 labeled `(query, response, context, faithful?)` samples per stratum | `qa`, `dialogue`, `summary` if applicable |
| ~10 GB free disk for sweep artifacts | calibration writes JSON reports |
| ~30 min wall clock on a 1× A100 (or ~90 min on CPU) | encoder + NLI sweep |

---

## Step 1 — Capture a labeled dataset

Format -- one JSONL line per sample::

```json
{"id": "ticket-001", "stratum": "qa", "query": "When is my SLA renewal?",
 "response": "Your SLA renews on July 14.",
 "context": ["Customer SLA renews annually on the anniversary of signing (2024-07-14)..."],
 "faithful": true}
```

Required keys: `id`, `query`, `response`, `context`, `faithful`.
Recommended: `stratum` (so the calibrator can fit per-domain
thresholds). Aim for at least:

- `qa`: 80 faithful + 80 hallucinated
- `dialogue`: 60 + 60 (skip if you don't run a chat product)
- `summary`: 60 + 60 (skip if you don't summarise)

Save it as `data/calibration/<customer>.jsonl`.

---

## Step 2 — Run a baseline sweep

```bash
latence-trace calibrate -- \
    --pairs-per-stratum 80 \
    --precision-target 0.75 \
    --enable-nli \
    --atomic-claims \
    --profile balanced \
    --out latence_trace/core/thresholds.acme-corp.json
```

What this does:

1. Re-runs scoring for every sample under `balanced` defaults.
2. Builds the score distribution per stratum.
3. Solves the constrained optimisation:
   *maximise recall of `red` band subject to* `precision >= 0.75`
   *and* `green band false-positive rate <= 0.05`.
4. Writes the new thresholds JSON next to your customer label.

The output JSON looks like this::

```json
{
  "primary_metric": "groundedness_v2",
  "thresholds": {
    "qa": {"green_min": 0.71, "red_max": 0.42},
    "dialogue": {"green_min": 0.66, "red_max": 0.39}
  },
  "evaluation": {
    "qa":       {"precision": 0.81, "recall": 0.74, "n": 160},
    "dialogue": {"precision": 0.78, "recall": 0.69, "n": 120}
  }
}
```

---

## Step 3 — Validate against a held-out slice

Always reserve 10-20% of your labels as a **validation set** the
calibrator never sees. Run::

```bash
latence-trace calibrate -- \
    --pairs-per-stratum 0 \
    --evaluate-only \
    --thresholds latence_trace/core/thresholds.acme-corp.json \
    --validation data/calibration/acme-corp.holdout.jsonl
```

Acceptance gate:

- `precision >= 0.75` per stratum.
- `recall >= 0.65` per stratum.
- `green-band FPR <= 0.05` (i.e. hallucinations slipping through as
  green should be < 5% on the validation set).

If any gate fails, do **not** ship -- collect more labels for the
failing stratum and re-fit.

---

## Step 4 — A/B against the shipped defaults

Stand up a second deployment with `LATENCE_TRACE_THRESHOLDS_FILE`
pointing at your refit. Mirror 5% of production traffic to both for
a week. Compare:

| Signal | Use shipped defaults if | Use refit if |
|---|---|---|
| Reviewer agreement on "amber" cases | >= 0.75 | < 0.75 |
| Recall@red on internal hallucination labels | >= 0.65 | < 0.65 |
| Green-band FPR | <= 0.05 | > 0.05 |

If both meet the gate, **ship the refit anyway** -- it's strictly
more representative of your traffic.

---

## Step 5 — Roll out

Two supported rollout modes — pick one:

### 5a. Hot-reload (zero-downtime, Business+ tier)

The scoring service watches `LATENCE_TRACE_THRESHOLDS_FILE` for
changes and hot-swaps the thresholds artefact without dropping in-flight
requests.  This is the recommended path when the `ConfigMap` is
mounted from a `RollingUpdate`-friendly volume.

```bash
kubectl --namespace rag-platform create configmap \
  lt-thresholds --from-file=thresholds.json=latence_trace/core/thresholds.acme-corp.json \
  --dry-run=client -o yaml | kubectl apply -f -
```

The pod receives the new `ConfigMap` contents within ~60 seconds and
log-lines `thresholds.hot_reload path=... version=...` confirm the
swap.  No restart required.

### 5b. Explicit rollout (initial bootstrap only)

Use this form only on the first deployment, when the mount path was
not previously configured:

```bash
helm upgrade lt latence-ai/latence-trace \
  --reuse-values \
  --set extraVolumes[0].name=thresholds \
  --set extraVolumes[0].configMap.name=lt-thresholds \
  --set extraVolumeMounts[0].name=thresholds \
  --set extraVolumeMounts[0].mountPath=/etc/latence-trace/thresholds \
  --set extraEnv[0].name=LATENCE_TRACE_THRESHOLDS_FILE \
  --set extraEnv[0].value=/etc/latence-trace/thresholds/thresholds.json
```

### Per-tenant routing

For hosted SaaS and multi-tenant self-hosted deployments, store
tenant-specific thresholds at
`/etc/latence-trace/thresholds/{tenant}.json`.  The gateway picks the
file matching the request's tenant claim; fall-back is the default
`thresholds.json` artefact.  This is the B2 per-tenant threshold
routing; see `commercial/hosted-sla.md` for tier availability.

Verify::

```bash
kubectl exec -it deploy/lt-latence-trace -- \
  curl -s http://localhost:8090/agent-help | jq '.thresholds'
```

---

## Step 6 — Set a recalibration schedule

Calibration **drifts** because:

- Your retriever changes (new embedding model, different chunk size).
- Your LLM upgrades (GPT-4o -> GPT-5 changes the hallucination
  surface).
- Your domain shifts (new product, new vertical).

Default cadence:

| Trigger | Action |
|---|---|
| Quarterly | re-fit with the latest 800-1500 labeled samples |
| Retriever / chunker change | re-fit immediately |
| LLM upgrade | re-fit within 2 weeks |
| `green-band FPR` alert from Grafana | re-fit immediately |

Wire the alert::

```promql
# alert when more than 2% of green-band requests get human-flagged
sum(rate(rag_human_flagged_total{lt_band="green"}[1h]))
  / sum(rate(rag_responses_total{lt_band="green"}[1h])) > 0.02
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All samples score below 0.30 | encoder mismatch (e.g. EN-only model on DE traffic) | switch to `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` |
| Calibrator picks `green_min == red_max` | dataset is degenerate (no hallucinations or all hallucinations) | rebalance the dataset |
| Validation precision is OK, recall is awful | false negatives concentrated on a stratum -- usually `summary` | turn on `--atomic-claims` and rerun |
| Refit is **worse** than the shipped defaults | dataset is too small or too noisy | go back to Step 1; aim for inter-rater agreement > 0.8 |
| Tabular / segment-table responses leak into `green` despite a wrong number / sign / unit | Typed Structured Evidence Lane is off (legacy fusion only) | export `VOYAGER_GROUNDEDNESS_STRUCTURED_GATE=1` (or run the `quality` profile, where it is on by default) and rerun. The lane fires only when the source is detected as `prose_table` / `numeric_fact` / `table_md` / `kv_pairs` / `json`, so prose strata are unaffected. See `docs/algorithm-audit.md` §K. |
| Pure-prose responses regress after enabling the structured gate | the `_looks_like_prose_table` detector is firing on something it should not | inspect `scores.structured_source*` in the response; if `structured_source_typed_count > 0` on a context that is genuinely prose, file the sample at `support@latence.ai` so we can tighten the detector heuristics |

---

## What we look for in a "good" calibration

A healthy refit, on a clean 200-sample dataset, lands here:

| Metric | Target |
|---|---|
| `precision@red` | 0.78 - 0.92 |
| `recall@red` | 0.65 - 0.82 |
| `green-band FPR` | <= 0.05 |
| `amber-band rate` | 8% - 25% (your reviewable budget) |

If you're outside any of those bands, open a ticket with the JSON
report attached at `support@latence.ai` -- we co-debug calibrations
for every paying customer.

---

## See also

- `docs/benchmarks.md` -- benchmark numbers for the shipped defaults.
- `docs/algorithm-audit.md` -- math + algorithm details.
- `docs/operations/observability.md` -- metrics to watch in Grafana.
- `docs/operations/deployment-checklist.md` -- pre-flight gate.
