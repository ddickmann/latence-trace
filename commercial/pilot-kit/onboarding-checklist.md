# Pilot Onboarding Checklist

> Day-by-day checklist used jointly by Provider SE and Customer
> Engineering during the first two weeks of a pilot. Keeps the
> 60-day clock honest.

## Day 1 -- kickoff

- [ ] Pilot Agreement signed; license JWT generated and delivered
      out-of-band to Customer security contact.
- [ ] Slack / Teams shared channel created; on-call rotation
      defined on both sides.
- [ ] Success criteria locked (`pilot-success-criteria-template.md`).
- [ ] Customer's representative environment identified (staging or
      shadow-prod).

## Day 1-3 -- environment

- [ ] Container image pulled (`ghcr.io/latence-ai/latence-trace:<tag>`).
- [ ] Helm chart values reviewed; resource requests + limits set.
- [ ] License JWT loaded via Customer's secret store (Vault /
      External Secrets / Sealed Secrets).
- [ ] vLLM-Factory sidecar (or Customer's own model server) wired in
      for ColBERT + NLI.
- [ ] `helm install` succeeds; `/healthz` and `/readyz` return 200.

## Day 3-5 -- integration

- [ ] Python SDK installed in Customer's RAG service
      (`pip install latence-trace-client`).
- [ ] First request scored end-to-end via SDK.
- [ ] OpenTelemetry traces visible in Customer's collector (if
      enabled).
- [ ] Prometheus scraping `/metrics`; the four golden signals are
      visible on Customer's dashboards.

## Day 5-10 -- calibration

- [ ] Pull a representative labelled sample (>= 200 items) from
      Customer's traffic.
- [ ] Run the calibration CLI per
      `docs/operations/calibration-runbook.md`.
- [ ] Refit thresholds; commit them to Customer's chart values.
- [ ] Re-test on a held-out 50-item sample; AUROC noted.

## Day 10-14 -- shadow + alerts

- [ ] Shadow-mode rollout in front of Customer's traffic (no user
      impact).
- [ ] Alerting in place: `latence_trace_request_errors_total`,
      `latence_trace_score_band_ratio{band="amber"}`,
      `latence_trace_request_duration_seconds_p95`,
      `latence_trace_license_days_to_expiry`.
- [ ] Day-14 health check call.

## Day 15-29 -- ramp

- [ ] Canary 5% -> 25% -> 50% -> 100% over the window.
- [ ] Daily metrics shipped to Customer + Provider shared dashboard.
- [ ] Document any incidents in the shared channel.

## Day 30 -- mid-pilot review

- [ ] Refresh `pilot-success-criteria-template.md` with day-30
      numbers.
- [ ] Refresh ROI calculator with real numbers.
- [ ] Review with exec sponsor; capture decisions in the tracker.

## Day 31-59 -- iterate

- [ ] Tackle issues from day-30 review.
- [ ] Optionally widen scope (more workflows, more languages).
- [ ] Calibrate again if the data distribution drifted.

## Day 60 -- end-of-pilot

- [ ] Final metrics captured.
- [ ] Day-60 review with exec sponsor.
- [ ] Decision recorded in the tracker (`won` / `lost` / `extended`).
- [ ] If `won`: kick off MSA + Order Form (target close in 30 days).
- [ ] If `won` and Customer agrees: draft the case study.
