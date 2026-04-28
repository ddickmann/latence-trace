# TRACE v1.1 bold-claims ledger

Single source of truth for every public or sales claim about TRACE v1.1.
Every row below maps exactly to an artifact under
`data/veracier-industries/proof_bundle_v1/` or under `commercial/`.
If a claim cannot be mapped to an artifact, it must not be shipped.

This document is read by marketing, sales, support, and legal before
anything goes live on latence.ai, in a pitch deck, on social, or in
a Product Hunt launch.

## Defence ledger

| # | Claim | Proof artifact | Caveats / qualifier |
|---|---|---|---|
| 1 | "TRACE flags hallucinated RAG outputs in real time" | `proof_bundle_v1/proof_report.md`, headline table; latency bench at `proof_bundle_v1/latency_bench/` | Real-time = p95 under sustained concurrency of 8 on A10 worker (see latency contract). |
| 2 | "100% red precision on the Veracier Industries benchmark" | `proof_bundle_v1/validation_summary.json` + `proof_report.md` section 2 | Scope: 118 curated enterprise RAG cases across 6 verticals (finance, legal, HR, compliance, engineering, marketing). |
| 3 | "100% green precision on the Veracier Industries benchmark" | same as #2 | Same scope. |
| 4 | "Zero-shot evaluator, no tenant fine-tuning required" | `docs/guides/zero-shot-calibration.md`, `latence_trace/core/thresholds.py` default strata | True for verticals shipped in v1.1; edge cases are routed to amber, not auto-accepted. |
| 5 | "First-class multilingual scoring: EN / DE / FR / ES / IT" | `docs/guides/multilingual.md` language matrix, `latence_trace/core/groundedness.py` `EPISTEMIC_HEDGE_CUES` + `DEFAULT_NULL_BANK_TEXTS` | Benchmarked on Veracier variants only; external multilingual benchmarks TBD for v1.2. |
| 6 | "Agent-ready via MCP, LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Haystack, Pydantic AI, OpenAI, n8n, and native TypeScript + Python SDKs" | `data/veracier-industries/proof_bundle_v1/integration_proof.json` | Live-run proof of 15/15 transports from `scripts/prove_integrations.py` against a GPU-backed worker. |
| 7 | "Sub-second isolated scoring on a single A10" | `proof_bundle_v1/latency_bench/isolated.json` (p50/p95/p99 by profile) | Isolated = concurrency=1 single-request lane. |
| 8 | "Linear latency up to concurrency 32" | `proof_bundle_v1/latency_bench/sustained.json` | p95 grows sub-linearly up to concurrency=32; beyond that plan capacity. |
| 9 | "99.5% monthly availability target with credits" | `commercial/hosted-sla.md` | Target only; credits are contractual, not a promise of zero outages. |
| 10 | "No customer payload is persisted on the hot path" | `commercial/hosted-dpa.md`; `gateway/cloudflare/src/index.ts`; `runpod/handler.py` audit-log opt-in gate | Audit log is opt-in and tenant-scoped; off by default on the free tier. |
| 11 | "Per-tenant thresholds, hot-reloaded from Cloudflare KV" | `latence_trace/core/thresholds.py`; `gateway/cloudflare/src/index.ts` `X-Latence-Tenant-Id` forwarding | Hot reload window <=30s. |
| 12 | "Append-only per-score audit log with teacher signals" | `latence_trace/middleware/audit_log.py`, tests under `tests/middleware/` | v2-ready: each record carries per-channel scores and top-k attributions. |
| 13 | "Transparent failure appendix documents every known weak case" | `proof_bundle_v1/failure_appendix.md` + `proof_bundle_v1/failure_modes_v1.md` | Updated on every weekly `v2_trigger_monitor.py` run. |
| 14 | "Reviewer queue with CSV export and decision recording" | `docs/amber_reviewer_queue.md`; `latence_trace/middleware/amber_queue.py`; tests under `tests/middleware/test_amber_queue.py` | Accept/edit/reject decisions are persisted per-tenant. |
| 15 | "Status + changelog pages live at status.latence.ai and /changelog" | `docs/operations/status-and-changelog.md`, `CHANGELOG.md` | Automated from GitHub release tags. |
| 16 | "Three named design partners" | `commercial/design-partners/*.md` | Names and quantified outcomes belong in the partner-signed briefs, not this public ledger. |

## Controlled vocabulary

When describing TRACE externally, always use the phrasing on the left.
The phrasing on the right is NOT defensible and must not appear in
any customer-facing surface.

| Use this | Avoid this |
|---|---|
| "real-time groundedness verifier" | "hallucination detector" (too broad) |
| "100% red/green precision on Veracier Industries v1" | "100% accuracy" (unscoped) |
| "zero-shot for the six shipped verticals" | "zero-shot for everything" |
| "we flag, customer decides" | "we fix hallucinations" |
| "p95 < Xms under concurrency N" | "sub-Xms latency" (unscoped) |
| "SOC 2 Type I in readiness" (until auditor signs) | "SOC 2 certified" |
| "ISO 27001 roadmap" | "ISO 27001 certified" |
| "GDPR-aligned hosting" | "GDPR certified" (no such cert exists) |

## External benchmark positioning

On external corpora TRACE v1.1 is competitive but not best-in-class:

* HaluEval QA @ standard red precision: **0.59**  (target 0.80)
* HaluEval Summ @ standard red precision: **0.68**  (target 0.85)
* RAGTruth QA @ quality red precision: **0.39**  (target 0.80)
* RAGTruth Summ @ standard red precision: **0.47**  (target 0.80)

These numbers are published in `proof_bundle_v1/external_benchmarks.md`
and `proof_bundle_v1/failure_modes_v1.md`. The sales guidance is:

> Lead with Veracier Industries. Acknowledge HaluEval / RAGTruth gap.
> The v2 biaffine student is in training (see `research/triangular_maxsim/
> student_v2/`) to close the gap.

Do **not** claim best-in-class on HaluEval or RAGTruth until the v2
student ships and clears its success criteria.

## Review cadence

| Owner | Cadence | Action |
|---|---|---|
| Head of Product | Weekly | Run `scripts/v2_trigger_monitor.py`, update this sheet if any row changed. |
| Head of Marketing | Bi-weekly | Diff this sheet against the latest blog / deck / social draft. |
| Head of Legal | Monthly | Confirm SLA / DPA / attestation rows still hold. |

Any edit to this sheet must come with a commit that touches at least
one proof artifact or a dated exec sign-off under
`commercial/sign-offs/`.
