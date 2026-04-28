# TRACE public roadmap

Last refreshed: 2026-04-28.  Monthly cadence; the next refresh is
committed before the 1st of each month.

## v1.0 — shipped (2026-04-12)

Stateless groundedness scorer for RAG + agent pipelines.

- `standard`, `quality`, `code` profiles.
- Self-hosted Helm chart and hosted `api.latence.ai`.
- Verified on the Veracier Industries benchmark
  (`data/veracier-industries/proof_bundle_v1/`) with 97% green
  precision, 95% red precision, 85% amber agreement.

## v1.1 — sellability (this release, 2026-04-28)

Everything in `/root/.cursor/plans/trace_v1_enterprise_sellability_7b897b50.plan.md`.

- **Product**
  - Per-tenant threshold routing with hot reload.
  - Append-only audit log with teacher-channel capture.
  - Multilingual: first-class FR, ES, IT in addition to EN + DE.
  - LangGraph, CrewAI, AutoGen, Haystack, Pydantic AI adapters.
  - TypeScript SDK (`@latence/trace`) with retries and OTel.
  - n8n community node + HTTP Request blueprint.
  - Remote MCP endpoint at `api.latence.ai/mcp` (SSE +
    streamable-HTTP).
  - Amber reviewer queue API: export + accept/edit/reject
    recording (v2 prep).
- **Platform**
  - Cloudflare gateway: API keys, Durable Object rate limits,
    Stripe-ready usage metering.
  - Worker right-sizing report with isolated + sustained latency
    lanes and autoscale guidance.
- **Commercial**
  - Hosted SLA, DPA, price card, trust center dates, incident-
    response runbook, order-form annex.
  - Six vertical one-pagers (finance / legal / HR / compliance /
    engineering / marketing).
  - External benchmark report (HaluEval QA + Summ, RAGTruth QA).
  - Failure appendix in the proof bundle.

## v1.2 — GA hardening (target 2026-06-15)

- **Product**
  - Public replayable sandbox at `latence.ai/try`.
  - ROI calculator at `latence.ai/roi`.
  - 5-minute quickstart + Colab + curl-only variants.
  - Self-serve signup with 1 000-score free tier.
- **Commercial**
  - Three signed design partners with quantified outcomes (one
    per vertical minimum).
  - SOC 2 Type I letter, ISO 27001 readiness, pentest pack.
  - CAIQ-Lite and SIG-Lite pre-fills.
  - Launch content pack (blog, HN, social, PH, demo video).
- **Platform**
  - Status page wired to uptime monitors + CHANGELOG mirror on
    portal.
  - Weekly v2-trigger monitor with go/no-go memo template.

## v2.0 — learned local-support head (target 2026-09-01, conditional)

Conditional on the v2-trigger monitor firing.  Triggers:

- External red precision (RAGTruth QA) stays < 0.55 for two weeks
  with the current thresholds, **or**
- External red recall (HaluEval QA) stays < 0.70 for two weeks,
  **or**
- Veracier headline metrics regress more than 2 pp, **or**
- A design-partner vertical reports sustained amber rate > 40%.

The v2 plan is fully scoped in
`/root/.cursor/plans/trace_v1_enterprise_sellability_7b897b50.plan.md`:

> tiny encoder + low-rank biaffine token-to-token scorer +
> MaxSim-style pooling, distilled from the v1 API's teacher
> channels and top-k attributions captured in the audit log since
> v1.1.

The v1.1 audit log already records everything the student needs;
no additional instrumentation is required before training.

## What we are deliberately not doing

- **No opaque end-to-end scorer.**  The point of the v2 student is
  a better local support function, not a replacement for the
  late-interaction geometry.  Opaque aggregators are harder to
  trust, regress, and explain.
- **No drift-in-one-head bundling.**  Drift is temporal, not a
  token-matching problem; it gets its own head fed by rolling
  task anchors and recent-turn summaries when we reach the drift
  release.
- **No code-channel deprecation.**  The code profile's exact-
  identifier / exact-literal / AST-role channels stay in; they
  are the reason TRACE is useful for engineering RAG pipelines
  that mix prose and code.

## Feedback and requests

File issues at `github.com/latence-ai/latence-trace/issues`,
email `support@latence.ai`, or ping the community
(`#trace-roadmap` in the Latence Discord).  Road-map requests go
through the same triage as bug reports; high-signal items get a
visible badge on this page at the next monthly refresh.
