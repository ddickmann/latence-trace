# Changelog

All notable changes to Latence TRACE are captured here.  The public
mirror of this file is available at `https://latence.ai/changelog`.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and TRACE adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Hosted MCP endpoint** (`c6_hosted_mcp_endpoint`): remote Model
  Context Protocol over SSE and streamable-HTTP at
  `https://api.latence.ai/mcp`, with drop-in manifests for Claude
  Desktop, Cursor, and the OpenAI Agents SDK.
- **TypeScript SDK** (`@latence/trace`, `c3_js_ts_sdk`): Node 18+,
  Deno, Bun, Cloudflare Workers. Retries with jitter, `Retry-After`
  honour, optional OTel tracer, and full types.
- **n8n community node** (`n8n-nodes-latence-trace`, `c7`): Score
  Groundedness and Route-by-Band actions plus an HTTP Request
  blueprint workflow for same-day installs.
- **LangGraph, CrewAI, AutoGen, Haystack 2, Pydantic AI adapters**
  (`c2`).
- **Multilingual hedge cue + null-bank coverage** for FR/ES/IT
  (`c1`); now five first-class languages with the language matrix
  published at `docs/guides/multilingual.md`.
- **Per-tenant threshold routing** with ConfigMap hot-reload
  (`b2_per_tenant_thresholds`).
- **Append-only audit log** with non-blocking background writer and
  teacher-channel capture for future V2 distillation
  (`b6_audit_log`).
- **Cloudflare gateway**: API key issuance / rotation / revocation,
  Durable-Object rate limits, D1 usage metering, Stripe export
  (`b1_tenant_api_keys`, `b3_billing`).
- **Enterprise docs**: hosted SLA (`b4`), DPA (`b5`), incident
  response runbook (`d3`), price card (`d2`), order form annex
  (`d4`), per-vertical briefs (`e2`), vertical risk-band semantics
  (`e3`).
- **Proof bundle v1** under `data/veracier-industries/proof_bundle_v1/`
  with curated variants, marketing use cases, and a reproducer
  script (`a1`-`a8`, `g3`).

### Changed

- `runpod/handler.py` emits an audit log record on every successful
  score call, wired through `latence_trace/middleware/audit_log.py`.

### Fixed

- `docs/enterprise_rag_guide.md` adapter paths corrected from
  `adapters/` to `integrations/` (`c5`).

## [1.0.0] - 2026-04-12

Initial public release of TRACE v1.

- Stateless groundedness scorer over ColBERT / MaxSim plus
  NLI-aggregate teacher channels.
- `standard`, `quality`, `code` profiles.
- Veracier benchmark proof at 97% green precision / 95% red
  precision / 85% amber agreement.
- Self-hosted (Helm chart) and hosted (`api.latence.ai`) deployments.
