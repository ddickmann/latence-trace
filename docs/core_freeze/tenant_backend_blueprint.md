# TRACE Tenant Backend Blueprint

This is post-core-freeze work. It is documented now so compute and SDK boundaries do not drift.

## Split Of Responsibilities

| Layer | Owns | Must not own |
|---|---|---|
| SDK/plugin | Native integration, local session events, idempotency, optional caller-carried memory state | Model scoring logic |
| Tenant backend | Auth, policies, sessions, event logs, traces, InfiniMem state, insights, dashboard API | GPU model runtime |
| Compute runtime | Grounding, redaction, compression, memory-step compute, drift scoring, span extraction | Long-term trace database |
| Portal control plane | Users, orgs, deployment registry, billing, license records | Raw trace logs |

## Deployment Modes

- Cloud evaluation: shared tenant backend with tenant isolation and shared or pooled RunPod compute.
- Dedicated cloud: dedicated backend/storage and dedicated or semi-dedicated compute.
- Customer VPC: backend, storage, and compute inside customer VPC.
- On-prem/air-gapped: backend, storage, dashboard, model bundles, and license/update packages inside customer environment.

## Minimal Tenant Backend Tables

- `tenants`
- `api_keys`
- `sessions`
- `events`
- `traces`
- `memory_spans`
- `policies`
- `insight_rollups`

Supabase in the shared Latence portal should store control-plane metadata only: organizations, users, deployment registry, deployment keys, license records, billing metadata, and enterprise requests.

## Migration Rule

Do not repurpose the portal for trace storage until the compute runtime and SDK contract are frozen. The portal should first expose deployment selection and quickstart surfaces around a stable runtime.
