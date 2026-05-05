Below is the architecture I would give engineers.

The key design decision:

> **TRACE should be a deployable runtime with a thin SDK, but the product should have a lightweight cloud control plane for onboarding, deployment selection, billing, and cloud-user dashboards.**

You should **not** keep the current heavy SaaS surface as the core product. The core product is the TRACE deployment. The portal only manages deployments.

---

# 1. Product architecture in one sentence

**Latence TRACE consists of:**

1. **A self-contained compute runtime** that runs models and exposes scoring APIs.
2. **A tenant backend** that handles auth, policies, sessions, event logs, traces, insights, and state.
3. **Thin SDKs/plugins** that capture agent events and call the tenant backend.
4. **A lightweight Latence portal** for user registration, deployment selection, cloud provisioning, and billing.

---

# 2. Main architecture

```text
User / Agent / Plugin
   ↓
Thin SDK or Plugin
   ↓
Customer TRACE API Endpoint
   ↓
Tenant Backend / TRACE Gateway
   - auth
   - policies
   - sessions
   - trace logs
   - InfiniMem state
   - insights
   - decision engine
   ↓
TRACE Compute Runtime
   - groundedness
   - PII redaction
   - compression
   - guardrail models
   - drift scoring
   - span extraction
   ↓
Tenant Backend stores results
   ↓
Portal / local dashboard displays insights
```

The clean split:

| Layer               | Owns                                       | Should it be shared?                                                 |
| ------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| **SDK/plugin**      | Agent integration, local session events    | Customer-side                                                        |
| **Tenant backend**  | Logs, auth, sessions, policies, state      | Isolated per tenant logically; dedicated physically for higher tiers |
| **Compute runtime** | Models and GPU inference                   | Shared in Cloud; dedicated in VPC/on-prem                            |
| **Latence portal**  | Users, orgs, deployment selection, billing | Shared SaaS control plane                                            |

---

# 3. Deployment modes

## 3.1 Cloud evaluation deployment

This is the fast, cheap, managed option.

Use case:

* free/low-cost testing
* demos
* low-risk data
* non-sensitive traces
* early developer adoption

Architecture:

```text
Customer SDK/plugin
   ↓
Latence Cloud TRACE endpoint
   ↓
Cloud tenant backend
   ↓
Shared Runpod compute runtime
   ↓
Tenant-isolated logs and insights
   ↓
Latence portal
```

Important wording:

> **Cloud is for evaluation and low-risk workloads. Sensitive production data should use VPC or on-prem.**

Cloud can use a shared GPU runtime, but logging and auth must be tenant-isolated.

## 3.2 Dedicated cloud deployment

Use case:

* early production
* one customer wants better isolation
* still managed by you

Architecture:

```text
Customer SDK/plugin
   ↓
Dedicated tenant backend
   ↓
Dedicated or semi-dedicated compute runtime
   ↓
Dedicated DB/storage
   ↓
Latence portal control plane
```

Price this much higher than Cloud.

## 3.3 Customer VPC

Use case:

* regulated customer
* sensitive prompts/code/data
* production workloads

Architecture:

```text
Customer SDK/plugin
   ↓
TRACE backend inside customer VPC
   ↓
TRACE compute runtime inside customer VPC
   ↓
Customer-owned DB/storage
   ↓
Optional connection to Latence control plane for license/update only
```

## 3.4 On-prem / air-gapped

Use case:

* finance
* healthcare
* public sector
* defense
* critical infrastructure

Architecture:

```text
Customer network
   ├── TRACE backend
   ├── TRACE compute runtime
   ├── local database
   ├── local dashboard
   ├── local model bundle
   └── offline license/update package
```

No dependency on Supabase, Vercel, Runpod, or your cloud.

---

# 4. Important correction: do not clone the whole backend into Vercel

Vercel should not host the TRACE backend.

Use Vercel for:

* marketing pages
* user portal frontend
* deployment selector
* billing UI
* questionnaire for VPC/on-prem
* docs links

Use Supabase for:

* portal auth
* organizations
* deployment registry
* billing metadata
* license records
* cloud tenant mapping

The **TRACE tenant backend** should be its own service, not Vercel functions.

Reason:

> TRACE needs long-running jobs, local logs, model calls, background workers, tenant state, and predictable networking. That does not belong in Vercel serverless.

---

# 5. Component spec

## 5.1 Latence Portal

Purpose:

> Shared control plane for user management, deployment selection, billing, and deployment discovery.

Tech:

* Vercel frontend
* Supabase auth
* Supabase Postgres for control-plane metadata
* Stripe or Lemon Squeezy later for billing
* Tally or embedded form for enterprise questionnaire

Portal features:

### User / org management

* create account
* create organization
* invite users later
* select deployment type
* view deployment endpoint
* rotate deployment token
* view billing plan
* submit VPC/on-prem request

### Deployment menu

Options:

```text
Cloud
Dedicated Cloud
VPC
On-Prem / Air-gapped
```

For Cloud:

* automatically provisions tenant
* returns API endpoint
* returns API key
* shows quickstart
* shows usage/logs/insights

For VPC/on-prem:

* starts questionnaire
* schedules sales/contact flow
* shows deployment requirements
* optionally generates architecture PDF later

### Portal database tables

Minimum Supabase schema:

```sql
organizations
- id
- name
- plan
- created_at

users
- id
- email
- name

organization_members
- org_id
- user_id
- role

deployments
- id
- org_id
- type              -- cloud | dedicated | vpc | onprem
- status            -- provisioning | active | paused | failed
- api_base_url
- region
- runtime_profile   -- cpu | small_gpu | medium_gpu | large_gpu
- tenant_id
- created_at

deployment_keys
- id
- deployment_id
- key_hash
- name
- created_at
- last_used_at
- revoked_at

licenses
- id
- org_id
- deployment_id
- features
- expires_at
- status

billing_accounts
- org_id
- provider
- customer_id
- subscription_id
- status

enterprise_requests
- id
- org_id
- deployment_type
- questionnaire_payload
- status
- created_at
```

Do **not** store raw traces in the Supabase control plane by default.

---

## 5.2 Tenant backend / TRACE Gateway

Purpose:

> The customer-facing API and stateful brain of TRACE.

Responsibilities:

* authenticate SDK/plugin calls
* manage sessions
* receive agent events
* store traces/logs
* manage InfiniMem state
* call compute runtime
* apply policies
* return decisions
* expose insights
* expose dashboard API

This service is deployment-specific.

For Cloud, you can run a shared tenant backend with strong logical isolation to keep costs down.

For Dedicated/VPC/on-prem, run one backend per customer.

### Tenant backend services

```text
trace-api
trace-worker
trace-db
trace-ui
trace-object-store
```

For MVP, you can combine API and worker.

### Tenant backend database

Cloud MVP:

* Postgres with tenant isolation
* tables partitioned by `tenant_id`
* row-level security where possible
* encrypted secrets
* short retention default

Dedicated/VPC/on-prem:

* customer-owned Postgres
* optional SQLite/DuckDB only for lightweight local demos

### Core tables

```sql
tenants
- id
- org_id
- deployment_id
- name
- created_at

api_keys
- id
- tenant_id
- key_hash
- name
- scopes
- last_used_at
- revoked_at

sessions
- id
- tenant_id
- external_session_id
- agent_type        -- rag | coding | tool | workflow
- integration       -- cursor | claude_code | codex | n8n | langgraph
- started_at
- ended_at
- metadata_json

events
- id
- tenant_id
- session_id
- event_type
- source
- content_ref
- content_hash
- redacted_content
- metadata_json
- created_at

traces
- id
- tenant_id
- session_id
- event_id
- mode              -- rag | coding | memory | redaction | compression
- input_hash
- output_hash
- decision
- severity
- scores_json
- labels_json
- spans_json
- created_at

memory_spans
- id
- tenant_id
- session_id
- span_text
- span_type
- layer             -- hot | warm | cold | tombstone
- survival_score
- freshness_score
- source_event_id
- supersedes_span_id
- created_at
- updated_at

policies
- id
- tenant_id
- name
- config_yaml
- active
- created_at

insight_rollups
- id
- tenant_id
- period
- metric_name
- metric_value
- dimensions_json
- created_at
```

---

## 5.3 TRACE Compute Runtime

Purpose:

> Self-contained model and scoring service.

This is the Dockerized runtime that can run on Runpod, customer VPC, or on-prem.

It should be as stateless as possible.

Responsibilities:

* groundedness scoring
* unsupported-span detection
* context utilization
* dead-weight detection
* context compression
* PII detection/redaction
* guardrail classification later
* drift scoring
* exact-critical extraction
* memory span extraction
* coding-agent scoring helpers

It should **not** be the long-term trace database.

### Runtime API

Internal API between tenant backend and compute runtime:

```http
POST /v1/compute/redact
POST /v1/compute/groundedness
POST /v1/compute/compress
POST /v1/compute/context-utilization
POST /v1/compute/drift
POST /v1/compute/extract-spans
POST /v1/compute/guardrails
POST /v1/compute/code-grounding
GET  /health
GET  /models
```

### Runtime configuration

```yaml
models:
  pii: gliner_pii
  groundedness: latence_groundedness_v1
  compression: llmlingua2
  drift: latence_drift_v1
  guardrails: disabled

runtime:
  device: cuda
  max_batch_size: 16
  max_input_tokens: 8192
  log_raw_payloads: false
```

### Runtime output example

```json
{
  "model_version": "latence-groundedness-v1",
  "scores": {
    "groundedness": 0.81,
    "unsupported": 0.19,
    "context_utilization": 0.63
  },
  "spans": [
    {
      "text": "refund will be approved within 48 hours",
      "label": "unsupported_claim",
      "score": 0.87
    }
  ]
}
```

---

# 6. SDK and plugin design

The SDK should be extremely thin.

The SDK does not own business logic. It only:

* starts sessions
* sends events
* calls `check`
* receives decisions
* optionally wraps common agent frameworks
* exposes typed responses

## 6.1 SDK core

Public SDK methods:

```typescript
trace.startSession()
trace.event()
trace.check()
trace.redact()
trace.verify()
trace.compress()
trace.memoryStep()
trace.endSession()
```

Example:

```typescript
const trace = new TraceClient({
  endpoint: process.env.TRACE_ENDPOINT,
  apiKey: process.env.TRACE_API_KEY
});

const result = await trace.check({
  mode: "rag",
  input: userMessage,
  context: retrievedDocs,
  output: assistantAnswer
});

if (result.decision === "block") {
  throw new Error("TRACE blocked unsupported answer");
}
```

## 6.2 Standard TRACE event schema

Every integration must emit the same event format.

```json
{
  "session_id": "sess_123",
  "event_type": "tool_result",
  "integration": "claude_code",
  "agent_type": "coding",
  "content": "...",
  "metadata": {
    "repo": "acme/backend",
    "file": "src/auth.ts",
    "tool_name": "grep",
    "commit_sha": "abc123"
  }
}
```

Core event types:

```text
user_message
assistant_message
tool_call
tool_result
file_read
file_write
code_diff
test_run
command_run
retrieval_result
decision
error
memory_update
```

## 6.3 Plugin principle

For Cursor, Claude Code, Codex, OpenCode, n8n, LangChain, LangGraph, LlamaIndex:

> **The plugin captures session/state/events in the native way of that tool, then converts them into the standard TRACE event schema.**

Do not make every plugin implement its own scoring logic.

Plugins own:

* local hooks
* session lifecycle
* event capture
* local metadata
* optional local buffering
* calling the SDK

Tenant backend owns:

* policies
* sessions
* trace logs
* memory state
* decisions

Compute runtime owns:

* models
* scoring
* extraction
* compression

---

# 7. Integration specs

## 7.1 Claude Code

Mechanism:

* hooks
* MCP server where helpful
* local config

Capture:

* user prompt
* assistant output
* tool calls
* tool results
* file reads/writes
* command runs
* code diffs
* test output

Special feature:

* block or warn before risky next step
* inject repair packet when TRACE says repair

## 7.2 Codex

Mechanism:

* CLI wrapper first
* optional plugin later

Example:

```bash
latence trace run codex "fix failing tests"
```

Capture:

* prompt
* file changes
* command runs
* diffs
* final answer

## 7.3 Cursor

Mechanism:

* VS Code/Cursor extension
* MCP server
* optional local sidecar

Capture:

* selected context
* current file
* prompts
* accepted/rejected diffs
* agent actions where available
* PR metadata if connected

## 7.4 OpenCode

Mechanism:

* CLI wrapper
* MCP/plugin if available

Capture same as Codex.

## 7.5 n8n

Mechanism:

* custom node

Nodes:

```text
TRACE Redact
TRACE Verify
TRACE Check Agent Step
TRACE Compress Context
TRACE Log Event
```

## 7.6 LangChain

Mechanism:

* callback handler
* runnable wrapper

Capture:

* chain input/output
* retriever results
* tool calls
* LLM outputs
* intermediate steps

## 7.7 LangGraph

Mechanism:

* node wrapper
* graph middleware
* checkpoint integration

Capture:

* node execution
* state transitions
* tool events
* interrupts
* human review points
* final answer

## 7.8 LlamaIndex

Mechanism:

* callback manager
* query engine wrapper
* retriever wrapper

Capture:

* retrieved nodes
* generated response
* source citations
* query transforms

---

# 8. Cloud provisioning flow

## 8.1 User flow

```text
User signs up
   ↓
Creates organization
   ↓
Selects deployment: Cloud
   ↓
Chooses runtime profile
   ↓
Portal provisions tenant
   ↓
Portal returns:
   - TRACE endpoint
   - API key
   - SDK snippet
   - plugin setup
   ↓
User sends first trace
   ↓
Portal shows logs/insights
```

## 8.2 Cloud runtime profiles

```text
Free Sandbox
- shared CPU/GPU
- low throughput
- short retention
- non-sensitive testing

Cloud Starter
- shared GPU runtime
- tenant-isolated backend
- basic logs

Cloud Pro
- higher throughput
- longer retention
- more model features
- priority queue

Dedicated Cloud
- dedicated backend
- dedicated or reserved compute
```

## 8.3 Cloud backend isolation options

You have three options.

### Option A — shared backend, tenant-isolated DB

Cheapest. Best for free and starter.

```text
One trace-api
One Postgres
tenant_id isolation
shared Runpod runtime
```

### Option B — shared API, separate schema per tenant

Middle ground.

```text
One trace-api
One Postgres cluster
schema per tenant
shared Runpod runtime
```

### Option C — backend per tenant

Most isolated, more expensive.

```text
trace-api per tenant
db per tenant
shared or dedicated runtime
```

Recommendation:

* use **Option A** for free/sandbox
* use **Option B** for paid cloud
* use **Option C** for dedicated cloud

Do not promise physically separate backend for the cheapest cloud tier. It will destroy margins.

Phrase it like:

> **Cloud uses tenant-isolated managed infrastructure. Dedicated and VPC deployments provide stronger physical isolation.**

---

# 9. VPC/on-prem flow

## 9.1 User flow

```text
User selects VPC or On-Prem
   ↓
Questionnaire opens
   ↓
Collects requirements
   ↓
You qualify manually
   ↓
Engineer/sales call
   ↓
Deployment proposal
   ↓
Helm/Docker bundle delivery
```

## 9.2 Questionnaire fields

Ask:

* company name
* use case: RAG / coding / both
* data sensitivity
* expected traces/day
* expected context sizes
* cloud provider
* region
* GPU availability
* Kubernetes availability
* desired retention
* SSO required?
* audit/export requirements?
* air-gapped?
* current agent stack
* deadline
* budget range

## 9.3 VPC package

Deliver:

* Docker images
* Helm chart
* model weights
* config templates
* license file
* deployment guide
* health checks
* upgrade guide
* support channel

---

# 10. Dashboard / insights

## 10.1 Cloud users

Use the Latence portal to show logs and insights by querying the cloud tenant backend.

Do not store full raw trace logs in Supabase control plane.

Portal displays:

* traces
* sessions
* decisions
* groundedness score
* PII detections
* context utilization
* compression savings
* drift score
* blocked/repaired/warned events
* integration breakdown

## 10.2 VPC/on-prem users

They need a local dashboard.

Options:

### MVP

Bundle `trace-ui` inside deployment.

```text
http://trace.local:3000
```

### Later

Optional remote-control-plane sync with aggregate metadata only.

But by default:

> VPC/on-prem dashboards run inside the customer environment.

---

# 11. API design

Public customer-facing TRACE API:

```http
POST /v1/sessions
POST /v1/events
POST /v1/check
POST /v1/redact
POST /v1/verify
POST /v1/compress
POST /v1/memory/step
GET  /v1/traces
GET  /v1/sessions/{id}
GET  /v1/insights
GET  /health
```

## `/v1/check`

Main endpoint.

```json
{
  "mode": "rag",
  "input": "user question",
  "context": ["retrieved document 1", "retrieved document 2"],
  "output": "assistant answer",
  "options": {
    "redact": true,
    "verify": true,
    "compress": true,
    "log": true
  }
}
```

Response:

```json
{
  "trace_id": "trc_123",
  "decision": "warn",
  "scores": {
    "groundedness": 0.74,
    "context_utilization": 0.42,
    "pii_risk": 0.13
  },
  "labels": [
    "unsupported_span_detected",
    "low_context_utilization"
  ],
  "spans": [
    {
      "text": "refund will be approved in 48 hours",
      "label": "unsupported_claim",
      "score": 0.86
    }
  ],
  "recommended_action": "repair"
}
```

---

# 12. Security requirements

Minimum for Cloud:

* API keys hashed
* tenant isolation
* short default retention
* raw payload logging disabled by default where possible
* redacted logs preferred
* encryption at rest
* encryption in transit
* audit log for portal actions
* no training on customer data
* rate limits
* abuse monitoring
* admin-only deployment access

Minimum for VPC/on-prem:

* no external calls required
* local model weights
* local DB
* local dashboard
* license file support
* configurable retention
* optional telemetry disabled by default
* export/delete tools
* upgrade/rollback support

---

# 13. Logging policy

Cloud default:

```yaml
logging:
  store_raw_inputs: false
  store_redacted_inputs: true
  store_outputs: configurable
  store_context: false
  store_scores: true
  store_spans: true
  retention_days: 7
```

Production/VPC default:

```yaml
logging:
  store_raw_inputs: customer_configurable
  store_redacted_inputs: true
  store_outputs: customer_configurable
  store_context: customer_configurable
  store_scores: true
  store_spans: true
  retention_days: customer_configurable
```

This is important for GDPR positioning.

---

# 14. Repository structure

Recommended monorepo:

```text
latence/
  apps/
    portal/                 # Vercel frontend
    trace-api/              # tenant backend
    trace-ui/               # local/cloud dashboard
    trace-worker/           # async processing
    trace-runtime/          # model runtime service
  packages/
    sdk-python/
    sdk-typescript/
    sdk-core/
    schemas/
    policies/
  integrations/
    langchain/
    langgraph/
    llamaindex/
    n8n/
    claude-code/
    codex-cli/
    opencode/
    cursor-extension/
  deploy/
    docker-compose/
    helm/
    runpod/
    onprem/
  models/
    manifests/
    download-scripts/
  docs/
```

---

# 15. Roadmap

## Phase 0 — Architecture cleanup

Goal:

> Separate control plane, tenant backend, compute runtime, and SDK.

Deliverables:

* final event schema
* final API schema
* Dockerized compute runtime
* tenant backend skeleton
* portal deployment menu
* API key per deployment
* one Cloud deployment path

---

## Phase 1 — Cloud MVP

Goal:

> User can sign up, select Cloud, receive endpoint/API key, send traces, and view results.

Deliverables:

* Supabase auth/orgs/deployments
* automatic Cloud tenant creation
* shared Runpod runtime
* tenant-isolated logs
* `/v1/check`
* `/v1/redact`
* `/v1/verify`
* basic dashboard
* Python SDK
* TypeScript SDK
* docs

Do not build every plugin yet.

---

## Phase 2 — RAG integrations

Goal:

> Make RAG agent adoption easy.

Deliverables:

* LangChain adapter
* LangGraph adapter
* LlamaIndex adapter
* n8n nodes
* RAG quickstart
* context utilization report
* groundedness report
* compression report

---

## Phase 3 — Coding-agent integrations

Goal:

> Make TRACE useful for coding agents.

Deliverables:

* CLI wrapper
* Claude Code integration
* Codex wrapper
* OpenCode wrapper
* GitHub Action
* basic Cursor extension or MCP server
* coding drift score
* test-evidence detection
* codebase-groundedness report

---

## Phase 4 — InfiniMem v1

Goal:

> Add memory survival management.

Deliverables:

* memory span extraction
* hot/warm/cold/tombstone layers
* immutable event vault
* survival scoring
* supersession logic
* repair packets
* memory dashboard

---

## Phase 5 — Dedicated/VPC deployment

Goal:

> Sell serious customers.

Deliverables:

* Docker Compose bundle
* Helm chart
* deployment license file
* model bundle
* local dashboard
* local DB
* VPC deployment guide
* enterprise questionnaire flow
* setup pricing

---

## Phase 6 — Guardrails private beta

Goal:

> Add prompt attack/exfiltration detection when reliable.

Deliverables:

* prompt injection classifier
* exfiltration classifier
* secret-seeking detection
* custom labels
* policy decisions
* private eval set
* false-positive review workflow

---

# 16. Engineering requirements

## Functional requirements

### Portal

* user can register
* user can create org
* user can select deployment
* Cloud deployment provisions automatically
* user can view endpoint/key
* user can rotate/revoke key
* user can access docs for selected integration
* VPC/on-prem opens questionnaire

### Tenant backend

* accepts API-key authenticated requests
* creates sessions
* stores events
* calls compute runtime
* stores trace results
* applies policy decisions
* exposes insights
* supports retention config
* supports export/delete

### Runtime

* starts from Docker image
* exposes health check
* lists model versions
* supports CPU/GPU mode
* runs PII, groundedness, compression, drift
* returns structured JSON only
* does not persist raw customer data by default

### SDK

* simple config
* typed responses
* retries/timeouts
* no hidden business logic
* open-source
* works with any endpoint

### Integrations

* all integrations convert native events to standard TRACE events
* all integrations use SDK
* no integration talks directly to compute runtime

---

# 17. Non-functional requirements

## Latency

Target:

* PII/redaction: low latency
* groundedness: acceptable for online agent path
* compression: optional or async if heavy
* memory updates: async where possible
* `/v1/check` should support timeout and partial results

## Reliability

* retries between backend and runtime
* queue for async processing
* graceful degradation
* if compute unavailable, backend returns configurable fail-open/fail-closed decision

## Privacy

* raw logs off by default in Cloud
* redacted logs preferred
* configurable retention
* no training by default
* customer export/delete

## Portability

Same product must run:

* Cloud
* Dedicated
* VPC
* On-prem

Same API everywhere.

---

# 18. Pricing model implied by this architecture

Use deployment-based pricing.

Do not make per-request billing the center.

## Cloud

* Free sandbox
* Cloud Starter: fixed monthly
* Cloud Pro: fixed monthly by runtime profile

## Dedicated

* fixed monthly + setup

## VPC

* high monthly + setup

## On-prem

* annual contract + setup

Phrase:

> **Flat monthly pricing by deployment capacity. No artificial per-request limits. Throughput depends on selected runtime profile.**

---

# 19. Biggest architectural risks

## Risk 1: Cloud isolation is oversold

Do not claim Cloud is physically isolated if it is shared.

Use:

> tenant-isolated managed cloud

Reserve:

> dedicated

for physically isolated backend/runtime.

## Risk 2: Compute runtime stores sensitive data

Keep compute stateless. Logs live in tenant backend.

## Risk 3: Plugin complexity explodes

Do not custom-build full state logic in every plugin.

Each plugin should only map native events into TRACE’s standard schema.

## Risk 4: Portal becomes the product again

Avoid rebuilding a heavy SaaS. Portal is only deployment and visibility.

## Risk 5: VPC/on-prem divergence

Same API, same images, same configs. Deployment mode changes infrastructure, not product behavior.

---

# 20. Final target architecture

This is the clean version:

```text
                     Latence Control Plane
                Vercel Portal + Supabase Metadata
                  users, orgs, deployments, billing
                              │
                              │ provisions / displays
                              ▼

Customer SDK / Plugin ──► TRACE Tenant Backend ──► TRACE Compute Runtime
Claude Code              auth, sessions, logs       models, scoring
Codex                    policies, memory           redaction
Cursor                   insights, decisions        compression
n8n                      dashboard API              groundedness
LangChain
LangGraph

Cloud:
- tenant backend managed by Latence
- compute on shared/dedicated Runpod
- dashboard in Latence portal

VPC/on-prem:
- tenant backend runs inside customer environment
- compute runs inside customer environment
- dashboard runs locally
- portal only manages license/support, optional
```

---

# My blunt recommendation

Yes: rebuild around this architecture.

Do **not** keep TRACE as a heavy multi-tenant SaaS.

Make TRACE:

> **a deployable real-time protection runtime for AI agents**

with:

* self-contained compute
* tenant backend
* thin SDK
* standard event schema
* simple portal for deployment selection
* Cloud for testing
* VPC/on-prem for serious customers

That matches the product, the trust story, and your solo-founder constraints.
