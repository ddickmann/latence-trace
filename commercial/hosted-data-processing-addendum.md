# Hosted Data Processing Addendum — TRACE v1

**Effective date:** 2026-04-28  
**Service:** `api.latence.ai` — TRACE hosted RAG validation API  
**Controller:** Customer  
**Processor:** Latence SAS, incorporated in France

This addendum supplements the Master Subscription Agreement and
governs processing of personal data submitted to the hosted TRACE
service.

## 1. Subject-matter and duration

### 1.1 Nature of processing

Scoring customer-supplied question / response / evidence triples
against a hosted TRACE worker and returning a `groundedness_score` +
`band` + `evidence_trace` payload.  Latence does not retrieve,
generate, store, or otherwise transform the customer payload beyond
what is required to produce the score response.

### 1.2 Duration

For each scoring request, the processor handles the payload for the
duration of the synchronous API call only.  See clause 4 for
retention.

### 1.3 Categories of data subjects

Determined by the customer.  Personal data is submitted only if the
customer includes it in the question, response, or evidence pack.
Customers are responsible for the lawfulness of that inclusion.

### 1.4 Categories of personal data

Determined by the customer.  Customers may optionally flag a payload
as containing special-category data (Article 9 GDPR), in which case
additional controls in clause 5 apply.

## 2. Sub-processors

The following sub-processors are engaged for the hosted service:

| Name | Purpose | Location | SCC status |
| --- | --- | --- | --- |
| Cloudflare, Inc. | Edge + API gateway (CDN, rate-limit, auth) | Global; EU edge for EU customers | SCCs 2021 |
| RunPod, Inc. | GPU compute for TRACE scoring workers | Primary: Ireland (EU). Fail-over: Oregon (US). | SCCs 2021 + DPA |
| Datadog, Inc. | Log aggregation, infra monitoring | EU region (Frankfurt) | SCCs 2021 |
| PagerDuty, Inc. | On-call paging | US; EU region available on request | SCCs 2021 |
| Stripe Payments Europe Ltd | Billing and invoicing | Ireland | SCCs 2021 |

New sub-processors will be announced 30 days in advance on
`https://latence.ai/sub-processors` with an opt-out right for
existing customers.

## 3. Security measures

Latence implements the technical and organizational measures defined
in `SECURITY.md` and `commercial/soc2-control-mapping.md`, including:

- TLS 1.3 for data in transit; ephemeral TLS keys via Cloudflare.
- At-rest encryption for the (optional) sampled logs store (see
  clause 4.2) using AES-256-GCM.
- Role-based access control; production access is multi-factor and
  short-lived (1-hour sessions).
- Quarterly third-party penetration test; results available on
  request under NDA.
- SOC 2 Type I audit in progress; Type II window open.
- Annual GDPR Article 32 review.

## 4. Data retention

### 4.1 Default (no sampled logging)

- Request payload (question, response, evidence) is held in worker
  memory for the duration of the scoring call and discarded on
  response; it is **not** written to durable storage.
- Request metadata (tenant ID, timestamp, latency, band, hash-of-body)
  is retained in the audit log for 90 days by default, configurable
  per tenant up to 13 months or down to 0 days on the Enterprise tier.
- No customer payload content is stored by Latence by default.

### 4.2 Opt-in sampled logging

Customers on the Business or Enterprise tiers may opt in to sampled
logging for support and improvement purposes.  Sampling rate, retention
window (max 30 days), and redaction policy are set in the tenant
configuration.  Sampled logs are stored in the same region as the
customer's primary worker.  Opt-in is recorded in the audit log and
can be revoked at any time with effect for future requests.

### 4.3 Deletion on termination

Within 30 days of termination, Latence will:

- Purge the tenant's API keys and quotas from Cloudflare.
- Purge audit-log entries and any opt-in sampled logs from storage.
- Provide a deletion certificate on request.

## 5. Special categories (Article 9 GDPR)

If the customer flags a payload as containing special-category data
(header: `X-Latence-Data-Class: special`), the scoring worker:

- Disables sampled logging for that request irrespective of tenant
  configuration;
- Records only a redacted audit-log entry (no payload hash beyond the
  request ID);
- Prohibits spill into telemetry (OTel spans carry no payload bytes).

## 6. International transfers

- EU customers are served from EU-region workers by default.
- Any transfer outside the EEA is performed under the EU Commission
  Standard Contractual Clauses (2021/914) with each sub-processor.
- A Transfer Impact Assessment template is available on request.

## 7. Data-subject requests

Latence does not control customer data and cannot respond to
data-subject requests directly.  Upon receipt of a request addressed
to Latence that relates to a customer tenant, Latence will notify
the customer within 3 business days and assist the customer as
reasonably required.

## 8. Breach notification

Latence will notify the customer's primary security contact without
undue delay, and in any event within 48 hours, of any confirmed
personal-data breach affecting the customer's tenant.

## 9. Contact

- Data Protection Officer: `dpo@latence.ai`
- Security incidents: `security@latence.ai`
- Trust Center: `https://latence.ai/trust`
