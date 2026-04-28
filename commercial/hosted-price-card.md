# TRACE Hosted SKU and Price Card

**Effective date:** 2026-04-28

## Plans

| Plan | Monthly commitment | Scores included | Overage | Profile access | Support |
| --- | ---: | ---: | ---: | --- | --- |
| Free | $0 | 1,000 / month (standard profile only) | Hard cap | `standard` | Community only |
| Starter | $299 | 50,000 / month | $0.008 / score | `standard`, `quality` | Email, next business day |
| Business | $1,499 | 300,000 / month | $0.006 / score | `standard`, `quality`, `code` | Email + chat, 4 biz-hr response |
| Enterprise | From $5,000 | From 1,000,000 / month | Negotiated | All profiles + priority queue | 24x7 P1 paging, TAM |

All plans include:

- Hosted `api.latence.ai` endpoint
- Tenant API key issuance + rotation + revocation
- Per-tenant threshold routing (Business+)
- 90-day audit log (configurable to 13 months on Enterprise)
- Vertical one-pagers and proof bundle access

## Discount / commitment options

- Annual prepay: 15% discount.
- Multi-year (3y) prepay: 25% discount.
- Design-partner pricing (first 10 enterprise customers): 50% off
  year-one list, subject to reference + case-study cooperation.

## Overage billing

- Overage is invoiced monthly in arrears.
- Tier caps enforced at the API gateway per tenant.  Hard cap on Free
  tier; soft cap + pay-as-you-go on paid tiers.

## What's included on Enterprise

- Named Technical Account Manager (TAM).
- Monthly reliability review against the Veracier proof bundle and
  `external_benchmarks.md` regression report.
- Quarterly architecture review.
- Access to Grafana dashboard JSON at `docs/operations/grafana/` to
  import into the customer's observability stack.
- Private Slack / Teams channel with the engineering team.
- 24x7 P1 paging via PagerDuty.
- 13-month audit-log retention with customer-exported CSV / JSON.
- Private network peering options (Cloudflare Private Links, AWS
  PrivateLink) on request.

## Self-hosted (for comparison)

The OSS / self-hosted option remains free under the Apache 2.0
license.  Hosted plans exist for customers that prefer to consume
TRACE as a service (no RunPod deployment, no licence-server
operation).

## Payment and terms

- Stripe-invoiced; card, ACH, or SEPA supported.
- NET-30 payment terms on Business and Enterprise.
- Multi-year prepay invoiced annually unless otherwise negotiated.

## Contact

Sales: `sales@latence.ai`  
Procurement and custom contracts: `finance@latence.ai`
