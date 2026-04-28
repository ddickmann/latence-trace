# Status page + public changelog

## status.latence.ai

The hosted control plane publishes uptime and latency at
`https://status.latence.ai`.  The page is backed by an external
provider (Statuspage by Atlassian) and ingests signals from:

| Source | Metric | Cadence |
| --- | --- | --- |
| UptimeRobot synthetic monitor | HTTP 200 on `GET https://api.latence.ai/v1/health` from five regions (us-east, us-west, eu-west, eu-central, ap-southeast) | 1 minute |
| Cloudflare health check | Worker `error_rate > 1%` trigger | 1 minute |
| Prometheus → Alertmanager → PagerDuty | `latence_trace_score_duration_seconds{quantile="0.95"}` per profile | 30 seconds |
| Manual operator update | Incident comms (SEV-1 / SEV-2) | on-demand |

Components exposed on the public page:

- `Scoring API` (us-east, us-west, eu, ap)
- `MCP endpoint`
- `Gateway authentication`
- `Portal / dashboard`

Incident taxonomy and response times are defined in
`docs/operations/incident-response.md` (SEV-1 / SEV-2 / SEV-3).

### Wiring

Statuspage Webhooks route to a private Slack channel and to the
`#announcements` channel on the community Discord.  The Prometheus
alert rule that triggers a state change lives under
`docs/operations/grafana/` alongside the dashboard JSON.

## /changelog

`CHANGELOG.md` at the repo root is the canonical changelog.  The
Next.js portal renders the same file at `https://latence.ai/changelog`
via a tiny build-time step:

```ts
// latence-ai/app/changelog/page.tsx (sketch)
import { readFileSync } from "fs";
import { remark } from "remark";
import html from "remark-html";

export default async function Changelog() {
  const md = readFileSync("../latence-trace/CHANGELOG.md", "utf-8");
  const rendered = await remark().use(html).process(md);
  return <article dangerouslySetInnerHTML={{ __html: String(rendered) }} />;
}
```

Keep the file in Keep-a-Changelog format.  Each release candidate
that touches behaviour must add an entry under `[Unreleased]` before
it ships.

## Release notes vs changelog

- **CHANGELOG.md** is the single source of truth for every merged
  change that a customer can observe from the API or the portal.
- **Release notes** (the per-tag GitHub Release body) is a curated
  superset: headline summary + Changelog excerpt + migration notes.
- **docs/roadmap.md** looks forward, not backward; do not update it
  when shipping.

## RSS / email

The `/changelog` page exposes an RSS feed at
`https://latence.ai/changelog/rss.xml` so customers can subscribe.
Major release notes are also emailed to the Starter, Business, and
Enterprise tiers' declared notification addresses.
