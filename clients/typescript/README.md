# @latence/trace

Official TypeScript / JavaScript SDK for [Latence TRACE](https://latence.ai):
groundedness verification plus real-time PII compliance redaction.

Runs in Node 18+, Deno, Bun, and the Cloudflare Workers runtime.
Zero required runtime dependencies; OTel tracing integration is
optional via a peer dependency.

## Install

```bash
npm install @latence/trace
```

## Usage

```ts
import { LatenceTrace } from "@latence/trace";

const trace = new LatenceTrace({
  apiKey: process.env.LATENCE_API_KEY!,
});

const result = await trace.scoreGroundedness({
  question: "What was our 2023 ARR?",
  responseText: "ARR reached 12.4M USD in 2023.",
  rawContext: "FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
  profile: "standard",
});

if (result.band === "red") {
  throw new Error("ungrounded answer; retry with more evidence");
}

const compliance = await trace.redactCompliance({
  text: "Send Jane Doe at jane@example.com into the model prompt.",
  labels: ["person", "email"],
  redact: true,
  redactionMode: "mask",
});

console.log(compliance.redacted_text, compliance.entity_count);
```

## Features

- **Retries with jitter and Retry-After**. The SDK retries on 429 and
  5xx, honouring the gateway's `Retry-After` header. Non-retryable
  4xx errors throw `LatenceTraceError` immediately.
- **Per-tenant thresholds**. Pass `tenantId` to route through a
  tenant-specific threshold policy.
- **OpenTelemetry integration**. Pass a `Tracer` instance (e.g.
  `trace.getTracer("app")` from `@opentelemetry/api`) and every score
  or redaction call becomes a span with useful runtime attributes.
- **Compliance redaction**. `redactCompliance()` calls
  `/v1/compliance/redact` and returns typed entities, label usage, chunk
  counts, redacted text, and timings.
- **Custom fetch**. Pass `fetchImpl` to integrate with Workers, Edge
  runtimes, or mocked tests.

## Self-hosting

```ts
const trace = new LatenceTrace({
  apiKey: "ignored",
  baseUrl: "http://trace.internal:8000",
});
```

The self-hosted server accepts the same request / response shape as
the hosted gateway.

## Error handling

```ts
import { LatenceTraceError } from "@latence/trace";

try {
  await trace.scoreGroundedness({ ... });
} catch (err) {
  if (err instanceof LatenceTraceError) {
    console.error({ status: err.status, code: err.code, requestId: err.requestId });
  }
}
```

## License

Apache-2.0 © Latence AI.
