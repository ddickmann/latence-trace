import { describe, expect, it, vi } from "vitest";
import { LatenceTrace, LatenceTraceError } from "./index.js";

function makeFetch(
  responses: Array<{
    status: number;
    body?: unknown;
    headers?: Record<string, string>;
  }>,
): { fetchImpl: typeof globalThis.fetch; calls: number } {
  let calls = 0;
  const fetchImpl = vi.fn(async () => {
    const r = responses[calls] ?? responses[responses.length - 1];
    calls += 1;
    return new Response(JSON.stringify(r.body ?? {}), {
      status: r.status,
      headers: r.headers,
    });
  });
  return { fetchImpl: fetchImpl as unknown as typeof globalThis.fetch, get calls() { return calls; } };
}

describe("LatenceTrace", () => {
  it("returns the score on 200", async () => {
    const { fetchImpl } = makeFetch([
      {
        status: 200,
        body: { band: "green", groundedness: 0.91, profile: "standard" },
      },
    ]);
    const client = new LatenceTrace({
      apiKey: "k",
      fetchImpl,
      maxRetries: 0,
    });
    const res = await client.scoreGroundedness({
      question: "q",
      responseText: "r",
      rawContext: "c",
    });
    expect(res.band).toBe("green");
    expect(res.groundedness).toBeCloseTo(0.91);
  });

  it("retries on 429 with Retry-After", async () => {
    const { fetchImpl } = makeFetch([
      { status: 429, headers: { "retry-after": "0" }, body: { message: "rate limited" } },
      { status: 200, body: { band: "amber", groundedness: 0.7, profile: "standard" } },
    ]);
    const client = new LatenceTrace({ apiKey: "k", fetchImpl, retryBackoffMs: 1 });
    const res = await client.scoreGroundedness({
      question: "q",
      responseText: "r",
      rawContext: "c",
    });
    expect(res.band).toBe("amber");
  });

  it("throws LatenceTraceError on non-retryable 4xx", async () => {
    const { fetchImpl } = makeFetch([
      { status: 402, body: { code: "quota_exceeded", message: "tier cap" } },
    ]);
    const client = new LatenceTrace({ apiKey: "k", fetchImpl, maxRetries: 3 });
    await expect(
      client.scoreGroundedness({ question: "q", responseText: "r", rawContext: "c" }),
    ).rejects.toBeInstanceOf(LatenceTraceError);
  });

  it("sends per-tenant header when tenantId is provided", async () => {
    let captured: Record<string, string> = {};
    const fetchImpl = vi.fn(async (_url, init) => {
      captured = (init?.headers ?? {}) as Record<string, string>;
      return new Response(
        JSON.stringify({ band: "green", groundedness: 0.99, profile: "standard" }),
        { status: 200 },
      );
    }) as unknown as typeof globalThis.fetch;
    const client = new LatenceTrace({ apiKey: "k", fetchImpl, maxRetries: 0 });
    await client.scoreGroundedness({
      question: "q",
      responseText: "r",
      rawContext: "c",
      tenantId: "acme",
    });
    expect(captured["x-latence-tenant-id"]).toBe("acme");
  });
});
