/**
 * api.latence.ai - Cloudflare Worker gateway for TRACE hosted.
 *
 * Implements:
 *   B1: Tenant API key verification, rotation, revocation (KV + D1).
 *   B2: Per-tenant threshold routing header forwarding.
 *   B3: Usage metering per tenant+band via Analytics Engine + D1.
 *   C6: Remote MCP endpoint dispatch (SSE + streamable-HTTP).
 *   Rate limiting: Durable Object token bucket per tenant.
 *
 * This is the production-path skeleton.  Deployment is via Wrangler;
 * see gateway/cloudflare/README.md.
 */

export interface Env {
  API_KEYS: KVNamespace;            // key: sha256(api_key) -> tenant row JSON
  USAGE_DB: D1Database;             // usage metering
  AUDIT_DB: D1Database;             // audit log mirror (optional)
  ANALYTICS: AnalyticsEngineDataset;// per-score usage events
  THRESHOLDS: KVNamespace;          // per-tenant thresholds JSON
  RATE_LIMITER: DurableObjectNamespace;
  TRACE_ORIGIN: string;             // upstream RunPod URL
  TRACE_ORIGIN_KEY: string;         // upstream auth header
  MCP_ORIGIN: string;               // remote MCP server origin
}

interface TenantRow {
  tenant_id: string;
  status: "active" | "revoked";
  plan: "free" | "starter" | "business" | "enterprise";
  rps_ceiling: number;
  monthly_quota: number;
  rotated_at?: string;
}

async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text)
  );
  return [...new Uint8Array(buf)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function errorJson(code: string, message: string, status: number): Response {
  return new Response(
    JSON.stringify({ error: { code, message, type: "api_error" } }),
    {
      status,
      headers: { "content-type": "application/json" },
    }
  );
}

async function authenticate(
  req: Request,
  env: Env
): Promise<{ tenant: TenantRow; apiKey: string } | Response> {
  const auth = req.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(\S+)$/.exec(auth);
  if (!match) return errorJson("unauthenticated", "Missing bearer token", 401);
  const apiKey = match[1];
  const kvKey = await sha256(apiKey);
  const row = await env.API_KEYS.get<TenantRow>(kvKey, "json");
  if (!row || row.status !== "active") {
    return errorJson("unauthenticated", "Invalid or revoked API key", 401);
  }
  return { tenant: row, apiKey };
}

async function enforceRateLimit(
  tenant: TenantRow,
  env: Env
): Promise<Response | null> {
  const id = env.RATE_LIMITER.idFromName(tenant.tenant_id);
  const stub = env.RATE_LIMITER.get(id);
  const res = await stub.fetch(
    "https://rl/consume?rps=" + tenant.rps_ceiling
  );
  if (res.status === 429) {
    return errorJson("rate_limited", "Tenant rate limit exceeded", 429);
  }
  return null;
}

async function enforceQuota(
  tenant: TenantRow,
  env: Env
): Promise<Response | null> {
  const yearMonth = new Date().toISOString().slice(0, 7);
  const row = await env.USAGE_DB.prepare(
    "SELECT SUM(count) AS consumed FROM usage WHERE tenant_id=? AND month=?"
  )
    .bind(tenant.tenant_id, yearMonth)
    .first<{ consumed: number | null }>();
  const consumed = row?.consumed ?? 0;
  if (consumed >= tenant.monthly_quota) {
    const status = tenant.plan === "free" ? 429 : 402;
    return errorJson("quota_exceeded", "Monthly quota exceeded", status);
  }
  return null;
}

async function logUsage(
  tenant: TenantRow,
  env: Env,
  band: string
): Promise<void> {
  const yearMonth = new Date().toISOString().slice(0, 7);
  env.ANALYTICS.writeDataPoint({
    blobs: [tenant.tenant_id, tenant.plan, band],
    doubles: [1],
    indexes: [tenant.tenant_id],
  });
  await env.USAGE_DB.prepare(
    `INSERT INTO usage(tenant_id, month, band, count)
     VALUES (?, ?, ?, 1)
     ON CONFLICT(tenant_id, month, band)
     DO UPDATE SET count = count + 1`
  )
    .bind(tenant.tenant_id, yearMonth, band)
    .run();
}

async function handleScore(req: Request, env: Env): Promise<Response> {
  const auth = await authenticate(req, env);
  if (auth instanceof Response) return auth;
  const { tenant } = auth;
  const rl = await enforceRateLimit(tenant, env);
  if (rl) return rl;
  const qt = await enforceQuota(tenant, env);
  if (qt) return qt;

  // Per-tenant threshold header forwarding (B2).
  const thresholdKey = await env.THRESHOLDS.get(`thresholds.${tenant.tenant_id}`);
  const upstreamHeaders = new Headers(req.headers);
  upstreamHeaders.set("authorization", `Bearer ${env.TRACE_ORIGIN_KEY}`);
  upstreamHeaders.set("x-latence-tenant-id", tenant.tenant_id);
  if (thresholdKey) {
    upstreamHeaders.set(
      "x-latence-tenant-thresholds",
      btoa(thresholdKey)
    );
  }
  const upstream = await fetch(env.TRACE_ORIGIN + new URL(req.url).pathname, {
    method: req.method,
    headers: upstreamHeaders,
    body: req.body,
  });
  const body = await upstream.text();
  let band: string = "unknown";
  try {
    const parsed = JSON.parse(body);
    band = parsed?.trace?.band ?? parsed?.band ?? "unknown";
  } catch (_) {
    // non-JSON upstream response - don't break the gateway
  }
  await logUsage(tenant, env, band);
  return new Response(body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}

async function handleKeyRotate(req: Request, env: Env): Promise<Response> {
  const auth = await authenticate(req, env);
  if (auth instanceof Response) return auth;
  const { tenant, apiKey } = auth;
  // Generate a new key, write the new KV entry, revoke the old.
  const newKeyBytes = new Uint8Array(32);
  crypto.getRandomValues(newKeyBytes);
  const newKey =
    "ltk_" +
    [...newKeyBytes]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  const newHash = await sha256(newKey);
  const now = new Date().toISOString();
  await env.API_KEYS.put(
    newHash,
    JSON.stringify({ ...tenant, rotated_at: now }),
    { expirationTtl: 60 * 60 * 24 * 365 * 2 }
  );
  const oldHash = await sha256(apiKey);
  await env.API_KEYS.put(
    oldHash,
    JSON.stringify({ ...tenant, status: "revoked" }),
    { expirationTtl: 60 * 60 * 24 * 30 }
  );
  return new Response(
    JSON.stringify({ api_key: newKey, rotated_at: now }),
    { headers: { "content-type": "application/json" } }
  );
}

async function handleKeyRevoke(req: Request, env: Env): Promise<Response> {
  const auth = await authenticate(req, env);
  if (auth instanceof Response) return auth;
  const { tenant, apiKey } = auth;
  const hash = await sha256(apiKey);
  await env.API_KEYS.put(
    hash,
    JSON.stringify({ ...tenant, status: "revoked" })
  );
  return new Response(null, { status: 204 });
}

async function handleMcp(req: Request, env: Env): Promise<Response> {
  const auth = await authenticate(req, env);
  if (auth instanceof Response) return auth;
  const url = new URL(req.url);
  const upstreamUrl = env.MCP_ORIGIN + url.pathname + url.search;
  const headers = new Headers(req.headers);
  headers.set("authorization", `Bearer ${env.TRACE_ORIGIN_KEY}`);
  headers.set("x-latence-tenant-id", auth.tenant.tenant_id);
  return fetch(upstreamUrl, {
    method: req.method,
    headers,
    body: req.body,
  });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/healthz") {
      return new Response("ok");
    }
    if (url.pathname === "/v1/groundedness/score" && req.method === "POST") {
      return handleScore(req, env);
    }
    if (url.pathname === "/v1/code/score" && req.method === "POST") {
      return handleScore(req, env);
    }
    if (url.pathname === "/v1/keys/rotate" && req.method === "POST") {
      return handleKeyRotate(req, env);
    }
    if (url.pathname === "/v1/keys/revoke" && req.method === "POST") {
      return handleKeyRevoke(req, env);
    }
    if (url.pathname.startsWith("/mcp")) {
      return handleMcp(req, env);
    }
    return errorJson("not_found", "Unknown path", 404);
  },
};

/** Rate-limiter Durable Object: per-tenant token bucket. */
export class RateLimiter {
  state: DurableObjectState;
  tokens: number;
  lastRefill: number;
  constructor(state: DurableObjectState) {
    this.state = state;
    this.tokens = 0;
    this.lastRefill = 0;
  }
  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const rps = Number(url.searchParams.get("rps") ?? "20");
    const now = Date.now();
    if (this.lastRefill === 0) {
      this.tokens = rps;
      this.lastRefill = now;
    } else {
      const elapsed = (now - this.lastRefill) / 1000;
      this.tokens = Math.min(rps, this.tokens + elapsed * rps);
      this.lastRefill = now;
    }
    if (this.tokens < 1) {
      return new Response(null, { status: 429 });
    }
    this.tokens -= 1;
    return new Response(null, { status: 204 });
  }
}
