export type Profile = "standard" | "quality" | "code";
export type Band = "green" | "amber" | "red";

export interface ScoreRequest {
  question: string;
  responseText: string;
  rawContext: string;
  profile?: Profile;
  runtimeHeadFeatures?: Record<string, number>;
  trajectoryFeatures?: Record<string, number>;
  tenantId?: string;
  requestId?: string;
}

export interface TeacherChannels {
  nli_aggregate?: number;
  maxsim_support?: number;
  context_coverage_ratio?: number;
  context_usage_ratio?: number;
  epistemic_hedge_gate?: number;
  [key: string]: number | undefined;
}

export interface Attribution {
  response_span: string;
  evidence_span: string;
  score: number;
}

export interface ScoreResponse {
  band: Band;
  groundedness: number;
  profile: Profile;
  runtime_decision?: {
    action: "allow" | "auto_repair" | "block";
    score: number;
    score_channel: string;
    class_key: string;
    head_id?: string;
    head_enabled?: boolean;
    head_score?: number;
    head_features_used?: string[];
    head_reason_codes?: string[];
    [key: string]: unknown;
  };
  teacher_channels?: TeacherChannels;
  top_k_attributions?: Attribution[];
  request_id?: string;
  [key: string]: unknown;
}

export interface ClientOptions {
  apiKey: string;
  baseUrl?: string;
  maxRetries?: number;
  retryBackoffMs?: number;
  fetchImpl?: typeof globalThis.fetch;
  tracer?: Tracer;
  timeoutMs?: number;
}

export interface Tracer {
  startSpan(name: string, attrs?: Record<string, unknown>): Span;
}

export interface Span {
  setAttribute(key: string, value: string | number | boolean): void;
  recordException(err: unknown): void;
  setStatus(status: { code: "ok" | "error"; message?: string }): void;
  end(): void;
}

export class LatenceTraceError extends Error {
  readonly status: number;
  readonly code: string | undefined;
  readonly retryAfterMs?: number;
  readonly requestId?: string;

  constructor(
    message: string,
    opts: { status: number; code?: string; retryAfterMs?: number; requestId?: string },
  ) {
    super(message);
    this.name = "LatenceTraceError";
    this.status = opts.status;
    this.code = opts.code;
    this.retryAfterMs = opts.retryAfterMs;
    this.requestId = opts.requestId;
  }

  get retryable(): boolean {
    return this.status === 429 || (this.status >= 500 && this.status < 600);
  }
}

const DEFAULT_BASE_URL = "https://api.latence.ai";
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_BACKOFF_MS = 250;
const DEFAULT_TIMEOUT_MS = 60_000;

export class LatenceTrace {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly maxRetries: number;
  private readonly retryBackoffMs: number;
  private readonly fetchImpl: typeof globalThis.fetch;
  private readonly tracer?: Tracer;
  private readonly timeoutMs: number;

  constructor(opts: ClientOptions) {
    if (!opts.apiKey) {
      throw new Error("LatenceTrace: apiKey is required");
    }
    this.apiKey = opts.apiKey;
    this.baseUrl = (opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.maxRetries = opts.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.retryBackoffMs = opts.retryBackoffMs ?? DEFAULT_RETRY_BACKOFF_MS;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
    this.tracer = opts.tracer;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  async scoreGroundedness(req: ScoreRequest): Promise<ScoreResponse> {
    const span = this.tracer?.startSpan("latence_trace.score_groundedness", {
      "latence.profile": req.profile ?? "standard",
      "latence.tenant_id": req.tenantId ?? "",
    });

    try {
      const body = {
        question: req.question,
        response_text: req.responseText,
        raw_context: req.rawContext,
        profile: req.profile ?? "standard",
        runtime_head_features: req.runtimeHeadFeatures,
        trajectory_features: req.trajectoryFeatures,
      };
      const headers: Record<string, string> = {
        "content-type": "application/json",
        accept: "application/json",
        authorization: `Bearer ${this.apiKey}`,
      };
      if (req.tenantId) headers["x-latence-tenant-id"] = req.tenantId;
      if (req.requestId) headers["x-request-id"] = req.requestId;

      const response = await this.requestWithRetry(
        `${this.baseUrl}/v1/score/groundedness`,
        headers,
        body,
      );
      span?.setAttribute("latence.band", String(response.band));
      span?.setAttribute("latence.groundedness", Number(response.groundedness) || 0);
      span?.setStatus({ code: "ok" });
      return response;
    } catch (err) {
      span?.recordException(err);
      span?.setStatus({ code: "error", message: (err as Error).message });
      throw err;
    } finally {
      span?.end();
    }
  }

  private async requestWithRetry(
    url: string,
    headers: Record<string, string>,
    body: unknown,
  ): Promise<ScoreResponse> {
    let attempt = 0;
    let lastErr: unknown;
    while (attempt <= this.maxRetries) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        let response: Response;
        try {
          response = await this.fetchImpl(url, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timer);
        }
        if (response.ok) {
          return (await response.json()) as ScoreResponse;
        }

        const retryAfter = Number(response.headers.get("retry-after") ?? "0");
        const reqId = response.headers.get("x-request-id") ?? undefined;
        let code: string | undefined;
        let message = response.statusText;
        try {
          const errBody = (await response.json()) as {
            code?: string;
            message?: string;
          };
          code = errBody.code;
          if (errBody.message) message = errBody.message;
        } catch {
          // body was not JSON; fall through
        }
        const err = new LatenceTraceError(message, {
          status: response.status,
          code,
          retryAfterMs: Number.isFinite(retryAfter) ? retryAfter * 1000 : undefined,
          requestId: reqId,
        });
        if (!err.retryable || attempt === this.maxRetries) {
          throw err;
        }
        await this.sleep(err.retryAfterMs ?? this.backoffMs(attempt));
      } catch (err) {
        if (err instanceof LatenceTraceError) {
          if (!err.retryable || attempt === this.maxRetries) {
            throw err;
          }
          lastErr = err;
        } else if (attempt === this.maxRetries) {
          throw err;
        } else {
          lastErr = err;
          await this.sleep(this.backoffMs(attempt));
        }
      }
      attempt += 1;
    }
    throw lastErr instanceof Error
      ? lastErr
      : new LatenceTraceError("Exhausted retries", { status: 0 });
  }

  private backoffMs(attempt: number): number {
    const base = this.retryBackoffMs * Math.pow(2, attempt);
    const jitter = Math.random() * this.retryBackoffMs;
    return base + jitter;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

export default LatenceTrace;
