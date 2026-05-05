# latence-trace

> Enterprise AI compliance runtime for **real-time PII redaction** and
> calibrated groundedness verification across **RAG** and **coding agents**.
> **Multilingual: English + German out of the box.** Part of the
> latence.ai product family.

`latence-trace` is the Latence TRACE runtime. It scores how well an LLM
response is grounded in supplied context, returns auditable per-claim evidence,
classifies outputs into calibrated `green` / `amber` / `red` risk bands, and
redacts PII in real time before prompts or responses leave a trust boundary.

## Verification + Compliance, One Service

Pick the lane per request via `scoring_mode` in the `/groundedness` body.
Shared encoder, shared kernels, shared observability — the domain-specific
signals fan out on the scoring path:

| Lane | `scoring_mode` | Who it's for | What it answers |
| --- | --- | --- | --- |
| **RAG** | `"rag"` (default) | Enterprises running retrieval-augmented LLM apps | *"Is this answer anchored in the retrieved context? Which chunks are dead weight?"* |
| **Code** | `"code"` | Teams shipping coding agents (Claude Code, Cursor, Codex, OpenCode …) | *"Is the generated code grounded in the opened files? Did the agent drift? Which files in the context window are genuinely unused?"* |

The compliance lane is exposed at `POST /v1/compliance/redact`. It uses
GLiNER PII detection, token-aware chunking, deterministic validators, custom
regex overrides, masking, and synthetic replacement redaction via
`redaction_mode="replace"`. Portal insights store only privacy-safe aggregate
metadata: entity counts, label counts, chunk counts, redaction mode, latency,
and error rates.

Compliance requests and responses use stable canonical GDPR labels. The runtime
translates those labels to benchmarked GLiNER-facing aliases only at inference
time, then maps predictions back before redaction and analytics. This keeps the
public API stable while improving recall for labels such as `person`, which is
sent to GLiNER as `name`.

Address handling supports both full-address redaction (`address`) and focused
components (`street_address`, `postal_code`, `city`, `country`) so customers can
choose broad removal or more precise masking.

The RAG lane remains untouched — same models, same thresholds, bitwise
parity guaranteed by
[`tests/api/test_rag_lane_parity.py`](tests/api/test_rag_lane_parity.py).
The code lane adds AST-grounded literal matching, an ambiguity-triggered
NLI cascade, a logistic composite, and per-session multi-turn signals on
top of the shared MaxSim scorer. See the
[coding-agent guide](docs/coding_agent_guide.md) and
[docs/code_lane_v3.md](docs/code_lane_v3.md).

> **Scope.** `latence-trace` answers **"is this response anchored in the
> supplied context?"** (RAG-grounding / faithfulness / code-grounding).
> It does **not** answer **"is this response factually correct against
> world knowledge?"** (open-domain factuality). For the latter, pair
> with a knowledge-base fact-checker. See
> [`docs/algorithm-audit.md`](docs/algorithm-audit.md) §"Scope and
> Known Mismatches" for the empirical evidence.

> **New here?** Start with the end-to-end tutorial:
> [`docs/guides/tutorial.md`](docs/guides/tutorial.md) — covers boot,
> first request, the three premise lanes, the per-token heatmap, the
> retrieval-coverage observability metric, profiles, and integrations
> (HTTP / Python SDK / MCP / OpenAI tools).

## Why

Open-source LLM observability tools either
- emit a single opaque score from a 7-70B "judge" model, or
- run a proprietary hosted service against your private context.

`latence-trace` runs locally or as a sidecar, in ~100 ms p95, with five
explainable channels and per-stratum calibrated thresholds. It is licensed for
use inside the latence.ai product family; it is not OSS.

## Headline numbers

Honest numbers, A5000 batch=1, **n = 120 samples per stratum** (no n=20 lucky
slices), production config: GTE-ModernColBERT bf16 + multilingual mDeBERTa NLI
+ BGE-reranker-v2-m3 + atomic claims, response-chunked path, headline =
`groundedness_v2`. Source report:
[`research/triangular_maxsim/reports/truth_bench_n120.json`](research/triangular_maxsim/reports/truth_bench_n120.json).

| Lane | Stratum | n | Metric | Value | 95% CI |
|------|---------|--:|:------:|------:|:------:|
| Internal minimal pairs | lexical (entity / date / number / unit swap) | 120 | paired_acc | **0.95** | [0.63, 1.00] |
| Internal minimal pairs | semantic (negation / role swap) | 60 | paired_acc | **0.98** | [0.90, 1.00] |
| Internal minimal pairs | partial support | 30 | paired_acc | **1.00** | [1.00, 1.00] |
| Internal minimal pairs | hard compound facts | 30 | paired_acc | **1.00** | [1.00, 1.00] |
| Internal minimal pairs | hard structured (JSON / md table) | 30 | paired_acc | **0.93** | [0.83, 1.00] |
| Internal minimal pairs | hard distributed dialogue | 30 | paired_acc | 0.57 | [0.40, 0.73] |
| Internal minimal pairs | German | 26 | paired_acc | **1.00** | [1.00, 1.00] |
| Domain-pair: finance prose (10-K, earnings calls, BLS / Fed) | finance prose | 25 | paired_acc | **0.98** ¶ | [0.93, 1.00] |
| Domain-pair: legal prose (statutes, rulings, GDPR) | legal prose | 25 | paired_acc | **0.96** | [0.88, 1.00] |
| Domain-pair: **EN table-adversarial** (segment tables, balance sheets, FOMC, ETF holdings) | English tables | 100 | paired_acc | **0.875** ¶ | [0.81, 0.93] |
| Domain-pair: **DE table-adversarial** (SAP, Siemens, ECB, Bundesbank, Volkswagen) | German tables | 70 | paired_acc | **0.836** ¶ | [0.75, 0.91] |
| RAGTruth | macro F1 (qa / summ / data2text) | 360 | F1@median | **0.61** | — |
| RAGTruth | qa | 120 | F1@median | **0.73** (precision 0.98) | — |
| RAGTruth | summarization | 120 | F1@median | **0.65** (precision 0.80) | — |
| RAGTruth | data2text | 120 | F1@median | 0.45 | — |
| HaluEval QA | paired ranking (right > halu) — multilingual NLI | 60 | paired_acc | 0.67 | — |
| HaluEval QA | paired ranking — **English NLI** † | 60 | paired_acc | **0.78** | — |
| HaluEval Summarization | paired ranking — multilingual NLI | 60 | paired_acc | 0.65 | — |
| HaluEval Summarization | paired ranking — **English NLI** † | 60 | paired_acc | **0.75** | — |
| HaluEval Dialogue | paired ranking (either NLI) | 60 | paired_acc | 0.57–0.58 ‡ | — |
| Latency | end-to-end (NLI on, reranker on, atomic on) | — | p95 | **118 ms** | — |

HaluEval numbers come from saved diagnostic runs under
[`research/triangular_maxsim/reports/halueval_diagnose_mdeberta_n60.json`](research/triangular_maxsim/reports/halueval_diagnose_mdeberta_n60.json)
and [`..._deberta_en_n60.json`](research/triangular_maxsim/reports/halueval_diagnose_deberta_en_n60.json);
each is an n=60 paired diagnostic (60 paired right/hallucinated answers per
stratum). Reproduce: `python scripts/diagnose_halueval.py --nli-model <id> --limit 60`.

† English NLI peer = `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Set
`VOYAGER_GROUNDEDNESS_NLI_MODEL` to switch. Recommended for English-only
deployments — gains ≈ +10 percentage points paired ranking accuracy on
HaluEval QA / Summarization vs. the multilingual default.<br>
¶ **Typed Structured Evidence Lane** (AND-gate over entity/value/unit/sign
matched against typed source cells) is on by default for `quality` profile
and contributes most of the lift on the table-adversarial sets. Disable
with `VOYAGER_GROUNDEDNESS_STRUCTURED_GATE=0` to fall back to the legacy
narrative-only fusion. Reports + reproduction datasets:
[`research/triangular_maxsim/reports/domain_pairs/`](research/triangular_maxsim/reports/domain_pairs/)
(`finance_pairs.jsonl`, `legal_pairs.jsonl`,
`tables_adversarial_en.jsonl`, `tables_adversarial_de.jsonl`, plus
gate-on/off and ablation reports under `structured_lane/`). Removing the
lane drops EN-table delta from `0.345` to `0.256` (the score gap between
grounded and ungrounded responses) while removing NLI drops paired_acc
by `−6.5pp`. **Green precision = 99.55%** across all four domain-pair
sets (1 false-green out of 220 ungrounded responses; the single FP is a
DE rate-decision bps↔% unit swap on an 88-character source that does
not yield enough cells for reliable AND-gating).<br>
‡ HaluEval Dialogue is **a known mismatch lane** for context-grounded scoring:
hallucinations introduce real-world facts that are *not in the dialogue
context*, so both right and hallucinated continuations score "ungrounded"
relative to the supplied context. We surface this honestly rather than tune
to it. See [`docs/algorithm-audit.md`](docs/algorithm-audit.md) §"Scope and
Known Mismatches".

§ FActScore biographies are now evaluated under the **canonical FActScore
protocol** (per-claim atomic precision at the F1-optimal threshold),
not response-level F1. Each biography's per-annotation `is_supported` gold
label is compared one-by-one against the atomic claim's score against the
matching Wikipedia article (enriched via
[`scripts/enrich_factscore_with_wiki.py`](scripts/enrich_factscore_with_wiki.py)).
Per-claim **precision = 0.61, recall = 0.62, F1 = 0.62** at n = 748
atomic claims (30 biographies, multilingual mDeBERTa NLI); the criterion
target of 0.65 is missed by ≈ 4 points and we report the actual number
rather than tune to the test set. See
[`research/triangular_maxsim/reports/truth_bench_n120_factscore_per_claim.json`](research/triangular_maxsim/reports/truth_bench_n120_factscore_per_claim.json).

### Where it shines, where it does not

- ✅ **RAG QA + summarization grounding** — the headline use case
  (RAGTruth qa F1 0.73 / precision 0.98, summarization F1 0.65 /
  precision 0.80, HaluEval QA paired 0.78 with English NLI).
- ✅ **Bilingual EN+DE** — German minimal pairs 1.00 paired, identical
  schema, same `/groundedness` endpoint.
- ✅ **Tabular / structured-source pairs** — 0.93 paired on hard
  JSON / markdown table stratum (NLI + structured triples), plus
  **0.875 paired on EN segment-tables** and **0.836 paired on DE
  Geschäftsbericht tables** with the new Typed Structured Evidence
  Lane (n=170 paired adversarial samples, AND-gate over entity ∧
  value ∧ unit ∧ sign — one wrong cell collapses the score).
  **Green precision 99.55%** across the union of legal + finance +
  EN-tables + DE-tables (1 FP / 220 ungrounded responses).
- ✅ **Per-claim atomic verification (FActScore-style)** — 0.61 precision /
  0.62 F1 at n = 748 atomic claims, Wikipedia-grounded, evaluated under
  the canonical FActScore protocol (was previously reported as `skipped`
  because the loader collapsed per-claim labels into one response-level
  sample; v1.1 emits one sample per atomic annotation).
- ⚠️ **Distributed-evidence dialogue** where the support is split across
  speaker turns (0.57 paired, n=30 — wide CI). Pair with a
  context-rewriter or a longer-premise reranker run; see
  [`docs/algorithm-audit.md`](docs/algorithm-audit.md) for the
  per-stratum diagnosis.
- ❌ **Open-domain factuality without a source document** — if you cannot
  hand the sidecar a context to verify against, no algorithm can. Pair
  with a retrieval step (or with `scripts/enrich_factscore_with_wiki.py`-
  style enrichment) so each claim has its source paragraph.

See [`docs/algorithm-audit.md`](docs/algorithm-audit.md) for the per-stratum
breakdown, the math behind every channel, and reproduction instructions
([`docs/benchmarks.md`](docs/benchmarks.md)).

## Pick your profile

Three Pareto-optimal default profiles ship out of the box. Each is a
single environment variable away; the runtime never overwrites a value
the operator already exported, so you keep full override control.

| Profile | Use when | NLI | Reranker | Atomic + concat | Semantic entropy | DE min-pair acc | p95 (A5000, batch=1) | Peak VRAM |
|---|---|---|---|---|---|---|---|---|
| `fast` | sub-200 ms p95 SLO; cheap "is it grounded at all?" check | off | off | off | off | 100% (literal-only) | ~160 ms | ~3.7 GB |
| `balanced` (default) | typical RAG QA serving | on (mDeBERTa) | off | off | off | **92%** | ~190 ms | ~4.2 GB |
| `quality` | high-stakes outputs; long multi-premise contexts | on (mDeBERTa) | on (bge v2-m3) | on | off (opt-in via env, requires recalibration) | **100%** † | ~195 ms | ~4.5 GB |

† DE accuracy from
[`research/triangular_maxsim/reports/truth_bench_n120.json`](research/triangular_maxsim/reports/truth_bench_n120.json)
(`quality`-equivalent config, mDeBERTa NLI + BGE reranker + atomic claims,
n = 26 DE pairs).

Numbers are from the per-profile sweep in
[`research/triangular_maxsim/reports/profile_pareto.md`](research/triangular_maxsim/reports/profile_pareto.md);
each profile bundles a calibrated thresholds artefact under
[`latence_trace/data/thresholds.<profile>.json`](latence_trace/data/) and
a fusion-weight artefact under
[`latence_trace/data/fusion_weights.<profile>.json`](latence_trace/data/).

```bash
# Default (balanced)
latence-trace-server

# Lowest p95
LATENCE_TRACE_PROFILE=fast latence-trace-server
# or
latence-trace-server --profile fast

# Maximum coverage
latence-trace-server --profile quality

# Opt out of all presets and rely on your own env vars
latence-trace-server --profile none
```

See [`docs/guides/profiles.md`](docs/guides/profiles.md) for the full
per-profile config dump, the evaluation methodology, and reproduction
commands.

## How

Five fused channels:

1. **Calibrated reverse MaxSim** - dense token-level support score against the
   context, calibrated against an internal null bank.
2. **Literal guardrails** - rule-based extraction and matching of dates,
   numbers, units, currencies, URLs, identifiers.
3. **NLI peer** - DeBERTa-MNLI verification with cross-encoder premise
   reranking (BAAI/bge-reranker-v2-m3) and atomic-fact decomposition.
4. **Semantic entropy** - bidirectional NLI clustering over multiple
   verification samples, Shannon entropy over clusters.
5. **Structured-source verification** - JSON / markdown-table triple extraction
   with numeric tolerance and alias resolution.
6. **Typed Structured Evidence Lane** *(new)* — for prose-formatted segment
   tables, KV lists, and short single-fact numeric statements (rate
   decisions, KPI summaries, weather/grid records) the runtime extracts
   typed cells from the source `(anchor, value, unit, currency, sign,
   period)` and typed claims from the response `(anchor, value, unit,
   currency, sign)`. Each claim is aligned to its best source cell and
   scored under an **AND-gate** `min(entity_align, value_match,
   unit_match, sign_match)`; the aggregate is `min` over all aligned
   claims. When the lane fires, the headline becomes
   `min(narrative_score, structured_score)` so a single wrong cell
   collapses the score to ~0 — exactly what catches segment-table
   number / sign / unit flips that lexical and dense-similarity
   channels soften through. Behind feature flag
   `VOYAGER_GROUNDEDNESS_STRUCTURED_GATE` (default ON for `quality`
   profile). Lane stays silent on pure prose so the legal / FActScore
   strata see no regression. EN/DE supported (German number formats,
   `Mio./Mrd. EUR`, `Basispunkte`, `Prozentpunkte`).

All channels are renormalized into a single `groundedness_v2` headline; the
runtime classifies that headline into a per-stratum risk band using calibrated
thresholds (`thresholds.json`).

### Retrieval-efficiency observability (context coverage)

Every response carries two scalars and a per-unit flag that turn the
scoring engine into a feedback channel for your retriever:

| Field | Where | What it tells you |
|---|---|---|
| `scores.context_coverage_ratio` | global | **Absolute-strength signal.** Fraction of fetched support units whose `coverage_score` crossed `coverage_threshold` (default 0.5). Range `[0, 1]`. **`0.4` means 60% of the chunks the retriever pulled were dead weight.** Tune the threshold via `coverage_threshold`. |
| `scores.context_attribution_ratio` | global | **Competitive-placement signal.** Fraction of units that won the argmax for at least one response token. Independent of `coverage_threshold`. Useful for dedup-style questions ("which chunks dominated for some span?"). |
| `support_units[i].coverage_score` | per-unit | Max reverse-context similarity any response token had to this unit. Independent of argmax and of threshold. Range `[0, 1]`. |
| `support_units[i].used` | per-unit | `True` when `coverage_score >= coverage_threshold`. Filter `used == False` to surface dead-weight chunks. |

Coverage and attribution are **independent observability axes** — neither
strictly dominates the other. A unit can lose every argmax (low
attribution) yet still have one strong match (high coverage), and
under tight thresholds the reverse holds. Pick the one that maps to
your retrieval-tuning question:

- *"Is the retriever pulling dead weight?"* → `context_coverage_ratio`.
- *"Are sibling chunks crowding each other out for the same spans?"* →
  watch units with `matched_response_tokens > 0` but `used == False`.

Cost: one `max(dim=0)` reduction over the already-computed `(R, U)`
per-unit similarity matrix. Microbench (CPU) shows median ≤ 0.3 ms even
at 4096 response tokens × 256 support units — well below the per-request
encoder/scorer baseline. No extra encoder calls, no extra Triton kernels.

Empirical signal on real HaluEval-QA data
([`scripts/bench_context_coverage.py`](scripts/bench_context_coverage.py)):

| Scenario                                       | `coverage_ratio` (median) | `support_units_used` (median) |
|------------------------------------------------|---------------------------|-------------------------------|
| Grounded response, 1 retrieved chunk           | 1.000                     | 1 / 1                         |
| Grounded response, 1 relevant + 4 distractors  | **0.117**                 | **1.5 / 12**                  |

The signal collapses from 1.0 to ~0.12 when the retriever over-fetches
— exactly the diagnostic we want. Range invariant `0 ≤ ratio ≤ 1` held
for **400 / 400** real samples; reproduction in
[`docs/guides/tutorial.md`](docs/guides/tutorial.md) §5b.

### Long contexts and long responses

Both `raw_context` and `response_text` are **sentence-packed into windows
that fit the encoder's max sequence length and scored chunk-by-chunk**, so
neither side is silently truncated when input exceeds the encoder limit:

| Field | API knob | Default | What it does |
|---|---|---|---|
| `raw_context` | `raw_context_chunk_tokens` | 256 | Splits the context on sentence boundaries into windows of ~N tokens; each window becomes a `support_unit` and is encoded + scored independently. |
| `response_text` | `response_chunk_tokens` | 256 | Splits the response on sentence boundaries into windows of ~N tokens; each window is encoded, scored against the **full** support set, and per-token scores stitched back to global response positions. Single-window responses skip the chunker entirely (parity-preserving fast path). |

The math is exact: `g_t = max_u m_{t,u}` is row-independent, so splitting
the response along the token axis and concatenating per-window results
produces bitwise-identical headline scores to a hypothetical "encode the
full response in one shot" path that would otherwise OOM beyond the
encoder limit. See
[`tests/test_response_chunking_parity.py`](tests/test_response_chunking_parity.py)
for the parity proof and
[`docs/perf/response_chunking_bench.md`](docs/perf/response_chunking_bench.md)
for the linear-scaling microbench.

## Quickstart

```bash
pip install -e ".[dev]"
latence-trace-server --host 0.0.0.0 --port 8090
```

The server boots with the `balanced` profile by default (NLI peer on,
no reranker, ~190 ms p95). Pick `fast` for tighter SLOs or `quality`
for the full stack - see [Pick your profile](#pick-your-profile).

The default encoder is the multilingual
[`VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT)
ModernColBERT checkpoint loaded in `bf16`. Override either with
`VOYAGER_GROUNDEDNESS_MODEL` and `VOYAGER_GROUNDEDNESS_TORCH_DTYPE`.

```bash
# English
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Paris is the capital of France.",
    "query_text": "What is the capital of France?",
    "response_text": "The capital of France is Paris."
  }'

# German
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Berlin ist die Hauptstadt Deutschlands seit 1990.",
    "query_text": "Was ist die Hauptstadt Deutschlands?",
    "response_text": "Die Hauptstadt Deutschlands ist Berlin."
  }'
```

The service returns `scores`, `risk_band`, per-token heatmaps,
`literal_diagnostics`, `structured_diagnostics`, and per-claim NLI evidence
for both languages with the same response schema.

Code-lane request (same endpoint, `scoring_mode` discriminator):

```bash
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_mode": "code",
    "session_id": "ide-session-abc123",
    "response_language_hint": "python",
    "query_text": "Rename `calibrate` to `calibrate_threshold` across the tracer.",
    "raw_context": "# file: tracer.py\nclass Tracer:\n    def calibrate_threshold(self, x): ...\n",
    "response_text": "```python\ntracer.calibrate(x)\n```"
  }'
```

The response adds a `code_lane_diagnostics` block with AST-level drift
counters, literal novelty, the NLI cascade verdict (when it fires), and
per-file ownership with reason codes. Temporal signals (drift, EMA
groundedness, eviction recommendations) ride on an optional
caller-portable `session_state` blob — the API stays stateless, the
caller carries memory. See
[`docs/coding_agent_guide.md`](docs/coding_agent_guide.md) and
[`docs/session_semantics.md`](docs/session_semantics.md) for
copy-paste integration recipes for Claude Code, Cursor, OpenAI Codex,
and OpenCode.

Agents and humans can self-discover the request shape, active profile, and
all sibling endpoints in a single GET:

```bash
curl http://127.0.0.1:8090/agent-help            # canonical request shape, profiles block, endpoint map
curl http://127.0.0.1:8090/.well-known/ai-plugin.json  # ChatGPT-style plugin descriptor
curl http://127.0.0.1:8090/openapi.json          # OpenAPI 3.1 schema
```

Validation errors and service errors share a single structured envelope
(`code`, `message`, `hint`, `docs_url`), so retry / repair loops do not
need to special-case 4xx vs 5xx parsing.

## Multilingual support (English + German)

The default models, regex guardrails, stopwords, conjunction splits, and
calibration null bank are all bilingual EN+DE out of the box, so a German
request like the example above works without any per-request flag:

| Component | Default | Coverage |
|---|---|---|
| ColBERT encoder | `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` (bf16) | EN + DE multilingual |
| NLI peer (default) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | EN + DE + 100 langs |
| NLI peer (English-only deployments, **+10 pp paired acc on HaluEval QA / Summ** vs default) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` via `VOYAGER_GROUNDEDNESS_NLI_MODEL` | EN (best); DE works but weaker than mDeBERTa |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` (opt-in) | Multilingual |
| Atomic-claim splitter | spaCy auto-routes `en_core_web_sm` / `de_core_news_sm` | EN + DE |
| Literal guardrails | Date / number / currency / percent / measurement regex | EN + DE formats |
| Calibration null bank | 16 EN + 16 DE diverse sentences | EN + DE |

See [`docs/guides/multilingual.md`](docs/guides/multilingual.md) for the
full configuration matrix, environment overrides, and notes on adding more
languages.

## Layout

- `latence_trace/core/` - scoring core (groundedness, NLI, claims, semantic
  entropy, structured, thresholds).
- `latence_trace/kernels/` - Triton triangular MaxSim kernel.
- `latence_trace/api/` - FastAPI router, Pydantic models, service layer.
- `latence_trace/auth/` - JWT (Ed25519) license verifier + middleware.
- `latence_trace/middleware/` - request-id and token-bucket rate-limit middleware.
- `latence_trace/observability/` - Prometheus metrics, OpenTelemetry tracing,
  structured JSON logging.
- `latence_trace/providers/` - encoder providers (pylate local, vLLM-factory
  remote ModernColBERT pooling).
- `clients/` - non-Python integration clients and manifests. The public Python
  SDK lives in the sibling `latence-trace-python` repository and publishes the
  `latence` PyPI package.
- `deploy/helm/latence-trace/` - production Helm chart (Deployment +
  Service + ConfigMap + Secret + HPA + PDB + NetworkPolicy +
  ServiceMonitor + Ingress).
- `commercial/` - procurement + buying-side artefacts: security
  one-pager, MSA + DPA + Order Form templates, SOC 2 control mapping,
  Trust Center page, audit-evidence inventory, and the
  [pilot kit](commercial/pilot-kit/) (pilot agreement, success
  criteria, ROI calculator, case-study template, tracker).
- `docs/operations/` - operator-facing runbooks (deployment checklist,
  calibration runbook, observability, licensing).
- `server/` - uvicorn entry point.
- `research/triangular_maxsim/` - evaluation harness, minimal pairs,
  pre-registered targets, committed reports.
- `scripts/` - threshold calibration and fusion-weight sweeps.

## Operate it in production

For Fortune-500 ops teams there is now a complete operator surface:

| Layer | Where |
|---|---|
| Hardened container (uid 65532, RO root, no shells, OCI labels, tini) | [`Dockerfile`](Dockerfile) |
| Helm chart (Deployment / Service / ConfigMap / Secret / HPA / PDB / NetworkPolicy / ServiceMonitor / Ingress) | [`deploy/helm/latence-trace/`](deploy/helm/latence-trace/) |
| Ed25519-signed JWT license enforcement | [`latence_trace/auth/`](latence_trace/auth/), CLI `latence-trace license inspect|verify|fingerprint` |
| Prometheus `/metrics`, OpenTelemetry tracing, structured JSON logs, request-ID propagation | [`latence_trace/observability/`](latence_trace/observability/) |
| Token-bucket rate limiter (per-license / per-IP, burst-capable, `Retry-After` aware) | [`latence_trace/middleware/`](latence_trace/middleware/) |
| Python SDK (sync + async + retries + OTel + framework adapters) | sibling repo `latence-trace-python`, PyPI package `latence` |
| Per-customer threshold / weight refit playbook | [`docs/operations/calibration-runbook.md`](docs/operations/calibration-runbook.md) |
| Procurement + buying artefacts (security one-pager, MSA, DPA, SOC 2 mapping, Trust Center, pilot kit) | [`commercial/`](commercial/) |

See [`docs/operations/`](docs/operations/) for the full operator
walkthrough and [`SECURITY.md`](SECURITY.md) for the security posture
and disclosure process.

## Reproducing benchmarks

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git \
  research/triangular_maxsim/external_data/RAGTruth
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git \
  research/triangular_maxsim/external_data/HaluEval

# FActScore biographies ship without source context (open-domain factuality);
# run scripts/enrich_factscore_with_wiki.py first to backfill Wikipedia text.

export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data
export VOYAGER_GROUNDEDNESS_FACTSCORE_DIR=$PWD/research/triangular_maxsim/external_data/factscore  # optional
export VOYAGER_GROUNDEDNESS_TORCH_DTYPE=bfloat16
export VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1
export VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
export VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 30 --max-external-per-stratum 120 \
  --enable-nli --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/truth_bench_n120.json

# HaluEval-only paired ranking diagnostic (the metric the README cites)
python scripts/diagnose_halueval.py --limit 60 \
  --nli-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
HALUEVAL_DIAGNOSE_OUT=research/triangular_maxsim/reports/halueval_diagnose_deberta_en_n60.json \
  python scripts/diagnose_halueval.py --limit 60 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

Use `n = 20` for fast iteration only — at `n = 20` the F1 95% CI is roughly
±0.20, so single-run F1 deltas are pure noise. The headline numbers above
are at `n = 120` (RAGTruth, internal pairs) and `n = 60` per label per stratum
(HaluEval paired diagnostics).

## License

Proprietary. See [LICENSE](LICENSE).
