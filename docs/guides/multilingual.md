# Multilingual Configuration (English + German)

Latence Trace ships with a bilingual default stack. The same `POST /groundedness`
request shape works for English and German content without any per-request flag,
and the response schema is identical across languages.

This page documents:

- the multilingual default models loaded at runtime,
- which guardrails are language-aware (literal regexes, stopwords, atomic
  claim splitter, calibration null bank),
- the environment overrides for each layer, and
- what you need to swap or recalibrate to add a third language.

## Default models

| Layer | Default | Notes |
|-------|---------|-------|
| ColBERT encoder | `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` | German+English ModernColBERT fine-tune; same tokenizer / IO contract as the English-only `lightonai/GTE-ModernColBERT-v1` it replaces. Loaded in `bf16` by default to halve VRAM with negligible cosine impact. |
| NLI peer (opt-in) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Multilingual mDeBERTa with the same 3-class entail/neutral/contradict head as the English DeBERTa-MNLI it replaces. |
| Cross-encoder reranker (opt-in) | `BAAI/bge-reranker-v2-m3` | Multilingual reranker; no swap needed. |
| Atomic-claim splitter (opt-in) | `en_core_web_sm` and `de_core_news_sm` | Auto-routed per response sentence; both pipelines are cached after first load. Falls back to the multilingual regex splitter if a model is not installed. |

> Install the German spaCy pipeline once per host:
>
> ```bash
> python -m spacy download de_core_news_sm
> python -m spacy download en_core_web_sm
> ```
>
> The runtime tolerates either model being absent: it will degrade to the
> multilingual regex fallback and emit a structured log entry. No request
> ever fails because spaCy is missing.

## Environment overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `VOYAGER_GROUNDEDNESS_MODEL` | `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` | ColBERT encoder name passed to pylate (`models.ColBERT`). |
| `VOYAGER_GROUNDEDNESS_TORCH_DTYPE` | `bfloat16` | One of `bfloat16`, `float16`, `float32`, or `default` to let pylate decide. |
| `VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT` | unset | When set, the encoder is served via vLLM-factory `/pooling`. |
| `VOYAGER_GROUNDEDNESS_VLLM_MODEL` | falls back to `VOYAGER_GROUNDEDNESS_MODEL` then to the multilingual default | Model id used by the vLLM endpoint. |
| `VOYAGER_GROUNDEDNESS_NLI_ENABLED` | `0` | Set to `1` to enable the NLI peer channel. |
| `VOYAGER_GROUNDEDNESS_NLI_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Override the NLI checkpoint. |
| `VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL` | `en_core_web_sm` | English atomic-claim spaCy pipeline. |
| `VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL_DE` | `de_core_news_sm` | German atomic-claim spaCy pipeline. |

All other tunables (fusion weights, NLI batch size, latency budget,
calibration thresholds path, structured-source toggles) keep their existing
behaviour and apply uniformly across languages.

## Language-aware guardrails

### Literal extraction

The literal extractor recognizes English and German formats in the same
pass. After extraction the literals are normalized to a canonical form so
that, for example, the German `1.234,56 EUR` and the English `\u20ac1234.56`
collide on a hash and are treated as the same literal.

| Literal kind | English examples | German examples |
|--------------|------------------|-----------------|
| `date` | `2024-01-15`, `15/01/2024`, `January 15, 2024`, `15 January 2024` | `15.01.2024`, `15. Januar 2024`, `Januar 15, 2024` |
| `currency` | `$1,234.56`, `\u00a3500` | `1.234,56 EUR`, `5,99 \u20ac`, `1.234 CHF` |
| `percent` | `20.5%` | `20,5%` |
| `measurement` | `42.5 km`, `26.2 miles` | `42,5 km`, `1.234,5 Kilometer`, `5 Stunden` |
| `number` | `1234`, `1,234.56`, `1234567` | `1.234,56`, `1.234.567`, `42,5` |
| `url`, `email`, `identifier` | unchanged | unchanged |

Cross-language matches happen when the canonical normalized form is equal:
the German `20,5%` collapses to `20.5%`, which matches an English
`20.5%`. Currency symbols stay distinct (`$` vs `\u20ac`) because mixing them
across a request would be a real factual error, not a formatting quirk.

### Stopwords and content-token weighting

`_STOPWORDS` is a union of common English and German function words
(articles, prepositions, copulas, conjunctions, common pronouns,
negations). The weighting helper used by the headline `reverse_context`
score downweights any token whose lowercased form is in this set, so the
same content mask works for monolingual EN, monolingual DE, and mixed
responses without language detection.

### Atomic-claim splitter

When the NLI peer is enabled, the atomic-claim splitter:

1. Uses a fast function-word + umlaut heuristic per sentence to decide
   English vs German.
2. Routes to the matching cached spaCy pipeline (`en_core_web_sm` or
   `de_core_news_sm`).
3. Falls back to the multilingual regex splitter when neither model is
   installed.

The regex fallback honors English coordinations
(`and`, `but`, `while`, `whereas`) and German coordinations
(`und`, `aber`, `sondern`, `w\u00e4hrend`, `jedoch`, `doch`) plus their
relative-clause introducers (`which`, `who`, `where`, `welche*`, `wobei`,
`woher`, `wohin`).

### Calibration null bank

The default calibration null bank carries 16 English and 16 German short
sentences from disjoint domains (history, science, geography, music, law).
Per-token z-scores and the `reverse_context_calibrated` headline are
computed against this combined null distribution, which keeps the
calibration well-defined for both languages without needing a per-request
language flag.

To swap in a domain-specific or language-specific null bank, build a list
of short, topically diverse sentences in the target language and inject
them via the existing
`compute_null_distribution` API or by overriding `default_null_bank_texts`
at startup.

## Risk-band thresholds

The shipped `thresholds.json` was calibrated on the internal English
minimal-pair suite plus the external English benchmarks (RAGTruth +
HaluEval + FActScore). The calibrated cut-points still apply to German
content because the underlying score distribution is dominated by the
multilingual encoder + NLI signal, not by a language-specific bias term.

That said, if you operate a heavy German workload and want
production-grade per-stratum coverage in the German lane, run
`scripts/calibrate_thresholds.py` on a German minimal-pair set and point
`VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH` at the new artefact. The
calibration script is language-agnostic; only the input pairs change.

## Adding a third language

The same architecture extends to any language with reasonable model
availability:

1. Pick a multilingual or language-specific ColBERT checkpoint that
   exposes the same ModernColBERT or pylate-compatible IO contract; set
   `VOYAGER_GROUNDEDNESS_MODEL`.
2. If you keep the NLI peer, the multilingual mDeBERTa default already
   covers 100+ languages; otherwise point
   `VOYAGER_GROUNDEDNESS_NLI_MODEL` at a language-specific checkpoint.
3. Install or build a spaCy pipeline for atomic-claim splitting and
   point `VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL_<LANG>` at it (the runtime
   reads `..._SPACY_MODEL` for English and `..._SPACY_MODEL_DE` for
   German; add another env-keyed variant by extending `claims.py` if you
   need a third language).
4. Append language-specific function words to `_STOPWORDS`, language
   coordinations to `_COORD_PATTERN`, and date/number/measurement
   patterns to `_LITERAL_PATTERNS` in
   `latence_trace/core/groundedness.py`.
5. Append diverse short sentences in the new language to
   `DEFAULT_NULL_BANK_TEXTS` (or inject a custom bank at startup).
6. Re-run `scripts/calibrate_thresholds.py` against a minimal-pair set in
   the new language and ship the resulting `thresholds.json`.

Each step is additive and non-breaking - existing English and German
traffic is unaffected.
