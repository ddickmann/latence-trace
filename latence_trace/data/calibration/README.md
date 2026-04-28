# rag_prose calibration set

This directory holds the labeled calibration examples used by the
`latence-trace-calibrate-rag-prose` entry point. The fitter consumes
`rag_prose.jsonl` to compute `green_min` / `amber_min` thresholds for the
`rag_prose` stratum and writes them back into
`latence_trace/data/thresholds.balanced.json` and
`latence_trace/data/thresholds.quality.json`.

## Targets

The production targets for the fit are:

- Green precision on a held-out split ≥ 0.97 (a green verdict implies the
  answer is truly grounded in at least 97 % of cases).
- Red precision on a held-out split ≥ 0.95.
- Amber band agreement is measured on the Veracier pilot and the full
  labeled amber subset; no explicit precision target is enforced at fit time
  because amber is a reviewer-queue band by product contract (see
  `latence-trace/docs/amber_reviewer_queue.md`).

## Dataset composition

The full labeled set targets 200 examples with the following mix:

| Label | Target count | Source                                              |
| ----- | ------------ | --------------------------------------------------- |
| green | 80           | Perfect extractive variants across all archetypes.  |
| red   | 80           | Wrong unsupported variants across all archetypes.   |
| amber | 40           | Post-refinement ambiguous variants, human-reviewed. |

The 17 Veracier pilot use cases contribute between 2 and 8 examples each so
archetypes stay balanced. Archetypes currently represented:

- `finance_tax`
- `legal_litigation`
- `legal_contracts`
- `cyber_security`
- `technical_architecture`
- `quality_audit`
- `procurement_supplier`
- `sales_export`
- `operations_capacity`
- `hr_employment`
- `executive_contract_risk`
- `default_enterprise`

## File schema

`rag_prose.jsonl` is a JSON Lines file. Each line is a single example:

```json
{
  "id": "FIN-01:perfect:001",
  "archetype": "finance_tax",
  "language": "fr",
  "label": "green",
  "query_text": "...",
  "response_text": "...",
  "raw_context": "...",
  "source_doc_ids": ["FIN-01:DOC-1"],
  "notes": "Extractive perfect variant with two verbatim evidence quotes."
}
```

Fields:

- `id` (str, required): stable identifier, unique inside the file.
- `archetype` (str, required): one of the archetypes listed above.
- `language` (str, optional): ISO 639-1 code; defaults to `fr` for Veracier
  when missing.
- `label` (str, required): `green`, `amber`, or `red`.
- `query_text` (str, required): the user question fed to TRACE.
- `response_text` (str, required): the answer to be scored.
- `raw_context` (str, required): the evidence pack used during TRACE RAG
  scoring. Include document delimiters exactly as they would appear in
  production (the benchmark orchestrator uses `_raw_context_from_pack`).
- `source_doc_ids` (list[str], optional): doc ids for reviewer traceability.
- `notes` (str, optional): 1-sentence rationale.

## Seeding the file

The initial seed is produced from a live Veracier benchmark run:

```bash
python -m scripts.calibrate_rag_prose build-seed \
    --benchmark-run data/veracier_trace_validation_2026-04-22 \
    --out latence_trace/data/calibration/rag_prose.jsonl \
    --label-from expected-band
```

The `build-seed` subcommand reads `variants.jsonl` + `trace_results.jsonl`
from a completed benchmark run, keeps rows where the live TRACE band matches
the expected band (high-confidence labels), and writes them in the schema
above. Human reviewers then:

1. Promote additional high-quality rows (especially amber) to the file.
2. Remove outliers where the expected band is wrong (eg. unstable amber from
   refinement).
3. Rebalance the three classes so the 80/80/40 target is met.

## Running the fit

```bash
python -m scripts.calibrate_rag_prose fit \
    --calibration-jsonl latence_trace/data/calibration/rag_prose.jsonl \
    --green-precision-target 0.97 \
    --red-precision-target 0.95 \
    --holdout-fraction 0.3 \
    --seed 42 \
    --profiles standard quality
```

This calls TRACE via the SDK once per example per profile, fits the two
thresholds per profile, writes the `rag_prose` stratum back into
`latence_trace/data/thresholds.balanced.json` and
`latence_trace/data/thresholds.quality.json`, and emits a
`rag_prose_fit.json` report alongside the calibration jsonl.
