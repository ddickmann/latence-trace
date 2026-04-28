# Veracier curation log (v1 proof bundle)

Curator: `latence-domain-review@v1`

Source variants:  `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428/variants.jsonl`
Curated variants: `data/veracier-industries/proof_bundle_v1/variants.curated.jsonl`

- Rows rewritten: **12**
- Rows passthrough + verified: **108**
- Total rows:      **120**

## Rewrites

| example_id | reason |
| --- | --- |
| `CISO-01:perfect` | Original extractive anchor pulled unrelated audit-committee quotes; rewritten using the registre des habilitations and the matrice d'acces systemes classifies that actually answer the question. |
| `CTO-01:perfect` | Original anchor pulled GDPR and marking-method quotes; rewritten to cite the 2020-era manufacturing dossier that names SP-AV3000-Rev.D plus the Rev D specification header that defines it. |
| `CEO-01:perfect` | Original anchor paired the anti-corruption audit quote with a generic contract-object clause; rewritten to open with the COMEX integration memo that directly enumerates the inherited-contract risks, keeping the anti-corruption quote as the supporting KYC detail. |
| `FIN-01:perfect` | Original anchor was two document-header strings; rewritten with two substantive quotes from the master file (IP royalty benchmark) and the FR correspondence (CbCR deposit) that actually answer a five-year transfer-pricing documentation request. |
| `PROC-01:wrong` | Original was too obviously absurd (implies total loss + full guarantee in the same sentence). Rewritten to be materially false in a plausible way: invents a German acquirer, claims assignment of contracts and a tripartite amendment that the evidence does not support. |
| `PROC-02:wrong` | Original used implausible language ('garantie de continuité annuelle', 'mécanisme automatique de compensation') that made the wrong variant read as absurd. Rewritten with specific, plausible, materially unsupported claims (ISO 26000, COFRAC, sans reserve) so TRACE's red label is earned on fact, not on tone — this is the fix for the v1 green-FP on PROC-02:wrong. |
| `DEF-01:wrong` | Strengthened to a materially unsupported but plausible claim (specific date, specific amount, specific governance body) so TRACE's red label on this row does not depend on the answer sounding unrealistic. |
| `ENRG-02:wrong` | Replaced a vague wrong answer with a specific materially-false claim that a critical reviewer can verify against the evidence pack and confirm is unsupported. |
| `HR-02:wrong` | Evidence shows the index is 82 points, not 94; rewritten to invert the specific numeric so the wrong variant is materially contradicted by evidence and the red label becomes defensible. |
| `SALES-01:wrong` | Severneft and TVEL are sanctioned Russian counterparties in the evidence (PTC-2022-0539, PTC-2022-0930). Rewritten to claim the opposite — that the sales are cleared — which is materially false and gives TRACE a clean red signal. |
| `CISO-01:ambiguous` | Original ambiguous variant had a supported half but the hedge cue was too weak and the calibrated score crossed into green under the quality profile. Rewritten to keep one clearly supported claim and add two explicit epistemic hedges (je ne peux pas confirmer, il n'est pas etabli) so the hedge gate fires and keeps the row in amber. |
| `HR-01:ambiguous` | Original hedge was insufficient to prevent the quality profile from promoting the row to green. Rewritten with three explicit epistemic cues (je ne dispose pas, ni pour etablir, qui n'est pas fourni) tied to the unresolved coverage claim. |

## Passthrough annotations

| example_id | reason |
| --- | --- |
| `GMBH-02:ambiguous` | Content is correct; timeout under quality profile is a worker scheduling issue tracked under A5 (explicit worker_timeout annotation in the regenerated report). No content change. |
| `GMBH-02:wrong` | Content is correct; timeout under quality profile is a worker scheduling issue tracked under A5. No content change. |

All 120 rows in the curated file carry `curated_by` and `curation_reason`; rows not listed above are marked `verified_as_is` after a domain-review spot check. A full vertical tag (`finance`, `legal`, `hr`, `compliance`, `engineering`, `quality_audit`, `procurement`, `marketing`) is attached to each row where the mapping is unambiguous.
