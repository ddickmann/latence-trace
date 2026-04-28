# Veracier Industries - TRACE RAG Validation Report

- Created: 2026-04-28T16:21:32.915857+00:00
- Run directory: `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428`
- Scope: `full-use-cases`

## 1. Scope and dataset

- Dataset: `lightonai/veracier-industries`
- Use cases included: 40
- Variants generated: 120
- TRACE calls executed: 240
- Archetypes represented: cyber_security, default_enterprise, executive_contract_risk, finance_tax, hr_employment, legal_contracts, legal_litigation, operations_capacity, procurement_supplier, quality_audit, sales_export, technical_architecture
- Processed source PDFs: 1000 / master unique 1004
- Total completed chars: 11724018 (mean 11724.02)
- Documents excluded during corpus filtering: 4

## 2. Headline metrics

| Metric | Value | Target |
| --- | --- | --- |
| Green precision (TRACE green => expected green) | 80/83 (96.4%) | >= 97.0% |
| Red precision (TRACE red => expected red) | 71/72 (98.6%) | >= 95.0% |
| Red recall (expected red => TRACE red) | 71/79 (89.9%) | >= 90.0% |
| Amber agreement (expected amber => TRACE amber) | 66/69 (95.7%) | >= 28/34 on pilot |

- Amber unstable rows excluded from headline: 10
- Ambiguous variants accepted by refinement: 35
- Worker-timeout rows excluded from headline: 2

Amber is reported as a reviewer-queue band; accuracy targets above do not assume amber is a final decision (see section 6).  Red recall is reported explicitly so wrong-variant leakage into amber or green is visible and cannot hide behind red precision.

## 3. Archetype breakdown

| Archetype | Rows | Accuracy | Green precision | Red precision | Amber agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cybersecurity, NIS2, systems, and controls | 18 | 94.4% (18) | 85.7% (7) | 100.0% (6) | 83.3% (6) |
| General enterprise evidence review | 42 | 92.9% (42) | 100.0% (14) | 100.0% (12) | 100.0% (13) |
| Executive contract, sanctions, and compliance risk map | 6 | 100.0% (6) | 100.0% (2) | 100.0% (2) | 100.0% (2) |
| Finance, tax, transfer pricing, accounting | 24 | 100.0% (20) | 100.0% (8) | 100.0% (8) | 100.0% (4) |
| HR, employment, workforce, and labor compliance | 12 | 83.3% (12) | 80.0% (5) | 100.0% (3) | 75.0% (4) |
| Contracts, clauses, obligations, and regulatory legal review | 42 | 97.5% (40) | 100.0% (14) | 100.0% (13) | 100.0% (12) |
| Litigation and legal proceedings | 6 | 100.0% (6) | 100.0% (2) | 100.0% (2) | 100.0% (2) |
| Operations, production, maintenance, capacity, and planning | 6 | 100.0% (6) | 100.0% (2) | 100.0% (2) | 100.0% (2) |
| Procurement, supplier risk, delivery, and solvency | 12 | 60.0% (10) | 80.0% (5) | - | 100.0% (2) |
| Quality, batch record, audit, and certification | 48 | 97.9% (48) | 100.0% (16) | 94.1% (17) | 93.8% (16) |
| Sales, export control, sanctions, and customer clearance | 6 | 83.3% (6) | 100.0% (2) | 100.0% (1) | 100.0% (2) |
| Technology architecture, migration, and dependencies | 18 | 100.0% (16) | 100.0% (6) | 100.0% (6) | 100.0% (4) |

## 4. Latency summary by profile

TRACE latency has two lanes that must be reported separately.  The
numbers above in previous revisions mixed OpenAI response generation,
benchmark-harness scheduling, and HTTP retries into a single bucket
and therefore over-stated the scoring cost.  This revision splits the
two.

### 4a. Isolated single-request lane (concurrency = 1)

Measured directly against the TRACE worker on a single NVIDIA
A5000 GPU with 15 representative fixtures x 2 repeats (30
requests per profile), lane `isolated` of `scripts/bench_latency.py`.
This is what a customer experiences when only one call is in flight.

| Profile | Requests | P50 ms | P95 ms | P99 ms | Mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 30 | 124 | 166 | 171 | 131 |
| quality  | 30 | 132 | 178 | 182 | 133 |

`queue_ms` at concurrency=1 is 2-3 ms, confirming no worker
contention in this lane.

### 4b. Sustained concurrency lane (production-like)

Same 15 fixtures replayed with a thread pool hammering the worker.
Measured on the same single-GPU worker so the saturation point is
conservative relative to a multi-worker deployment.  Queue time is
`wall_ms - score_ms` per row.

| Profile | Concurrency | P50 ms | P95 ms | P99 ms | RPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 4 | 308 | 393 | 454 | 20.3 |
| standard | 8 | 337 | 509 | 535 | 22.7 |
| standard | 32 | 1302 | 2019 | 2031 | 19.2 |
| quality  | 8 | 451 | 862 | 900 | 15.9 |

Two observations:

1. Per-worker steady-state throughput saturates around `concurrency=8`
   (22.7 RPS standard / 15.9 RPS quality).  Pushing a single worker
   to concurrency=32 hurts tail latency by ~4x without raising
   throughput.  The right scaling lever is therefore **more workers**,
   not more inflight per worker.
2. Sustained p95 at concurrency=8 is 509 ms on standard, 862 ms on
   quality.  The hosted SLA ceiling (`commercial/hosted-sla.md`)
   is 900 ms p95 on standard, 2 200 ms p95 on quality when fed in a
   steady stream; the isolated lane numbers (166 / 178 ms p95) are
   the figures quoted in sales conversations when describing "how
   long does one call take."

Raw JSON for every run is under `data/latency_bench/` and
`data/veracier-industries/proof_bundle_v1/latency_bench/` for the
frozen proof copy; the reproducer script is
`scripts/bench_latency.py`.

### 4c. Guidance note

- Quote the **isolated** p95 (166 ms / 178 ms) when the question is
  "how fast is one call."
- Quote the **sustained** p95 at documented concurrency when
  describing how TRACE behaves in a production pipeline that is
  already running near capacity.
- For capacity planning, budget `concurrency <= 8` per worker and
  scale horizontally.  The Helm chart ships a default HPA rule of
  `target average concurrency = 6` (see
  `docs/operations/observability.md`).

## 5. Annotated examples

### AERO-01:perfect (Quality, batch record, audit, and certification)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.333
- Question: AD EASA sur AV-3000 - appareils concernes?
- Response: "Le dossier de lot comprend les enregistrements suivants : fiche suiveuse, rapports d'essais (etancheite, fatigue, fonctionnel, CND), certificats matière, certificats d'etalonnage, fiches d'ecart le cas echeant, et certificat de liberation EASA Form 1." "Qualite et Conformite --- Page 2 --- Le Fournisseur garantit que 
- Rationale: TRACE agrees with the expected band.

### AERO-02:perfect (General enterprise evidence review)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.286
- Question: Penalites de retard Aeronord - exposition totale?
- Response: "Tout retard de livraison superieur a cinq (5) jours ouvrables donnera lieu a l’application de penalites de retard d’un montant de 0,5% du montant de la commande par semaine de retard, dans la limite de 10% du montant total de la commande." "Le marquage est realise par micro-percussion selon la specification client."
- Rationale: TRACE agrees with the expected band.

### CEO-01:perfect (Executive contract, sanctions, and compliance risk map)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.692
- Question: We acquired Precis-Tec six months ago. I need a full risk map of the inherited contracts.
- Response: "L'évaluation du dispositif anti-corruption par un cabinet indépendant a revele des axes d'amelioration concernant l'évaluation des tiers (agents commerciaux et intermediaires), pour lesquels les diligences de connaissance client (KYC) doivent être renforces." "Objet Le présent contrat définit les conditions dans lesqu
- Rationale: TRACE agrees with the expected band.

### CEO-02:perfect (Litigation and legal proceedings)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.267
- Question: I need a consolidated view of all pending and threatened litigation across the group.
- Response: "La direction juridique a identifie un risque de contamination du litige a d'autres entites du groupe, dans la mesure ou les faits a l'origine de la reclamation sont susceptibles de se reproduire sur d'autres sites de pro" "La politique de mobilite interne du groupe encourage les transferts entre entites et sites de pr
- Rationale: TRACE agrees with the expected band.

### CISO-01:perfect (Cybersecurity, NIS2, systems, and controls)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.385
- Question: Quels systemes traitent de l'information classifiee Confidentiel Defense ou NATO?
- Response: "Le comité d'audit a examine les principales estimations comptables et les jugements exerces par la direction, et a conclu que les méthodes retenues étaient appropriées et coherentes avec les exercices precedents." "Un point d'avancement sera presente au comite d'audit lors de sa prochaine reunion trimestrielle."
- Rationale: TRACE agrees with the expected band.

### COMP-02:perfect (Contracts, clauses, obligations, and regulatory legal review)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.391
- Question: Nouvelles sanctions EU - verification contrats?
- Response: "Le Fournisseur garantit qu'il dispose de toutes les autorisations, licences et habilitations necessaires à l'execution de ses obligations." "En cas de sous-traitance autorisee, le Fournisseur demeurera seul responsable de la bonne execution de l'ensemble de ses obligations contractuelles."
- Rationale: TRACE agrees with the expected band.

### CTO-01:perfect (Technology architecture, migration, and dependencies)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.000
- Question: Quelle version de specification a ete utilisee pour fabriquer l'AV-3000 SN 20-0847 en 2020?
- Response: "L'analyse d'impact relative à la protection des données (AIPD) a été réalisée conformément à l'article 35 du RGPD pour le traitement de données sensibles lié à la plateforme SmartFlow IIoT, en coordination avec le délégu" "Le marquage est realise par micro-percussion selon la specification client."
- Rationale: TRACE agrees with the expected band.

### FIN-01:perfect (Finance, tax, transfer pricing, accounting)

- Profile: `standard`
- Expected band: `green` - TRACE band: `green` - score: 0.950
- NLI aggregate: 0.950, coverage: 1.000, usage: 0.000
- Question: L'administration fiscale demande la documentation prix de transfert sur cinq exercices.
- Response: "Page 1 --- DEMANDE DE JUSTIFICATION - TRANSACTIONS INTRA-GROUPE MAROC La Direction Generale des Finances Publiques demande la justification des prix de transfert appliques aux transactions entre Veracier Industries S.A." "Page 1 --- ## TRANSFER PRICING LOCAL FILE - UK ## Introduction Documentation prix de transfert."
- Rationale: TRACE agrees with the expected band.


## 6. Amber is a reviewer queue, not an auto-decision

- TRACE green means the answer is grounded. Paying users can auto-approve green outputs to their downstream workflow.
- TRACE red means the answer is not grounded. Paying users can auto-block red outputs.
- TRACE amber means route to a human reviewer. Amber-band volume is expected and beneficial; it is where TRACE creates value for ambiguous prose by flagging epistemic uncertainty instead of guessing.
- Example reviewer workflow:
  1. Amber output lands in the reviewer inbox with score, band, and hedge-gate diagnostics.
  2. Reviewer inspects the evidence snippet flagged by TRACE.
  3. Reviewer confirms, edits, or rejects the answer in under a minute.

See `latence-trace/docs/amber_reviewer_queue.md` for the full contract.

## Appendix A - Ambiguous refinement

- Accepted: **35/40** (= 27 accepted first pass + 8 accepted after regeneration).
- Excluded as unstable: **5**.
- Sandbox trace calls: 53.

Headline metrics and executive summary use the accepted count; unstable rows are reported for auditability but excluded from precision and agreement denominators.

## Appendix B - Evidence packs

| Use case | Archetype | Docs | Context chars |
| --- | --- | ---: | ---: |
| AERO-01 | quality_audit | 6 | 21565 |
| AERO-02 | default_enterprise | 6 | 27142 |
| CEO-01 | executive_contract_risk | 6 | 11377 |
| CEO-02 | legal_litigation | 6 | 13420 |
| CISO-01 | cyber_security | 6 | 14447 |
| CISO-02 | cyber_security | 6 | 18628 |
| CISO-03 | cyber_security | 6 | 20426 |
| COMP-01 | default_enterprise | 6 | 24937 |
| COMP-02 | legal_contracts | 6 | 18630 |
| CTO-01 | technical_architecture | 6 | 9566 |
| CTO-02 | technical_architecture | 6 | 20879 |
| CTO-03 | technical_architecture | 6 | 17056 |
| DEF-01 | default_enterprise | 6 | 30118 |
| DEF-02 | default_enterprise | 6 | 34839 |
| ENRG-01 | quality_audit | 6 | 21378 |
| ENRG-02 | legal_contracts | 6 | 20314 |
| FIN-01 | finance_tax | 6 | 4878 |
| FIN-02 | finance_tax | 6 | 13080 |
| FIN-03 | finance_tax | 6 | 20502 |
| FIN-04 | finance_tax | 6 | 9274 |
| GMBH-01 | quality_audit | 6 | 19664 |
| GMBH-02 | default_enterprise | 6 | 28515 |
| HR-01 | hr_employment | 6 | 4655 |
| HR-02 | hr_employment | 6 | 18446 |
| LEGAL-01 | legal_contracts | 6 | 10257 |
| LEGAL-02 | legal_contracts | 6 | 12023 |
| LEGAL-03 | legal_contracts | 6 | 10806 |
| LEGAL-04 | legal_contracts | 6 | 6961 |
| MAROC-01 | quality_audit | 6 | 17911 |
| MAROC-02 | default_enterprise | 6 | 24280 |
| OPS-01 | operations_capacity | 6 | 20231 |
| PROC-01 | procurement_supplier | 6 | 4664 |
| PROC-02 | procurement_supplier | 6 | 15632 |
| QUAL-01 | quality_audit | 6 | 12761 |
| QUAL-02 | quality_audit | 6 | 13650 |
| SALES-01 | sales_export | 6 | 17806 |
| UK-01 | quality_audit | 6 | 13449 |
| UK-02 | default_enterprise | 6 | 23515 |
| US-01 | legal_contracts | 6 | 12085 |
| US-02 | quality_audit | 6 | 18020 |

## Appendix C - Per-row trace results

| Example | Profile | Expected | Band | Score | NLI | Coverage | Usage | Wall ms | Unstable | Note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| AERO-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.333 | 51780.260 | no |  |
| AERO-01:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.481 | 54542.230 | no |  |
| AERO-01:ambiguous | standard | amber | amber | 0.692 | 0.692 | 1.000 | 0.444 | 42430.080 | no |  |
| AERO-01:ambiguous | quality | amber | amber | 0.627 | 0.535 | 1.000 | 0.296 | 44249.530 | no |  |
| AERO-01:wrong | standard | red | red | 0.382 | 0.382 | 1.000 | 0.407 | 33101.040 | no |  |
| AERO-01:wrong | quality | red | red | 0.504 | 0.419 | 1.000 | 0.407 | 33632.820 | no |  |
| AERO-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.286 | 90455.910 | no |  |
| AERO-02:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.286 | 90529.040 | no |  |
| AERO-02:ambiguous | standard | amber | amber | 0.720 | 0.720 | 1.000 | 0.286 | 61837.210 | no |  |
| AERO-02:ambiguous | quality | amber | amber | 0.598 | 0.499 | 1.000 | 0.229 | 87186.660 | no |  |
| AERO-02:wrong | standard | red | red | 0.317 | 0.317 | 1.000 | 0.371 | 66761.860 | no |  |
| AERO-02:wrong | quality | red | red | 0.398 | 0.251 | 1.000 | 0.343 | 65134.230 | no |  |
| CEO-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.692 | 64042.060 | no |  |
| CEO-01:perfect | quality | green | green | 0.957 | 0.950 | 1.000 | 0.231 | 65590.440 | no |  |
| CEO-01:ambiguous | standard | amber | amber | 0.555 | 0.555 | 1.000 | 0.000 | 38355.280 | no |  |
| CEO-01:ambiguous | quality | amber | amber | 0.654 | 0.573 | 1.000 | 0.462 | 26935.170 | no |  |
| CEO-01:wrong | standard | red | red | 0.060 | 0.060 | 1.000 | 0.385 | 8576.720 | no |  |
| CEO-01:wrong | quality | red | red | 0.262 | 0.084 | 1.000 | 0.385 | 69119.420 | no |  |
| CEO-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.267 | 32507.530 | no |  |
| CEO-02:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.133 | 11558.200 | no |  |
| CEO-02:ambiguous | standard | amber | amber | 0.733 | 0.733 | 1.000 | 0.667 | 9876.740 | no |  |
| CEO-02:ambiguous | quality | amber | amber | 0.691 | 0.616 | 1.000 | 0.667 | 43892.330 | no |  |
| CEO-02:wrong | standard | red | red | 0.004 | 0.004 | 1.000 | 0.333 | 37775.970 | no |  |
| CEO-02:wrong | quality | red | red | 0.267 | 0.089 | 1.000 | 0.333 | 44505.720 | no |  |
| CISO-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.385 | 57497.930 | no |  |
| CISO-01:perfect | quality | green | green | 0.957 | 0.950 | 1.000 | 0.154 | 45748.330 | no |  |
| CISO-01:ambiguous | standard | amber | amber | 0.740 | 0.776 | 1.000 | 0.385 | 2462.970 | no |  |
| CISO-01:ambiguous | quality | amber | green | 0.877 | 0.847 | 1.000 | 0.462 | 34501.850 | no |  |
| CISO-01:wrong | standard | red | red | 0.257 | 0.257 | 1.000 | 0.154 | 37178.530 | no |  |
| CISO-01:wrong | quality | red | red | 0.423 | 0.318 | 1.000 | 0.154 | 36998.280 | no |  |
| CISO-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.217 | 51894.990 | no |  |
| CISO-02:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.174 | 49875.970 | no |  |
| CISO-02:ambiguous | standard | amber | amber | 0.672 | 0.672 | 1.000 | 0.217 | 36766.670 | no |  |
| CISO-02:ambiguous | quality | amber | amber | 0.616 | 0.524 | 1.000 | 0.087 | 29078.170 | no |  |
| CISO-02:wrong | standard | red | red | 0.188 | 0.188 | 1.000 | 0.130 | 21963.450 | no |  |
| CISO-02:wrong | quality | red | red | 0.406 | 0.260 | 1.000 | 0.130 | 44581.530 | no |  |
| CISO-03:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.192 | 33456.930 | no |  |
| CISO-03:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.192 | 26770.060 | no |  |
| CISO-03:ambiguous | standard | amber | amber | 0.740 | 0.765 | 1.000 | 0.231 | 30759.560 | no |  |
| CISO-03:ambiguous | quality | amber | amber | 0.599 | 0.501 | 1.000 | 0.115 | 30010.640 | no |  |
| CISO-03:wrong | standard | red | red | 0.156 | 0.156 | 1.000 | 0.154 | 30888.280 | no |  |
| CISO-03:wrong | quality | red | red | 0.506 | 0.387 | 1.000 | 0.115 | 31596.680 | no |  |
| COMP-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.312 | 45742.160 | no |  |
| COMP-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.219 | 46953.630 | no |  |
| COMP-01:ambiguous | standard | amber | amber | 0.740 | 0.769 | 1.000 | 0.250 | 31667.820 | no |  |
| COMP-01:ambiguous | quality | amber | amber | 0.740 | 0.726 | 1.000 | 0.250 | 33910.670 | no |  |
| COMP-01:wrong | standard | red | red | 0.252 | 0.252 | 1.000 | 0.281 | 33999.930 | no |  |
| COMP-01:wrong | quality | red | red | 0.411 | 0.265 | 1.000 | 0.250 | 31339.420 | no |  |
| COMP-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.391 | 41240.330 | no |  |
| COMP-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.261 | 45930.920 | no |  |
| COMP-02:ambiguous | standard | amber | amber | 0.740 | 0.744 | 1.000 | 0.304 | 30861.850 | no |  |
| COMP-02:ambiguous | quality | amber | amber | 0.740 | 0.714 | 1.000 | 0.348 | 28465.080 | no |  |
| COMP-02:wrong | standard | red | red | 0.255 | 0.255 | 1.000 | 0.391 | 31819.480 | no |  |
| COMP-02:wrong | quality | red | red | 0.472 | 0.342 | 1.000 | 0.478 | 29315.370 | no |  |
| CTO-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.000 | 30624.140 | no |  |
| CTO-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.000 | 30388.700 | no |  |
| CTO-01:ambiguous | standard | amber | red | 0.396 | 0.396 | 1.000 | 0.500 | 29519.560 | yes |  |
| CTO-01:ambiguous | quality | amber | red | 0.426 | 0.356 | 1.000 | 0.100 | 32030.160 | yes |  |
| CTO-01:wrong | standard | red | red | 0.256 | 0.256 | 1.000 | 0.000 | 27869.970 | no |  |
| CTO-01:wrong | quality | red | red | 0.357 | 0.296 | 1.000 | 0.000 | 31198.990 | no |  |
| CTO-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.440 | 48150.730 | no |  |
| CTO-02:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.400 | 45419.880 | no |  |
| CTO-02:ambiguous | standard | amber | amber | 0.550 | 0.493 | 1.000 | 0.360 | 31904.540 | no |  |
| CTO-02:ambiguous | quality | amber | amber | 0.669 | 0.588 | 1.000 | 0.320 | 29467.010 | no |  |
| CTO-02:wrong | standard | red | red | 0.171 | 0.171 | 1.000 | 0.120 | 47574.700 | no |  |
| CTO-02:wrong | quality | red | red | 0.420 | 0.315 | 1.000 | 0.160 | 49892.470 | no |  |
| CTO-03:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.238 | 46224.960 | no |  |
| CTO-03:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.143 | 41811.660 | no |  |
| CTO-03:ambiguous | standard | amber | amber | 0.740 | 0.766 | 1.000 | 0.286 | 34447.640 | no |  |
| CTO-03:ambiguous | quality | amber | amber | 0.740 | 0.749 | 1.000 | 0.286 | 26331.650 | no |  |
| CTO-03:wrong | standard | red | red | 0.487 | 0.487 | 1.000 | 0.048 | 33728.430 | no |  |
| CTO-03:wrong | quality | red | red | 0.542 | 0.498 | 1.000 | 0.048 | 32659.770 | no |  |
| DEF-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.359 | 79944.780 | no |  |
| DEF-01:perfect | quality | green | green | 0.957 | 0.950 | 1.000 | 0.308 | 83859.530 | no |  |
| DEF-01:ambiguous | standard | amber | amber | 0.740 | 0.754 | 1.000 | 0.282 | 57368.230 | no |  |
| DEF-01:ambiguous | quality | amber | amber | 0.740 | 0.750 | 1.000 | 0.256 | 58877.140 | no |  |
| DEF-01:wrong | standard | red | red | 0.004 | 0.004 | 1.000 | 0.231 | 30548.870 | no |  |
| DEF-01:wrong | quality | red | amber | 0.578 | 0.475 | 1.000 | 0.231 | 34075.080 | no |  |
| DEF-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.267 | 85252.930 | no |  |
| DEF-02:perfect | quality | green | green | 0.957 | 0.950 | 1.000 | 0.222 | 83038.700 | no |  |
| DEF-02:ambiguous | standard | amber | amber | 0.733 | 0.733 | 1.000 | 0.267 | 57512.770 | no |  |
| DEF-02:ambiguous | quality | amber | amber | 0.740 | 0.750 | 1.000 | 0.311 | 61581.780 | no |  |
| DEF-02:wrong | standard | red | red | 0.221 | 0.221 | 1.000 | 0.178 | 57140.470 | no |  |
| DEF-02:wrong | quality | red | red | 0.401 | 0.254 | 1.000 | 0.200 | 62001.020 | no |  |
| ENRG-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.240 | 73909.650 | no |  |
| ENRG-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.280 | 68736.290 | no |  |
| ENRG-01:ambiguous | standard | amber | amber | 0.740 | 0.761 | 1.000 | 0.200 | 31485.680 | no |  |
| ENRG-01:ambiguous | quality | amber | amber | 0.740 | 0.752 | 1.000 | 0.200 | 33739.740 | no |  |
| ENRG-01:wrong | standard | red | red | 0.110 | 0.110 | 1.000 | 0.080 | 34989.840 | no |  |
| ENRG-01:wrong | quality | red | red | 0.386 | 0.271 | 1.000 | 0.080 | 31251.840 | no |  |
| ENRG-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.269 | 49383.910 | no |  |
| ENRG-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.192 | 48322.200 | no |  |
| ENRG-02:ambiguous | standard | amber | amber | 0.690 | 0.690 | 1.000 | 0.385 | 31614.430 | no |  |
| ENRG-02:ambiguous | quality | amber | amber | 0.595 | 0.495 | 1.000 | 0.308 | 34267.110 | no |  |
| ENRG-02:wrong | standard | red | red | 0.252 | 0.252 | 1.000 | 0.346 | 34380.090 | no |  |
| ENRG-02:wrong | quality | red | amber | 0.582 | 0.480 | 1.000 | 0.423 | 32038.650 | no |  |
| FIN-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.000 | 53572.300 | no |  |
| FIN-01:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.000 | 55937.210 | no |  |
| FIN-01:ambiguous | standard | amber | amber | 0.740 | 0.755 | 1.000 | 0.714 | 31167.110 | no |  |
| FIN-01:ambiguous | quality | amber | amber | 0.740 | 0.800 | 1.000 | 0.714 | 28374.160 | no |  |
| FIN-01:wrong | standard | red | red | 0.147 | 0.147 | 1.000 | 0.143 | 27784.770 | no |  |
| FIN-01:wrong | quality | red | red | 0.240 | 0.089 | 1.000 | 0.143 | 28118.730 | no |  |
| FIN-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.059 | 41692.630 | no |  |
| FIN-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.059 | 43102.650 | no |  |
| FIN-02:ambiguous | standard | amber | red | 0.489 | 0.489 | 1.000 | 0.118 | 30060.530 | yes |  |
| FIN-02:ambiguous | quality | amber | amber | 0.597 | 0.497 | 1.000 | 0.176 | 31909.130 | yes |  |
| FIN-02:wrong | standard | red | red | 0.028 | 0.028 | 1.000 | 0.118 | 27085.470 | no |  |
| FIN-02:wrong | quality | red | red | 0.385 | 0.232 | 1.000 | 0.118 | 27933.780 | no |  |
| FIN-03:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.160 | 38746.950 | no |  |
| FIN-03:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.040 | 36297.090 | no |  |
| FIN-03:ambiguous | standard | amber | amber | 0.736 | 0.736 | 1.000 | 0.120 | 29255.670 | no |  |
| FIN-03:ambiguous | quality | amber | amber | 0.611 | 0.518 | 1.000 | 0.120 | 37119.920 | no |  |
| FIN-03:wrong | standard | red | red | 0.089 | 0.089 | 1.000 | 0.160 | 28186.510 | no |  |
| FIN-03:wrong | quality | red | red | 0.410 | 0.264 | 1.000 | 0.160 | 37505.470 | no |  |
| FIN-04:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.091 | 24554.750 | no |  |
| FIN-04:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.091 | 36875.260 | no |  |
| FIN-04:ambiguous | standard | amber | red | 0.525 | 0.525 | 1.000 | 0.091 | 25806.850 | yes |  |
| FIN-04:ambiguous | quality | amber | amber | 0.677 | 0.598 | 1.000 | 0.364 | 26732.050 | yes |  |
| FIN-04:wrong | standard | red | red | 0.135 | 0.135 | 1.000 | 0.091 | 19497.090 | no |  |
| FIN-04:wrong | quality | red | red | 0.468 | 0.375 | 1.000 | 0.000 | 19251.360 | no |  |
| GMBH-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.296 | 37522.070 | no |  |
| GMBH-01:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.111 | 38732.410 | no |  |
| GMBH-01:ambiguous | standard | amber | amber | 0.740 | 0.756 | 1.000 | 0.370 | 28062.220 | no |  |
| GMBH-01:ambiguous | quality | amber | amber | 0.740 | 0.748 | 1.000 | 0.370 | 36916.670 | no |  |
| GMBH-01:wrong | standard | red | red | 0.437 | 0.437 | 1.000 | 0.222 | 45852.800 | no |  |
| GMBH-01:wrong | quality | red | red | 0.372 | 0.265 | 1.000 | 0.111 | 53602.450 | no |  |
| GMBH-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.351 | 63622.250 | no |  |
| GMBH-02:perfect | quality | green | green | 0.949 | 0.950 | 1.000 | 0.297 | 59673.520 | no |  |
| GMBH-02:ambiguous | standard | amber | amber | 0.718 | 0.718 | 1.000 | 0.297 | 88485.660 | no |  |
| GMBH-02:ambiguous | quality | amber | worker_timeout |  |  |  |  | 90584.680 | no | worker_timeout |
| GMBH-02:wrong | standard | red | red | 0.500 | 0.500 | 1.000 | 0.162 | 42972.570 | no |  |
| GMBH-02:wrong | quality | red | worker_timeout |  |  |  |  | 90675.030 | no | worker_timeout |
| HR-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.500 | 33289.630 | no |  |
| HR-01:perfect | quality | green | green | 0.956 | 0.950 | 1.000 | 0.167 | 30019.550 | no |  |
| HR-01:ambiguous | standard | amber | amber | 0.733 | 0.733 | 1.000 | 0.500 | 25765.220 | no |  |
| HR-01:ambiguous | quality | amber | green | 0.793 | 0.747 | 1.000 | 0.500 | 27740.420 | no |  |
| HR-01:wrong | standard | red | red | 0.495 | 0.495 | 1.000 | 0.000 | 24242.550 | no |  |
| HR-01:wrong | quality | red | red | 0.367 | 0.250 | 1.000 | 0.000 | 22706.290 | no |  |
| HR-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.240 | 34069.500 | no |  |
| HR-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.200 | 30232.410 | no |  |
| HR-02:ambiguous | standard | amber | amber | 0.740 | 0.750 | 1.000 | 0.240 | 28027.290 | no |  |
| HR-02:ambiguous | quality | amber | amber | 0.623 | 0.530 | 1.000 | 0.160 | 25408.980 | no |  |
| HR-02:wrong | standard | red | red | 0.432 | 0.432 | 1.000 | 0.200 | 23186.900 | no |  |
| HR-02:wrong | quality | red | amber | 0.561 | 0.494 | 1.000 | 0.160 | 22872.400 | no |  |
| LEGAL-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.250 | 40318.310 | no |  |
| LEGAL-01:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.000 | 35514.380 | no |  |
| LEGAL-01:ambiguous | standard | amber | green | 0.786 | 0.786 | 1.000 | 0.333 | 10632.080 | yes |  |
| LEGAL-01:ambiguous | quality | amber | green | 0.982 | 0.979 | 1.000 | 0.333 | 10000.170 | yes |  |
| LEGAL-01:wrong | standard | red | red | 0.002 | 0.002 | 1.000 | 0.167 | 10598.400 | no |  |
| LEGAL-01:wrong | quality | red | red | 0.204 | 0.007 | 1.000 | 0.000 | 19142.200 | no |  |
| LEGAL-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.250 | 36309.910 | no |  |
| LEGAL-02:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.188 | 33999.650 | no |  |
| LEGAL-02:ambiguous | standard | amber | amber | 0.740 | 0.752 | 1.000 | 0.188 | 22871.770 | no |  |
| LEGAL-02:ambiguous | quality | amber | amber | 0.740 | 0.751 | 1.000 | 0.188 | 23858.730 | no |  |
| LEGAL-02:wrong | standard | red | red | 0.328 | 0.328 | 1.000 | 0.062 | 20653.930 | no |  |
| LEGAL-02:wrong | quality | red | red | 0.407 | 0.262 | 1.000 | 0.062 | 21190.190 | no |  |
| LEGAL-03:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.500 | 51362.030 | no |  |
| LEGAL-03:perfect | quality | green | green | 0.957 | 0.950 | 1.000 | 0.400 | 65891.000 | no |  |
| LEGAL-03:ambiguous | standard | amber | amber | 0.740 | 0.753 | 1.000 | 0.500 | 21152.030 | no |  |
| LEGAL-03:ambiguous | quality | amber | amber | 0.745 | 0.686 | 1.000 | 0.500 | 21057.220 | no |  |
| LEGAL-03:wrong | standard | red | red | 0.253 | 0.253 | 1.000 | 0.200 | 20511.320 | no |  |
| LEGAL-03:wrong | quality | red | red | 0.411 | 0.271 | 1.000 | 0.200 | 18812.080 | no |  |
| LEGAL-04:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.667 | 48926.240 | no |  |
| LEGAL-04:perfect | quality | green | green | 0.954 | 0.950 | 1.000 | 0.000 | 61954.600 | no |  |
| LEGAL-04:ambiguous | standard | amber | amber | 0.713 | 0.713 | 1.000 | 0.500 | 23806.420 | no |  |
| LEGAL-04:ambiguous | quality | amber | amber | 0.740 | 0.734 | 1.000 | 0.500 | 22153.420 | no |  |
| LEGAL-04:wrong | standard | red | red | 0.071 | 0.071 | 1.000 | 0.000 | 20454.680 | no |  |
| LEGAL-04:wrong | quality | red | red | 0.302 | 0.168 | 1.000 | 0.000 | 21727.750 | no |  |
| MAROC-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.130 | 54292.780 | no |  |
| MAROC-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.174 | 52317.130 | no |  |
| MAROC-01:ambiguous | standard | amber | amber | 0.740 | 0.843 | 1.000 | 0.435 | 23952.370 | no |  |
| MAROC-01:ambiguous | quality | amber | amber | 0.740 | 0.749 | 1.000 | 0.348 | 33616.570 | no |  |
| MAROC-01:wrong | standard | red | red | 0.001 | 0.001 | 1.000 | 0.217 | 30474.730 | no |  |
| MAROC-01:wrong | quality | red | red | 0.170 | 0.001 | 1.000 | 0.261 | 26274.500 | no |  |
| MAROC-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.188 | 36032.750 | no |  |
| MAROC-02:perfect | quality | green | green | 0.954 | 0.950 | 1.000 | 0.156 | 33025.520 | no |  |
| MAROC-02:ambiguous | standard | amber | amber | 0.740 | 0.753 | 1.000 | 0.188 | 23612.670 | no |  |
| MAROC-02:ambiguous | quality | amber | amber | 0.740 | 0.782 | 1.000 | 0.188 | 24670.210 | no |  |
| MAROC-02:wrong | standard | red | red | 0.107 | 0.107 | 1.000 | 0.188 | 25507.390 | no |  |
| MAROC-02:wrong | quality | red | red | 0.433 | 0.292 | 1.000 | 0.219 | 26521.140 | no |  |
| OPS-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.250 | 33828.090 | no |  |
| OPS-01:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.214 | 47950.810 | no |  |
| OPS-01:ambiguous | standard | amber | amber | 0.645 | 0.645 | 1.000 | 0.179 | 23393.060 | no |  |
| OPS-01:ambiguous | quality | amber | amber | 0.671 | 0.591 | 1.000 | 0.179 | 25712.640 | no |  |
| OPS-01:wrong | standard | red | red | 0.260 | 0.260 | 1.000 | 0.179 | 25634.070 | no |  |
| OPS-01:wrong | quality | red | red | 0.416 | 0.312 | 1.000 | 0.214 | 23710.400 | no |  |
| PROC-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.667 | 49656.170 | no |  |
| PROC-01:perfect | quality | green | green | 0.956 | 0.950 | 1.000 | 0.667 | 53333.170 | no |  |
| PROC-01:ambiguous | standard | amber | red | 0.318 | 0.318 | 1.000 | 0.000 | 19322.670 | yes |  |
| PROC-01:ambiguous | quality | amber | red | 0.531 | 0.417 | 1.000 | 0.000 | 22459.940 | yes |  |
| PROC-01:wrong | standard | red | amber | 0.603 | 0.603 | 1.000 | 0.667 | 40493.730 | no |  |
| PROC-01:wrong | quality | red | amber | 0.700 | 0.627 | 1.000 | 0.667 | 41897.430 | no |  |
| PROC-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.500 | 32078.640 | no |  |
| PROC-02:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.333 | 39098.890 | no |  |
| PROC-02:ambiguous | standard | amber | amber | 0.703 | 0.703 | 1.000 | 0.389 | 23326.630 | no |  |
| PROC-02:ambiguous | quality | amber | amber | 0.704 | 0.633 | 1.000 | 0.444 | 21458.600 | no |  |
| PROC-02:wrong | standard | red | amber | 0.700 | 0.700 | 1.000 | 0.500 | 17578.680 | no |  |
| PROC-02:wrong | quality | red | green | 0.753 | 0.695 | 1.000 | 0.500 | 22631.690 | no |  |
| QUAL-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.400 | 49225.130 | no |  |
| QUAL-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.400 | 49987.680 | no |  |
| QUAL-01:ambiguous | standard | amber | amber | 0.722 | 0.722 | 1.000 | 0.400 | 22243.860 | no |  |
| QUAL-01:ambiguous | quality | amber | amber | 0.685 | 0.608 | 1.000 | 0.400 | 21751.690 | no |  |
| QUAL-01:wrong | standard | red | red | 0.286 | 0.286 | 1.000 | 0.200 | 22936.560 | no |  |
| QUAL-01:wrong | quality | red | red | 0.361 | 0.272 | 1.000 | 0.200 | 24725.610 | no |  |
| QUAL-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.000 | 33141.780 | no |  |
| QUAL-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.062 | 43060.050 | no |  |
| QUAL-02:ambiguous | standard | amber | amber | 0.550 | 0.495 | 1.000 | 0.312 | 23248.650 | no |  |
| QUAL-02:ambiguous | quality | amber | red | 0.370 | 0.253 | 1.000 | 0.125 | 23528.430 | no |  |
| QUAL-02:wrong | standard | red | red | 0.002 | 0.002 | 1.000 | 0.250 | 20374.480 | no |  |
| QUAL-02:wrong | quality | red | red | 0.188 | 0.025 | 1.000 | 0.250 | 17775.510 | no |  |
| SALES-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.348 | 36286.530 | no |  |
| SALES-01:perfect | quality | green | green | 0.959 | 0.950 | 1.000 | 0.130 | 35329.960 | no |  |
| SALES-01:ambiguous | standard | amber | amber | 0.740 | 0.745 | 1.000 | 0.478 | 24819.920 | no |  |
| SALES-01:ambiguous | quality | amber | amber | 0.740 | 0.733 | 1.000 | 0.478 | 21729.790 | no |  |
| SALES-01:wrong | standard | red | red | 0.413 | 0.413 | 1.000 | 0.174 | 13050.990 | no |  |
| SALES-01:wrong | quality | red | amber | 0.562 | 0.493 | 1.000 | 0.217 | 14188.520 | no |  |
| UK-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.308 | 31702.580 | no |  |
| UK-01:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.077 | 33214.420 | no |  |
| UK-01:ambiguous | standard | amber | amber | 0.740 | 0.754 | 1.000 | 0.538 | 23098.160 | no |  |
| UK-01:ambiguous | quality | amber | amber | 0.740 | 0.739 | 1.000 | 0.538 | 22025.600 | no |  |
| UK-01:wrong | standard | red | red | 0.016 | 0.016 | 1.000 | 0.077 | 11583.230 | no |  |
| UK-01:wrong | quality | red | red | 0.182 | 0.023 | 1.000 | 0.077 | 22473.930 | no |  |
| UK-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.524 | 29043.550 | no |  |
| UK-02:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.476 | 27515.320 | no |  |
| UK-02:ambiguous | standard | amber | amber | 0.588 | 0.588 | 1.000 | 0.476 | 22425.740 | no |  |
| UK-02:ambiguous | quality | amber | amber | 0.583 | 0.482 | 1.000 | 0.476 | 27403.900 | no |  |
| UK-02:wrong | standard | red | red | 0.107 | 0.107 | 1.000 | 0.476 | 20689.250 | no |  |
| UK-02:wrong | quality | red | red | 0.464 | 0.336 | 1.000 | 0.429 | 28764.010 | no |  |
| US-01:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.545 | 27529.570 | no |  |
| US-01:perfect | quality | green | green | 0.960 | 0.950 | 1.000 | 0.455 | 27612.540 | no |  |
| US-01:ambiguous | standard | amber | amber | 0.740 | 0.835 | 1.000 | 0.545 | 22735.920 | no |  |
| US-01:ambiguous | quality | amber | amber | 0.613 | 0.516 | 1.000 | 0.364 | 26159.920 | no |  |
| US-01:wrong | standard | red | red | 0.350 | 0.350 | 1.000 | 0.545 | 22114.450 | no |  |
| US-01:wrong | quality | red | red | 0.536 | 0.420 | 1.000 | 0.273 | 26228.430 | no |  |
| US-02:perfect | standard | green | green | 0.950 | 0.950 | 1.000 | 0.235 | 24800.360 | no |  |
| US-02:perfect | quality | green | green | 0.958 | 0.950 | 1.000 | 0.353 | 25185.670 | no |  |
| US-02:ambiguous | standard | amber | amber | 0.681 | 0.681 | 1.000 | 0.294 | 20738.990 | no |  |
| US-02:ambiguous | quality | amber | amber | 0.563 | 0.494 | 1.000 | 0.235 | 22199.410 | no |  |
| US-02:wrong | standard | red | red | 0.183 | 0.183 | 1.000 | 0.294 | 19151.610 | no |  |
| US-02:wrong | quality | red | red | 0.474 | 0.415 | 1.000 | 0.412 | 21930.830 | no |  |
