# TRACE security questionnaires

Pre-filled responses to the two most common enterprise security and
privacy questionnaires.  Every answer maps to an artifact under
`commercial/trust-center/` or `docs/operations/`.  Do not change an
answer without updating the underlying artifact.

## Files

* `CAIQ-Lite.md` - Cloud Security Alliance **Consensus Assessments
  Initiative Questionnaire (Lite)**.  Shortest questionnaire most
  enterprises will accept for a <10k ARR pilot.
* `SIG-Lite.md` - Shared Assessments **Standardized Information
  Gathering (Lite)**.  Expected for anything touching regulated
  verticals (finance, legal, HR, compliance).
* `evidence-index.md` - maps each CAIQ / SIG answer to the specific
  evidence document, commit hash, or signed letter that backs it.

## Operating model

| Stage | Owner | Next step |
|---|---|---|
| Prospect requests questionnaire | Sales | Send `CAIQ-Lite.md` verbatim + this README link |
| Prospect returns with edits | Security | Accept edits that do not expand scope; route expansions to Legal |
| Prospect requests SOC 2 / ISO 27001 report | Head of Security | Issue Trust Center read-only link (not in this repo); NDA flow in `commercial/trust-center/` |
| Customer signs pilot | Legal | Append the questionnaire to `commercial/pilot-contract-template.md` so DPA + SLA survive |

Answers are deliberately short.  Each one links to a full explanation
in the Trust Center or the ops runbook.
