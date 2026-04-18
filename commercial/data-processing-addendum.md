# Data Processing Addendum (Template)

> **Template only.** Customer and Provider counsel must review and
> tailor this DPA before execution. It is incorporated by reference
> into the Master Subscription Agreement (`master-subscription-agreement.md`).

This Data Processing Addendum (the **"DPA"**) supplements the Master
Subscription Agreement (the **"MSA"**) between **latence.ai**
("**Provider**" / "**Processor**") and **<Customer Legal Name>**
("**Customer**" / "**Controller**") and applies whenever Customer
processes Personal Data through the Software.

---

## 1. Definitions

1.1 Terms not defined here have the meaning given in the MSA.
"**Personal Data**", "**Processing**", "**Controller**",
"**Processor**", "**Sub-processor**", "**Data Subject**", and
"**Supervisory Authority**" have the meanings given in
Regulation (EU) 2016/679 ("**GDPR**") and its UK equivalent.

1.2 "**Customer Personal Data**" means Personal Data contained in
Customer Data submitted to the Software for Processing.

---

## 2. Roles

Customer is the Controller of Customer Personal Data. Provider acts
as a Processor strictly on Customer's documented instructions. The
MSA, this DPA, and the documentation constitute Customer's complete
and final instructions to Provider; additional or different
instructions require written agreement.

---

## 3. Subject Matter, Nature, Purpose, Duration, Categories

| Item | Description |
|---|---|
| Subject matter | Self-hosted execution of the Software inside Customer's infrastructure to compute groundedness scores. |
| Nature | Stateless inference; no persistence of inputs or outputs. |
| Purpose | Detect ungrounded LLM outputs in Customer's RAG / agent workloads. |
| Duration | The Subscription Term. |
| Data subject categories | As determined by Customer's RAG inputs (typically end users of Customer's product). |
| Personal data categories | As determined by Customer's RAG inputs. Provider does not require any specific category. |
| Special categories (Art. 9 GDPR) | Only if Customer's inputs include them; Customer warrants it has a lawful basis. |

Because the Software runs **inside the Customer's perimeter** and
does not exfiltrate inputs to Provider, Provider is in practice a
processor only with respect to (a) license-level metadata (license
subject identifier) and (b) any optional support data Customer
shares (logs, repros).

---

## 4. Provider Obligations

4.1 **Confidentiality.** Provider personnel with access to Customer
Personal Data are bound by confidentiality obligations.

4.2 **Security.** Provider implements the technical and
organisational measures described in Annex II.

4.3 **Sub-processors.** Provider does not appoint any Sub-processor
to Process Customer Personal Data, because Customer Personal Data
never leaves the Customer's infrastructure under the standard
deployment model. Should Provider receive Customer Personal Data
in the course of support, the Sub-processors listed in Annex III
may be involved; Customer is notified at least 30 days before any
addition and may object on reasonable grounds.

4.4 **Assistance.** Provider will provide Customer with reasonable
assistance to (a) respond to Data Subject requests, (b) carry out
data protection impact assessments, and (c) consult with
Supervisory Authorities, taking into account the nature of
Processing and the information available to Provider.

4.5 **Personal Data Breach.** Provider will notify Customer without
undue delay (and in any event within 72 hours) after becoming aware
of a Personal Data Breach affecting Customer Personal Data
processed by Provider, and will provide such information as
Customer reasonably requires to comply with Articles 33-34 GDPR.

4.6 **Audit.** Once per twelve-month period (or more often if
required by a Supervisory Authority or following a Personal Data
Breach), Customer may audit Provider's compliance via (a)
Provider's most recent SOC 2 Type II report or equivalent or (b) a
written questionnaire. On-site audits require 30 days written
notice and reasonable scoping; costs are borne by Customer unless
the audit reveals material non-compliance.

4.7 **Return / Deletion.** On termination of the MSA, Provider will
delete any Customer Personal Data in its possession (typically only
support correspondence) within 30 days, except as required by law.

---

## 5. International Transfers

The Software runs inside Customer's chosen cloud region; Provider
does not transfer Customer Personal Data internationally as part of
the standard deployment. If a transfer occurs in the course of
support (Section 4.3), the parties incorporate the EU Standard
Contractual Clauses (Module 2: Controller-to-Processor, Decision
2021/914) and the UK International Data Transfer Addendum, as
applicable, by reference. Provider will implement supplementary
measures as required by the Schrems II case-law.

---

## 6. Liability

Liability under this DPA is governed by the limitation of liability
provisions of the MSA. Nothing in this DPA limits Data Subject
rights under applicable data protection law.

---

## Annex I -- Description of Processing

| Field | Value |
|---|---|
| Controller | <Customer Legal Name> |
| Processor | latence.ai |
| Categories of Data Subjects | As defined by Customer's RAG / agent inputs. |
| Categories of Personal Data | As defined by Customer's RAG / agent inputs. |
| Sensitive data | Only if Customer's inputs include them. |
| Frequency of transfer | Continuous, on a per-request basis -- but **inside Customer's perimeter**, not to Provider. |
| Nature of Processing | Stateless inference (similarity scoring, NLI, score fusion). |
| Purpose | Groundedness scoring of LLM outputs. |
| Storage period | None; inputs are not persisted by the engine. |
| Sub-processors | See Annex III. |

---

## Annex II -- Technical and Organisational Measures

| Domain | Control |
|---|---|
| Access control | Role-based access at the Provider; least-privilege; SSO + MFA. |
| Data minimisation | Engine is stateless; logs redact request bodies by default. |
| Encryption | Software supports TLS for inbound (terminated at Customer gateway) and outbound (vLLM). License JWT signed Ed25519. |
| Pseudonymisation | License `subject` is the only identifier persisted in Provider's billing records. |
| Resilience | HPA + PDB; rate limiter; ASGI inflight semaphore. |
| Vulnerability management | Pinned base image, multi-stage build, Trivy + Grype scans, SBOM, signed releases. |
| Incident response | Documented in `SECURITY.md`; 72-hour breach notification. |
| Personnel | Background-checked; trained on data protection and confidentiality. |
| Sub-processor management | None for in-perimeter Processing; Annex III otherwise. |

---

## Annex III -- Sub-processors (support-only)

| Sub-processor | Purpose | Location | Safeguard |
|---|---|---|---|
| <CRM provider> | Support ticket storage | <region> | DPA + SCCs |
| <Email provider> | Support email | <region> | DPA + SCCs |
| <Cloud backup> | Encrypted backup of support correspondence | <region> | DPA + SCCs |

(Customer is notified at least 30 days before any change.)

---

| Provider | Customer |
|---|---|
| latence.ai | <Customer Legal Name> |
| Signature: | Signature: |
| Name: | Name: |
| Title: | Title: |
| Date: | Date: |
