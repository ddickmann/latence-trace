# Master Subscription Agreement (Template)

> **Template only.** This is a starting point for negotiations between
> latence.ai (the "Provider") and an Enterprise Customer (the
> "Customer"). It must be reviewed and tailored by both sides' counsel
> before execution. It is **not** a substitute for legal advice.

This Master Subscription Agreement (this **"Agreement"**) is entered
into as of the date last signed below (the **"Effective Date"**) by
and between **latence.ai** ("Provider") and **<Customer Legal Name>**
("Customer").

---

## 1. Definitions

1.1 **"Software"** means the latence-trace Groundedness Tracker
self-hosted runtime, the official Helm chart, the official Python
SDK, and the documentation, in each case in object-code form, that
Provider makes available under this Agreement.

1.2 **"License Key"** means the cryptographically-signed JWT issued
by Provider that authorises Customer to use the Software for the
agreed Subscription Term, Tier, and entitlement set.

1.3 **"Subscription Term"** means the term defined in the applicable
Order Form. Initial term is one (1) year unless otherwise stated.

1.4 **"Tier"** means one of *trial*, *team*, or *enterprise* as
defined in the Order Form.

1.5 **"Customer Data"** means any data Customer or Customer's users
submit to the Software for processing.

---

## 2. License Grant

2.1 **Grant.** Subject to Customer's compliance with this Agreement
and timely payment of Fees, Provider grants Customer a non-exclusive,
non-transferable, non-sublicensable license during the Subscription
Term to install and use the Software inside Customer's own
infrastructure for Customer's internal business purposes, scoped to
the entitlements encoded in the License Key (subject identifier,
tier, max QPS, max workers, optional fingerprint, feature flags).

2.2 **Restrictions.** Customer shall not (a) sublicense, resell,
host as a multi-tenant SaaS, or otherwise make the Software
available to any third party as a standalone product, (b) reverse
engineer, decompile, or attempt to derive source code except to the
extent expressly permitted by applicable law, (c) circumvent or
disable the License Key enforcement, (d) remove or alter Provider's
proprietary notices, (e) use the Software in violation of applicable
laws.

2.3 **Open Source Components.** The Software incorporates third-party
open-source software listed in the SBOM published with each release.
Such components remain governed by their respective licenses.

---

## 3. Fees and Payment

3.1 **Fees.** Customer shall pay the fees set forth in the Order
Form ("**Fees**"). Unless otherwise stated, Fees are invoiced
annually in advance and due net 30 from the invoice date.

3.2 **Taxes.** Fees are exclusive of taxes. Customer is responsible
for all applicable sales, use, VAT, GST, and similar taxes.

3.3 **Late Payment.** Past-due amounts accrue interest at the lesser
of 1.0% per month or the maximum allowed by law.

---

## 4. Customer Data; Privacy

4.1 **Ownership.** Customer retains all right, title, and interest
in Customer Data. Provider acquires no rights in Customer Data
beyond what is necessary to perform under this Agreement.

4.2 **No Vendor Telemetry by Default.** The Software does not
transmit Customer Data to Provider. Optional telemetry is documented
in the SECURITY.md and is disabled unless Customer enables it.

4.3 **Data Processing Addendum.** When Customer processes personal
data through the Software, the Data Processing Addendum
(`commercial/data-processing-addendum.md`) is incorporated by
reference. The DPA prevails over any conflicting term of this
Agreement with respect to such personal data.

---

## 5. Support

5.1 **Standard Support.** Tier *team* and *enterprise* customers
receive support during business hours (09:00 - 18:00 CET,
Mon-Fri excluding bank holidays) via `support@latence.ai`. Tier
*trial* customers receive best-effort community support only.

5.2 **Severity / Response.** Provider will respond to support
requests as follows (target initial response):

| Severity | Definition | Initial response |
|---|---|---|
| 1 | Production outage; no workaround | 4 business hours |
| 2 | Major functional impairment; workaround exists | 1 business day |
| 3 | Minor issue or question | 3 business days |

5.3 **Updates.** During an active Subscription Term, Customer may
download and use any update or new version of the Software released
by Provider, subject to the License Key entitlements.

---

## 6. Warranties; Disclaimer

6.1 **Mutual.** Each party represents that it has the legal authority
to enter into this Agreement.

6.2 **Provider Warranty.** Provider warrants that for ninety (90)
days from delivery the Software will materially conform to the
documentation. Customer's exclusive remedy for breach of this
warranty is, at Provider's option, to (a) repair or replace the
Software or (b) refund the Fees paid for the affected Subscription
Term on a pro-rata basis.

6.3 **Disclaimer.** EXCEPT AS EXPRESSLY STATED IN SECTION 6.2, THE
SOFTWARE IS PROVIDED "AS IS" AND PROVIDER DISCLAIMS ALL OTHER
WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
NON-INFRINGEMENT. PROVIDER MAKES NO WARRANTY THAT GROUNDEDNESS
SCORES PRODUCED BY THE SOFTWARE WILL BE 100% ACCURATE; CUSTOMER IS
RESPONSIBLE FOR HUMAN REVIEW OF AMBER-BAND OUTPUTS AND FOR
CALIBRATING THRESHOLDS TO ITS DOMAIN.

---

## 7. Indemnification

7.1 **Provider.** Provider will defend Customer against any third-
party claim alleging that the Software, as delivered and used in
accordance with the documentation, infringes a valid US, EU, or UK
patent, copyright, or trademark, and will indemnify Customer for
amounts finally awarded against Customer by a court or paid in a
settlement approved by Provider in writing.

7.2 **Customer.** Customer will defend Provider against any third-
party claim arising from Customer Data or Customer's use of the
Software in violation of this Agreement.

7.3 **Procedure.** The indemnified party will (a) promptly notify
the indemnifying party in writing, (b) provide reasonable
cooperation, and (c) allow the indemnifying party sole control over
defense and settlement (provided no settlement imposes obligations
on the indemnified party without consent).

---

## 8. Limitation of Liability

8.1 EXCEPT FOR LIABILITY ARISING FROM (A) BREACH OF SECTION 2.2
(LICENSE RESTRICTIONS), (B) THE INDEMNIFICATION OBLIGATIONS IN
SECTION 7, OR (C) GROSS NEGLIGENCE OR WILLFUL MISCONDUCT, EACH
PARTY'S AGGREGATE LIABILITY UNDER THIS AGREEMENT WILL NOT EXCEED
THE FEES PAID OR PAYABLE BY CUSTOMER IN THE TWELVE MONTHS PRECEDING
THE EVENT GIVING RISE TO LIABILITY.

8.2 IN NO EVENT WILL EITHER PARTY BE LIABLE FOR INDIRECT,
INCIDENTAL, CONSEQUENTIAL, SPECIAL, EXEMPLARY, OR PUNITIVE DAMAGES,
INCLUDING LOST PROFITS, LOST REVENUE, OR LOST DATA, EVEN IF
ADVISED OF THE POSSIBILITY.

---

## 9. Term and Termination

9.1 **Term.** This Agreement begins on the Effective Date and
continues for the Subscription Term, automatically renewing for
additional one (1) year terms unless either party gives written
notice of non-renewal at least 60 days before the end of the
then-current term.

9.2 **For cause.** Either party may terminate this Agreement for
material breach by the other party that remains uncured 30 days
after written notice.

9.3 **Effect of termination.** Upon termination, Customer's License
Key is revoked and the Software's enforcement layer will refuse new
requests. Customer must remove all copies of the Software from its
infrastructure within thirty (30) days. Sections 2.2, 4.1, 6.3, 7,
8, 9.3, and 10 survive termination.

---

## 10. General

10.1 **Governing Law.** This Agreement is governed by the laws of
**[jurisdiction]**, excluding its conflict-of-laws rules.

10.2 **Notices.** All notices must be in writing and sent to the
addresses on the Order Form (or by email to the contacts identified
there).

10.3 **Assignment.** Neither party may assign this Agreement without
the other's written consent, except in connection with a merger,
acquisition, or sale of all or substantially all of its assets.

10.4 **Force Majeure.** Neither party is liable for failure to
perform due to causes beyond its reasonable control.

10.5 **Entire Agreement.** This Agreement (including the Order Form,
DPA, and any incorporated exhibits) is the entire agreement between
the parties regarding the Software and supersedes all prior
agreements.

---

| Provider | Customer |
|---|---|
| latence.ai | <Customer Legal Name> |
| Signature: | Signature: |
| Name: | Name: |
| Title: | Title: |
| Date: | Date: |
