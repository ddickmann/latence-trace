"""Passage + response templates for the synthetic enterprise lane.

Each ``Template`` is a pair of skeletons:

* ``passage`` - what goes into ``raw_context``; the evidence the model
                is expected to cite.
* ``response`` - a faithful response that *correctly* summarises or
                 answers from the passage. Adversarial perturbations
                 (entity swap / numeric flip / support drop) operate on
                 this pair in :mod:`adversarials`.

Each skeleton uses ``{slot}`` placeholders that are filled from the
industry banks in :mod:`industry_banks`. Three "shapes" are defined
per sub-domain so the generator emits:

* ``short``  - 1-2 sentence passage, 1-sentence response
* ``medium`` - 3-5 sentence passage, 2-3 sentence response
* ``long``   - 1-2 paragraph passage, 3-4 sentence response

Per (industry, sub_domain, shape) there are between 2 and 5 templates
so that combining templates x slot banks x paraphrases yields >500
unique passages per sub-domain. Slot names are scoped per industry's
entity/numeric banks - they must resolve to an industry slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SHAPES: tuple[str, ...] = ("short", "medium", "long")


@dataclass(frozen=True)
class Template:
    passage: str
    response: str
    # Slots must be resolvable from the industry banks for the given
    # sub_domain (entities + numerics + slot_defaults).
    required_slots: tuple[str, ...]


# Each entry is keyed by (industry_name, sub_domain, shape).
# Values are lists (there can be multiple templates per bucket).
TEMPLATES: Mapping[tuple[str, str, str], tuple[Template, ...]] = {

    # ---------- Finance / earnings ----------
    ("finance", "earnings", "short"): (
        Template(
            passage=(
                "{company} reported {quarter} {fy} revenue of ${revenue_billions} billion, "
                "up {growth_pct}% year over year."
            ),
            response=(
                "{company} grew {quarter} {fy} revenue {growth_pct}% to ${revenue_billions} billion."
            ),
            required_slots=("company", "quarter", "fy", "revenue_billions", "growth_pct"),
        ),
        Template(
            passage=(
                "Operating margin at {company} expanded to {margin_pct}% "
                "in {quarter} {fy}, while diluted EPS reached ${eps_usd}."
            ),
            response=(
                "{company}'s {quarter} {fy} operating margin was {margin_pct}% with "
                "diluted EPS of ${eps_usd}."
            ),
            required_slots=("company", "quarter", "fy", "margin_pct", "eps_usd"),
        ),
    ),
    ("finance", "earnings", "medium"): (
        Template(
            passage=(
                "{company} announced {quarter} {fy} revenue of ${revenue_billions} billion, "
                "up {growth_pct}% year over year, with {segment} revenue an area of strength. "
                "Operating margin expanded to {margin_pct}% and diluted EPS rose to ${eps_usd}, "
                "according to the {executive_title}."
            ),
            response=(
                "In {quarter} {fy}, {company} delivered ${revenue_billions} billion in revenue "
                "(+{growth_pct}% YoY). {segment} was a strength driver; operating margin was "
                "{margin_pct}% and diluted EPS was ${eps_usd}."
            ),
            required_slots=(
                "company", "quarter", "fy", "revenue_billions", "growth_pct",
                "segment", "margin_pct", "eps_usd", "executive_title",
            ),
        ),
        Template(
            passage=(
                "{analyst_firm} noted that {company}'s {product_line} franchise "
                "drove {region} growth of {growth_pct}% in {quarter} {fy}, "
                "lifting the installed base past {installed_base_billions} billion units. "
                "Management reaffirmed the {fy} full-year outlook."
            ),
            response=(
                "{analyst_firm} highlighted {company}'s {product_line} growth of {growth_pct}% "
                "in {region} during {quarter} {fy}, with the installed base crossing "
                "{installed_base_billions} billion units. The {fy} outlook was reaffirmed."
            ),
            required_slots=(
                "analyst_firm", "company", "product_line", "region",
                "growth_pct", "quarter", "fy", "installed_base_billions",
            ),
        ),
    ),
    ("finance", "earnings", "long"): (
        Template(
            passage=(
                "{company} reported {quarter} {fy} results in line with the {executive_title}'s "
                "prior guidance. Revenue totalled ${revenue_billions} billion, up {growth_pct}% "
                "year over year, driven by strength in {segment} and incremental contribution "
                "from the {product_line} franchise in {region}. Operating margin expanded to "
                "{margin_pct}% and diluted EPS rose to ${eps_usd}.\n\n"
                "{analyst_firm} observed that the installed base has now crossed "
                "{installed_base_billions} billion units and that services attach continues to "
                "compound. Management reiterated the {fy} full-year outlook and declined to "
                "update long-range targets at this time."
            ),
            response=(
                "{company}'s {quarter} {fy} revenue was ${revenue_billions} billion (+{growth_pct}% "
                "YoY), with {segment} strength and {product_line} traction in {region}. Operating "
                "margin hit {margin_pct}% and diluted EPS was ${eps_usd}. {analyst_firm} flagged the "
                "{installed_base_billions}-billion installed base; the {fy} outlook was reaffirmed."
            ),
            required_slots=(
                "company", "quarter", "fy", "executive_title", "revenue_billions",
                "growth_pct", "segment", "product_line", "region", "margin_pct",
                "eps_usd", "analyst_firm", "installed_base_billions",
            ),
        ),
    ),

    # ---------- Finance / risk_reports ----------
    ("finance", "risk_reports", "short"): (
        Template(
            passage=(
                "Issue {issue_label} was assigned to {remediation_owner} by the {committee} "
                "with a {remediation_days}-day remediation target under {control_framework}."
            ),
            response=(
                "{committee} assigned {issue_label} to {remediation_owner} (target: "
                "{remediation_days} days) under {control_framework}."
            ),
            required_slots=(
                "issue_label", "remediation_owner", "committee",
                "remediation_days", "control_framework",
            ),
        ),
    ),
    ("finance", "risk_reports", "medium"): (
        Template(
            passage=(
                "The {committee} reviewed {risk_category} exposure and noted a {breach_count}-breach "
                "increase in the quarter, raising daily VaR to ${var_usd_millions} million. "
                "{remediation_owner} was tasked with closing issue {issue_label} within "
                "{remediation_days} days under {control_framework}."
            ),
            response=(
                "The {committee} flagged {risk_category} with {breach_count} breaches this quarter "
                "and daily VaR of ${var_usd_millions} million; {remediation_owner} owns "
                "{issue_label} (SLA {remediation_days} days, {control_framework})."
            ),
            required_slots=(
                "committee", "risk_category", "breach_count", "var_usd_millions",
                "remediation_owner", "issue_label", "remediation_days",
                "control_framework",
            ),
        ),
    ),
    ("finance", "risk_reports", "long"): (
        Template(
            passage=(
                "The {committee} convened to review {risk_category} exposure for {quarter} {fy}. "
                "Daily VaR rose to ${var_usd_millions} million against a capital ratio of "
                "{capital_ratio_pct}%, and {breach_count} threshold breaches were logged during "
                "the period. Under {control_framework}, issue {issue_label} was formally assigned "
                "to {remediation_owner} with a {remediation_days}-day remediation target.\n\n"
                "The committee directed a follow-up review in the next cycle and requested that "
                "{remediation_owner} provide weekly status updates until the issue is closed."
            ),
            response=(
                "The {committee} reviewed {risk_category} for {quarter} {fy}: daily VaR of "
                "${var_usd_millions} million, capital ratio {capital_ratio_pct}%, and {breach_count} "
                "threshold breaches. Issue {issue_label} was assigned to {remediation_owner} under "
                "{control_framework} with a {remediation_days}-day SLA and weekly status updates."
            ),
            required_slots=(
                "committee", "risk_category", "quarter", "fy", "var_usd_millions",
                "capital_ratio_pct", "breach_count", "control_framework",
                "issue_label", "remediation_owner", "remediation_days",
            ),
        ),
    ),

    # ---------- Legal / contracts ----------
    ("legal", "contracts", "short"): (
        Template(
            passage=(
                "{clause_reference} of the {contract_type} with {counterparty} sets a "
                "{term_months}-month term governed by {governing_law} law."
            ),
            response=(
                "The {contract_type} with {counterparty} has a {term_months}-month term under "
                "{governing_law} law ({clause_reference})."
            ),
            required_slots=(
                "clause_reference", "contract_type", "counterparty",
                "term_months", "governing_law",
            ),
        ),
    ),
    ("legal", "contracts", "medium"): (
        Template(
            passage=(
                "Under {clause_reference} of the {contract_type} with {counterparty}, either party "
                "may terminate for material breach on {notice_days} days' written notice, subject "
                "to a {termination_cure_days}-day cure period. Liability is capped at "
                "${liability_cap_millions} million under {governing_law} law, negotiated by {attorney}."
            ),
            response=(
                "{clause_reference} of the {contract_type} with {counterparty} allows termination "
                "for material breach on {notice_days} days' notice with a {termination_cure_days}-day "
                "cure. Liability cap is ${liability_cap_millions} million under {governing_law} law "
                "(counsel: {attorney})."
            ),
            required_slots=(
                "clause_reference", "contract_type", "counterparty", "notice_days",
                "termination_cure_days", "liability_cap_millions", "governing_law", "attorney",
            ),
        ),
    ),
    ("legal", "contracts", "long"): (
        Template(
            passage=(
                "The {contract_type} between the parties and {counterparty}, effective for a "
                "{term_months}-month initial term under {governing_law} law and negotiated by "
                "{attorney}, contains the following key provisions. {clause_reference} sets a "
                "material-breach termination right exercisable on {notice_days} days' written "
                "notice, with a {termination_cure_days}-day cure period.\n\n"
                "Aggregate liability is capped at ${liability_cap_millions} million per contract "
                "year, excluding fraud, gross negligence, and indemnification obligations. The "
                "agreement auto-renews for successive twelve-month terms unless either party "
                "provides notice of non-renewal prior to the then-current expiration date."
            ),
            response=(
                "The {contract_type} with {counterparty} runs {term_months} months under "
                "{governing_law} law (negotiated by {attorney}). {clause_reference} allows "
                "material-breach termination on {notice_days} days' notice with a "
                "{termination_cure_days}-day cure. Liability is capped at "
                "${liability_cap_millions} million per year (with the standard carve-outs), and "
                "the agreement auto-renews in twelve-month increments absent timely non-renewal notice."
            ),
            required_slots=(
                "contract_type", "counterparty", "term_months", "governing_law", "attorney",
                "clause_reference", "notice_days", "termination_cure_days",
                "liability_cap_millions",
            ),
        ),
    ),

    # ---------- Legal / case_law ----------
    ("legal", "case_law", "short"): (
        Template(
            passage=(
                "In {case_cite}, the {jurisdiction} decided a question of {doctrine} under "
                "{statute} by a vote of {vote_split}."
            ),
            response=(
                "{case_cite} ({jurisdiction}, {vote_split}) addressed {doctrine} under {statute}."
            ),
            required_slots=(
                "case_cite", "jurisdiction", "doctrine", "statute", "vote_split",
            ),
        ),
    ),
    ("legal", "case_law", "medium"): (
        Template(
            passage=(
                "{case_cite} in the {jurisdiction} held, {vote_split}, that {doctrine} under "
                "{statute} does not bar the claim. Damages of ${damages_millions} million "
                "awarded in {citation_year} were reversed in part."
            ),
            response=(
                "{case_cite} ({jurisdiction}, {vote_split}) ruled that {doctrine} under {statute} "
                "does not bar the claim, reversing part of the {citation_year} "
                "${damages_millions} million damages award."
            ),
            required_slots=(
                "case_cite", "jurisdiction", "vote_split", "doctrine",
                "statute", "damages_millions", "citation_year",
            ),
        ),
    ),
    ("legal", "case_law", "long"): (
        Template(
            passage=(
                "In {case_cite}, decided in {citation_year}, the {jurisdiction} addressed a "
                "question of {doctrine} arising under {statute}. The court, by a vote of "
                "{vote_split}, reversed the lower court's ${damages_millions} million damages "
                "award and remanded for further proceedings on the narrower ground the parties "
                "had briefed.\n\n"
                "The majority emphasised that {doctrine} is a threshold inquiry and must be "
                "resolved on a complete evidentiary record rather than on a facial pleading. The "
                "dissent would have affirmed in full, reasoning that {statute} forecloses the "
                "relief sought on the stipulated facts."
            ),
            response=(
                "{case_cite} ({jurisdiction}, {citation_year}, {vote_split}) reversed the lower "
                "court's ${damages_millions} million damages award and remanded on a question of "
                "{doctrine} under {statute}. The majority required a complete evidentiary record "
                "on the threshold question; the dissent would have affirmed."
            ),
            required_slots=(
                "case_cite", "citation_year", "jurisdiction", "doctrine",
                "statute", "vote_split", "damages_millions",
            ),
        ),
    ),

    # ---------- HR / handbook ----------
    ("hr", "handbook", "short"): (
        Template(
            passage=(
                "{policy_name} ({policy_revision}) applies to {audience} in {jurisdiction} "
                "and is owned by {function}."
            ),
            response=(
                "{policy_name} {policy_revision} covers {audience} in {jurisdiction}, owned by {function}."
            ),
            required_slots=(
                "policy_name", "policy_revision", "audience", "jurisdiction", "function",
            ),
        ),
    ),
    ("hr", "handbook", "medium"): (
        Template(
            passage=(
                "The {policy_name} ({policy_revision}) is reviewed every {review_interval_months} "
                "months by {function} and applies to {audience} working in {jurisdiction}. "
                "Exceptions must be approved in writing by the {function} lead."
            ),
            response=(
                "{policy_name} ({policy_revision}) covers {audience} in {jurisdiction}, is owned "
                "and reviewed every {review_interval_months} months by {function}, and requires "
                "written approval for exceptions."
            ),
            required_slots=(
                "policy_name", "policy_revision", "review_interval_months",
                "function", "audience", "jurisdiction",
            ),
        ),
    ),
    ("hr", "handbook", "long"): (
        Template(
            passage=(
                "The {policy_name} ({policy_revision}) is a {function}-owned policy applicable to "
                "{audience} working in {jurisdiction}. Per the policy's scope statement, it is "
                "reviewed every {review_interval_months} months and amended to reflect changes in "
                "local law, internal practice, and audit findings.\n\n"
                "Exceptions may be granted only on a case-by-case basis by the {function} lead "
                "and must be documented in writing. Employees who are unsure whether a particular "
                "arrangement is permitted should consult {function} before proceeding."
            ),
            response=(
                "The {function}-owned {policy_name} ({policy_revision}) applies to {audience} in "
                "{jurisdiction}, is reviewed every {review_interval_months} months, and allows "
                "exceptions only with documented written approval from the {function} lead. "
                "Employees should consult {function} when in doubt."
            ),
            required_slots=(
                "policy_name", "policy_revision", "function", "audience",
                "jurisdiction", "review_interval_months",
            ),
        ),
    ),

    # ---------- HR / benefits ----------
    ("hr", "benefits", "short"): (
        Template(
            passage=(
                "The 401(k) match is {match_pct}% and enrollment happens during "
                "{enrollment_window}."
            ),
            response=(
                "401(k) match is {match_pct}%; enroll during {enrollment_window}."
            ),
            required_slots=("match_pct", "enrollment_window"),
        ),
        Template(
            passage=(
                "{benefit_type} is provided through {carrier} with a ${deductible_usd} annual deductible."
            ),
            response=(
                "{carrier} provides {benefit_type} with a ${deductible_usd} deductible."
            ),
            required_slots=("benefit_type", "carrier", "deductible_usd"),
        ),
    ),
    ("hr", "benefits", "medium"): (
        Template(
            passage=(
                "{benefit_type} is offered through {carrier} with a ${deductible_usd} annual "
                "deductible, and the 401(k) employer match is {match_pct}%. Parental leave is "
                "{leave_weeks} weeks fully paid."
            ),
            response=(
                "{carrier} provides {benefit_type} (deductible ${deductible_usd}); 401(k) match "
                "is {match_pct}% and parental leave is {leave_weeks} fully paid weeks."
            ),
            required_slots=(
                "benefit_type", "carrier", "deductible_usd",
                "match_pct", "leave_weeks",
            ),
        ),
    ),
    ("hr", "benefits", "long"): (
        Template(
            passage=(
                "Core benefits include {benefit_type} through {carrier} with a ${deductible_usd} "
                "annual deductible, a {match_pct}% employer 401(k) match (vesting immediately), "
                "and {leave_weeks} weeks of fully paid parental leave. A wellness stipend of "
                "${stipend_usd} per year is available to all benefits-eligible employees.\n\n"
                "Enrollment windows are restricted to {enrollment_window} and qualifying "
                "life events. Contact People Operations for a full summary plan description."
            ),
            response=(
                "Benefits: {benefit_type} ({carrier}, ${deductible_usd} deductible), {match_pct}% "
                "immediate-vest 401(k) match, {leave_weeks} weeks fully paid parental leave, and a "
                "${stipend_usd} annual wellness stipend. Enrollment is limited to "
                "{enrollment_window} or qualifying life events."
            ),
            required_slots=(
                "benefit_type", "carrier", "deductible_usd", "match_pct",
                "leave_weeks", "stipend_usd", "enrollment_window",
            ),
        ),
    ),

    # ---------- Compliance / soc2_iso ----------
    ("compliance", "soc2_iso", "short"): (
        Template(
            passage=(
                "Control {control_id} under {framework} requires {evidence_type}; "
                "SLA is {remediation_sla_days} days."
            ),
            response=(
                "{framework} control {control_id} expects {evidence_type} with a "
                "{remediation_sla_days}-day SLA."
            ),
            required_slots=(
                "control_id", "framework", "evidence_type", "remediation_sla_days",
            ),
        ),
    ),
    ("compliance", "soc2_iso", "medium"): (
        Template(
            passage=(
                "{auditor} tested control {control_id} under {framework} and returned "
                "{finding_count} findings. Remediation is tracked with a {remediation_sla_days}-day "
                "SLA; evidence expected is {evidence_type}."
            ),
            response=(
                "{auditor}'s test of {control_id} ({framework}) produced {finding_count} findings; "
                "remediation SLA is {remediation_sla_days} days with {evidence_type} as evidence."
            ),
            required_slots=(
                "auditor", "control_id", "framework", "finding_count",
                "remediation_sla_days", "evidence_type",
            ),
        ),
    ),
    ("compliance", "soc2_iso", "long"): (
        Template(
            passage=(
                "As part of the current {framework} audit, {auditor} tested control "
                "{control_id} across the sample period and produced {finding_count} findings. "
                "The evidence type requested to support remediation is {evidence_type}, and "
                "tracked issues carry a {remediation_sla_days}-day SLA managed by the Compliance "
                "PMO.\n\n"
                "Audit closure is contingent on clean re-test after remediation; partial "
                "remediation does not discharge the finding."
            ),
            response=(
                "{auditor}'s current-period test of {control_id} under {framework} produced "
                "{finding_count} findings. Remediation requires {evidence_type} within "
                "{remediation_sla_days} days and a clean re-test; partial remediation does not "
                "discharge the finding."
            ),
            required_slots=(
                "framework", "auditor", "control_id", "finding_count",
                "evidence_type", "remediation_sla_days",
            ),
        ),
    ),

    # ---------- Compliance / gdpr ----------
    ("compliance", "gdpr", "short"): (
        Template(
            passage=(
                "{data_category} is processed under {legal_basis} and retained for "
                "{retention_months} months."
            ),
            response=(
                "{data_category} is processed on {legal_basis} with a {retention_months}-month "
                "retention."
            ),
            required_slots=(
                "data_category", "legal_basis", "retention_months",
            ),
        ),
    ),
    ("compliance", "gdpr", "medium"): (
        Template(
            passage=(
                "The {processor} handles {data_category} under {legal_basis} with a "
                "{retention_months}-month retention. DPIA score is {dpia_score}/10 and subject "
                "requests are answered within {subject_request_days} days."
            ),
            response=(
                "{processor} processes {data_category} under {legal_basis}, retaining it for "
                "{retention_months} months (DPIA {dpia_score}/10); subject requests are answered "
                "within {subject_request_days} days."
            ),
            required_slots=(
                "processor", "data_category", "legal_basis",
                "retention_months", "dpia_score", "subject_request_days",
            ),
        ),
    ),
    ("compliance", "gdpr", "long"): (
        Template(
            passage=(
                "The {processor} processes {data_category} on behalf of the controller under "
                "{legal_basis}. Retention is {retention_months} months, scheduled deletion follows "
                "the data lifecycle policy, and the associated DPIA was re-scored at "
                "{dpia_score}/10 with the {dpa} notified of the assessment.\n\n"
                "Data-subject requests are triaged by the DPO team and answered within "
                "{subject_request_days} days as required by Article 12(3). Cross-border transfers, "
                "where applicable, rely on the standard contractual clauses with supplementary "
                "measures documented per the supervisory authority's guidance."
            ),
            response=(
                "{processor} processes {data_category} under {legal_basis} with a "
                "{retention_months}-month retention (DPIA {dpia_score}/10, notified to {dpa}). "
                "Data-subject requests are answered within {subject_request_days} days, and "
                "cross-border transfers rely on SCCs with supplementary measures."
            ),
            required_slots=(
                "processor", "data_category", "legal_basis", "retention_months",
                "dpia_score", "dpa", "subject_request_days",
            ),
        ),
    ),

    # ---------- Engineering / adr ----------
    ("engineering", "adr", "short"): (
        Template(
            passage=(
                "{rfc_label}: {system} adopts {technology} for a target p99 of {slo_p99_ms} ms."
            ),
            response=(
                "{rfc_label} picks {technology} for {system} (p99 target {slo_p99_ms} ms)."
            ),
            required_slots=("rfc_label", "system", "technology", "slo_p99_ms"),
        ),
    ),
    ("engineering", "adr", "medium"): (
        Template(
            passage=(
                "{rfc_label} by {team}: {system} adopts {technology}. Target throughput is "
                "{throughput_rps} RPS and p99 latency is {slo_p99_ms} ms; replication lag budget "
                "is {replication_lag_ms} ms."
            ),
            response=(
                "{team}'s {rfc_label} adopts {technology} for {system}: target {throughput_rps} RPS, "
                "p99 {slo_p99_ms} ms, replication-lag budget {replication_lag_ms} ms."
            ),
            required_slots=(
                "rfc_label", "team", "system", "technology",
                "throughput_rps", "slo_p99_ms", "replication_lag_ms",
            ),
        ),
    ),
    ("engineering", "adr", "long"): (
        Template(
            passage=(
                "{rfc_label}, authored by {team}, proposes that {system} migrate from its current "
                "store to {technology}. The target SLOs are a sustained {throughput_rps} RPS at "
                "p99 {slo_p99_ms} ms, with replication lag held under {replication_lag_ms} ms "
                "during steady-state operation.\n\n"
                "Rollout is proposed as a dual-write shadow followed by a read-cutover window, "
                "guarded by an error-budget policy; reverting is scoped to a single DNS flip "
                "should the shadow reveal correctness regressions."
            ),
            response=(
                "{rfc_label} ({team}) migrates {system} to {technology} for {throughput_rps} RPS at "
                "p99 {slo_p99_ms} ms and <{replication_lag_ms} ms replication lag. Rollout is a "
                "dual-write shadow then read-cutover, with a single-DNS-flip rollback."
            ),
            required_slots=(
                "rfc_label", "team", "system", "technology",
                "throughput_rps", "slo_p99_ms", "replication_lag_ms",
            ),
        ),
    ),

    # ---------- Engineering / runbooks ----------
    ("engineering", "runbooks", "short"): (
        Template(
            passage=(
                "{paging_alert} on {service}: timeout {timeout_seconds}s, {retry_count} retries "
                "(oncall: {oncall_rotation})."
            ),
            response=(
                "Alert {paging_alert} for {service} uses {timeout_seconds}s timeout and "
                "{retry_count} retries, paged to {oncall_rotation}."
            ),
            required_slots=(
                "paging_alert", "service", "timeout_seconds",
                "retry_count", "oncall_rotation",
            ),
        ),
    ),
    ("engineering", "runbooks", "medium"): (
        Template(
            passage=(
                "When {paging_alert} fires on {service}, {oncall_rotation} is paged. The "
                "prescribed timeout is {timeout_seconds}s with {retry_count} retries; error "
                "budget for the service is {error_budget_pct}% per 28-day window."
            ),
            response=(
                "{paging_alert} on {service} pages {oncall_rotation} with {timeout_seconds}s timeout, "
                "{retry_count} retries, against a {error_budget_pct}% 28-day error budget."
            ),
            required_slots=(
                "paging_alert", "service", "oncall_rotation",
                "timeout_seconds", "retry_count", "error_budget_pct",
            ),
        ),
    ),
    ("engineering", "runbooks", "long"): (
        Template(
            passage=(
                "{paging_alert} on {service} pages {oncall_rotation} directly. The first-line "
                "responder is expected to acknowledge within five minutes, capture an initial "
                "hypothesis, and update the incident channel. Timeouts are set to "
                "{timeout_seconds}s with {retry_count} retries; the service runs against a "
                "{error_budget_pct}% error budget per rolling 28-day window.\n\n"
                "If the runbook does not resolve the incident within thirty minutes, escalate to "
                "the backup rotation and page the subsystem owner. Post-incident reviews must be "
                "scheduled within three business days."
            ),
            response=(
                "{paging_alert} on {service} pages {oncall_rotation} (ack within 5 min); timeout "
                "{timeout_seconds}s, {retry_count} retries, {error_budget_pct}% 28-day error budget. "
                "Escalate to the backup rotation + subsystem owner after 30 minutes; PIR within 3 "
                "business days."
            ),
            required_slots=(
                "paging_alert", "service", "oncall_rotation", "timeout_seconds",
                "retry_count", "error_budget_pct",
            ),
        ),
    ),

    # ---------- Marketing / campaigns ----------
    ("marketing", "campaigns", "short"): (
        Template(
            passage=(
                "{campaign_name} targets {audience} via {channel} with a budget of "
                "${budget_millions} million."
            ),
            response=(
                "{campaign_name} spends ${budget_millions}M via {channel} on {audience}."
            ),
            required_slots=(
                "campaign_name", "audience", "channel", "budget_millions",
            ),
        ),
    ),
    ("marketing", "campaigns", "medium"): (
        Template(
            passage=(
                "{campaign_name} targets the {market_segment} segment ({audience}) across "
                "{channel}, with a ${budget_millions} million budget, {target_mqls} MQL target, "
                "and a CTR goal of {ctr_pct}%."
            ),
            response=(
                "{campaign_name} covers {market_segment} {audience} via {channel}: "
                "${budget_millions}M budget, {target_mqls} MQLs, {ctr_pct}% CTR goal."
            ),
            required_slots=(
                "campaign_name", "market_segment", "audience", "channel",
                "budget_millions", "target_mqls", "ctr_pct",
            ),
        ),
    ),
    ("marketing", "campaigns", "long"): (
        Template(
            passage=(
                "{campaign_name} is positioned against the {market_segment} segment, with "
                "{audience} as the primary ICP. Media spend is allocated across {channel} with a "
                "total budget of ${budget_millions} million; the target is {target_mqls} MQLs at "
                "a {ctr_pct}% CTR, measured in the marketing attribution platform.\n\n"
                "Creative is refreshed every six weeks and performance is reviewed weekly in the "
                "growth ops meeting. Budget is rebalanced toward highest-performing channel at "
                "mid-flight once statistical significance is reached."
            ),
            response=(
                "{campaign_name} hits {market_segment} {audience} through {channel} with a "
                "${budget_millions}M budget: {target_mqls} MQLs at {ctr_pct}% CTR. Creative "
                "refreshes every six weeks and budget rebalances mid-flight toward the best "
                "channel once significant."
            ),
            required_slots=(
                "campaign_name", "market_segment", "audience", "channel",
                "budget_millions", "target_mqls", "ctr_pct",
            ),
        ),
    ),

    # ---------- Marketing / positioning ----------
    ("marketing", "positioning", "short"): (
        Template(
            passage=(
                "{brand}'s share in the segment is {market_share_pct}%, with NPS of {nps}."
            ),
            response=(
                "{brand} has {market_share_pct}% share and {nps} NPS."
            ),
            required_slots=("brand", "market_share_pct", "nps"),
        ),
    ),
    ("marketing", "positioning", "medium"): (
        Template(
            passage=(
                "{brand}'s positioning centres on {value_prop}; against {competitor}, the brand "
                "holds {market_share_pct}% share and an NPS of {nps}."
            ),
            response=(
                "{brand} positions on {value_prop}. Vs. {competitor}, it holds {market_share_pct}% "
                "share and NPS {nps}."
            ),
            required_slots=(
                "brand", "value_prop", "competitor", "market_share_pct", "nps",
            ),
        ),
    ),
    ("marketing", "positioning", "long"): (
        Template(
            passage=(
                "{brand}'s positioning centres on {value_prop}, a deliberate choice to "
                "differentiate from {competitor}'s legacy-incumbency narrative. Segment research "
                "shows {brand} holding {market_share_pct}% share and a customer-voice NPS of "
                "{nps}, which materially outperforms the segment median.\n\n"
                "The next two quarters will double down on this narrative with a refreshed "
                "analyst briefing cycle, a customer-reference push, and a competitor-displacement "
                "play book that flows into direct-sales enablement."
            ),
            response=(
                "{brand} positions on {value_prop} to separate from {competitor}; the brand holds "
                "{market_share_pct}% share and NPS {nps}. The next two quarters feature analyst "
                "refresh, customer references, and a competitor-displacement playbook into direct "
                "sales."
            ),
            required_slots=(
                "brand", "value_prop", "competitor", "market_share_pct", "nps",
            ),
        ),
    ),

    # ---------- Healthcare / clinical_trials (OOD only) ----------
    ("healthcare", "clinical_trials", "short"): (
        Template(
            passage=(
                "{sponsor}'s {study_phase} study in {indication} enrolled {enrollment_count} "
                "subjects with {endpoint} as primary endpoint."
            ),
            response=(
                "{sponsor}'s {study_phase} {indication} trial enrolled {enrollment_count} "
                "subjects; primary endpoint is {endpoint}."
            ),
            required_slots=(
                "sponsor", "study_phase", "indication", "enrollment_count", "endpoint",
            ),
        ),
    ),
    ("healthcare", "clinical_trials", "medium"): (
        Template(
            passage=(
                "{sponsor} reported interim results from the {study_phase} trial in {indication}: "
                "{enrollment_count} subjects with {followup_months} months of follow-up; hazard "
                "ratio for the primary endpoint ({endpoint}) was {hr_ratio}."
            ),
            response=(
                "{sponsor}'s {study_phase} {indication} interim: {enrollment_count} subjects, "
                "{followup_months} months follow-up, HR {hr_ratio} on {endpoint}."
            ),
            required_slots=(
                "sponsor", "study_phase", "indication", "enrollment_count",
                "followup_months", "hr_ratio", "endpoint",
            ),
        ),
    ),
    ("healthcare", "clinical_trials", "long"): (
        Template(
            passage=(
                "{sponsor} announced interim results from the {study_phase} trial in "
                "{indication}. {enrollment_count} subjects were randomised; at a median follow-up "
                "of {followup_months} months, the primary endpoint ({endpoint}) showed a hazard "
                "ratio of {hr_ratio} in favour of the investigational arm. Completion rate across "
                "the arms was {completion_pct}%.\n\n"
                "The sponsor stated that the safety profile was consistent with prior "
                "{study_phase} experience and that the full dataset is expected to read out in the "
                "second half of next year ahead of a potential regulatory submission."
            ),
            response=(
                "{sponsor}'s {study_phase} {indication} interim: {enrollment_count} subjects, "
                "median {followup_months}-month follow-up, HR {hr_ratio} on {endpoint}, "
                "{completion_pct}% completion. Safety matches prior {study_phase}; full dataset "
                "expected H2 next year ahead of a potential filing."
            ),
            required_slots=(
                "sponsor", "study_phase", "indication", "enrollment_count",
                "followup_months", "endpoint", "hr_ratio", "completion_pct",
            ),
        ),
    ),

    # ---------- Healthcare / care_plans (OOD only) ----------
    ("healthcare", "care_plans", "short"): (
        Template(
            passage=(
                "{diagnosis_code} patients in {care_setting} are prescribed {regimen}."
            ),
            response=(
                "{regimen} is the standard prescription for {diagnosis_code} in {care_setting}."
            ),
            required_slots=(
                "diagnosis_code", "care_setting", "regimen",
            ),
        ),
    ),
    ("healthcare", "care_plans", "medium"): (
        Template(
            passage=(
                "The {provider_type} prescribes {regimen} for {diagnosis_code} and schedules "
                "follow-up every {follow_up_weeks} weeks; target adherence is {adherence_pct}%."
            ),
            response=(
                "{provider_type} prescribes {regimen} for {diagnosis_code}, follow-up every "
                "{follow_up_weeks} weeks, adherence target {adherence_pct}%."
            ),
            required_slots=(
                "provider_type", "regimen", "diagnosis_code",
                "follow_up_weeks", "adherence_pct",
            ),
        ),
    ),
    ("healthcare", "care_plans", "long"): (
        Template(
            passage=(
                "In {care_setting}, the {provider_type} prescribes {regimen} for patients with "
                "{diagnosis_code}. Dosing starts at {dose_mg} mg and is titrated at subsequent "
                "visits. Follow-up is scheduled every {follow_up_weeks} weeks, with a target "
                "adherence of {adherence_pct}% over the follow-up window.\n\n"
                "Patients are counselled on side effects and asked to contact the clinic if "
                "symptoms change substantively between visits. All changes to the regimen are "
                "documented in the shared care plan."
            ),
            response=(
                "{provider_type} in {care_setting} prescribes {regimen} for {diagnosis_code}, "
                "starting at {dose_mg} mg with titration; follow-up is every {follow_up_weeks} "
                "weeks, target adherence {adherence_pct}%. Changes go to the shared care plan."
            ),
            required_slots=(
                "care_setting", "provider_type", "regimen", "diagnosis_code",
                "dose_mg", "follow_up_weeks", "adherence_pct",
            ),
        ),
    ),

    # ---------- Public sector / grants (OOD only) ----------
    ("public_sector", "grants", "short"): (
        Template(
            passage=(
                "{agency} awarded {program} to {principal_investigator}: "
                "${award_usd_millions} million."
            ),
            response=(
                "{principal_investigator} received an {agency} {program} award of "
                "${award_usd_millions}M."
            ),
            required_slots=(
                "agency", "program", "principal_investigator", "award_usd_millions",
            ),
        ),
    ),
    ("public_sector", "grants", "medium"): (
        Template(
            passage=(
                "The {agency} granted {program} to {principal_investigator} for "
                "${award_usd_millions} million over {period_of_performance_months} months at an "
                "indirect rate of {indirect_rate_pct}%."
            ),
            response=(
                "{agency}'s {program} to {principal_investigator}: ${award_usd_millions}M over "
                "{period_of_performance_months} months, {indirect_rate_pct}% indirect."
            ),
            required_slots=(
                "agency", "program", "principal_investigator",
                "award_usd_millions", "period_of_performance_months",
                "indirect_rate_pct",
            ),
        ),
    ),
    ("public_sector", "grants", "long"): (
        Template(
            passage=(
                "The {agency} announced a {program} award of ${award_usd_millions} million to "
                "{principal_investigator}, with a {period_of_performance_months}-month period of "
                "performance and an indirect cost rate of {indirect_rate_pct}%. The proposed "
                "scope focuses on research milestones tied to the programme's core objectives.\n\n"
                "The award is subject to annual reporting and a mid-term programmatic review; "
                "deviations from the approved budget require written agency approval before funds "
                "can be reallocated."
            ),
            response=(
                "{agency}'s {program} to {principal_investigator}: ${award_usd_millions}M over "
                "{period_of_performance_months} months at {indirect_rate_pct}% indirect, with "
                "annual reporting, a mid-term programmatic review, and agency approval required "
                "for budget deviations."
            ),
            required_slots=(
                "agency", "program", "principal_investigator", "award_usd_millions",
                "period_of_performance_months", "indirect_rate_pct",
            ),
        ),
    ),

    # ---------- Public sector / policy_briefs (OOD only) ----------
    ("public_sector", "policy_briefs", "short"): (
        Template(
            passage=(
                "The {agency_brief} estimates that {policy_lever} would affect "
                "{beneficiary_thousands} thousand beneficiaries."
            ),
            response=(
                "{agency_brief}: {policy_lever} touches {beneficiary_thousands}K beneficiaries."
            ),
            required_slots=(
                "agency_brief", "policy_lever", "beneficiary_thousands",
            ),
        ),
    ),
    ("public_sector", "policy_briefs", "medium"): (
        Template(
            passage=(
                "The {agency_brief} models {policy_lever} as producing a {gdp_impact_pct}% GDP "
                "impact and reaching {beneficiary_thousands} thousand beneficiaries over the "
                "forecast horizon."
            ),
            response=(
                "Per {agency_brief}, {policy_lever} yields {gdp_impact_pct}% GDP impact and "
                "reaches {beneficiary_thousands}K beneficiaries."
            ),
            required_slots=(
                "agency_brief", "policy_lever", "gdp_impact_pct", "beneficiary_thousands",
            ),
        ),
    ),
    ("public_sector", "policy_briefs", "long"): (
        Template(
            passage=(
                "The {agency_brief} examines {policy_lever} and models it as delivering a "
                "{gdp_impact_pct}% GDP impact over the forecast horizon, reaching approximately "
                "{beneficiary_thousands} thousand beneficiaries. The brief argues that the lever's "
                "effectiveness depends on rapid administrative implementation and robust outcome "
                "measurement.\n\n"
                "The brief recommends that implementing agencies coordinate with existing "
                "programmes to avoid duplication, and that the lever be subject to a three-year "
                "sunset review tied to measurable outcomes."
            ),
            response=(
                "The {agency_brief} projects {policy_lever} at {gdp_impact_pct}% GDP impact and "
                "{beneficiary_thousands}K beneficiaries, conditional on fast administration and "
                "strong outcome measurement. A three-year sunset review tied to outcomes is "
                "recommended."
            ),
            required_slots=(
                "agency_brief", "policy_lever", "gdp_impact_pct", "beneficiary_thousands",
            ),
        ),
    ),
}


def list_shapes() -> tuple[str, ...]:
    return SHAPES


def templates_for(industry: str, sub_domain: str, shape: str) -> tuple[Template, ...]:
    key = (industry, sub_domain, shape)
    tpls = TEMPLATES.get(key, ())
    return tpls


__all__ = [
    "SHAPES",
    "TEMPLATES",
    "Template",
    "list_shapes",
    "templates_for",
]
