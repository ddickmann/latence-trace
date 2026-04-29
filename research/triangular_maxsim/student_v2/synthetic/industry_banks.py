"""Industry-specific entity banks + fact tables for synthetic data.

Eight industries are defined: six for **training** (legal, finance,
HR, compliance, engineering, marketing) and two held out entirely for
**OOD evaluation** (healthcare, public sector). Each industry declares:

* ``sub_domains``    - 2 sub-domains per industry (so disjoint entity
                        banks keep entity-swap adversarials plausible).
* ``entities``       - per-sub-domain banks of companies, people,
                        products, jurisdictions, systems, etc. Banks
                        within an industry across sub-domains are
                        intentionally disjoint so entity-swap never
                        re-uses the source entity.
* ``numerics``       - banks of dates, percentages, counts, amounts.
* ``slot_defaults``  - typical values for string slots common across
                        industries (quarter names, severity levels).

The banks are deliberately compact (~20-40 entries per slot) but wide
enough that the combinatorial passage generator can produce > 500
unique passages per sub-domain via slot filling driven by deterministic
seeds.

Nothing here depends on torch or network access. The module is pure
data so it can be imported from CI and unit-tested cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Common slot defaults shared across industries.
# ---------------------------------------------------------------------------


QUARTERS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
FISCAL_YEARS: tuple[str, ...] = (
    "FY2022", "FY2023", "FY2024", "FY2025", "FY2026",
)
SEVERITY: tuple[str, ...] = ("low", "medium", "high", "critical")
STATUSES: tuple[str, ...] = (
    "signed off", "cascaded", "under review", "scheduled", "pending remediation",
)


@dataclass(frozen=True)
class IndustryBank:
    """All the knobs a generator needs to produce one industry's lane."""

    name: str  # e.g. "finance"
    training: bool  # True for the 6 training industries, False for OOD holdouts
    sub_domains: Sequence[str]
    entities: Mapping[str, Mapping[str, Sequence[str]]]
    numerics: Mapping[str, Mapping[str, Sequence[str]]]
    slot_defaults: Mapping[str, Sequence[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finance (training)
# ---------------------------------------------------------------------------


_FINANCE = IndustryBank(
    name="finance",
    training=True,
    sub_domains=("earnings", "risk_reports"),
    entities={
        "earnings": {
            "company": (
                "Aetherium Semiconductor", "Northwind Industries",
                "Velocity Fintech", "Meridian Cloud Networks", "Orion Pharmacompanies",
                "Pinnacle Capital Group", "Azurelake Logistics", "Crestwave Media",
                "Brightspring Biosciences", "Summit Retail Holdings",
                "Catalyst Autonomics", "Vanguard Analytics Corp",
                "Horizonview Telecom", "Ironforge Industrial Supply",
                "Silverarc Robotics", "Keystone Energy Systems",
            ),
            "segment": (
                "services", "consumer devices", "enterprise cloud",
                "data infrastructure", "automotive", "licensing",
                "advertising", "healthcare", "payments", "logistics",
            ),
            "product_line": (
                "Atlas Cloud", "Helix Platform", "Orbit AI Suite",
                "Summit Analytics", "NovaLink Secure", "GridForge",
                "LumenOps", "Pulse Data Warehouse",
            ),
            "executive_title": (
                "CEO", "CFO", "COO", "Chief Revenue Officer",
            ),
            "region": (
                "North America", "EMEA", "APAC", "Latin America",
                "emerging markets", "Greater China",
            ),
            "analyst_firm": (
                "Granite Research", "Merrington Partners", "Rivermere Securities",
                "Peabody Jones Capital", "Oakmere Analytics",
            ),
        },
        "risk_reports": {
            "risk_category": (
                "credit risk", "operational risk", "market risk",
                "liquidity risk", "counterparty risk", "cyber risk",
                "climate transition risk", "regulatory risk",
            ),
            "committee": (
                "Board Risk Committee", "Audit Committee", "ALCO",
                "Executive Risk Council", "Model Risk Oversight Board",
            ),
            "control_framework": (
                "COSO ERM", "ISO 31000", "Basel IV standards",
                "internal ERM framework", "OCC heightened standards",
            ),
            "remediation_owner": (
                "Treasury", "Risk Operations", "Second Line",
                "Internal Audit", "Model Risk Management",
            ),
            "issue_label": (
                "RISK-2024-0147", "RISK-2024-0213", "RISK-2025-0008",
                "AUDIT-Q3-0092", "MODEL-VALIDATION-7714",
            ),
        },
    },
    numerics={
        "earnings": {
            "revenue_billions": tuple(f"{x:.1f}" for x in (12.4, 18.7, 28.2, 41.3, 48.2, 62.1, 85.8, 101.3, 124.5)),
            "growth_pct": tuple(str(x) for x in (2, 4, 5, 7, 9, 11, 14, 18, 22, 27, 33)),
            "margin_pct": tuple(f"{x:.1f}" for x in (18.4, 22.1, 25.6, 29.4, 31.8, 34.2)),
            "eps_usd": tuple(f"{x:.2f}" for x in (0.82, 1.17, 1.40, 1.88, 2.10, 2.54, 3.17)),
            "installed_base_billions": tuple(f"{x:.1f}" for x in (1.2, 1.6, 2.0, 2.2, 2.8, 3.4)),
        },
        "risk_reports": {
            "var_usd_millions": tuple(str(x) for x in (42, 87, 114, 203, 318, 497, 612, 744)),
            "breach_count": tuple(str(x) for x in (0, 1, 2, 3, 5, 8)),
            "remediation_days": tuple(str(x) for x in (15, 30, 45, 60, 90, 120, 180)),
            "capital_ratio_pct": tuple(f"{x:.2f}" for x in (10.25, 11.50, 12.75, 13.40, 14.80, 15.60)),
        },
    },
    slot_defaults={
        "quarter": QUARTERS,
        "fy": FISCAL_YEARS,
    },
)


# ---------------------------------------------------------------------------
# Legal (training)
# ---------------------------------------------------------------------------


_LEGAL = IndustryBank(
    name="legal",
    training=True,
    sub_domains=("contracts", "case_law"),
    entities={
        "contracts": {
            "contract_type": (
                "Master Services Agreement", "Mutual NDA",
                "Software Licensing Agreement", "Statement of Work",
                "Data Processing Addendum", "Reseller Agreement",
                "Professional Services Agreement",
            ),
            "counterparty": (
                "Ashford Holdings LLC", "Braeburn Capital Inc.",
                "Citrine Digital Corp.", "Daventry Group Ltd.",
                "Evergreen Technology AG", "Fairchild Partners",
                "Gatewood Media Co.", "Hollybrook Industries",
            ),
            "clause_reference": (
                "Section 4.2", "Section 7.1(b)", "Section 9.3",
                "Article III", "Schedule A", "Exhibit 2",
                "Clause 12.4", "Annex B",
            ),
            "governing_law": (
                "Delaware", "New York", "England and Wales",
                "California", "Singapore", "Ontario",
            ),
            "attorney": (
                "Pearson & Wade LLP", "Crawford Abernathy Brooks",
                "Winton Eastly Partners", "Huxley Langford Group",
            ),
        },
        "case_law": {
            "jurisdiction": (
                "U.S. Supreme Court", "Ninth Circuit", "Second Circuit",
                "Delaware Chancery Court", "EU Court of Justice",
                "High Court of England and Wales", "Federal Court of Australia",
            ),
            "doctrine": (
                "arbitrability", "forum non conveniens", "preemption",
                "personal jurisdiction", "fair use", "piercing the corporate veil",
                "de facto merger", "anti-SLAPP",
            ),
            "case_cite": (
                "Carter v. Wellcraft (2019)", "Raines & Co. v. Dempsey (2021)",
                "In re: Mirabelle Industries (2020)", "Odenheimer v. State (2023)",
                "Perryville Holdings v. Standish (2022)",
            ),
            "statute": (
                "Securities Exchange Act § 10(b)", "DMCA § 512(c)",
                "CCPA § 1798.140", "Sherman Act § 2",
                "Delaware General Corporation Law § 141",
            ),
        },
    },
    numerics={
        "contracts": {
            "term_months": tuple(str(x) for x in (12, 24, 36, 48, 60)),
            "notice_days": tuple(str(x) for x in (15, 30, 60, 90, 120)),
            "liability_cap_millions": tuple(str(x) for x in (1, 2, 5, 10, 25, 50)),
            "termination_cure_days": tuple(str(x) for x in (5, 10, 15, 30)),
        },
        "case_law": {
            "citation_year": tuple(str(x) for x in (2015, 2017, 2019, 2020, 2021, 2022, 2023, 2024)),
            "damages_millions": tuple(str(x) for x in (0, 3, 12, 28, 87, 150, 410)),
            "vote_split": ("9-0", "6-3", "5-4", "7-2", "unanimous"),
        },
    },
)


# ---------------------------------------------------------------------------
# HR / People Ops (training)
# ---------------------------------------------------------------------------


_HR = IndustryBank(
    name="hr",
    training=True,
    sub_domains=("handbook", "benefits"),
    entities={
        "handbook": {
            "policy_name": (
                "Remote Work Policy", "Bring Your Own Device Policy",
                "Time Off and Leave Policy", "Code of Conduct",
                "Anti-Harassment Policy", "Travel and Expense Policy",
                "Data Protection and Privacy Policy",
            ),
            "jurisdiction": (
                "United States", "United Kingdom", "Germany",
                "France", "Ireland", "Japan", "Australia", "Canada",
            ),
            "function": (
                "People Operations", "Legal", "Finance",
                "Compliance", "Workplace Services",
            ),
            "audience": (
                "all employees", "managers", "new hires",
                "EMEA staff", "APAC staff", "remote workers",
            ),
        },
        "benefits": {
            "benefit_type": (
                "medical coverage", "dental coverage", "vision coverage",
                "401(k) match", "equity refresh grant", "parental leave",
                "sabbatical program", "wellness stipend",
            ),
            "carrier": (
                "Aetna", "Cigna", "UnitedHealthcare",
                "Blue Cross Blue Shield", "Kaiser Permanente",
                "Humana", "Guardian",
            ),
            "enrollment_window": (
                "annual open enrollment", "new-hire 30-day window",
                "qualifying life event", "mid-year adjustment window",
            ),
        },
    },
    numerics={
        "handbook": {
            "policy_revision": ("v1.0", "v1.2", "v2.0", "v2.1", "v3.0"),
            "review_interval_months": tuple(str(x) for x in (6, 12, 18, 24)),
        },
        "benefits": {
            "match_pct": tuple(str(x) for x in (3, 4, 5, 6, 8, 10)),
            "deductible_usd": tuple(str(x) for x in (250, 500, 1000, 1500, 2500)),
            "leave_weeks": tuple(str(x) for x in (4, 6, 12, 16, 20, 26)),
            "stipend_usd": tuple(str(x) for x in (100, 250, 500, 1000, 1500)),
        },
    },
)


# ---------------------------------------------------------------------------
# Compliance (training)
# ---------------------------------------------------------------------------


_COMPLIANCE = IndustryBank(
    name="compliance",
    training=True,
    sub_domains=("soc2_iso", "gdpr"),
    entities={
        "soc2_iso": {
            "control_id": (
                "CC6.1", "CC6.2", "CC6.3", "CC7.1", "CC8.1",
                "A.5.1", "A.8.1", "A.8.25", "A.12.4", "A.18.1.4",
            ),
            "framework": (
                "SOC 2 Type II", "ISO/IEC 27001:2022", "NIST CSF",
                "PCI DSS v4.0", "CSA STAR Level 2",
            ),
            "auditor": (
                "Coalfire", "Schellman", "A-LIGN", "BARR Advisory",
                "Prescient Assurance",
            ),
            "evidence_type": (
                "configuration attestation", "access review log",
                "change management ticket", "penetration-test report",
                "vulnerability scan output",
            ),
        },
        "gdpr": {
            "legal_basis": (
                "Article 6(1)(a) consent", "Article 6(1)(b) contract",
                "Article 6(1)(c) legal obligation",
                "Article 6(1)(f) legitimate interest",
            ),
            "data_category": (
                "contact data", "employment records", "financial data",
                "health data (Article 9)", "biometric data (Article 9)",
                "behavioural telemetry", "location data",
            ),
            "dpa": (
                "Irish DPC", "French CNIL", "German LDI NRW",
                "Dutch AP", "Italian Garante", "UK ICO",
            ),
            "processor": (
                "Northern Cloud EU", "Helix Analytics GmbH",
                "Quadrant Payments S.A.", "Brightline Logging Ltd.",
            ),
        },
    },
    numerics={
        "soc2_iso": {
            "finding_count": tuple(str(x) for x in (0, 1, 2, 4, 7, 11)),
            "remediation_sla_days": tuple(str(x) for x in (7, 14, 30, 60, 90)),
        },
        "gdpr": {
            "retention_months": tuple(str(x) for x in (3, 6, 12, 24, 36, 60, 84)),
            "dpia_score": tuple(f"{x:.1f}" for x in (2.0, 3.4, 4.6, 6.8, 8.2)),
            "subject_request_days": tuple(str(x) for x in (15, 30, 45, 60, 90)),
        },
    },
)


# ---------------------------------------------------------------------------
# Engineering (training)
# ---------------------------------------------------------------------------


_ENGINEERING = IndustryBank(
    name="engineering",
    training=True,
    sub_domains=("adr", "runbooks"),
    entities={
        "adr": {
            "system": (
                "ingestion-api", "graph-indexer", "auth-gateway",
                "event-bus", "metric-aggregator", "feature-store",
                "recommendation-service", "media-transcoder",
            ),
            "technology": (
                "Kafka", "NATS JetStream", "PostgreSQL 16",
                "ClickHouse", "Redis Cluster", "etcd", "Temporal",
                "Apache Flink", "DynamoDB",
            ),
            "team": (
                "Core Platform", "Data Infra", "Growth Platform",
                "Developer Experience", "Site Reliability",
            ),
            "rfc_label": (
                "ADR-0012", "ADR-0023", "ADR-0047", "ADR-0061",
                "RFC-107", "RFC-134",
            ),
        },
        "runbooks": {
            "service": (
                "payments-processor", "notifications-service",
                "search-api", "billing-aggregator", "identity-provider",
                "cdn-edge", "analytics-pipeline",
            ),
            "oncall_rotation": (
                "rotation-A", "rotation-B", "rotation-weekend",
                "escalation-primary", "escalation-secondary",
            ),
            "paging_alert": (
                "HighErrorRate", "P99LatencyBreach", "ReplicaLagBacklog",
                "CircuitBreakerOpen", "DiskSpaceCritical",
                "ExpiringCertificate",
            ),
        },
    },
    numerics={
        "adr": {
            "slo_p99_ms": tuple(str(x) for x in (20, 50, 100, 250, 500)),
            "throughput_rps": tuple(str(x) for x in (500, 1000, 5000, 10000, 50000)),
            "replication_lag_ms": tuple(str(x) for x in (5, 20, 50, 100, 250)),
        },
        "runbooks": {
            "timeout_seconds": tuple(str(x) for x in (5, 10, 30, 60, 120)),
            "retry_count": tuple(str(x) for x in (0, 1, 3, 5, 7)),
            "error_budget_pct": tuple(f"{x:.1f}" for x in (0.1, 0.5, 1.0, 2.5, 5.0)),
        },
    },
)


# ---------------------------------------------------------------------------
# Marketing (training)
# ---------------------------------------------------------------------------


_MARKETING = IndustryBank(
    name="marketing",
    training=True,
    sub_domains=("campaigns", "positioning"),
    entities={
        "campaigns": {
            "campaign_name": (
                "Operation Northstar", "Project Sunrise",
                "Campaign Pathfinder", "Initiative Horizon",
                "Program Evergreen",
            ),
            "channel": (
                "paid search", "programmatic display", "influencer",
                "content syndication", "event sponsorship",
                "outbound email", "organic social",
            ),
            "audience": (
                "enterprise security buyers", "DevTools early adopters",
                "mid-market CFOs", "SMB HR leaders",
                "EMEA compliance directors",
            ),
            "market_segment": (
                "Fortune 500", "upper mid-market", "SMB",
                "public sector", "regulated industries",
            ),
        },
        "positioning": {
            "brand": (
                "Northwind", "Azurelake", "Brightspring",
                "Crestwave", "Meridian", "Velocity",
            ),
            "competitor": (
                "IncumbentCorp", "LegacySystems Co.",
                "BoutiqueRivals Inc.", "NewEntrants Ltd.",
            ),
            "value_prop": (
                "real-time verification", "zero-touch onboarding",
                "auditable automation", "cost-optimised storage",
                "compliance-by-default",
            ),
        },
    },
    numerics={
        "campaigns": {
            "budget_millions": tuple(f"{x:.1f}" for x in (0.2, 0.5, 1.2, 2.8, 4.5, 8.0)),
            "target_mqls": tuple(str(x) for x in (150, 400, 800, 1500, 3200)),
            "ctr_pct": tuple(f"{x:.2f}" for x in (0.15, 0.42, 0.78, 1.25, 2.10, 3.40)),
        },
        "positioning": {
            "market_share_pct": tuple(f"{x:.1f}" for x in (3.2, 7.8, 12.4, 18.0, 24.5, 31.2)),
            "nps": tuple(str(x) for x in (12, 27, 41, 54, 68, 74)),
        },
    },
)


# ---------------------------------------------------------------------------
# Healthcare (HELD OUT FOR OOD EVAL - never seen in training)
# ---------------------------------------------------------------------------


_HEALTHCARE = IndustryBank(
    name="healthcare",
    training=False,
    sub_domains=("clinical_trials", "care_plans"),
    entities={
        "clinical_trials": {
            "study_phase": ("Phase I", "Phase II", "Phase IIa", "Phase IIb", "Phase III"),
            "indication": (
                "type 2 diabetes", "non-small-cell lung cancer",
                "major depressive disorder", "ulcerative colitis",
                "hypertrophic cardiomyopathy", "age-related macular degeneration",
            ),
            "sponsor": (
                "Brightspring Therapeutics", "Helixbay Biosciences",
                "Lindenmere Pharmaceuticals", "Aldencrest Oncology",
                "Thornhill Neurosciences",
            ),
            "endpoint": (
                "progression-free survival", "HbA1c reduction",
                "Mayo score improvement", "objective response rate",
                "change in MADRS from baseline",
            ),
        },
        "care_plans": {
            "diagnosis_code": ("E11.9", "I10", "J45.40", "F32.9", "M54.5"),
            "care_setting": (
                "primary care", "specialist clinic", "telehealth",
                "inpatient ward", "home health",
            ),
            "provider_type": (
                "attending physician", "nurse practitioner",
                "physician assistant", "care coordinator",
            ),
            "regimen": (
                "metformin 500 mg BID", "lisinopril 10 mg QD",
                "fluticasone-salmeterol 250/50",
                "sertraline 50 mg QD",
            ),
        },
    },
    numerics={
        "clinical_trials": {
            "enrollment_count": tuple(str(x) for x in (42, 128, 296, 487, 812, 1140)),
            "hr_ratio": tuple(f"{x:.2f}" for x in (0.42, 0.58, 0.67, 0.81, 0.93, 1.07)),
            "followup_months": tuple(str(x) for x in (3, 6, 12, 18, 24, 36)),
            "completion_pct": tuple(str(x) for x in (68, 74, 82, 89, 94)),
        },
        "care_plans": {
            "follow_up_weeks": tuple(str(x) for x in (2, 4, 8, 12, 24)),
            "adherence_pct": tuple(str(x) for x in (62, 71, 78, 84, 91)),
            "dose_mg": tuple(str(x) for x in (5, 10, 25, 50, 100, 250, 500)),
        },
    },
)


# ---------------------------------------------------------------------------
# Public sector (HELD OUT FOR OOD EVAL - never seen in training)
# ---------------------------------------------------------------------------


_PUBLIC_SECTOR = IndustryBank(
    name="public_sector",
    training=False,
    sub_domains=("grants", "policy_briefs"),
    entities={
        "grants": {
            "agency": (
                "Department of Commerce", "National Institutes of Health",
                "Department of Energy", "Small Business Administration",
                "Department of Transportation", "UK Innovate",
                "European Research Council",
            ),
            "program": (
                "SBIR Phase II", "STTR Phase I", "ARPA-E OPEN 2024",
                "NIH R01", "Horizon Europe Cluster 4",
                "ERC Consolidator Grant",
            ),
            "principal_investigator": (
                "Dr. Alvarez-Hahn", "Dr. Okonkwo", "Dr. Prasad-Mukherjee",
                "Dr. Schwendtner", "Dr. Fujimori",
            ),
        },
        "policy_briefs": {
            "agency_brief": (
                "OECD Innovation Policy Review", "GAO Report to Congress",
                "CBO Budget Outlook", "NAO Value-for-Money Study",
                "IMF Article IV Consultation",
            ),
            "policy_lever": (
                "tax credit expansion", "procurement set-aside",
                "performance-based rulemaking", "compliance safe harbor",
                "fee waiver program",
            ),
        },
    },
    numerics={
        "grants": {
            "award_usd_millions": tuple(f"{x:.2f}" for x in (0.25, 0.75, 1.20, 2.50, 4.80, 7.50)),
            "period_of_performance_months": tuple(str(x) for x in (12, 18, 24, 36, 48)),
            "indirect_rate_pct": tuple(str(x) for x in (20, 25, 30, 40, 50, 55, 60)),
        },
        "policy_briefs": {
            "gdp_impact_pct": tuple(f"{x:.2f}" for x in (0.05, 0.12, 0.28, 0.42, 0.81, 1.25)),
            "beneficiary_thousands": tuple(str(x) for x in (8, 24, 72, 150, 340, 820)),
        },
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


INDUSTRIES: tuple[IndustryBank, ...] = (
    _FINANCE,
    _LEGAL,
    _HR,
    _COMPLIANCE,
    _ENGINEERING,
    _MARKETING,
    _HEALTHCARE,
    _PUBLIC_SECTOR,
)

_INDUSTRY_INDEX: dict[str, IndustryBank] = {b.name: b for b in INDUSTRIES}


def get_industry(name: str) -> IndustryBank:
    if name not in _INDUSTRY_INDEX:
        raise KeyError(
            f"unknown industry {name!r}; valid: {sorted(_INDUSTRY_INDEX)}"
        )
    return _INDUSTRY_INDEX[name]


def training_industries() -> tuple[IndustryBank, ...]:
    return tuple(b for b in INDUSTRIES if b.training)


def ood_industries() -> tuple[IndustryBank, ...]:
    return tuple(b for b in INDUSTRIES if not b.training)


__all__ = [
    "INDUSTRIES",
    "IndustryBank",
    "get_industry",
    "ood_industries",
    "training_industries",
]
