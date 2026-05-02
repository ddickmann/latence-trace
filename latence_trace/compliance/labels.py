"""Stable GDPR-oriented label catalog for compliance redaction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

ComplianceLabelMode = Literal["open", "category"]

GDPR_CATEGORIES: dict[str, list[str]] = {
    "identity_and_contact": [
        "person",
        "date_of_birth",
        "age",
        "gender",
        "nationality",
        "address",
        "phone_number",
        "email",
        "maiden_name",
        "marital_status",
    ],
    "government_and_legal": [
        "national_id",
        "passport_number",
        "drivers_license",
        "social_security_number",
        "voter_id",
        "visa_number",
        "vehicle_registration_number",
    ],
    "digital_and_location": [
        "ip_address",
        "mac_address",
        "device_id",
        "cookie_id",
        "gps_coordinates",
        "username",
        "password",
        "browser_fingerprint",
        "advertising_id",
    ],
    "financial": [
        "bank_account",
        "insurance_number",
        "swift_code",
        "account_number",
        "credit_card_number",
        "tax_identification_number",
        "transaction_id",
        "crypto_wallet_address",
        "credit_score",
        "income_bracket",
        "mortgage_number",
    ],
    "employment_and_education": [
        "organization",
        "organization_id",
        "employee_id",
        "job_title",
        "student_id",
        "university",
        "school",
        "performance_review",
    ],
    "health_and_medical_article_9": [
        "medical_condition",
        "medical_record_number",
        "health_insurance_number",
        "medical_procedure",
        "prescription_drug",
        "blood_type",
        "disability_status",
        "mental_health_status",
        "genetic_data",
    ],
    "sensitive_and_biometric_article_9": [
        "biometric_data",
        "facial_recognition_data",
        "voice_signature",
        "race_or_ethnicity",
        "religious_belief",
        "political_affiliation",
        "sexual_orientation",
        "trade_union_membership",
        "criminal_record",
    ],
}


def _dedupe_preserve_order(labels: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        label = str(raw).strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def all_gdpr_labels() -> list[str]:
    """Return the full ordered GDPR label set."""

    return _dedupe_preserve_order(
        label for labels in GDPR_CATEGORIES.values() for label in labels
    )


def known_categories() -> list[str]:
    return list(GDPR_CATEGORIES.keys())


def known_labels() -> set[str]:
    return set(all_gdpr_labels())


def resolve_label_set(
    *,
    mode: ComplianceLabelMode = "open",
    categories: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Resolve request-level labels deterministically.

    ``labels`` is an explicit allowlist override and must still belong to
    the shipped GDPR catalog. This keeps prompt order and replacement mapping
    stable while allowing tenants to focus the runtime further than category
    mode when needed.
    """

    allowed = known_labels()
    if labels:
        requested = _dedupe_preserve_order(labels)
        unknown = [label for label in requested if label not in allowed]
        if unknown:
            raise ValueError(f"Unknown compliance labels: {', '.join(unknown)}")
        return requested

    if mode == "open":
        return all_gdpr_labels()

    requested_categories = _dedupe_preserve_order(categories or [])
    if not requested_categories:
        raise ValueError("category mode requires at least one category")
    unknown_categories = [
        category for category in requested_categories if category not in GDPR_CATEGORIES
    ]
    if unknown_categories:
        raise ValueError(f"Unknown compliance categories: {', '.join(unknown_categories)}")

    return _dedupe_preserve_order(
        label
        for category in requested_categories
        for label in GDPR_CATEGORIES[category]
    )
