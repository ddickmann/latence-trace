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
        "street_address",
        "postal_code",
        "city",
        "country",
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

# Model-facing aliases discovered with native
# ``knowledgator/gliner-pii-large-v1.0`` at threshold 0.3 over a synthetic
# exact-span benchmark. The public API, analytics, masks, and replacement
# dataset stay on canonical GDPR labels; only the GLiNER prompt label set uses
# these aliases.
GLINER_MODEL_LABEL_ALIASES: dict[str, str] = {
    "person": "name",
    "date_of_birth": "person date of birth",
    "age": "age",
    "gender": "gender",
    "nationality": "nationality",
    "address": "address",
    "street_address": "street",
    "postal_code": "zip code",
    "city": "city name",
    "country": "country",
    "phone_number": "phone_number",
    "email": "email",
    "maiden_name": "maiden_name",
    "marital_status": "marital_status",
    "national_id": "national_id",
    "passport_number": "passport_number",
    "drivers_license": "driver's license number",
    "social_security_number": "ssn",
    "voter_id": "electoral id",
    "visa_number": "visa number",
    "vehicle_registration_number": "vehicle_registration_number",
    "ip_address": "internet protocol address",
    "mac_address": "hardware address",
    "device_id": "imei",
    "cookie_id": "cookie id",
    "gps_coordinates": "latitude and longitude",
    "username": "username",
    "password": "passcode",
    "browser_fingerprint": "browser_fingerprint",
    "advertising_id": "mobile advertising id",
    "bank_account": "bank account number",
    "insurance_number": "insurance_number",
    "swift_code": "bic code",
    "account_number": "account_number",
    "credit_card_number": "credit_card_number",
    "tax_identification_number": "tax_identification_number",
    "transaction_id": "transaction id",
    "crypto_wallet_address": "crypto_wallet_address",
    "credit_score": "credit_score",
    "income_bracket": "salary range",
    "mortgage_number": "mortgage_number",
    "organization": "company name",
    "organization_id": "organization id",
    "employee_id": "employee_id",
    "job_title": "job_title",
    "student_id": "matriculation number",
    "university": "college",
    "school": "school name",
    "performance_review": "performance rating",
    "medical_condition": "medical_condition",
    "medical_record_number": "medical_record_number",
    "health_insurance_number": "insurance member id",
    "medical_procedure": "medical procedure",
    "prescription_drug": "prescription_drug",
    "blood_type": "blood_type",
    "disability_status": "accessibility status",
    "mental_health_status": "psychiatric condition",
    "genetic_data": "genetic test result",
    "biometric_data": "biometric data",
    "facial_recognition_data": "faceprint",
    "voice_signature": "voiceprint",
    "race_or_ethnicity": "racial origin",
    "religious_belief": "religion",
    "political_affiliation": "political party",
    "sexual_orientation": "sexual preference",
    "trade_union_membership": "labor union membership",
    "criminal_record": "criminal_record",
}

GLINER_ALIAS_BENCHMARK_F1: dict[str, float] = {
    "person": 1.0,
    "date_of_birth": 1.0,
    "age": 1.0,
    "gender": 1.0,
    "nationality": 1.0,
    "address": 1.0,
    "street_address": 0.6667,
    "postal_code": 1.0,
    "city": 1.0,
    "country": 1.0,
    "phone_number": 1.0,
    "email": 1.0,
    "maiden_name": 1.0,
    "marital_status": 1.0,
    "national_id": 1.0,
    "passport_number": 1.0,
    "drivers_license": 1.0,
    "social_security_number": 1.0,
    "voter_id": 1.0,
    "visa_number": 1.0,
    "vehicle_registration_number": 1.0,
    "ip_address": 1.0,
    "mac_address": 0.6667,
    "device_id": 1.0,
    "cookie_id": 1.0,
    "gps_coordinates": 0.5,
    "username": 1.0,
    "password": 1.0,
    "browser_fingerprint": 1.0,
    "advertising_id": 1.0,
    "bank_account": 1.0,
    "insurance_number": 1.0,
    "swift_code": 1.0,
    "account_number": 1.0,
    "credit_card_number": 1.0,
    "tax_identification_number": 1.0,
    "transaction_id": 1.0,
    "crypto_wallet_address": 1.0,
    "credit_score": 1.0,
    "income_bracket": 1.0,
    "mortgage_number": 1.0,
    "organization": 1.0,
    "organization_id": 1.0,
    "employee_id": 1.0,
    "job_title": 1.0,
    "student_id": 1.0,
    "university": 1.0,
    "school": 1.0,
    "performance_review": 1.0,
    "medical_condition": 1.0,
    "medical_record_number": 1.0,
    "health_insurance_number": 1.0,
    "medical_procedure": 1.0,
    "prescription_drug": 1.0,
    "blood_type": 1.0,
    "disability_status": 0.6667,
    "mental_health_status": 1.0,
    "genetic_data": 1.0,
    "biometric_data": 1.0,
    "facial_recognition_data": 0.5,
    "voice_signature": 0.5,
    "race_or_ethnicity": 0.4,
    "religious_belief": 1.0,
    "political_affiliation": 1.0,
    "sexual_orientation": 1.0,
    "trade_union_membership": 1.0,
    "criminal_record": 0.5,
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

    return _dedupe_preserve_order(label for labels in GDPR_CATEGORIES.values() for label in labels)


def known_categories() -> list[str]:
    return list(GDPR_CATEGORIES.keys())


def known_labels() -> set[str]:
    return set(all_gdpr_labels())


def model_label_alias(canonical_label: str) -> str:
    """Return the optimized GLiNER-facing label for a canonical GDPR label."""

    return GLINER_MODEL_LABEL_ALIASES.get(canonical_label, canonical_label)


def to_model_label_set(canonical_labels: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Translate canonical labels to GLiNER aliases and return alias->canonical."""

    aliases: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    for canonical in canonical_labels:
        alias = model_label_alias(canonical)
        if alias not in alias_to_canonical:
            aliases.append(alias)
        alias_to_canonical[alias] = canonical
    return aliases, alias_to_canonical


def canonicalize_model_label(label: str, alias_to_canonical: dict[str, str]) -> str:
    """Map a GLiNER prediction label back to the public canonical label."""

    return alias_to_canonical.get(label, label)


def model_alias_metadata() -> dict[str, dict[str, str | float]]:
    return {
        label: {
            "model_label": model_label_alias(label),
            "single_label_benchmark_f1": GLINER_ALIAS_BENCHMARK_F1.get(label, 0.0),
        }
        for label in all_gdpr_labels()
    }


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
        label for category in requested_categories for label in GDPR_CATEGORIES[category]
    )
