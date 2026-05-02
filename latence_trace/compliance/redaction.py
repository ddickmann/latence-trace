"""Mask and replacement redaction helpers for compliance entities."""

from __future__ import annotations

import random
import threading
from typing import Any, Literal

GDPR_TO_REPLACEMENT_LABEL: dict[str, str] = {
    "person": "identity.full_name",
    "date_of_birth": "demographics.date_of_birth",
    "age": "demographics.age",
    "gender": "demographics.gender",
    "nationality": "demographics.nationality",
    "address": "address.full_address",
    "phone_number": "contact.phone_number",
    "email": "contact.email",
    "maiden_name": "family.mothers_maiden_name",
    "marital_status": "family.marital_status",
    "national_id": "gov.national_id_number",
    "passport_number": "gov.passport_number",
    "drivers_license": "gov.drivers_license",
    "social_security_number": "gov.ssn",
    "vehicle_registration_number": "vehicle.plate_number",
    "ip_address": "device.ip_address",
    "mac_address": "device.mac_address",
    "device_id": "device.imei",
    "cookie_id": "device.cookie_id",
    "username": "device.username",
    "password": "security.password",
    "advertising_id": "adid.apple_idfa",
    "bank_account": "finance.bank_account",
    "insurance_number": "insurance.policy_number",
    "swift_code": "finance.bic",
    "account_number": "finance.bank_account",
    "credit_card_number": "finance.credit_card",
    "tax_identification_number": "finance.tax_id",
    "transaction_id": "finance.transaction_id",
    "credit_score": "finance.credit_score",
    "employee_id": "employment.employee_id",
    "job_title": "employment.job_title",
    "organization": "employment.employer_name",
    "medical_condition": "health.allergies_conditions",
    "medical_record_number": "health.medical_record_number",
    "health_insurance_number": "insurance.health_policy_number",
    "prescription_drug": "health.prescriptions",
    "disability_status": "accessibility.disability_status",
    "race_or_ethnicity": "sensitive.racial_ethnic_origin",
    "religious_belief": "sensitive.religious_beliefs",
    "political_affiliation": "sensitive.political_opinions",
    "sexual_orientation": "sensitive.sex_life_orientation",
    "trade_union_membership": "sensitive.trade_union",
    "criminal_record": "legal.criminal_conviction",
}


class ReplacementDataset:
    """Lazy loader for the PII replacement dataset."""

    def __init__(self, dataset_path: str | None) -> None:
        self.dataset_path = dataset_path or "doubledsbv/pii-replacement-dataset"
        self._loaded = False
        self._rows: list[dict[str, Any]] = []
        self._columns: set[str] = set()
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from datasets import load_dataset

                dataset = load_dataset(self.dataset_path, split="train")
                self._rows = [dict(row) for row in dataset]
                self._columns = set(dataset.column_names)
            except Exception:
                self._rows = []
                self._columns = set()
            self._loaded = True

    def sample(self, label: str, *, country: str | None = None) -> str | None:
        self._load()
        if label not in self._columns or not self._rows:
            return None
        rows = self._rows
        if country:
            filtered = [
                row
                for row in self._rows
                if str(row.get("country", "")).lower() == country.lower()
            ]
            if filtered:
                rows = filtered
        for _ in range(min(len(rows), 20)):
            row = random.choice(rows)
            value = row.get(label)
            if value is not None and str(value):
                return str(value)
        return None


class ComplianceRedactionEngine:
    def __init__(self, dataset_path: str | None = None) -> None:
        self.dataset = ReplacementDataset(dataset_path)

    @staticmethod
    def mask_entity(entity: dict[str, Any]) -> str:
        label = str(entity.get("label", "PII")).upper()
        return f"[{label}]"

    def replace_entity(
        self,
        entity: dict[str, Any],
        *,
        country: str | None = None,
    ) -> str | None:
        source_label = str(entity.get("label", ""))
        replacement_label = GDPR_TO_REPLACEMENT_LABEL.get(source_label, source_label)
        return self.dataset.sample(replacement_label, country=country)

    def redact_text(
        self,
        text: str,
        entities: list[dict[str, Any]],
        *,
        mode: Literal["mask", "replace"] = "mask",
        country: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not entities:
            return text, entities

        redacted = text
        out: list[dict[str, Any]] = []
        for entity in sorted(entities, key=lambda item: int(item["start"]), reverse=True):
            replacement: str | None
            actual_mode = "mask"
            if mode == "replace" and entity.get("source") != "custom_regex":
                replacement = self.replace_entity(entity, country=country)
                if replacement is not None:
                    actual_mode = "replace"
                else:
                    replacement = self.mask_entity(entity)
            else:
                replacement = self.mask_entity(entity)

            start = int(entity["start"])
            end = int(entity["end"])
            redacted = redacted[:start] + replacement + redacted[end:]
            updated = dict(entity)
            updated["redacted_value"] = replacement
            updated["redaction_mode"] = actual_mode
            out.append(updated)

        out.reverse()
        return redacted, out
