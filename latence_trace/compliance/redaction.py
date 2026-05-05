"""Mask and replacement redaction helpers for compliance entities."""

from __future__ import annotations

import random
import threading
from collections import deque
from pathlib import Path
from typing import Any, Literal

GDPR_TO_REPLACEMENT_LABEL: dict[str, str] = {
    "person": "identity.full_name",
    "date_of_birth": "demographics.date_of_birth",
    "age": "demographics.age",
    "gender": "demographics.gender",
    "nationality": "demographics.nationality",
    "address": "address.full_address",
    "street_address": "address.street",
    "postal_code": "address.postal_code",
    "city": "address.city",
    "country": "address.country",
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

REPLACEMENT_LABEL_FALLBACKS: dict[str, str] = {
    "gov.ssn": "gov.national_id_number",
}


class TabuList:
    """Thread-safe set of recently used replacement row indices."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max(0, int(max_size))
        self._items: set[int] = set()
        self._queue: deque[int] = deque(maxlen=self.max_size or 1)
        self._lock = threading.Lock()

    def is_tabu(self, index: int) -> bool:
        with self._lock:
            return int(index) in self._items

    def add(self, index: int) -> None:
        if self.max_size <= 0:
            return
        index = int(index)
        with self._lock:
            if len(self._queue) >= self.max_size:
                oldest = self._queue.popleft()
                self._items.discard(oldest)
            self._queue.append(index)
            self._items.add(index)

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._items.clear()


class ReplacementDataset:
    """Lazy in-memory loader for the PII replacement dataset."""

    def __init__(self, dataset_path: str | None, *, records: list[dict[str, Any]] | None = None) -> None:
        self.dataset_path = dataset_path or "doubledsbv/pii-replacement-dataset"
        self._loaded = False
        self._data: dict[str, list[Any]] = {}
        self._columns: set[str] = set()
        self._country_index: dict[str, list[int]] = {}
        self._all_indices: list[int] = []
        self._load_error: str | None = None
        self._lock = threading.Lock()
        if records is not None:
            self._install_records(records)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> ReplacementDataset:
        return cls("__in_memory__", records=records)

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                dataset = self._load_dataset()
                rows = [dict(row) for row in dataset]
                self._install_records(rows)
            except Exception as exc:
                self._data = {}
                self._columns = set()
                self._country_index = {}
                self._all_indices = []
                self._load_error = f"{type(exc).__name__}: {exc}"
            self._loaded = True

    def _load_dataset(self) -> Any:
        from datasets import load_dataset, load_from_disk

        path = Path(self.dataset_path).expanduser()
        if path.exists():
            dataset = load_from_disk(str(path))
            if hasattr(dataset, "keys") and "train" in dataset:
                return dataset["train"]
            return dataset
        return load_dataset(self.dataset_path, split="train")

    def _install_records(self, rows: list[dict[str, Any]]) -> None:
        columns: set[str] = set()
        for row in rows:
            columns.update(str(key) for key in row)
        self._columns = columns
        self._data = {column: [row.get(column) for row in rows] for column in columns}
        self._all_indices = list(range(len(rows)))
        self._country_index = self._build_country_index(rows)
        self._load_error = None
        self._loaded = True

    def _build_country_index(self, rows: list[dict[str, Any]]) -> dict[str, list[int]]:
        country_column = self._country_column()
        if country_column is None:
            return {}
        index: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            country = str(row.get(country_column) or "").strip()
            if not country:
                continue
            index.setdefault(country.lower(), []).append(idx)
        return index

    def _country_column(self) -> str | None:
        if "address.country" in self._columns:
            return "address.country"
        if "country" in self._columns:
            return "country"
        return None

    def has_label(self, label: str) -> bool:
        self._load()
        return label in self._columns

    def sample(
        self,
        label: str,
        *,
        country: str | None = None,
        tabu_list: TabuList | None = None,
    ) -> str | None:
        self._load()
        if label not in self._columns or not self._all_indices:
            return None
        candidates = self._candidate_indices(country)
        if not candidates:
            return None
        values = self._data.get(label, [])
        for idx in self._sample_indices(candidates, tabu_list):
            if idx >= len(values):
                continue
            value = values[idx]
            if value is not None and str(value):
                if tabu_list is not None:
                    tabu_list.add(idx)
                return str(value)
        return None

    def _candidate_indices(self, country: str | None) -> list[int]:
        if country:
            candidates = self._country_index.get(country.strip().lower())
            if candidates:
                return list(candidates)
        usa = self._country_index.get("usa")
        if usa:
            return list(usa)
        return list(self._all_indices)

    @staticmethod
    def _sample_indices(candidates: list[int], tabu_list: TabuList | None) -> list[int]:
        pool = list(candidates)
        random.shuffle(pool)
        if tabu_list is not None and len(pool) > 1:
            fresh = [idx for idx in pool if not tabu_list.is_tabu(idx)]
            if fresh:
                pool = fresh
        return pool

    def stats(self, *, load: bool = True) -> dict[str, Any]:
        if load:
            self._load()
        return {
            "dataset_path": self.dataset_path,
            "loaded": self._loaded and bool(self._data),
            "load_error": self._load_error,
            "num_samples": len(self._all_indices),
            "available_labels": len(self._columns),
            "country_column": self._country_column(),
            "countries_available": sorted(self._country_index),
        }


class ComplianceRedactionEngine:
    def __init__(self, dataset_path: str | None = None, *, tabu_size: int = 1000) -> None:
        self.dataset = ReplacementDataset(dataset_path)
        self.tabu_list = TabuList(max_size=tabu_size)

    @staticmethod
    def mask_entity(entity: dict[str, Any]) -> str:
        label = str(entity.get("label", "PII")).upper()
        return f"[{label}]"

    @staticmethod
    def detect_language_country(text: str) -> str:
        try:
            from langdetect import detect

            lang = detect(text)
        except Exception:
            return "USA"
        return {
            "de": "Deutschland",
            "fr": "France",
            "en": "USA",
        }.get(str(lang).lower(), "USA")

    def replace_entity(
        self,
        entity: dict[str, Any],
        *,
        country: str | None = None,
    ) -> str | None:
        source_label = str(entity.get("label", ""))
        replacement_label = GDPR_TO_REPLACEMENT_LABEL.get(source_label, source_label)
        if not self.dataset.has_label(replacement_label):
            replacement_label = REPLACEMENT_LABEL_FALLBACKS.get(replacement_label, replacement_label)
        return self.dataset.sample(
            replacement_label,
            country=country,
            tabu_list=self.tabu_list,
        )

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

        resolved_country = country or self.detect_language_country(text)
        redacted = text
        out: list[dict[str, Any]] = []
        for entity in sorted(entities, key=lambda item: int(item["start"]), reverse=True):
            replacement: str | None
            actual_mode = "mask"
            if mode == "replace" and entity.get("source") != "custom_regex":
                replacement = self.replace_entity(entity, country=resolved_country)
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

    def stats(self, *, load: bool = True) -> dict[str, Any]:
        dataset_stats = self.dataset.stats(load=load)
        return {
            **dataset_stats,
            "tabu_size": self.tabu_list.size(),
            "tabu_max_size": self.tabu_list.max_size,
        }
