"""Conservative deterministic cleanup for GLiNER PII entities."""

from __future__ import annotations

import ipaddress
import re
from typing import Any

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
_MAC_RE = re.compile(r"^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")
_SWIFT_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?$")
_DATE_RE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})$"
)
_GPS_RE = re.compile(r"^-?\d{1,2}(?:\.\d+)?\s*,\s*-?\d{1,3}(?:\.\d+)?$")
_VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{11,17}$", re.IGNORECASE)
_CRYPTO_RE = re.compile(r"^(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[ac-hj-np-z02-9]{11,71})$")

_TRIM_CHARS = " \t\r\n\"'`.,;:()[]{}<>"


def _luhn_ok(value: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) < 12 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def clean_entity_boundaries(text: str, entity: dict[str, Any]) -> dict[str, Any] | None:
    start = int(entity.get("start", 0))
    end = int(entity.get("end", 0))
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    if start >= end:
        return None
    cleaned = dict(entity)
    cleaned["start"] = start
    cleaned["end"] = end
    cleaned["text"] = text[start:end]
    return cleaned


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


def passes_sanity_check(entity: dict[str, Any]) -> bool:
    """Return False only for obvious format errors."""

    label = str(entity.get("label", ""))
    value = str(entity.get("text", "")).strip()
    if not value:
        return False
    if label == "email":
        return bool(_EMAIL_RE.match(value))
    if label == "phone_number":
        return bool(_PHONE_RE.match(value)) and sum(ch.isdigit() for ch in value) >= 7
    if label == "ip_address":
        return _is_valid_ip(value)
    if label == "mac_address":
        return bool(_MAC_RE.match(value))
    if label == "credit_card_number":
        return _luhn_ok(value)
    if label == "swift_code":
        return bool(_SWIFT_RE.match(value.upper()))
    if label == "gps_coordinates":
        return bool(_GPS_RE.match(value))
    if label == "date_of_birth":
        return bool(_DATE_RE.match(value))
    if label == "vehicle_registration_number":
        return 2 <= len(value) <= 16
    if label == "crypto_wallet_address":
        return bool(_CRYPTO_RE.match(value))
    return True


def apply_sanity_checks(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for entity in entities:
        bounded = clean_entity_boundaries(text, entity)
        if bounded is not None and passes_sanity_check(bounded):
            cleaned.append(bounded)
    return cleaned
