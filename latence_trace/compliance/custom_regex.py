"""User-defined regex extraction for compliance redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CustomLabelConfig:
    label_name: str
    extractor: str
    compiled_pattern: re.Pattern[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomLabelConfig:
        label_name = str(data.get("label_name", "")).strip()
        extractor = str(data.get("extractor", ""))
        if not label_name:
            raise ValueError("custom label missing label_name")
        if not extractor:
            raise ValueError(f"custom label {label_name!r} missing extractor")
        try:
            compiled = re.compile(extractor)
        except re.error as exc:
            raise ValueError(f"invalid regex for {label_name!r}: {exc}") from exc
        return cls(label_name=label_name, extractor=extractor, compiled_pattern=compiled)


def validate_custom_configs(
    configs: list[dict[str, Any]],
) -> tuple[list[CustomLabelConfig], list[str]]:
    valid: list[CustomLabelConfig] = []
    errors: list[str] = []
    for idx, config in enumerate(configs):
        try:
            valid.append(CustomLabelConfig.from_dict(config))
        except ValueError as exc:
            errors.append(f"custom_labels[{idx}]: {exc}")
    return valid, errors


def extract_custom_entities(
    text: str,
    configs: list[CustomLabelConfig],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for config in configs:
        for match in config.compiled_pattern.finditer(text):
            entities.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "label": config.label_name,
                    "score": 1.0,
                    "source": "custom_regex",
                }
            )
    return entities
