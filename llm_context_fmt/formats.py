"""Format template and prompt-constraint helpers."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_FORMATS = ("plain", "json", "xml", "markdown", "csv", "list")

BUILTIN_FORMAT_TEMPLATES = {
    "plain": "",
    "json": "Respond in JSON format with appropriate keys and values.",
    "xml": "Respond in XML format with appropriate tags.",
    "markdown": (
        "Respond in Markdown format with headings, lists, "
        "and emphasis as appropriate."
    ),
    "csv": "Respond as a CSV table with column headers and rows.",
    "list": "Respond as a bullet list with items prefixed by dashes.",
    "json-schema": (
        "Respond as a JSON object with the following schema: {schema}"
    ),
}


def parse_format_templates(raw_templates: tuple[str, ...]) -> dict[str, str]:
    """
    Parse custom templates.

    Supported syntax:
    - "name=template text"
    - "template text" (auto-assigned as custom_1, custom_2, ...)
    """
    templates: dict[str, str] = {}
    custom_idx = 1

    for raw in raw_templates:
        value = raw.strip()
        if not value:
            continue
        if "=" in value:
            name, template = value.split("=", 1)
            name = name.strip()
            template = template.strip()
            if not name or not template:
                raise ValueError(f"Invalid format template: {raw!r}")
            templates[name] = template
        else:
            templates[f"custom_{custom_idx}"] = value
            custom_idx += 1
    return templates


def parse_formats(raw_formats: str) -> list[str]:
    formats = [fmt.strip() for fmt in raw_formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one format must be provided.")
    return formats


def resolve_schema(schema_value: str | None) -> str | None:
    """Load inline JSON schema or schema from file path."""
    if schema_value is None:
        return None

    value = schema_value.strip()
    if not value:
        raise ValueError("Schema value cannot be empty.")

    candidate = Path(value)
    if candidate.exists():
        raw = candidate.read_text(encoding="utf-8")
    else:
        raw = value

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema: {exc}") from exc
    return json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))


def build_system_prompt(
    format_name: str,
    schema: str | None,
    custom_templates: dict[str, str] | None = None,
) -> str:
    templates = dict(BUILTIN_FORMAT_TEMPLATES)
    if custom_templates:
        templates.update(custom_templates)

    if format_name not in templates:
        valid = ", ".join(sorted(templates))
        raise ValueError(f"Unknown format '{format_name}'. Valid formats: {valid}")

    template = templates[format_name]
    if format_name == "plain":
        return (
            "You are a helpful assistant. Provide your best answer. "
            "No special output format is required."
        )

    if "{schema}" in template:
        if not schema:
            raise ValueError(
                "The 'json-schema' format requires --schema with valid JSON."
            )
        template = template.replace("{schema}", schema)

    return (
        "You are a helpful assistant. Follow the required output format exactly. "
        f"{template}"
    )


def validate_formats(
    formats: list[str],
    custom_templates: dict[str, str] | None = None,
) -> None:
    valid_names = set(BUILTIN_FORMAT_TEMPLATES)
    if custom_templates:
        valid_names.update(custom_templates)
    unknown = [fmt for fmt in formats if fmt not in valid_names]
    if unknown:
        valid = ", ".join(sorted(valid_names))
        raise ValueError(f"Unknown format(s): {', '.join(unknown)}. Valid: {valid}")
