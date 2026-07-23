"""CLI entrypoint for llm-context-fmt."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from llm_context_fmt.analyzer import (
    aggregate_metric_dicts,
    analyze_output,
    compare_formats,
    detect_structural_features,
)
from llm_context_fmt.formats import (
    DEFAULT_FORMATS,
    parse_format_templates,
    parse_formats,
    resolve_schema,
    validate_formats,
)
from llm_context_fmt.presets import DEMO_PICK_A_WORD_PROMPT
from llm_context_fmt.runner import (
    DEFAULT_MODEL,
    ProviderConfigurationError,
    RunRequest,
    run_all_formats,
)

OUTPUT_TRUNCATE = 500


def _truncate(text: str, size: int = OUTPUT_TRUNCATE) -> str:
    if len(text) <= size:
        return text
    return text[: size - 3] + "..."


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _aggregate_format_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    valid_texts = [r["output"] for r in records if (not r["refused"] and not r["error"])]
    had_refusal = any(r["refused"] for r in records)
    errors = [r["error"] for r in records if r["error"]]

    if not valid_texts:
        output = "REFUSED" if had_refusal and not errors else ""
        metrics = {
            "token_count": 0.0,
            "word_count": 0.0,
            "lexical_diversity": 0.0,
            "sentiment_score": 0.0,
        }
        return (
            {
                "output": output,
                **metrics,
                "structural_features": ["none"],
                "status": "refused" if had_refusal and not errors else "error",
                "errors": [e for e in errors if e],
            },
            False,
        )

    per_run_metrics = [analyze_output(text) for text in valid_texts]
    avg_metrics = aggregate_metric_dicts(per_run_metrics)
    representative = valid_texts[0]
    return (
        {
            "output": representative,
            "token_count": round(avg_metrics["token_count"], 2),
            "word_count": round(avg_metrics["word_count"], 2),
            "lexical_diversity": round(avg_metrics["lexical_diversity"], 4),
            "sentiment_score": round(avg_metrics["sentiment_score"], 4),
            "structural_features": detect_structural_features(representative),
            "status": "ok",
            "errors": [e for e in errors if e],
        },
        True,
    )


def _render_table(
    prompt: str,
    model: str,
    formats_payload: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    console = Console()
    table = Table(title="llm-context-fmt Results")
    table.add_column("Format", style="bold cyan")
    table.add_column("Output (truncated)")
    table.add_column("Token Cnt", justify="right")
    table.add_column("Word Cnt", justify="right")
    table.add_column("Lex Div", justify="right")
    table.add_column("Status")

    for fmt, payload in formats_payload.items():
        table.add_row(
            fmt,
            _truncate(str(payload["output"])),
            f"{payload['token_count']}",
            f"{payload['word_count']}",
            f"{payload['lexical_diversity']}",
            str(payload["status"]),
        )
    console.print(table)
    console.print(f"Prompt: {_truncate(prompt, 120)}")
    console.print(f"Model: {model}")
    console.print(
        "Format Sensitivity Score: "
        f"{comparison['format_sensitivity_score']} "
        f"(Length variance: {comparison['length_variance']})"
    )


@click.command(context_settings={"help_option_names": ["--help"]})
@click.argument("prompt", required=False)
@click.option(
    "--formats",
    default=",".join(DEFAULT_FORMATS),
    show_default=True,
    help="Comma-separated formats to test.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to use.")
@click.option(
    "--provider",
    default="openrouter",
    show_default=True,
    type=click.Choice(["openrouter", "openai", "anthropic"], case_sensitive=False),
    help="API provider.",
)
@click.option("--api-base", default=None, help="Custom API base URL.")
@click.option("--api-key", default=None, help="API key (defaults to env var).")
@click.option(
    "--schema",
    default=None,
    help="JSON Schema for json-schema format (file path or inline JSON).",
)
@click.option("--demo", is_flag=True, help='Run the built-in "Pick a word" census demo.')
@click.option(
    "--output",
    "output_mode",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json", "both"], case_sensitive=False),
    help="Output format.",
)
@click.option("--max-tokens", default=1024, show_default=True, type=int, help="Max tokens.")
@click.option(
    "--temperature",
    default=0.7,
    show_default=True,
    type=float,
    help="Temperature.",
)
@click.option(
    "--runs",
    default=1,
    show_default=True,
    type=int,
    help="Number of runs per format.",
)
@click.option(
    "--format-template",
    "format_templates",
    multiple=True,
    help="Custom format template. Use name=template or template text.",
)
def main(
    prompt: str | None,
    formats: str,
    model: str,
    provider: str,
    api_base: str | None,
    api_key: str | None,
    schema: str | None,
    demo: bool,
    output_mode: str,
    max_tokens: int,
    temperature: float,
    runs: int,
    format_templates: tuple[str, ...],
) -> None:
    """Run one prompt across multiple output format constraints."""
    if demo:
        prompt = DEMO_PICK_A_WORD_PROMPT
    if not prompt:
        raise click.UsageError("PROMPT is required unless --demo is used.")
    if runs < 1:
        raise click.UsageError("--runs must be >= 1.")
    if max_tokens < 1:
        raise click.UsageError("--max-tokens must be >= 1.")

    try:
        parsed_formats = parse_formats(formats)
        custom_templates = parse_format_templates(format_templates)
        validate_formats(parsed_formats, custom_templates=custom_templates)
        normalized_schema = resolve_schema(schema)
        if "json-schema" in parsed_formats and not normalized_schema:
            raise ValueError(
                "The 'json-schema' format requires --schema with valid JSON."
            )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    req = RunRequest(
        prompt=prompt,
        formats=parsed_formats,
        model=model,
        provider=provider.lower(),
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        runs=runs,
        schema=normalized_schema,
        custom_templates=custom_templates,
    )

    try:
        _progress("Starting API requests...")
        by_format = asyncio.run(run_all_formats(req, progress_cb=_progress))
    except ProviderConfigurationError as exc:
        click.echo(f"Provider configuration error: {exc}", err=True)
        raise SystemExit(1) from exc

    serializable_records: dict[str, list[dict[str, Any]]] = {}
    for fmt, records in by_format.items():
        serializable_records[fmt] = [
            {
                "format_name": r.format_name,
                "run_index": r.run_index,
                "output": r.output,
                "refused": r.refused,
                "error": r.error,
            }
            for r in records
        ]

    formats_payload: dict[str, dict[str, Any]] = {}
    valid_texts_for_comparison: dict[str, str] = {}

    for fmt in parsed_formats:
        payload, has_valid = _aggregate_format_records(serializable_records[fmt])
        formats_payload[fmt] = payload
        if has_valid:
            valid_texts_for_comparison[fmt] = str(payload["output"])

    if not valid_texts_for_comparison:
        click.echo("No valid responses received.", err=True)
        raise SystemExit(3)

    comparison = compare_formats(valid_texts_for_comparison)
    result_payload = {
        "prompt": prompt,
        "model": model,
        "provider": provider.lower(),
        "formats": formats_payload,
        "comparison": comparison,
        "raw_runs": serializable_records,
    }

    if output_mode in ("table", "both"):
        _render_table(prompt, model, formats_payload, comparison)
    if output_mode in ("json", "both"):
        click.echo(json.dumps(result_payload, indent=2, ensure_ascii=True))
