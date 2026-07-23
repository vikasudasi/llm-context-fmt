from __future__ import annotations

from click.testing import CliRunner

from llm_context_fmt.analyzer import compare_formats
from llm_context_fmt.cli import main
from llm_context_fmt.formats import build_system_prompt, parse_formats, resolve_schema


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Run one prompt across multiple output format constraints" in result.output
    assert "--formats" in result.output
    assert "--provider" in result.output


def test_parse_formats():
    assert parse_formats("plain,json,xml") == ["plain", "json", "xml"]


def test_schema_validation_rejects_invalid_json():
    try:
        resolve_schema("{bad")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid schema to raise ValueError")


def test_build_system_prompt_for_json_schema():
    prompt = build_system_prompt(
        "json-schema",
        schema='{"type":"object","properties":{"x":{"type":"string"}}}',
        custom_templates=None,
    )
    assert "schema" in prompt.lower()
    assert "type" in prompt


def test_compare_formats_metrics_present():
    comparison = compare_formats(
        {
            "plain": "apple banana",
            "json": '{"word":"apple"}',
        }
    )
    assert "format_sensitivity_score" in comparison
    assert "length_variance" in comparison
    assert "overlap_scores" in comparison
