# llm-context-fmt

Run one prompt across multiple output format constraints and measure how format instructions change model behavior.

## Why

Many teams assume output formatting instructions only change structure, not meaning. In practice, constraints like "JSON only" or "respond as CSV" often alter lexical choices, response length, and even the content direction.

`llm-context-fmt` helps you quantify that shift by:

- Running the same prompt across multiple format constraints
- Collecting per-format response metrics (token count, lexical diversity, sentiment proxy)
- Computing cross-format similarity and a format sensitivity score
- Producing terminal-friendly output and machine-readable JSON for analysis pipelines

## Installation

### Requirements

- Python `>=3.10`
- Access to at least one supported LLM provider API key

### Install from source

```bash
git clone <your-repo-url>
cd llm-context-fmt
pip install -e .
```

### Install developer dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Basic command

```bash
llm-context-fmt "Explain why caching improves API latency."
```

### Built-in demo prompt

Runs the built-in "Pick a word" prompt to showcase format sensitivity on a simple query.

```bash
llm-context-fmt --demo
```

### Select specific formats

```bash
llm-context-fmt \
  --formats plain,json,xml,markdown \
  "Describe the lifecycle of a butterfly."
```

### Use JSON Schema-constrained output

```bash
llm-context-fmt \
  --formats plain,json-schema \
  --schema '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}' \
  "Name one city in Japan."
```

You can also pass a schema file:

```bash
llm-context-fmt \
  --formats json-schema \
  --schema ./schema.json \
  "Summarize this topic in one sentence."
```

### Run multiple trials per format

Useful for reducing one-off sampling noise and averaging metric values.

```bash
llm-context-fmt \
  --runs 5 \
  --temperature 0.7 \
  "Give three practical stress-management tips."
```

### Emit JSON for downstream analysis

```bash
llm-context-fmt \
  --output json \
  --formats plain,json,xml \
  "Write a one-paragraph product summary."
```

### Print both table and JSON

```bash
llm-context-fmt \
  --output both \
  "Compare monolith and microservice architectures."
```

### Use custom format templates

Named custom template:

```bash
llm-context-fmt \
  --format-template "yaml=Respond in valid YAML with keys title and bullets." \
  --formats plain,yaml \
  "Outline a 2-week onboarding plan."
```

Unnamed templates are auto-named as `custom_1`, `custom_2`, etc.

## CLI Options Reference

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `PROMPT` | argument | none | Prompt to evaluate. Required unless `--demo` is used. |
| `--formats` | comma-separated string | `plain,json,xml,markdown,csv,list` | Formats to run. Supports built-ins plus custom template names. |
| `--model` | string | `deepseek/deepseek-v4-flash` | Model identifier passed to the selected provider API. |
| `--provider` | enum | `openrouter` | Provider to use: `openrouter`, `openai`, `anthropic`. |
| `--api-base` | string | provider default | Override base URL (useful for OpenAI-compatible endpoints). |
| `--api-key` | string | env var | Explicit API key (otherwise read from provider-specific env var). |
| `--schema` | string | none | JSON schema as inline JSON or a file path (required for `json-schema` format). |
| `--demo` | flag | `false` | Runs built-in prompt instead of requiring `PROMPT`. |
| `--output` | enum | `table` | Output mode: `table`, `json`, or `both`. |
| `--max-tokens` | integer | `1024` | Max tokens requested from model response. Must be `>= 1`. |
| `--temperature` | float | `0.7` | Sampling temperature passed to provider API. |
| `--runs` | integer | `1` | Number of calls per format. Must be `>= 1`. |
| `--format-template` | repeatable string | none | Custom format template (`name=template`) or unnamed template text. |
| `--help` | flag | n/a | Shows help message and exits. |

## Supported Built-in Formats

- `plain`
- `json`
- `xml`
- `markdown`
- `csv`
- `list`
- `json-schema` (requires `--schema`)

## Output Format

### Table output (`--output table`)

Displays a terminal table with:

- `Format`: format name tested
- `Output (truncated)`: response preview (up to 500 chars)
- `Token Cnt`: estimated token count (character-length heuristic)
- `Word Cnt`: total detected words
- `Lex Div`: lexical diversity (`unique_words / total_words`)
- `Status`: `ok`, `refused`, or `error`

Then prints:

- prompt preview
- model name
- aggregate `Format Sensitivity Score` and `Length variance`

### JSON output (`--output json` or `--output both`)

High-level shape:

```json
{
  "prompt": "string",
  "model": "string",
  "provider": "string",
  "formats": {
    "<format_name>": {
      "output": "string",
      "token_count": 0,
      "word_count": 0,
      "lexical_diversity": 0.0,
      "sentiment_score": 0.0,
      "structural_features": ["..."],
      "status": "ok|refused|error",
      "errors": []
    }
  },
  "comparison": {
    "overlap_scores": {
      "plain_json": 0.0
    },
    "format_sensitivity_score": 0.0,
    "length_variance": 0.0,
    "structural_features": {
      "<format_name>": ["..."]
    }
  },
  "raw_runs": {
    "<format_name>": [
      {
        "format_name": "string",
        "run_index": 0,
        "output": "string",
        "refused": false,
        "error": null
      }
    ]
  }
}
```

### Metrics and comparison semantics

- `format_sensitivity_score`: `1 - mean(pairwise_jaccard_similarity)`
- `overlap_scores`: pairwise Jaccard similarity on tokenized word sets
- `length_variance`: population stddev of token counts divided by mean token count
- `sentiment_score`: lightweight lexicon-based proxy over detected word list
- `structural_features`: simple detectors (lists, tables, JSON keys, XML tags, markdown headings, etc.)

## Architecture Overview

Project modules:

- `llm_context_fmt/cli.py`: CLI entrypoint, argument parsing, aggregation, rendering table/JSON output
- `llm_context_fmt/formats.py`: built-in format templates, custom template parsing, schema loading/validation, system prompt construction
- `llm_context_fmt/runner.py`: provider adapters, API key resolution, async request fan-out, retry and error/refusal handling
- `llm_context_fmt/analyzer.py`: output tokenization, per-output metrics, structural detection, cross-format comparisons
- `llm_context_fmt/presets.py`: built-in demo prompt constants
- `llm_context_fmt/__main__.py`: `python -m llm_context_fmt` entrypoint

Execution flow:

1. Parse CLI options and validate formats/schema.
2. Build per-format system prompts.
3. Send asynchronous API requests (`formats * runs` calls).
4. Aggregate run-level results into per-format summaries.
5. Compute cross-format comparison metrics.
6. Render table, JSON, or both.

## Environment Variables

- `OPENROUTER_API_KEY`: used when `--provider openrouter` (default provider)
- `OPENAI_API_KEY`: used when `--provider openai`
- `ANTHROPIC_API_KEY`: used when `--provider anthropic`

`--api-key` overrides environment variable lookup.  
`--api-base` overrides provider default base URL.

## Exit Codes

- `0`: success
- `1`: provider configuration error (for example, missing API key)
- `3`: no valid responses received

Argument and validation issues are reported by Click as usage errors.

## Testing

Run tests with:

```bash
pytest
```

## License

No license file is currently included in this repository.

If you plan to distribute or use this project beyond private/internal use, add a `LICENSE` file (for example, MIT, Apache-2.0, or BSD-3-Clause) and update this section accordingly.
