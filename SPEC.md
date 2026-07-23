# llm-context-fmt — SPEC

## What to Build

A CLI tool that runs the same prompt through different format constraints (plain text, JSON, XML, Markdown, CSV, JSON Schema) and shows how the output content shifts. Based on the July 2026 ArXiv paper showing that "Reply with JSON only" changes which answers models choose across 44 models.

## File Structure

```
llm-context-fmt/
├── pyproject.toml
├── README.md
├── SPEC.md
├── llm_context_fmt/
│   ├── __init__.py
│   ├── __main__.py          # python -m entry
│   ├── cli.py               # click CLI interface
│   ├── formats.py            # format template definitions
│   ├── runner.py             # API calling logic
│   ├── analyzer.py           # cross-format comparison analysis
│   └── presets.py            # built-in demo prompts
└── tests/
    └── test_basic.py
```

## Tech Stack & Dependencies

- **Python ≥ 3.10**
- **click** — CLI framework
- **rich** — terminal tables and formatting
- **httpx** — async HTTP for API calls
- **tabulate** — table output
- **nltk or textstat** — lexical diversity (optional, use stdlib if possible)

## Formats to Test

The tool wraps the user's prompt in these format constraints:

1. **plain** — No format constraint (baseline). Append nothing.
2. **json** — "Respond in JSON format with appropriate keys and values."
3. **xml** — "Respond in XML format with appropriate tags."
4. **markdown** — "Respond in Markdown format with headings, lists, and emphasis as appropriate."
5. **csv** — "Respond as a CSV table with column headers and rows."
6. **list** — "Respond as a bullet list with items prefixed by dashes."
7. **json-schema** — "Respond as a JSON object with the following schema: {schema}"

Additionally, the user can provide custom format templates using `--format-template '...'`.

## API Provider Support

- OpenRouter (default) — reads `OPENROUTER_API_KEY` env var
- OpenAI — reads `OPENAI_API_KEY` env var  
- Anthropic — reads `ANTHROPIC_API_KEY` env var
- Generic OpenAI-compatible endpoint via `--api-base`

Default model: `deepseek/deepseek-v4-flash` (from OpenRouter)

## CLI Interface

```
llm-context-fmt [OPTIONS] PROMPT

Options:
  --formats TEXT          Comma-separated formats to test (default: plain,json,xml,markdown,csv,list)
  --model TEXT            Model to use (default: deepseek/deepseek-v4-flash)
  --provider TEXT         API provider: openrouter, openai, anthropic (default: openrouter)
  --api-base TEXT         Custom API base URL for generic OpenAI-compatible endpoint
  --api-key TEXT          API key (defaults to env var)
  --schema TEXT           JSON Schema for json-schema format (file path or inline JSON)
  --demo                 Run the built-in "Pick a word" census demo
  --output TEXT           Output format: table, json, both (default: table)
  --max-tokens INT       Max tokens per response (default: 1024)
  --temperature FLOAT     Temperature (default: 0.7)
  --runs INT             Number of runs per format for statistical significance (default: 1)
  --help                 Show this message and exit.
```

## Output

### Table Mode (default)

A rich terminal table showing each format, the output (truncated), token count, and a format sensitivity score.

### JSON Mode

```json
{
  "prompt": "...",
  "model": "...",
  "formats": {
    "plain": {
      "output": "...",
      "token_count": 123,
      "word_count": 98,
      "lexical_diversity": 0.72
    },
    "json": { "...": "..." }
  },
  "comparison": {
    "overlap_scores": { "plain_json": 0.45, "plain_xml": 0.52, ... },
    "format_sensitivity_score": 0.38,
    "length_variance": 0.15,
    "structural_features": { "plain": ["paragraphs"], "json": ["keys"], ... }
  }
}
```

## Analysis Metrics

For each output:
- **Token count** (using model's tokenizer estimate ~4 chars/token)
- **Word count**
- **Lexical diversity** (unique words / total words)
- **Sentiment score** (using TextBlob or simple AFINN if available, else basic positive/negative word count ratio)

Cross-format comparison:
- **Pairwise Jaccard similarity** (word set overlap) between each format pair
- **Format Sensitivity Score** — aggregate: 1 - (mean pairwise similarity). Higher = more format-dependent output.
- **Length variance** — coefficient of variation of token counts across formats
- **Structural features detected** — paragraphs, lists, tables, JSON keys, XML tags, markdown headings

## Built-in Demo: "Pick a Word" Census

When `--demo` is used, the tool runs the prompt from the original paper:

> "Pick a word. Any word. Just pick a single word. What word did you pick?"

This is a simple question where the answer should be the same regardless of format. The demo shows how format constraints shift which word the model picks.

## Exit Codes

- 0: Success
- 1: API error (network, auth)
- 2: Invalid arguments (bad format name, bad schema)
- 3: No valid responses received

## Edge Cases

- Empty API key → error message with instructions
- Network failure → retry once, then report error per-format
- Model returns refusal → show "REFUSED" in output, exclude from analysis
- JSON Schema is invalid → error with parse details
- Output too long → truncate display to 500 chars, store full in JSON output
- Multiple runs per format → average metrics across runs

## Implementation Notes

- Use `httpx.AsyncClient` for concurrent API calls across formats
- Always use system prompt for the format constraint, user prompt for the actual query
- Use click's `option` for all CLI args
- The `analyzer.py` module should work with stdlib only (no numpy/pandas) for portability
- Print progress to stderr so stdout stays clean for piping
