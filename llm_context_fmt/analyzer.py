"""Output-level and cross-format analysis (stdlib only)."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from statistics import mean, pstdev

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_JSON_KEY_RE = re.compile(r'"([^"]+)"\s*:')
_XML_TAG_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_\-]*)[^>]*>")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LIST_LINE_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*[^,\n]+(?:\s*,\s*[^,\n]+)+\s*$", re.MULTILINE)
_JSON_KEY_FALLBACK_RE = re.compile(r'"[^"]+"\s*:\s*')
_MD_INLINE_MARKERS_RE = re.compile(r"(\*\*|~~|\*|`)")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)

_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "love",
    "wonderful",
    "positive",
    "success",
    "clear",
    "helpful",
}
_NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "sad",
    "hate",
    "poor",
    "negative",
    "failure",
    "confused",
    "harmful",
}


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def lexical_diversity(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def sentiment_score(words: list[str]) -> float:
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    return (pos - neg) / len(words)


def analyze_output(text: str) -> dict[str, float | int]:
    words = tokenize_words(text)
    return {
        "token_count": estimate_token_count(text),
        "word_count": len(words),
        "lexical_diversity": round(lexical_diversity(words), 4),
        "sentiment_score": round(sentiment_score(words), 4),
    }


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(tokenize_words(a))
    set_b = set(tokenize_words(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def min_overlap(a: str, b: str) -> float:
    set_a = set(tokenize_words(a))
    set_b = set(tokenize_words(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def clean_content(text: str, format_name: str) -> str:
    normalized = format_name.strip().lower()

    if normalized in {"json", "json-schema"}:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            cleaned = _JSON_KEY_FALLBACK_RE.sub("", text)
            return re.sub(r"[{}\[\]]", " ", cleaned)

        values: list[str] = []

        def collect_strings(value: object) -> None:
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    collect_strings(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_strings(nested)

        collect_strings(parsed)
        return " ".join(values)

    if normalized == "xml":
        return re.sub(r"<[^>]+>", " ", text)

    if normalized == "csv":
        rows: list[list[str]] = []
        try:
            reader = csv.reader(io.StringIO(text))
            rows = [row for row in reader if row]
        except csv.Error:
            lines = [line for line in text.splitlines() if line.strip()]
            rows = [line.split(",") for line in lines]

        if rows and any("**" in cell for cell in rows[0]):
            rows = rows[1:]

        return " ".join(
            cell.strip().replace("**", "")
            for row in rows
            for cell in row
            if cell.strip()
        )

    if normalized == "markdown":
        cleaned = _MD_HEADING_RE.sub("", text)
        cleaned = _BLOCKQUOTE_RE.sub("", cleaned)
        return _MD_INLINE_MARKERS_RE.sub("", cleaned)

    if normalized == "list":
        return _LIST_LINE_RE.sub("", text)

    return text


def detect_structural_features(text: str) -> list[str]:
    features: list[str] = []
    if "\n\n" in text:
        features.append("paragraphs")
    if _LIST_LINE_RE.search(text):
        features.append("lists")
    if _TABLE_ROW_RE.search(text):
        features.append("tables")
    if _JSON_KEY_RE.search(text):
        features.append("json_keys")
    if _XML_TAG_RE.search(text):
        features.append("xml_tags")
    if _MD_HEADING_RE.search(text):
        features.append("markdown_headings")
    if not features:
        features.append("plain_text")
    return features


def aggregate_metric_dicts(metric_dicts: list[dict[str, float | int]]) -> dict[str, float]:
    if not metric_dicts:
        return {
            "token_count": 0.0,
            "word_count": 0.0,
            "lexical_diversity": 0.0,
            "sentiment_score": 0.0,
        }
    keys = ("token_count", "word_count", "lexical_diversity", "sentiment_score")
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(mean(float(m[key]) for m in metric_dicts))
    return out


def compare_formats(format_outputs: dict[str, str]) -> dict[str, object]:
    names = sorted(format_outputs)
    cleaned_outputs = {
        name: clean_content(output, name)
        for name, output in format_outputs.items()
    }
    pairs: list[tuple[str, str]] = []
    pair_scores: dict[str, float] = {}
    min_overlap_scores: dict[str, float] = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            score = jaccard_similarity(cleaned_outputs[a], cleaned_outputs[b])
            overlap = min_overlap(cleaned_outputs[a], cleaned_outputs[b])
            key = f"{a}_{b}"
            pair_scores[key] = round(score, 4)
            min_overlap_scores[key] = round(overlap, 4)
            pairs.append((a, b))

    similarities = list(pair_scores.values())
    if similarities:
        mean_similarity = float(mean(similarities))
        sensitivity = 1.0 - mean_similarity
    else:
        sensitivity = 0.0

    token_counts = [
        estimate_token_count(format_outputs[name])
        for name in names
    ]
    if token_counts and mean(token_counts) > 0:
        length_variance = pstdev(token_counts) / mean(token_counts)
    else:
        length_variance = 0.0

    structural = {
        name: detect_structural_features(output)
        for name, output in format_outputs.items()
    }

    return {
        "overlap_scores": pair_scores,
        "min_overlap_scores": min_overlap_scores,
        "format_sensitivity_score": round(sensitivity, 4),
        "length_variance": round(length_variance, 4),
        "structural_features": structural,
    }
