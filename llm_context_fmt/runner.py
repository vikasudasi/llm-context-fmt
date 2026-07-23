"""Provider adapters and async execution engine."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from llm_context_fmt.formats import build_system_prompt

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com/v1"


class ProviderConfigurationError(Exception):
    """Raised when provider configuration is invalid."""


@dataclass(slots=True)
class RunRequest:
    prompt: str
    formats: list[str]
    model: str
    provider: str
    api_base: str | None
    api_key: str | None
    max_tokens: int
    temperature: float
    runs: int
    schema: str | None
    custom_templates: dict[str, str]


@dataclass(slots=True)
class ResponseRecord:
    format_name: str
    run_index: int
    output: str
    refused: bool
    error: str | None


def resolve_api_key(provider: str, explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key

    env_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = env_map.get(provider, "OPENAI_API_KEY")
    key = os.getenv(env_var)
    if not key:
        raise ProviderConfigurationError(
            f"Missing API key. Set --api-key or {env_var}."
        )
    return key


def detect_refusal(text: str) -> bool:
    lowered = text.lower()
    refusal_markers = (
        "i can't help with that",
        "i cannot help with that",
        "i can't comply",
        "i cannot comply",
        "i must refuse",
        "i'm unable to comply",
        "i am unable to comply",
        "i won't",
    )
    return any(marker in lowered for marker in refusal_markers)


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return ""


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content") or []
    chunks: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    return "\n".join(chunks).strip()


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


async def _single_call(
    client: httpx.AsyncClient,
    req: RunRequest,
    format_name: str,
    run_index: int,
) -> ResponseRecord:
    system_prompt = build_system_prompt(
        format_name,
        schema=req.schema,
        custom_templates=req.custom_templates,
    )

    provider = req.provider.lower()
    api_key = resolve_api_key(provider, req.api_key)

    if provider == "anthropic":
        base_url = req.api_base or DEFAULT_ANTHROPIC_BASE
        url = f"{base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        data = await _post_json(client, url, headers, payload)
        output = _extract_anthropic_text(data)
    else:
        if provider == "openrouter":
            base_url = req.api_base or DEFAULT_OPENROUTER_BASE
        else:
            base_url = req.api_base or DEFAULT_OPENAI_BASE
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        if provider == "openrouter":
            headers["http-referer"] = "https://github.com"
            headers["x-title"] = "llm-context-fmt"
        payload = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt},
            ],
        }
        data = await _post_json(client, url, headers, payload)
        output = _extract_openai_text(data)

    refused = detect_refusal(output)
    return ResponseRecord(
        format_name=format_name,
        run_index=run_index,
        output=output,
        refused=refused,
        error=None,
    )


async def _call_with_retry(
    client: httpx.AsyncClient,
    req: RunRequest,
    format_name: str,
    run_index: int,
) -> ResponseRecord:
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            return await _single_call(client, req, format_name, run_index)
        except ProviderConfigurationError:
            raise
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            return ResponseRecord(
                format_name=format_name,
                run_index=run_index,
                output="",
                refused=False,
                error=f"HTTP {exc.response.status_code}: {body}",
            )
        except httpx.HTTPError as exc:
            if attempt < attempts:
                await asyncio.sleep(0.5)
                continue
            return ResponseRecord(
                format_name=format_name,
                run_index=run_index,
                output="",
                refused=False,
                error=f"Network error: {exc}",
            )
        except Exception as exc:  # defensive
            return ResponseRecord(
                format_name=format_name,
                run_index=run_index,
                output="",
                refused=False,
                error=f"Unexpected error: {exc}",
            )
    # Unreachable, but helps static analyzers.
    return ResponseRecord(
        format_name=format_name,
        run_index=run_index,
        output="",
        refused=False,
        error="Unknown error",
    )


async def run_all_formats(
    req: RunRequest,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, list[ResponseRecord]]:
    timeout = httpx.Timeout(60.0, connect=15.0)
    limits = httpx.Limits(max_connections=30, max_keepalive_connections=15)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = []
        for format_name in req.formats:
            for run_idx in range(req.runs):
                if progress_cb:
                    progress_cb(
                        f"Requesting format={format_name} run={run_idx + 1}/{req.runs}"
                    )
                task = _call_with_retry(client, req, format_name, run_idx)
                tasks.append(task)

        raw_results = await asyncio.gather(*tasks)

    by_format: dict[str, list[ResponseRecord]] = {fmt: [] for fmt in req.formats}
    for result in raw_results:
        by_format[result.format_name].append(result)
    for records in by_format.values():
        records.sort(key=lambda r: r.run_index)
    return by_format
