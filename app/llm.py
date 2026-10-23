"""Thin Gemini client: retries with backoff, model fallback chain, JSON repair."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Callable, Optional

import requests

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = (429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def model_chain() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    fallbacks = [
        model.strip()
        for model in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
        if model.strip()
    ]
    return list(dict.fromkeys([primary, *fallbacks]))


def generate(
    prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    images: Optional[list[tuple[str, bytes]]] = None,
    on_retry: Optional[Callable[[str], None]] = None,
    timeout: int = 90,
) -> str:
    """Call Gemini. Retries transient errors, then falls back to the next model."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError("GEMINI_API_KEY is not configured")
    parts: list[dict] = [{"text": prompt}]
    for mime, data in images or []:
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    cfg: dict = {"temperature": temperature}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"contents": [{"parts": parts}], "generationConfig": cfg}

    note = on_retry or (lambda _m: None)
    last = "unknown error"
    for model in model_chain():
        for attempt in range(2):
            try:
                r = requests.post(
                    API.format(model=model),
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json=body,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last = f"{model}: network error ({type(exc).__name__})"
                time.sleep(1 + attempt)
                continue
            if r.status_code == 200:
                try:
                    cand = r.json()["candidates"][0]
                    text = "".join(p.get("text", "") for p in cand["content"]["parts"]).strip()
                except (KeyError, IndexError, ValueError):
                    text = ""
                if text:
                    return text
                last = f"{model}: empty or blocked response"
                break  # try the next model
            if r.status_code == 429:
                # Avoid repeatedly waiting on a rate-limited model.
                raise LLMError(
                    "Gemini returned HTTP 429. Check API quota and rate limits."
                )
            if r.status_code in RETRYABLE:
                last = f"{model}: HTTP {r.status_code}"
                if attempt == 1:
                    break
                delay = 2
                note(f"{last}; retrying in {delay}s")
                time.sleep(delay)
                continue
            if r.status_code == 404:
                last = f"{model}: model not available"
                break
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
        note(f"Switching Gemini model after: {last}")
    raise LLMError(f"All Gemini models failed ({last})")


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except ValueError:
                continue
    raise ValueError("Model did not return parseable JSON")


def generate_json(prompt: str, on_retry: Optional[Callable[[str], None]] = None, **kw):
    """Generate JSON; if parsing fails, ask the model once to repair its own output."""
    text = generate(prompt, json_mode=True, on_retry=on_retry, **kw)
    try:
        return parse_json(text)
    except ValueError:
        if on_retry:
            on_retry("Malformed JSON from model; asking it to repair the output")
        fixed = generate(
            "The following was meant to be valid JSON but does not parse. "
            "Return ONLY the corrected JSON, no commentary:\n" + text[:12000],
            temperature=0,
            json_mode=True,
            on_retry=on_retry,
        )
        return parse_json(fixed)
