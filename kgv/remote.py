"""Frontier LLMs (Claude, GPT) via an OpenAI-compatible relay.

Credentials come from the gitignored .env file (RELAY_BASE_URL, RELAY_API_KEY). Text embeddings stay local
(bge, see textenc.py) so every experiment uses the same encoder.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k in env})
    return env


ENV = _env()


def _post(url, key, payload, timeout=600, retries=6):
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}",
                                                                 "Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except urllib.error.HTTPError as ex:
            if ex.code == 429 and attempt >= 2:
                raise  # quota exhausted: fail fast instead of stalling every worker for hours
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as ex:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))
            last = ex  # noqa: F841


def chat(model: str, prompt: str, max_tokens: int = 64) -> str:
    """One completion. Claude models think before answering, so they get a larger budget; an empty answer cut
    off by the length limit is retried with a bigger one."""
    budget = max(max_tokens, 2048) if model.startswith("claude") else max_tokens
    for _ in range(3):
        d = _post(f"{ENV['RELAY_BASE_URL']}/chat/completions", ENV["RELAY_API_KEY"],
                  {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": budget,
                   "temperature": 0})
        choice = d["choices"][0]
        text = choice["message"].get("content") or ""
        if text.strip() or choice.get("finish_reason") != "length":
            return text
        budget *= 4
    return text


def chat_many(model: str, prompts: list[str], max_tokens: int = 64, workers: int = 16) -> list[str]:
    def one(p):
        try:
            return chat(model, p, max_tokens)
        except Exception as ex:  # noqa: BLE001
            return f"<error {type(ex).__name__}>"
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))
