"""Sentence encoder (bge-large) served by OpenRouter, with an on-disk cache.

bge is the one text encoder used throughout the project. The hosted baai/bge-large-en-v1.5 returns the same
vectors as the local checkpoint (cosine 1.00000 on a spot check), so earlier local results stay comparable.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .remote import ENV

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "embed_cache"
HOSTED = {"BAAI/bge-large-en-v1.5": "baai/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5": "baai/bge-base-en-v1.5"}


def _remote(chunk, model):
    body = json.dumps({"model": HOSTED[model], "input": chunk}).encode()
    for attempt in range(8):
        try:
            req = urllib.request.Request(f"{ENV['OPENROUTER_BASE_URL']}/embeddings", data=body, headers={
                "Authorization": f"Bearer {ENV['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return [np.asarray(x["embedding"], np.float32) for x in sorted(d["data"], key=lambda x: x["index"])]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError):
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt))


def encode(texts: list[str], model: str = "BAAI/bge-large-en-v1.5", batch: int = 128, workers: int = 6) -> np.ndarray:
    """L2-normalised embeddings, cached per (model, text)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / f"{model.replace('/', '_')}.npz"
    cache = dict(np.load(store)) if store.exists() else {}
    key = lambda t: hashlib.sha1(t.encode()).hexdigest()
    missing = sorted({t for t in texts if key(t) not in cache})
    if missing:
        chunks = [missing[i : i + batch] for i in range(0, len(missing), batch)]
        with ThreadPoolExecutor(workers) as ex:
            for chunk, vecs in zip(chunks, ex.map(lambda c: _remote(c, model), chunks)):
                for t, v in zip(chunk, vecs):
                    cache[key(t)] = v / np.linalg.norm(v)
        np.savez(store, **cache)
    return np.stack([cache[key(t)] for t in texts])
