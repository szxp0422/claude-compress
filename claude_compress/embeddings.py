"""Embedding provider used by the dedup and eigencontext stages.

Real semantic behaviour requires a real embedding model. We support, in order:
  1. sentence-transformers (local model) -- the intended production path.
  2. A deterministic hashing embedding (bag-of-character-ngrams) -- a degraded
     fallback so the pipeline RUNS with zero heavy deps / no model download.

The fallback captures lexical overlap, not true semantics. It is good enough to
catch near-duplicate / boilerplate text, which is the high-value, low-risk case.
It is NOT good enough to trust for aggressive eigencontext pruning -- which is
one more reason that stage defaults to off.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_MODEL = None
_MODE = "hash"


def _try_load_st(model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL, _MODE
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(model_name)
        _MODE = "sentence-transformers"
    except Exception:
        _MODEL = None
        _MODE = "hash"


_try_load_st()

_DIM = 256
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def mode() -> str:
    return _MODE


def _hash_embed(text: str) -> np.ndarray:
    """Cheap deterministic embedding: hashed char-3gram + word counts."""
    vec = np.zeros(_DIM, dtype=np.float32)
    toks = _TOKEN_RE.findall(text.lower())
    grams = toks + [text[i : i + 3] for i in range(0, max(0, len(text) - 2))]
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed(texts: List[str]) -> np.ndarray:
    """Return an (n, dim) array of L2-normalised embeddings."""
    if not texts:
        return np.zeros((0, _DIM), dtype=np.float32)
    if _MODEL is not None:
        arr = _MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype=np.float32)
    return np.vstack([_hash_embed(t) for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"embedding dimension mismatch: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))
