"""Hybrid BM25 + cosine retrieval over the prebuilt KB index."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend import config

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Okapi BM25 with an inverted index over a static corpus."""

    def __init__(self, corpus: list[list[str]], k1: float = _K1, b: float = _B) -> None:
        self._k1 = k1
        self._n = len(corpus)

        lengths = np.array([len(doc) for doc in corpus], dtype=np.float32)
        avg = float(lengths.mean()) if self._n else 0.0
        ratio = lengths / avg if avg else np.zeros_like(lengths)
        self._len_norm = (k1 * (1.0 - b + b * ratio)).astype(np.float32)

        postings: dict[str, dict[int, int]] = {}
        for doc_id, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                postings.setdefault(term, {})[doc_id] = tf

        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        for term, entry in postings.items():
            ids = np.fromiter(entry.keys(), dtype=np.int32, count=len(entry))
            tfs = np.fromiter(entry.values(), dtype=np.float32, count=len(entry))
            self._postings[term] = (ids, tfs)
            df = len(entry)
            self._idf[term] = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def scores(self, tokens: list[str]) -> np.ndarray:
        """BM25 score per document for the query tokens."""
        out = np.zeros(self._n, dtype=np.float32)
        for term in tokens:
            entry = self._postings.get(term)
            if entry is None:
                continue
            ids, tfs = entry
            out[ids] += self._idf[term] * (
                (tfs * (self._k1 + 1.0)) / (tfs + self._len_norm[ids])
            )
        return out


@lru_cache(maxsize=1)
def _get_encoder():
    """Load the sentence-transformers model once (lazy import)."""
    from sentence_transformers import SentenceTransformer

    # Use local model path – do NOT download from the internet.
    model_path = Path(config.EMBED_MODEL)
    if not model_path.exists():
        raise RuntimeError(f"Local embedding model not found: {model_path}")

    return SentenceTransformer(
        str(model_path),
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def _get_index() -> tuple[list[dict[str, Any]], np.ndarray, _BM25]:
    """Load chunks, normalized vectors and the BM25 index from KB_INDEX."""
    if not config.kb_is_built():
        raise ValueError(
            f"kb index missing at {config.KB_INDEX}; "
            "run: python -m backend.tools.build_kb"
        )

    with np.load(config.KB_INDEX, allow_pickle=True) as data:
        vectors = data["vectors"].astype(np.float32)
        chunks = json.loads(str(data["chunks"].item()))
        index_model = str(data["model"].item())

    if index_model != config.EMBED_MODEL:
        raise ValueError(
            f"kb index was built with '{index_model}' but EMBED_MODEL is "
            f"'{config.EMBED_MODEL}'; rebuild the index"
        )

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"corrupt index: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
        )

    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{chunk['source']}#{i}"

    bm25 = _BM25([_tokenize(chunk["text"]) for chunk in chunks])
    return chunks, vectors, bm25


def _rrf(scores: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank fusion weights from raw scores."""
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return 1.0 / (k + ranks)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return top_k chunks ranked by RRF, each with raw bm25 and cosine scores."""
    if not query.strip():
        raise ValueError("empty query")

    chunks, vectors, bm25 = _get_index()
    limit = min(top_k or config.KB_TOP_K, len(chunks))

    lexical = bm25.scores(_tokenize(query))
    query_vec = _get_encoder().encode(
        query,
        normalize_embeddings=True,
    ).astype(np.float32)
    cosine = vectors @ query_vec

    fused = _rrf(lexical, config.KB_RRF_K) + _rrf(cosine, config.KB_RRF_K)
    top = np.argsort(-fused)[:limit]

    return [
        {
            **chunks[int(i)],
            "cosine": round(float(cosine[i]), 6),
            "bm25": round(float(lexical[i]), 6),
            "fused": round(float(fused[i]), 6),
        }
        for i in top
    ]


def is_grounded(hits: list[dict[str, Any]]) -> bool:
    """True if the best hit's cosine reaches KB_MIN_COSINE."""
    best = 0.0
    for hit in hits:
        try:
            value = float(hit.get("cosine", 0.0))
        except (TypeError, ValueError):
            continue
        if value == value and value > best:  # skip NaN
            best = value
    return best >= config.KB_MIN_COSINE