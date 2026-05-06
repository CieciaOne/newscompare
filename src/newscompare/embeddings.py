"""Local embeddings via sentence-transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazy-load and return the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Return (n, dim) array of embeddings. Empty list returns (0, dim)."""
    if not texts:
        m = get_model(model_name)
        return np.zeros((0, m.get_sentence_embedding_dimension()))
    model = get_model(model_name)
    # Larger batches speed up CPU/MPS/GPU; sentence-transformers batches internally.
    return model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Single pair cosine similarity. a, b are 1d."""
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(an, bn))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """(n, dim) -> (n, n) pairwise cosine similarity."""
    if embeddings.size == 0:
        return np.array([]).reshape(0, 0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    unit = embeddings / norms
    return np.dot(unit, unit.T)
