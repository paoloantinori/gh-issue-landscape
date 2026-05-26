"""Text embedding strategies for issue content.

Supports three backends:
- ``local``  -- sentence-transformers all-MiniLM-L6-v2 (384-dim, runs on CPU)
- ``api``    -- any OpenAI-compatible ``/v1/embeddings`` endpoint
- ``openai`` -- shortcut for the official OpenAI API with text-embedding-3-small
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_API_BATCH_SIZE = 100
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
_API_DEFAULT_MODEL = "text-embedding-ada-002"

_local_model = None


def _get_local_model():
    """Return a cached SentenceTransformer model instance."""
    global _local_model  # noqa: PLW0603
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading local embedding model: %s", _LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
    return _local_model


def _compute_content_hash(texts: list[str]) -> str:
    """Streaming SHA-256 over texts (avoids building a giant intermediate string)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_paths(output_dir: str) -> tuple[Path, Path]:
    """Return (embeddings_npy_path, hash_txt_path)."""
    data_dir = Path(output_dir) / "data"
    return data_dir / "embeddings.npy", data_dir / "embeddings_hash.txt"


def _try_load_cache(content_hash: str, output_dir: str) -> np.ndarray | None:
    """Load cached embeddings if the content hash matches."""
    npy_path, hash_path = _cache_paths(output_dir)
    if not npy_path.exists() or not hash_path.exists():
        return None
    stored_hash = hash_path.read_text(encoding="utf-8").strip()
    if stored_hash != content_hash:
        log.info("Content hash mismatch — recomputing embeddings.")
        return None
    log.info("Loading cached embeddings from %s", npy_path)
    return np.load(npy_path)


def _save_cache(embeddings: np.ndarray, content_hash: str, output_dir: str) -> None:
    """Persist embeddings and content hash to disk."""
    npy_path, hash_path = _cache_paths(output_dir)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    hash_path.write_text(content_hash + "\n", encoding="utf-8")


def _embed_local(texts: list[str]) -> np.ndarray:
    """Embed texts with the local sentence-transformers model."""
    model = _get_local_model()
    log.info("Encoding %d texts with %s…", len(texts), _LOCAL_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings)


def _embed_api(
    texts: list[str],
    api_url: str,
    api_key: str,
    model: str = _API_DEFAULT_MODEL,
) -> np.ndarray:
    """Embed texts via an OpenAI-compatible /v1/embeddings endpoint."""
    import openai

    client = openai.OpenAI(base_url=api_url, api_key=api_key)

    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // _API_BATCH_SIZE)  # ceil division
    log.info(
        "Embedding %d texts via API (%s), %d batch(es)…",
        len(texts),
        api_url,
        total_batches,
    )

    for batch_idx in range(0, len(texts), _API_BATCH_SIZE):
        batch = texts[batch_idx : batch_idx + _API_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return np.asarray(all_embeddings, dtype=np.float32)


def embed(
    texts: list[str],
    backend: str = "local",
    output_dir: str = "./output",
    api_url: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Compute or load cached embeddings for *texts*.

    Returns array of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    content_hash = _compute_content_hash(texts)
    cached = _try_load_cache(content_hash, output_dir)
    if cached is not None:
        return cached

    backend = backend.lower()

    if backend == "local":
        embeddings = _embed_local(texts)

    elif backend == "api":
        if not api_url:
            raise ValueError("The 'api' backend requires --api-url to be set.")
        if not api_key:
            raise ValueError("The 'api' backend requires --api-key to be set.")
        embeddings = _embed_api(texts, api_url=api_url, api_key=api_key)

    elif backend == "openai":
        if not api_key:
            raise ValueError(
                "The 'openai' backend requires --api-key (or OPENAI_API_KEY)."
            )
        embeddings = _embed_api(
            texts,
            api_url="https://api.openai.com/v1",
            api_key=api_key,
            model=_OPENAI_DEFAULT_MODEL,
        )

    else:
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            "Choose from: local, api, openai."
        )

    _save_cache(embeddings, content_hash, output_dir)
    return embeddings
