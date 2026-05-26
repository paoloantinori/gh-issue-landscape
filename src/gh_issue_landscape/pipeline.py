"""BERTopic-based topic modeling pipeline.

Takes pre-computed embeddings, performs dimensionality reduction with UMAP,
clusters with HDBSCAN, and produces topic assignments plus 2D coordinates
suitable for interactive visualization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its"
    " me my no not of on or our she that the their them they this to was we"
    " were what when where which who will with you your"
    " null none true false add implement support update create remove use set get".split()
)


def _to_int(val) -> int:
    """Extract a plain int from a numpy/pandas scalar or pass through."""
    return int(val.item() if hasattr(val, "item") else val)


def _get_keywords(topic_model: BERTopic, tid: int, limit: int | None = None) -> list[str]:
    """Extract keyword strings for a topic, with optional truncation."""
    topic_words = topic_model.get_topic(tid)
    if not topic_words or not isinstance(topic_words, list):
        return []
    words = [w for w, _ in topic_words]
    return words[:limit] if limit else words


def clean_topic_label(keywords: list[str], max_words: int = 3) -> str:
    """Turn raw c-TF-IDF keywords into a readable topic label.

    Filters stopwords, deduplicates, title-cases, and joins with `` / ``.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for w in keywords:
        w_lower = w.lower().strip()
        if not w_lower or w_lower in seen or len(w_lower) > 25:
            continue
        seen.add(w_lower)
        digit_ratio = sum(c.isdigit() for c in w_lower) / len(w_lower)
        if w_lower not in _STOPWORDS and digit_ratio < 0.4:
            clean.append(w_lower.title())
        if len(clean) >= max_words:
            break

    if not clean:
        clean = [w.strip().title() for w in keywords[:max_words] if w.strip()]

    return " / ".join(clean) if clean else "Misc"


@dataclass
class PipelineResult:
    """Fitted BERTopic model with topic assignments and 2D coordinates."""

    topic_model: BERTopic
    topics: list[int]
    topic_info: pd.DataFrame
    umap_2d: np.ndarray
    topic_labels: dict[int, str] = field(default_factory=dict)
    umap_3d: np.ndarray | None = None

    def topic_to_doc_indices(self) -> dict[int, list[int]]:
        """Map each topic_id to its list of document indices."""
        mapping: dict[int, list[int]] = {}
        for doc_idx, tid in enumerate(self.topics):
            mapping.setdefault(tid, []).append(doc_idx)
        return mapping

    def get_label(self, tid: int) -> str:
        """Get the cleaned label for a topic, falling back to 'Topic N'."""
        return self.topic_labels.get(tid, f"Topic {tid}")


def _save_umap_coordinates(umap_2d: np.ndarray, data_dir: Path) -> None:
    """Save 2-D UMAP coordinates as a NumPy binary file."""
    out_path = data_dir / "umap-2d.npy"
    np.save(out_path, umap_2d)
    log.info("Saved 2-D UMAP coordinates → %s", out_path)


def _save_topics_json(result: PipelineResult, data_dir: Path) -> None:
    """Persist a structured JSON summary of discovered topics."""
    out_path = data_dir / "topics.json"
    topic_to_docs = result.topic_to_doc_indices()

    records: list[dict] = []
    for _, row in result.topic_info.iterrows():
        tid = _to_int(row["Topic"])
        keywords = _get_keywords(result.topic_model, tid)
        label = result.get_label(tid)

        records.append(
            {
                "topic_id": tid,
                "label": label,
                "count": _to_int(row.get("Count", len(topic_to_docs.get(tid, [])))),
                "document_indices": topic_to_docs.get(tid, []),
                "keywords": keywords,
            }
        )

    out_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Saved topic metadata → %s", out_path)


def _save_model(topic_model: BERTopic, data_dir: Path) -> None:
    """Save the BERTopic model, falling back to pickle if safetensors fails."""
    model_dir = data_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir)

    try:
        topic_model.save(
            model_path,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
        log.info("Saved BERTopic model (safetensors) → %s", model_path)
    except Exception:
        try:
            topic_model.save(
                model_path,
                serialization="pickle",
                save_ctfidf=True,
                save_embedding_model=False,
            )
            log.info("Saved BERTopic model (pickle) → %s", model_path)
        except Exception as exc:
            log.error("Model save failed: %s", exc)


def run_pipeline(
    texts: list[str],
    embeddings: np.ndarray,
    output_dir: str = "./output",
    reduce_3d: bool = False,
) -> PipelineResult:
    """Run the BERTopic topic-modeling pipeline on pre-computed embeddings."""
    if len(texts) != embeddings.shape[0]:
        raise ValueError(
            f"Length mismatch: {len(texts)} texts vs "
            f"{embeddings.shape[0]} embedding rows."
        )

    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Configuring UMAP and HDBSCAN…")

    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        prediction_data=True,
    )

    log.info(
        "Fitting BERTopic on %d documents (%d-dim embeddings).",
        len(texts),
        embeddings.shape[1],
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=None,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = topics.count(-1)

    # Build clean topic labels and register them on the model
    topic_labels: dict[int, str] = {}
    for tid_val in topic_info["Topic"]:
        tid = _to_int(tid_val)
        if tid == -1:
            topic_labels[tid] = "Uncategorized"
        else:
            topic_labels[tid] = clean_topic_label(_get_keywords(topic_model, tid))
    topic_model.set_topic_labels(topic_labels)

    log.info("BERTopic: %d topics, %d outliers.", n_topics, n_outliers)

    # Reuse the UMAP embedding computed during fit (avoids a redundant transform)
    umap_2d = np.asarray(topic_model.umap_model.embedding_, dtype=np.float32)
    log.info("2-D UMAP projection shape: %s", umap_2d.shape)

    # Optional 3-D UMAP reduction on the original embeddings
    umap_3d: np.ndarray | None = None
    if reduce_3d:
        log.info("Computing 3-D UMAP projection…")
        umap_3d_model = UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        umap_3d = umap_3d_model.fit_transform(embeddings)
        umap_3d = np.asarray(umap_3d, dtype=np.float32)
        log.info("3-D UMAP projection shape: %s", umap_3d.shape)

    log.info("Saving artifacts…")
    result = PipelineResult(
        topic_model=topic_model,
        topics=topics,
        topic_info=topic_info,
        umap_2d=umap_2d,
        topic_labels=topic_labels,
        umap_3d=umap_3d,
    )
    _save_umap_coordinates(umap_2d, data_dir)
    if umap_3d is not None:
        umap_3d_path = data_dir / "umap-3d.npy"
        np.save(umap_3d_path, umap_3d)
        log.info("Saved 3-D UMAP coordinates → %s", umap_3d_path)
    _save_topics_json(result, data_dir)
    _save_model(topic_model, data_dir)

    return result
