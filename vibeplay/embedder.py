"""Embedding pipeline for vibe sentences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import get_settings
from .io_utils import read_vibes_csv, write_csv
from .logging_utils import configure_logging, progress_iter, task_logger
from .llm_vibes import chunked


def _unit_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each embedding vector to unit length, guarding against zero norms."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _save_ids(ids: Sequence[str], path: Path) -> None:
    """Persist the ordered track identifiers alongside the index artifacts."""

    with path.open("w", encoding="utf-8") as handle:
        for track_id in ids:
            json.dump({"Id": track_id}, handle, ensure_ascii=False)
            handle.write("\n")


def embed_vibes(*, vibes_path: Path, index_dir: Path, model: SentenceTransformer | None = None) -> None:
    """Compute embeddings and build a FAISS index for vibe sentences."""

    settings = get_settings()
    logger = configure_logging()
    index_dir.mkdir(parents=True, exist_ok=True)

    frame = read_vibes_csv(vibes_path)
    sentences = frame["VibeSentence"].tolist()
    track_ids = frame["Id"].tolist()

    transformer: SentenceTransformer = (
        model if model is not None else SentenceTransformer(settings.embedding_model)
    )
    embeddings: list[np.ndarray] = []

    batches = list(chunked(sentences, settings.batch_size))
    total_batches = max(1, len(batches))

    with task_logger(logger, "embed vibes"):
        for batch in progress_iter(batches, desc="Embedding batches", total=total_batches):
            vectors = transformer.encode(
                list(batch),
                batch_size=settings.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            embeddings.append(vectors.astype("float32"))

    if embeddings:
        matrix = np.vstack(embeddings)
    else:
        dimension = transformer.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("Embedding dimension is undefined for the selected model")
        matrix = np.empty((0, dimension), dtype="float32")

    matrix = _unit_normalize(matrix)

    faiss_index = faiss.IndexFlatIP(matrix.shape[1])
    if matrix.size:
        faiss_index.add(matrix)

    np.save(index_dir / "embeddings.npy", matrix)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    _save_ids(track_ids, index_dir / "ids.jsonl")
    write_csv(frame, index_dir / "vibes.join.csv")


__all__ = ["embed_vibes"]

