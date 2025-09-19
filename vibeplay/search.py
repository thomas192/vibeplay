"""Similarity search for vibe playlists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pandas as pd

from .config import LIBRARY_COLUMNS, PLAYLIST_EXTRA_COLUMNS
from .io_utils import (
    merge_library_with_vibes,
    read_library_csv,
    read_vibes_csv,
    validate_playlist_columns,
    write_csv,
)
from .logging_utils import configure_logging, task_logger


def _load_ids(path: Path) -> list[str]:
    """Return the ordered list of track identifiers stored in ids.jsonl."""

    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            ids.append(str(payload["Id"]))
    return ids


def _apply_artist_cap(
    candidates: Sequence[tuple[str, float]],
    joined: pd.DataFrame,
    cap: int,
) -> list[tuple[str, float]]:
    """Drop candidates once their artist has already met the configured limit."""

    if cap <= 0:
        return list(candidates)
    counts: dict[str, int] = {}
    accepted: list[tuple[str, float]] = []
    artist_by_id = joined.set_index("Id")["Artists"].to_dict()
    for track_id, score in candidates:
        artist = artist_by_id.get(track_id)
        if artist is None:
            continue
        taken = counts.get(artist, 0)
        if taken >= cap:
            continue
        counts[artist] = taken + 1
        accepted.append((track_id, score))
    return accepted


def _build_playlist_frame(
    *,
    ranked: Sequence[tuple[str, float]],
    joined: pd.DataFrame,
) -> pd.DataFrame:
    """Compose the playlist rows in the required column order from ranked ids."""

    expected_columns = list(LIBRARY_COLUMNS) + list(PLAYLIST_EXTRA_COLUMNS)
    if not ranked:
        return pd.DataFrame(columns=expected_columns)
    order = [track_id for track_id, _ in ranked]
    subset = joined.set_index("Id").loc[order].reset_index()
    ordered = subset[[
        "Added At",
        "Track Name",
        "Artists",
        "Album",
        "Id",
        "VibeSentence",
    ]].copy()
    ordered["Similarity"] = [score for _, score in ranked]
    validate_playlist_columns(ordered)
    return ordered


def make_playlist(
    *,
    seed_id: str,
    library_path: Path,
    index_dir: Path,
    output_path: Path,
    k: int,
    artist_cap: int,
) -> None:
    """Search for the tracks most similar to the seed and write the playlist CSV."""

    logger = configure_logging()
    ids_path = index_dir / "ids.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    index_path = index_dir / "faiss.index"
    vibes_join_path = index_dir / "vibes.join.csv"

    for required in (ids_path, embeddings_path, index_path, vibes_join_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required index artifact: {required}")

    library = read_library_csv(library_path)
    vibes = read_vibes_csv(vibes_join_path)
    joined = merge_library_with_vibes(library, vibes)

    ids = _load_ids(ids_path)
    embedding_matrix = np.load(embeddings_path).astype("float32")
    faiss_index = faiss.read_index(str(index_path))

    id_to_pos = {track_id: idx for idx, track_id in enumerate(ids)}
    if seed_id not in id_to_pos:
        raise ValueError(f"Seed id {seed_id} not present in index")

    seed_vector = embedding_matrix[id_to_pos[seed_id]][None, :]
    search_k = min(k + 1, len(ids))
    distances, indices = faiss_index.search(seed_vector, search_k)
    raw_candidates: list[tuple[str, float]] = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        track_id = ids[idx]
        if track_id == seed_id:
            continue
        raw_candidates.append((track_id, float(max(0.0, min(1.0, score)))))

    raw_candidates.sort(key=lambda item: item[1], reverse=True)
    filtered = _apply_artist_cap(raw_candidates, joined, artist_cap)
    ranked = filtered[:k]
    ranked_with_seed = [(seed_id, 1.0)] + ranked

    with task_logger(logger, "make playlist"):
        playlist = _build_playlist_frame(ranked=ranked_with_seed, joined=joined)
        write_csv(playlist, output_path)


__all__ = ["make_playlist"]
