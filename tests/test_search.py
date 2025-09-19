from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from vibeplay.search import make_playlist


def _write_index_artifacts(
    *,
    index_dir: Path,
    embeddings: np.ndarray,
    ids: list[str],
    vibes_frame: pd.DataFrame,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    with (index_dir / "ids.jsonl").open("w", encoding="utf-8") as handle:
        for track_id in ids:
            handle.write(f"{{\"Id\": \"{track_id}\"}}\n")
    vibes_frame.to_csv(index_dir / "vibes.join.csv", index=False)


def test_make_playlist_orders_by_similarity(
    tmp_path: Path,
    sample_library_path: Path,
    sample_vibes_frame: pd.DataFrame,
    embedding_matrix: np.ndarray,
    sample_ids: list[str],
) -> None:
    index_dir = tmp_path / "index"
    _write_index_artifacts(
        index_dir=index_dir,
        embeddings=embedding_matrix,
        ids=sample_ids,
        vibes_frame=sample_vibes_frame,
    )

    output_path = tmp_path / "playlist.csv"
    make_playlist(
        seed_id="track-1",
        library_path=sample_library_path,
        index_dir=index_dir,
        output_path=output_path,
        k=3,
        artist_cap=2,
    )

    playlist = pd.read_csv(output_path)
    assert list(playlist.columns) == [
        "Added At",
        "Track Name",
        "Artists",
        "Album",
        "Id",
        "VibeSentence",
        "Similarity",
    ]
    assert playlist.iloc[0]["Id"] == "track-1"
    assert playlist.iloc[0]["Similarity"] == pytest.approx(1.0)
    others = playlist.iloc[1:]
    assert list(others["Id"])[:2] == ["track-2", "track-5"]
    assert others["Similarity"].iloc[0] >= others["Similarity"].iloc[1]


def test_make_playlist_honors_artist_cap(
    tmp_path: Path,
    sample_library_path: Path,
    sample_vibes_frame: pd.DataFrame,
    embedding_matrix: np.ndarray,
    sample_ids: list[str],
) -> None:
    index_dir = tmp_path / "index"
    _write_index_artifacts(
        index_dir=index_dir,
        embeddings=embedding_matrix,
        ids=sample_ids,
        vibes_frame=sample_vibes_frame,
    )

    output_path = tmp_path / "playlist.csv"
    make_playlist(
        seed_id="track-5",
        library_path=sample_library_path,
        index_dir=index_dir,
        output_path=output_path,
        k=5,
        artist_cap=1,
    )

    playlist = pd.read_csv(output_path)
    assert playlist.iloc[0]["Id"] == "track-5"
    assert playlist.iloc[0]["Similarity"] == pytest.approx(1.0)
    counts = playlist.iloc[1:].groupby("Artists").size().to_dict()
    assert all(count <= 1 for count in counts.values())
    assert "track-1" not in playlist["Id"].tolist() or "track-3" not in playlist["Id"].tolist()
