from pathlib import Path

import pandas as pd
import pytest

from vibeplay.io_utils import (
    CSVValidationError,
    merge_library_with_vibes,
    read_library_csv,
    validate_playlist_columns,
)


def test_read_library_csv_valid(sample_library_path: Path) -> None:
    frame = read_library_csv(sample_library_path)
    assert list(frame.columns) == [
        "Added At",
        "Track Name",
        "Artists",
        "Album",
        "Id",
    ]
    assert len(frame) == 5


def test_read_library_csv_invalid_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Track Name,Artists,Album,Id\nX,Y,Z,1\n", encoding="utf-8")
    with pytest.raises(CSVValidationError):
        read_library_csv(bad)


def test_merge_preserves_order(sample_library_path: Path) -> None:
    library = read_library_csv(sample_library_path)
    vibes = pd.DataFrame(
        {
            "Id": ["track-1", "track-3", "track-5"],
            "VibeSentence": ["v1", "v3", "v5"],
            "model": ["m", "m", "m"],
            "created_at": ["2025", "2025", "2025"],
            "prompt_tokens": [1, 1, 1],
            "completion_tokens": [1, 1, 1],
        }
    )
    merged = merge_library_with_vibes(library, vibes)
    assert list(merged["Id"]) == list(library["Id"])  # order preserved
    assert merged.loc[merged["Id"] == "track-3", "VibeSentence"].iloc[0] == "v3"


def test_validate_playlist_columns_accepts_valid(sample_library_path: Path) -> None:
    playlist = read_library_csv(sample_library_path).copy()
    playlist["VibeSentence"] = "vibe"
    playlist["Similarity"] = 0.9
    validate_playlist_columns(playlist)


def test_validate_playlist_columns_rejects_invalid(sample_library_path: Path) -> None:
    playlist = read_library_csv(sample_library_path).copy()
    playlist["Similarity"] = 0.9
    with pytest.raises(CSVValidationError):
        validate_playlist_columns(playlist)
