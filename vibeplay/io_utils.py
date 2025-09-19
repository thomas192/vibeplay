"""CSV input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import LIBRARY_COLUMNS, PLAYLIST_EXTRA_COLUMNS, VIBES_COLUMNS


class CSVValidationError(ValueError):
    """Raised when CSV headers are invalid."""


def _validate_headers(frame: pd.DataFrame, expected: Iterable[str]) -> None:
    """Ensure the dataframe columns exactly match the expected header sequence."""

    expected_list = list(expected)
    if list(frame.columns) != expected_list:
        raise CSVValidationError(
            f"Expected columns {expected_list} but received {list(frame.columns)}"
        )


def read_library_csv(path: Path) -> pd.DataFrame:
    """Read the music library CSV and validate headers."""

    frame = pd.read_csv(path, dtype=str)
    _validate_headers(frame, LIBRARY_COLUMNS)
    return frame


def read_vibes_csv(path: Path) -> pd.DataFrame:
    """Read vibes CSV and validate headers."""

    frame = pd.read_csv(path, dtype={"Id": str, "VibeSentence": str})
    _validate_headers(frame, VIBES_COLUMNS)
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe to CSV without the index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def merge_library_with_vibes(library: pd.DataFrame, vibes: pd.DataFrame) -> pd.DataFrame:
    """Join library rows with their vibe sentences while preserving original order."""

    merged = library.merge(vibes[["Id", "VibeSentence"]], on="Id", how="left")
    return merged


def validate_playlist_columns(frame: pd.DataFrame) -> None:
    """Ensure playlist columns follow the required schema."""

    expected = list(LIBRARY_COLUMNS) + list(PLAYLIST_EXTRA_COLUMNS)
    _validate_headers(frame, expected)
