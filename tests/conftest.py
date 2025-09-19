from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vibeplay.config import VIBES_COLUMNS


@pytest.fixture()
def sample_library_path() -> Path:
    return Path(__file__).parent / "fixtures" / "library_small.csv"


@pytest.fixture()
def sample_library_frame(sample_library_path: Path) -> pd.DataFrame:
    return pd.read_csv(sample_library_path, dtype=str)


@pytest.fixture()
def sample_vibes_frame(sample_library_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in sample_library_frame.itertuples(index=False):
        rows.append(
            {
                "Id": getattr(row, "Id"),
                "VibeSentence": f"vibe {getattr(row, 'Id')}",
                "model": "test-model",
                "created_at": "2025-01-01T00:00:00Z",
                "prompt_tokens": 10,
                "completion_tokens": 12,
            }
        )
    return pd.DataFrame(rows, columns=list(VIBES_COLUMNS))


@pytest.fixture()
def embedding_matrix() -> np.ndarray:
    # 5 unit vectors in 3 dimensions for deterministic similarity tests.
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.4, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.8, 0.0],
        ],
        dtype="float32",
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


@pytest.fixture()
def sample_ids(sample_library_frame: pd.DataFrame) -> list[str]:
    return sample_library_frame["Id"].tolist()
