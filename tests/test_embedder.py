from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd

from vibeplay.embedder import embed_vibes


class DummyModel:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        sentences: List[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(list(sentences))
        vectors = np.zeros((len(sentences), 4), dtype="float32")
        for idx in range(len(sentences)):
            vectors[idx, idx % 4] = 1.0
        return vectors


def test_embed_vibes_persists_artifacts(tmp_path: Path, sample_vibes_frame: pd.DataFrame) -> None:
    vibes_path = tmp_path / "vibes.csv"
    sample_vibes_frame.to_csv(vibes_path, index=False)
    index_dir = tmp_path / "index"

    model = DummyModel()
    embed_vibes(vibes_path=vibes_path, index_dir=index_dir, model=model)

    embeddings = np.load(index_dir / "embeddings.npy")
    assert embeddings.shape[0] == len(sample_vibes_frame)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)

    ids = [line.strip() for line in (index_dir / "ids.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(sample_vibes_frame)

    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    assert faiss_index.ntotal == len(sample_vibes_frame)

    joined = pd.read_csv(index_dir / "vibes.join.csv")
    assert list(joined.columns) == list(sample_vibes_frame.columns)
    assert len(model.calls) >= 1
