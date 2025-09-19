from __future__ import annotations

from pathlib import Path
from typing import List
from types import SimpleNamespace

import pandas as pd
import pytest

from vibeplay.llm_vibes import (
    LLMError,
    LLMGeneration,
    RateLimiter,
    chunked,
    generate_vibes,
)


class StubClient:
    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate_prompt(self, prompt: str) -> LLMGeneration:
        self.prompts.append(prompt)
        return LLMGeneration(
            text=f"stub vibe {len(self.prompts)}",
            model="stub-model",
            prompt_tokens=50,
            completion_tokens=25,
        )


class FailingClient(StubClient):
    def generate_prompt(self, prompt: str) -> LLMGeneration:
        raise LLMError("simulated failure")


def _write_library(path: Path, rows: int) -> None:
    frame = pd.DataFrame(
        {
            "Added At": ["2025-01-01T00:00:00Z"] * rows,
            "Track Name": [f"Track {i}" for i in range(rows)],
            "Artists": ["Artist"] * rows,
            "Album": ["Album"] * rows,
            "Id": [f"track-{i}" for i in range(rows)],
        }
    )
    frame.to_csv(path, index=False)


def test_generate_vibes_uses_batch_size_of_25(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_path = tmp_path / "library.csv"
    _write_library(library_path, rows=55)
    output_path = tmp_path / "vibes.csv"

    batches: List[int] = []
    original_chunked = chunked

    def tracking_chunked(sequence, size):  # type: ignore[override]
        for batch in original_chunked(sequence, size):
            batches.append(len(batch))
            yield batch

    monkeypatch.setattr("vibeplay.llm_vibes.chunked", tracking_chunked)

    client = StubClient()
    generate_vibes(
        library_path=library_path,
        output_path=output_path,
        cache_dir=tmp_path / "cache",
        client=client,
    )

    assert all(size == 25 for size in batches[:-1])
    assert batches[-1] == 5  # 55 % 25 == 5
    assert len(client.prompts) == 55


def test_generate_vibes_uses_cache(tmp_path: Path) -> None:
    library_path = tmp_path / "library.csv"
    _write_library(library_path, rows=5)
    output_path = tmp_path / "vibes.csv"
    cache_dir = tmp_path / "cache"

    first_client = StubClient()
    generate_vibes(
        library_path=library_path,
        output_path=output_path,
        cache_dir=cache_dir,
        client=first_client,
    )
    assert len(first_client.prompts) == 5

    second_client = StubClient()
    generate_vibes(
        library_path=library_path,
        output_path=tmp_path / "vibes2.csv",
        cache_dir=cache_dir,
        client=second_client,
    )
    assert len(second_client.prompts) == 0  # cache satisfied all rows


def test_generate_vibes_writes_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dummy_settings = SimpleNamespace(
        gemini_api_key=None,
        cache_dir=tmp_path / "cache",
        batch_size=25,
        rpm=15,
        tpm=250_000,
        rpd=1_000,
        gemini_model="models/gemini-2.5-flash-lite",
        gemini_api_url="https://generativelanguage.googleapis.com/v1beta",
        llm_timeout_seconds=30,
    )
    monkeypatch.setattr("vibeplay.llm_vibes.get_settings", lambda: dummy_settings)
    library_path = tmp_path / "library.csv"
    _write_library(library_path, rows=3)
    output_path = tmp_path / "vibes.csv"
    errors_path = tmp_path / "errors.csv"

    with pytest.raises(LLMError):
        # No API key passed and no client, expect configuration error.
        generate_vibes(
            library_path=library_path,
            output_path=output_path,
            errors_path=errors_path,
            cache_dir=tmp_path / "cache",
        )


def test_generate_vibes_records_errors(tmp_path: Path) -> None:
    library_path = tmp_path / "library.csv"
    _write_library(library_path, rows=2)
    output_path = tmp_path / "vibes.csv"
    errors_path = tmp_path / "errors.csv"
    cache_dir = tmp_path / "cache"

    failing = FailingClient()
    generate_vibes(
        library_path=library_path,
        output_path=output_path,
        errors_path=errors_path,
        cache_dir=cache_dir,
        client=failing,
    )

    assert errors_path.exists()
    errors = pd.read_csv(errors_path)
    assert len(errors) == 2
    assert set(errors.columns) == {"Id", "error"}


def test_rate_limiter_waits_when_rpm_exceeded() -> None:
    state = {"now": 0.0, "sleeps": []}

    def now() -> float:
        return state["now"]

    def sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)
        state["now"] += seconds

    limiter = RateLimiter(rpm=1, tpm=100, rpd=5, now=now, sleep=sleep)
    limiter.acquire(10)
    limiter.acquire(10)  # second call should wait for the next minute

    assert pytest.approx(state["sleeps"][0], rel=1e-3) == 60.0


def test_rate_limiter_enforces_daily_cap() -> None:
    limiter = RateLimiter(rpm=10, tpm=1000, rpd=1)
    limiter.acquire(10)
    with pytest.raises(LLMError):
        limiter.acquire(10)



