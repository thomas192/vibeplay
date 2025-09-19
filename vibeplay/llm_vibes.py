"""Generate vibe sentences via Gemini with caching and rate limiting."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Protocol, Sequence, TypeVar, cast

import pandas as pd
import requests
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import VIBES_COLUMNS, get_settings
from .io_utils import read_library_csv, write_csv
from .logging_utils import configure_logging, progress_iter, task_logger

SYSTEM_PROMPT = (
    "You are a music-vibe tagger. Write one compact, listener-friendly sentence that captures a "
    "track\u2019s vibe using only the given metadata. Keep it 12\u201320 words, natural language, no hashtags. "
    "Include genre, mood, energy, and notable instrumentation/era when evident from title/artist/album "
    "alone. No speculation. \u2264 2 commas. Avoid mentioning track or artist names; focus on atmosphere, ambiance, tempo, and instrumentation."
)

USER_PROMPT_TEMPLATE = (
    "TRACK METADATA\n"
    "Track Name: {track_name}\n"
    "Artist(s): {artists}\n"
    "Album: {album}\n\n"
    "INSTRUCTIONS\n"
    "- One sentence only, 12\u201320 words.\n"
    "- If uncertain, stay generic but musical (e.g., \u201cmelancholic synth-pop ballad with mid-tempo pulse\u201d).\n"
    "- Do not invent languages, places, years, or producers.\n"
    "- Do not mention track or artist names; describe atmosphere, ambiance, tempo, and instrumentation.\n"
    "- Tone: descriptive, not promotional.\n"
    "Return ONLY the sentence."
)


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMTransientError(LLMError):
    """Raised for retryable LLM failures."""


@dataclass(slots=True)
class Track:
    """Simplified representation of a track row."""

    track_id: str
    track_name: str
    artists: str
    album: str


@dataclass(slots=True)
class LLMGeneration:
    """Normalized LLM response data."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """Protocol for an LLM client."""

    def generate_prompt(self, prompt: str) -> LLMGeneration:
        """Generate a single response."""


class RateLimiter:
    """Token bucket rate limiter for RPM/TPM/RPD constraints."""

    def __init__(
        self,
        *,
        rpm: int,
        tpm: int,
        rpd: int,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize counters and suppliers used to enforce rate limits."""

        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._now = now
        self._sleep = sleep
        self._minute_window_start = self._now()
        self._minute_requests = 0
        self._minute_tokens = 0
        self._current_day = date.today()
        self._daily_requests = 0

    def acquire(self, tokens: int) -> None:
        """Block until the per-minute and daily buckets allow the requested tokens."""

        requested_tokens = max(tokens, 1)
        while True:
            self._refresh_counters()
            if self._daily_requests >= self.rpd:
                raise LLMError("Daily request cap reached, aborting to respect RPD limit")
            within_rpm = self._minute_requests < self.rpm
            within_tpm = self._minute_tokens + requested_tokens <= self.tpm
            if within_rpm and within_tpm:
                self._minute_requests += 1
                self._minute_tokens += requested_tokens
                self._daily_requests += 1
                return
            wait_seconds = self._minute_window_start + 60 - self._now()
            if wait_seconds > 0:
                self._sleep(wait_seconds)
            else:
                self._sleep(0.05)

    def _refresh_counters(self) -> None:
        """Reset minute and day counters when their windows advance."""

        now_value = self._now()
        if now_value - self._minute_window_start >= 60:
            self._minute_window_start = now_value
            self._minute_requests = 0
            self._minute_tokens = 0
        today = date.today()
        if today != self._current_day:
            self._current_day = today
            self._daily_requests = 0


class GeminiClient(LLMClient):
    """Minimal Gemini API client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str,
        timeout: int,
        rate_limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        """Persist Gemini connection details and a rate limiter for future calls."""

        self._api_key = api_key
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._rate_limiter = rate_limiter
        self._session = session or requests.Session()
        self._retryer = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(LLMTransientError),
            reraise=True,
        )

    def generate_prompt(self, prompt: str) -> LLMGeneration:
        """Request a Gemini completion for the prompt with retries and rate limiting."""

        estimated_tokens = estimate_tokens(prompt)
        self._rate_limiter.acquire(estimated_tokens)
        try:
            for attempt in self._retryer:
                with attempt:
                    return self._call_once(prompt)
        except RetryError as exc:  # pragma: no cover - tenacity wraps final error
            last = exc.last_attempt
            if last is not None:
                exception = last.exception()
                if exception is not None:
                    raise exception
            raise LLMError("Gemini retries exhausted") from exc
        raise LLMError("Gemini retries exhausted")

    def _call_once(self, prompt: str) -> LLMGeneration:
        """Issue a single Gemini API call and normalize the response payload."""

        url = f"{self._api_base}/{self._model}:generateContent"
        params = {"key": self._api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": 0.7,
            },
        }
        response = self._session.post(url, params=params, json=payload, timeout=self._timeout)
        if response.status_code in {429} or 500 <= response.status_code < 600:
            raise LLMTransientError(
                f"Gemini transient error {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise LLMError(f"Gemini error {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:  # pragma: no cover
            raise LLMError(f"Invalid Gemini response: {data}") from exc
        usage = data.get("usageMetadata", {})
        return LLMGeneration(
            text=content.strip(),
            model=data.get("model", self._model),
            prompt_tokens=_coerce_int(usage.get("promptTokenCount", 0)),
            completion_tokens=_coerce_int(usage.get("candidatesTokenCount", 0)),
        )


@dataclass(slots=True)
class VibeRecord:
    """Vibe CSV row representation."""

    track_id: str
    sentence: str
    model: str
    created_at: datetime
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, str | int]:
        """Serialize the record for CSV output."""

        return {
            "Id": self.track_id,
            "VibeSentence": self.sentence,
            "model": self.model,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def to_cache_payload(self) -> dict[str, str | int]:
        """Serialize the record for JSON cache storage."""

        payload = self.to_dict().copy()
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_cache(cls, payload: dict[str, Any]) -> "VibeRecord":
        """Rehydrate a cached JSON payload into a vibe record instance."""

        return cls(
            track_id=str(payload["Id"]),
            sentence=str(payload["VibeSentence"]),
            model=str(payload["model"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])).astimezone(UTC),
            prompt_tokens=_coerce_int(payload.get("prompt_tokens")),
            completion_tokens=_coerce_int(payload.get("completion_tokens")),
        )


class VibeCache:
    """Filesystem cache keyed by track id."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, track_id: str) -> Path:
        """Return the cache filename for the given track id."""

        return self._directory / f"{track_id}.json"

    def get(self, track_id: str) -> VibeRecord | None:
        """Load a cached vibe record if it exists."""

        candidate = self.path_for(track_id)
        if not candidate.exists():
            return None
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return VibeRecord.from_cache(cast(dict[str, Any], data))

    def set(self, record: VibeRecord) -> None:
        """Persist a vibe record to its cache location."""

        candidate = self.path_for(record.track_id)
        candidate.write_text(
            json.dumps(record.to_cache_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


T = TypeVar("T")


def chunked(sequence: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield fixed-size chunks from a sequence."""

    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def estimate_tokens(text: str) -> int:
    """Rudimentary token estimate compatible with rate limits."""

    return max(1, math.ceil(len(text) / 4))


def build_user_prompt(track: Track) -> str:
    """Render the user prompt for one track."""

    return USER_PROMPT_TEMPLATE.format(
        track_name=track.track_name,
        artists=track.artists,
        album=track.album,
    )


def _create_records_from_library(
    library_path: Path,
    cache: VibeCache,
) -> tuple[List[Track], dict[str, VibeRecord]]:
    """Load tracks from the library and split them into cached and uncached sets."""

    frame = read_library_csv(library_path)
    records: List[dict[str, str]] = [
        {
            "Added At": str(row["Added At"]),
            "Track Name": str(row["Track Name"]),
            "Artists": str(row["Artists"]),
            "Album": str(row["Album"]),
            "Id": str(row["Id"]),
        }
        for row in frame.to_dict("records")
    ]
    tracks: List[Track] = []
    cached: dict[str, VibeRecord] = {}
    for row in records:
        track_id = row["Id"]
        track = Track(
            track_id=track_id,
            track_name=row["Track Name"],
            artists=row["Artists"],
            album=row["Album"],
        )
        cached_record = cache.get(track_id)
        if cached_record:
            cached[track_id] = cached_record
        tracks.append(track)
    return tracks, cached


def _normalize_dataframe(records: dict[str, VibeRecord], order: List[Track]) -> pd.DataFrame:
    """Build a dataframe of vibe records ordered to match the source library."""

    rows = []
    for track in order:
        record = records.get(track.track_id)
        if not record:
            continue
        rows.append(record.to_dict())
    frame = pd.DataFrame(rows, columns=list(VIBES_COLUMNS))
    return frame


def generate_vibes(
    *,
    library_path: Path,
    output_path: Path,
    errors_path: Path | None = None,
    cache_dir: Path | None = None,
    client: LLMClient | None = None,
) -> None:
    """Generate vibe sentences for the provided library CSV."""

    settings = get_settings()
    logger = configure_logging()
    cache = VibeCache(cache_dir or settings.cache_dir)
    tracks, cached_records = _create_records_from_library(library_path, cache)
    uncached = [track for track in tracks if track.track_id not in cached_records]

    if client is None:
        api_key = settings.gemini_api_key
        if not api_key:
            raise LLMError("GEMINI_API_KEY must be set to call Gemini")
        rate_limiter = RateLimiter(rpm=settings.rpm, tpm=settings.tpm, rpd=settings.rpd)
        client = GeminiClient(
            api_key=api_key,
            model=settings.gemini_model,
            api_base=settings.gemini_api_url,
            timeout=settings.llm_timeout_seconds,
            rate_limiter=rate_limiter,
        )

    error_rows: list[dict[str, str]] = []
    generated: dict[str, VibeRecord] = dict(cached_records)

    with task_logger(logger, "generate vibes"):
        if uncached:
            batches = list(chunked(uncached, settings.batch_size))
            for batch in progress_iter(
                batches,
                desc="LLM batches",
                total=len(batches),
            ):
                for track in batch:
                    prompt = build_user_prompt(track)
                    try:
                        generation = client.generate_prompt(prompt)
                    except LLMError as exc:
                        logger.error("LLM failed for track %s: %s", track.track_id, exc)
                        error_rows.append({"Id": track.track_id, "error": str(exc)})
                        continue
                    record = VibeRecord(
                        track_id=track.track_id,
                        sentence=generation.text,
                        model=generation.model,
                        created_at=datetime.now(tz=UTC),
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                    )
                    cache.set(record)
                    generated[track.track_id] = record

    vibes_frame = _normalize_dataframe(generated, tracks)
    write_csv(vibes_frame, output_path)

    if errors_path and error_rows:
        error_frame = pd.DataFrame(error_rows, columns=["Id", "error"])
        write_csv(error_frame, errors_path)


def _coerce_int(value: Any) -> int:
    """Convert loose numeric fields from the API/cache into safe integers."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:  # pragma: no cover - defensive
                return 0
    return 0


__all__ = [
    "GeminiClient",
    "LLMClient",
    "LLMError",
    "LLMGeneration",
    "RateLimiter",
    "VibeCache",
    "VibeRecord",
    "build_user_prompt",
    "chunked",
    "estimate_tokens",
    "generate_vibes",
]
