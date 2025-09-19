"""Logging helpers with tqdm integration."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm


LOGGER_NAME = "vibeplay"

T = TypeVar("T")


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger."""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def progress_iter(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
    """Wrap an iterable with tqdm for consistent progress reporting."""

    yield from tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)


@contextmanager
def task_logger(logger: logging.Logger, task: str) -> Iterator[None]:
    """Context manager that logs start and finish of a task."""

    logger.info("start | %s", task)
    try:
        yield
    finally:
        logger.info("done  | %s", task)
