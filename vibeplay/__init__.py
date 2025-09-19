"""Vibe-based playlist generation package."""

from importlib import metadata

try:
    __version__ = metadata.version("vibeplay")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
