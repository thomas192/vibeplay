"""Command line interface for vibeplay."""

from __future__ import annotations

from pathlib import Path

import click

from .embedder import embed_vibes
from .llm_vibes import generate_vibes
from .search import make_playlist


@click.group(help="Utilities for generating vibe sentences, embeddings, and playlists.")
def cli() -> None:
    """Entry point for vibeplay utilities."""


@cli.command(name="gen-vibes", help="Generate vibe sentences for each track in the library CSV.")
@click.option(
    "--in",
    "library_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Input library CSV with the expected Spotify export columns.",
)
@click.option(
    "--out",
    "output_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Where to write the vibes CSV (Id,VibeSentence,...).",
)
@click.option(
    "--errors",
    "errors_path",
    required=False,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Optional CSV capturing rows that failed LLM generation.",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    required=False,
    type=click.Path(path_type=Path, file_okay=False),
    help="Override the directory used to cache Gemini responses.",
)
def gen_vibes_command(
    *,
    library_path: Path,
    output_path: Path,
    errors_path: Path | None,
    cache_dir: Path | None,
) -> None:
    """Generate vibe sentences for the input library."""

    generate_vibes(
        library_path=library_path,
        output_path=output_path,
        errors_path=errors_path,
        cache_dir=cache_dir,
    )


@cli.command(name="embed", help="Embed vibe sentences and persist the FAISS index artifacts.")
@click.option(
    "--vibes",
    "vibes_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Vibes CSV produced by the gen-vibes command.",
)
@click.option(
    "--index-dir",
    "index_dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory where embeddings.npy, faiss.index, ids.jsonl, and vibes.join.csv will be stored.",
)
def embed_command(*, vibes_path: Path, index_dir: Path) -> None:
    """Embed vibe sentences and persist the FAISS index."""

    embed_vibes(vibes_path=vibes_path, index_dir=index_dir)


@cli.command(name="make-playlist", help="Run similarity search to build a vibe-based playlist.")
@click.option(
    "--seed-id",
    "seed_id",
    required=True,
    help="Track Id (from the library CSV) to seed the similarity search.",
)
@click.option(
    "--library",
    "library_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Original library CSV used to provide metadata in the playlist output.",
)
@click.option(
    "--index-dir",
    "index_dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory containing embeddings.npy, faiss.index, ids.jsonl, and vibes.join.csv.",
)
@click.option(
    "--out",
    "output_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Where to write the playlist CSV.",
)
@click.option(
    "--k",
    "k",
    default=50,
    show_default=True,
    type=int,
    help="Number of similar tracks to return (seed excluded).",
)
@click.option(
    "--artist-cap",
    "artist_cap",
    default=2,
    show_default=True,
    type=int,
    help="Maximum number of tracks per artist in the playlist (0 disables the cap).",
)
def make_playlist_command(
    *,
    seed_id: str,
    library_path: Path,
    index_dir: Path,
    output_path: Path,
    k: int,
    artist_cap: int,
) -> None:
    """Create a playlist by similarity search."""

    make_playlist(
        seed_id=seed_id,
        library_path=library_path,
        index_dir=index_dir,
        output_path=output_path,
        k=k,
        artist_cap=artist_cap,
    )


def main() -> None:  # pragma: no cover - entrypoint
    cli(prog_name="vibeplay")


if __name__ == "__main__":  # pragma: no cover
    main()
