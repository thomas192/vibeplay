# Vibeplay

Convert a CSV music library into vibe-based playlists using Gemini for sentence tagging,
sentence-transformer embeddings, and FAISS cosine similarity search.

## Quick start

```bash
# one-time setup
uv sync

# generate vibe sentences (Gemini API key required)
uv run gen-vibes --in library.csv --out vibes.csv --errors errors.csv

# embed vibes and build FAISS index
uv run embed --vibes vibes.csv --index-dir ./index

# create a playlist from a seed track
uv run make-playlist \
  --seed-id 2nWCQC688OgFKIfzk6bqKk \
  --library library.csv \
  --index-dir ./index \
  --out playlist.csv \
  --k 50 \
  --artist-cap 2
```

## Library CSV format

The library must be a UTF-8 CSV with the exact header shown below. `Id` should be the
track identifier (e.g., Spotify ID) and all values are treated as strings.

```csv
Added At,Track Name,Artists,Album,Id
2025-08-12T09:14:03Z,Aurora Skies,Lumen Ensemble,Into the Dawn,4a1n9FYwA7v3JfQiPzZKQb
2025-08-13T18:27:54Z,Night Runner,City Pulse,Neon Drive,6Yjy1tzU88zFf2h8WGeV6c
```

## Configuration

Create a `.env` file (see `.env.example`) and provide `GEMINI_API_KEY`. Defaults for
batch size (25), cache directory, and model names live in `vibeplay.config.Settings`.

## Development

- Lint: `uv run lint`
- Format: `uv run fmt`
- Tests: `uv run test`

The codebase follows YAGNI/KISS/SRP/DRY guidelines with small, testable modules and
explicit interfaces. Each pipeline step (`gen-vibes`, `embed`, `make-playlist`) runs
independently and can be validated in isolation.
