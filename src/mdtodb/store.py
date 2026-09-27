"""ChromaDB client/collection plumbing."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.client import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from .embeddings import get_embedding


def open_collection(
    path: str | Path | None = None,
    *,
    host: str | None = None,
    port: int = 8000,
    name: str = "documents",
    embedding: str | EmbeddingFunction = "default",
    embedding_name: str | None = None,
    client: ClientAPI | None = None,
    **embedding_options,
) -> tuple[Collection, str]:
    """Open (or create) a collection; returns it and the embedding identifier.

    Exactly one of ``path`` (PersistentClient), ``host`` (HttpClient) or
    ``client`` (pre-built client, e.g. EphemeralClient in tests) is required.
    ``embedding`` may be an embedding name (see ``EMBEDDINGS``) or an
    ``EmbeddingFunction`` instance — pass ``embedding_name`` for the latter
    if the stored identifier should differ from ``ef.name()``.
    """
    if isinstance(embedding, str):
        ef, resolved_name = get_embedding(embedding, **embedding_options)
    else:
        ef = embedding
        resolved_name = embedding_name or ef.name()
    if client is None:
        if (path is None) == (host is None):
            raise ValueError("exactly one of path or host is required")
        client = chromadb.PersistentClient(str(path)) if path is not None else chromadb.HttpClient(host=host, port=port)
    collection = client.get_or_create_collection(name, embedding_function=ef)
    return collection, resolved_name


def open_existing_collection(
    path: str | Path,
    *,
    name: str = "documents",
    client: ClientAPI | None = None,
) -> Collection:
    """Open an existing Chroma collection for reading; never creates one.

    Raises ``FileNotFoundError`` when ``path`` does not look like an existing
    Chroma database directory, so a typo cannot create a fresh database.
    """
    root = Path(path).expanduser().resolve()
    if client is None:
        if not root.is_dir() or not (root / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Existing Chroma database not found; no database was created")
        client = chromadb.PersistentClient(
            path=str(root),
            settings=Settings(anonymized_telemetry=False, migrations="validate", allow_reset=False),
        )
    return client.get_collection(name=name, embedding_function=None)
