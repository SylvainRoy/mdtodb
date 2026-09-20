"""Embedding function resolution.

``get_embedding`` returns a ``(function, identifier)`` pair; the identifier
is a stable string stored in each item's ``embedding`` metadata so a change
of embedding backend or model is detected on the next plan.
"""

from __future__ import annotations

import os

from chromadb.api.types import EmbeddingFunction

EMBEDDINGS = ("default", "gemini")
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
_KEY_ENV_VAR = "_MDTODB_GEMINI_API_KEY"


def get_embedding(name: str, **options) -> tuple[EmbeddingFunction, str]:
    """Return the embedding function for ``name`` and its stable identifier."""
    name = name.lower()
    if name == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction(), "default"
    if name == "gemini":
        try:
            from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
        except ImportError as exc:  # pragma: no cover
            raise ValueError("gemini embedding is not available in this chromadb version") from exc
        model = options.get("model_name") or DEFAULT_GEMINI_MODEL
        api_key = options.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("gemini embedding needs an API key: --gemini-api-key or GEMINI_API_KEY/GOOGLE_API_KEY")
        # GoogleGeminiEmbeddingFunction only accepts a key through an env var.
        os.environ[_KEY_ENV_VAR] = api_key
        try:
            ef = GoogleGeminiEmbeddingFunction(model_name=model, api_key_env_var=_KEY_ENV_VAR)
        except ValueError as exc:
            if "not installed" in str(exc):
                raise ValueError(
                    "gemini embedding needs the google-genai package: `uv sync --extra gemini`"
                ) from exc
            raise
        return ef, f"gemini:{model}"
    raise ValueError(f"unknown embedding {name!r}: expected one of {EMBEDDINGS}")
