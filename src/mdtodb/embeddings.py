"""Embedding function resolution.

``get_embedding`` returns a ``(function, identifier)`` pair; the identifier
is a stable string stored in each item's ``embedding`` metadata so a change
of embedding backend or model is detected on the next plan.
"""

from __future__ import annotations

import os
import re

from chromadb.api.types import EmbeddingFunction

EMBEDDINGS = ("default", "gemini")
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
# gemini-embedding-001 reads at most 2,048 tokens per input, so 100 inputs per
# request stays around 200k tokens — well under the per-minute token quotas.
DEFAULT_EMBED_BATCH_SIZE = 100
DEFAULT_EMBED_RETRIES = 5
# gemini-embedding-001 only reads the first 2,048 tokens of an input anyway;
# ~8k characters covers that for Latin text, and bounds the tokens billed
# against the per-minute quota for documents that are mostly noise.
DEFAULT_EMBED_MAX_CHARS = 8000
_DATA_URI_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+")
_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)
_KEY_ENV_VAR = "_MDTODB_GEMINI_API_KEY"


def _gemini_class():
    from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

    class GeminiEmbedding(GoogleGeminiEmbeddingFunction):
        """Chroma's Gemini embedding with per-request batching and 429/5xx retries.

        Chroma builds the google-genai client without ``retry_options``, which
        makes the SDK give up on the first ``429 RESOURCE_EXHAUSTED``; one
        request per Chroma upsert batch also sums every document's tokens into
        a single rate-limit window.
        """

        def __init__(self, *, batch_size: int, retries: int, max_chars: int = DEFAULT_EMBED_MAX_CHARS, **kwargs) -> None:
            super().__init__(**kwargs)
            if batch_size < 1:
                raise ValueError("embed batch size must be >= 1")
            if max_chars < 1:
                raise ValueError("embed max chars must be >= 1")
            self.batch_size = batch_size
            self.max_chars = max_chars
            import chromadb
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    headers={"x-goog-api-client": f"chroma/{chromadb.__version__}"},
                    retry_options=types.HttpRetryOptions(
                        attempts=max(1, retries),
                        initial_delay=1.0,
                        max_delay=60.0,
                        http_status_codes=list(_RETRY_STATUS_CODES),
                    ),
                ),
            )

        def __call__(self, input):
            texts = [self.prepare(text) for text in input]
            embeddings = []
            for start in range(0, len(texts), self.batch_size):
                embeddings.extend(super().__call__(texts[start : start + self.batch_size]))
            return embeddings

        def prepare(self, text: str) -> str:
            """Drop base64 data URIs (embedded images) and cap the length sent to the API."""
            text = _DATA_URI_RE.sub("", text)
            return text[: self.max_chars] if len(text) > self.max_chars else text

    return GeminiEmbedding


def get_embedding(name: str, **options) -> tuple[EmbeddingFunction, str]:
    """Return the embedding function for ``name`` and its stable identifier."""
    name = name.lower()
    if name == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction(), "default"
    if name == "gemini":
        try:
            gemini_class = _gemini_class()
        except ImportError as exc:  # pragma: no cover
            raise ValueError("gemini embedding is not available in this chromadb version") from exc
        model = options.get("model_name") or DEFAULT_GEMINI_MODEL
        api_key = options.get("api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        batch_size = options.get("batch_size")
        if api_key is None:
            raise ValueError("gemini embedding needs an API key: --gemini-api-key or GEMINI_API_KEY/GOOGLE_API_KEY")
        # GoogleGeminiEmbeddingFunction only accepts a key through an env var.
        os.environ[_KEY_ENV_VAR] = api_key
        try:
            ef = gemini_class(
                model_name=model,
                api_key_env_var=_KEY_ENV_VAR,
                batch_size=DEFAULT_EMBED_BATCH_SIZE if batch_size is None else batch_size,
                max_chars=options.get("max_chars", DEFAULT_EMBED_MAX_CHARS),
                retries=options.get("retries", DEFAULT_EMBED_RETRIES),
            )
        except ValueError as exc:
            if "not installed" in str(exc):
                raise ValueError(
                    "gemini embedding needs the google-genai package: `uv sync --extra gemini`"
                ) from exc
            raise
        return ef, f"gemini:{model}"
    raise ValueError(f"unknown embedding {name!r}: expected one of {EMBEDDINGS}")
