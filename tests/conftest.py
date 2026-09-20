"""Deterministic hash-based embedding: 8-dim vectors, no model download."""
from __future__ import annotations

import hashlib
import uuid

import chromadb
import pytest
from chromadb.api.types import EmbeddingFunction


class FakeEmbedding(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    def __call__(self, input):
        out = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([b / 255.0 for b in digest[:8]])
        return out

    @staticmethod
    def name() -> str:
        return "fake"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FakeEmbedding":
        return FakeEmbedding()


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def collection(fake_embedding):
    # EphemeralClient shares one in-memory database per process: use a unique
    # collection name so tests do not see each other's items.
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"test_{uuid.uuid4().hex[:12]}", embedding_function=fake_embedding
    )
