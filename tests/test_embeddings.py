from __future__ import annotations

from types import SimpleNamespace

import pytest

import mdtodb.embeddings as embeddings_module
from mdtodb.embeddings import DEFAULT_EMBED_BATCH_SIZE, get_embedding

genai = pytest.importorskip("google.genai")


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_content(self, *, model, contents, config):
        self.calls.append(list(contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(text)), 1.0]) for text in contents])


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.models = FakeModels()
        FakeClient.instances.append(self)


@pytest.fixture
def fake_genai(monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return FakeClient


def test_gemini_embeds_in_batches(fake_genai):
    ef, name = get_embedding("gemini", batch_size=3)
    assert name == "gemini:gemini-embedding-001"
    docs = [f"doc {i}" for i in range(7)]
    vectors = ef(docs)
    assert len(vectors) == 7 and list(vectors[6]) == [5.0, 1.0]
    assert [len(c) for c in ef.client.models.calls] == [3, 3, 1]


def test_gemini_client_retries_rate_limits(fake_genai):
    ef, _ = get_embedding("gemini")
    assert ef.batch_size == DEFAULT_EMBED_BATCH_SIZE
    retry = ef.client.kwargs["http_options"].retry_options
    assert retry.attempts == embeddings_module.DEFAULT_EMBED_RETRIES
    assert 429 in retry.http_status_codes and 503 in retry.http_status_codes
    assert ef.client.kwargs["api_key"] == "test-key"


def test_gemini_rejects_bad_batch_size(fake_genai):
    with pytest.raises(ValueError, match="batch size"):
        get_embedding("gemini", batch_size=0)


def test_gemini_strips_data_uris_and_caps_length(fake_genai):
    ef, _ = get_embedding("gemini", batch_size=10, max_chars=20)
    blob = "![][image1]\n\n[image1]: <data:image/png;base64," + "iVBORw0KGgo" * 50 + ">\n"
    ef(["short text " + blob, "x" * 100])
    [sent] = ef.client.models.calls
    assert sent[0] == "short text ![][image1]\n\n[image1]: <>\n"[:20]
    assert "base64" not in sent[0]
    assert sent[1] == "x" * 20
