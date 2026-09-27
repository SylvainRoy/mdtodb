from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import chromadb
import pytest

from mdtodb import open_existing_collection

class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_collection(self, **kwargs):
        self.calls.append(("get_collection", kwargs))
        return object()

    def get_or_create_collection(self, **kwargs):
        self.calls.append(("get_or_create_collection", kwargs))
        raise AssertionError("get_or_create_collection must not be called")

    def reset(self):
        raise AssertionError("reset must not be called")

def test_missing_database_never_creates(tmp_path: Path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("PersistentClient must not be constructed")

    monkeypatch.setattr(chromadb, "PersistentClient", boom)
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        open_existing_collection(missing)
    assert not missing.exists()
    (tmp_path / "emptydir").mkdir()
    with pytest.raises(FileNotFoundError):
        open_existing_collection(tmp_path / "emptydir")

def test_injected_client_uses_get_collection_only():
    client = RecordingClient()
    open_existing_collection("unused", name="documents", client=client)
    assert client.calls == [
        ("get_collection", {"name": "documents", "embedding_function": None})
    ]

def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
    )

def test_missing_collection_propagates(tmp_path: Path):
    create = "import sys, chromadb; chromadb.PersistentClient(sys.argv[1])"
    proc = _run(create, str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    script = textwrap.dedent(
        """
        import sys
        from mdtodb import open_existing_collection
        try:
            open_existing_collection(sys.argv[1], name="documents")
        except Exception as exc:
            print(type(exc).__name__)
        """
    )
    proc = _run(script, str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()

def test_real_persistent_client_opens_read_only(tmp_path: Path):
    create = textwrap.dedent(
        """
        import sys, chromadb
        chromadb.PersistentClient(sys.argv[1]).get_or_create_collection("documents")
        """
    )
    proc = _run(create, str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    open_script = textwrap.dedent(
        """
        import sys
        from mdtodb import open_existing_collection
        collection = open_existing_collection(sys.argv[1])
        print(collection.name, collection.count())
        """
    )
    proc = _run(open_script, str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "documents 0"
