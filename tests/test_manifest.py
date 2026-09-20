from __future__ import annotations

import json
from pathlib import Path

from mdtodb import MANIFEST_NAME, Manifest, ManifestEntry, markdown_path_for


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), "utf-8")


def test_load_v2(tmp_path: Path):
    _write(
        tmp_path / MANIFEST_NAME,
        {
            "version": 2,
            "files": {
                "a/b.pdf": {
                    "fingerprint": "fp",
                    "size": 10,
                    "mtime": 1.0,
                    "engine": "marker",
                    "output": "a/b.md",
                    "generated_at": 2.0,
                }
            },
        },
    )
    m = Manifest.load(tmp_path / MANIFEST_NAME)
    e = m.entries["a/b.pdf"]
    assert (e.fingerprint, e.engine, e.output, e.size, e.mtime, e.generated_at) == (
        "fp",
        "marker",
        "a/b.md",
        10,
        1.0,
        2.0,
    )


def test_load_v1_sha256(tmp_path: Path):
    _write(
        tmp_path / MANIFEST_NAME,
        {"version": 1, "files": {"x.pdf": {"sha256": "h", "engine": "e", "output": "x.md"}}},
    )
    m = Manifest.load(tmp_path / MANIFEST_NAME)
    assert m.entries["x.pdf"].fingerprint == "h"
    assert m.entries["x.pdf"].size == 0  # missing keys tolerated


def test_extra_keys_tolerated(tmp_path: Path):
    _write(
        tmp_path / MANIFEST_NAME,
        {"files": {"x.pdf": {"fingerprint": "f", "engine": "e", "output": "x.md", "future": 1}}},
    )
    assert Manifest.load(tmp_path / MANIFEST_NAME).entries["x.pdf"].fingerprint == "f"


def test_from_entries():
    m = Manifest.from_entries(
        {
            "a.pdf": {"fingerprint": "f1", "engine": "e", "output": "a.md"},
            "b.pdf": ManifestEntry(fingerprint="f2", engine="e", output="b.md"),
        }
    )
    assert set(m.entries) == {"a.pdf", "b.pdf"}
    assert m.entries["b.pdf"].fingerprint == "f2"


def test_markdown_path_for():
    assert markdown_path_for("a/b.pdf") == "a/b.md"
    assert markdown_path_for("x.PNG") == "x.md"
