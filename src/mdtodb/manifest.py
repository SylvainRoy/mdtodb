"""Reader for the ``.pdftomd-manifest.json`` written by pdftomd.

Standalone: mdtodb only ever reads the manifest, never writes it. Version 2
stores ``{"version": 2, "files": {<rel source path>: {...}}}``; version 1 used
``sha256`` instead of ``fingerprint``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

MANIFEST_NAME = ".pdftomd-manifest.json"


@dataclass
class ManifestEntry:
    fingerprint: str
    engine: str
    output: str
    size: int = 0
    mtime: float = 0.0
    generated_at: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        d = dict(d)
        if "sha256" in d:  # manifest version 1
            d["fingerprint"] = d.pop("sha256")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def markdown_path_for(rel_path: str) -> str:
    """``sub/dir/report.pdf`` -> ``sub/dir/report.md``."""
    return str(PurePosixPath(rel_path).with_suffix(".md"))


class Manifest:
    def __init__(self, entries: Mapping[str, ManifestEntry] | None = None) -> None:
        self.entries: dict[str, ManifestEntry] = dict(entries or {})

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        data = json.loads(Path(path).read_text("utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        return cls({k: ManifestEntry.from_dict(v) for k, v in data.get("files", {}).items()})

    @classmethod
    def from_entries(cls, mapping: Mapping[str, "ManifestEntry | dict"]) -> "Manifest":
        return cls(
            {
                k: v if isinstance(v, ManifestEntry) else ManifestEntry.from_dict(v)
                for k, v in mapping.items()
            }
        )
