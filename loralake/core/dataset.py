import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

METADATA_FILE = "metadata.jsonl"


def dataset_name(tags: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", tags).strip(" ._") or "dataset"


@dataclass(frozen=True)
class Record:
    site: str
    id: int
    file: str
    rating: str = ""
    created_at: str = ""
    width: int = 0
    height: int = 0
    tags: dict[str, list[str]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.site}_{self.id}"

    @property
    def year(self) -> int | None:
        return int(self.created_at[:4]) if self.created_at[:4].isdigit() else None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(
            site=data.get("site") or data.get("source") or "",
            id=int(data["id"]),
            file=data.get("file") or "",
            rating=data.get("rating") or "",
            created_at=data.get("created_at") or "",
            width=data.get("width") or 0,
            height=data.get("height") or 0,
            tags=data.get("tags") or {},
        )


def read_records(dataset_dir: Path) -> Iterator[Record]:
    path = dataset_dir / METADATA_FILE
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield Record.from_json(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue


def resolve_dataset(name_or_path: str, dataset_dir: Path) -> Path | None:
    candidates = [Path(name_or_path), dataset_dir / name_or_path, dataset_dir / dataset_name(name_or_path)]
    return next((path for path in candidates if (path / METADATA_FILE).exists()), None)


def available_records(dataset_dir: Path) -> list[Record]:
    latest = {record.key: record for record in read_records(dataset_dir) if record.site and record.file}
    return [record for record in latest.values() if (dataset_dir / record.file).exists()]
