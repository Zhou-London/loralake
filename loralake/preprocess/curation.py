"""数据集本身的人工整理结果（llk --review），与模型无关，预处理和训练都会先应用它。

- dropped: 丢弃的图片
- reviewed: 已确认的疑似重复图片对
- tags: 改过 tag 的图片，保存修改后的完整 tag（按类别）
"""

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..core.dataset import Record
from .tags import canonical_tag

REVIEW_FILE = "review.json"
LEGACY_FILE = "duplicates.json"
TAG_CATEGORIES = ("artist", "copyright", "character", "general", "meta")

Pair = tuple[str, str]
Tags = dict[str, list[str]]


def _pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


def normalize_tags(tags: Mapping[str, Iterable[str]]) -> Tags:
    """统一 tag 写法、去掉重复和空类别，类别按 Danbooru 的顺序排列。"""
    seen: set[str] = set()
    result: Tags = {}
    for category in sorted(tags, key=lambda c: (TAG_CATEGORIES.index(c) if c in TAG_CATEGORIES else len(TAG_CATEGORIES), c)):
        values = []
        for tag in map(canonical_tag, tags[category]):
            if tag and tag not in seen:
                seen.add(tag)
                values.append(tag)
        if values:
            result[category] = values
    return result


@dataclass
class Curation:
    dropped: set[str] = field(default_factory=set)
    reviewed: set[Pair] = field(default_factory=set)
    tags: dict[str, Tags] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset_dir: Path) -> "Curation":
        path = dataset_dir / REVIEW_FILE
        if not path.exists():
            path = dataset_dir / LEGACY_FILE
            if not path.exists():
                return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(dropped=set(data.get("dropped", [])),
                   reviewed={_pair(a, b) for a, b in data.get("reviewed", [])},
                   tags={key: normalize_tags(tags) for key, tags in data.get("tags", {}).items()})

    def save(self, dataset_dir: Path) -> None:
        path = dataset_dir / REVIEW_FILE
        data: dict[str, Any] = {"dropped": sorted(self.dropped), "reviewed": sorted([a, b] for a, b in self.reviewed),
                                "tags": dict(sorted(self.tags.items()))}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        (dataset_dir / LEGACY_FILE).unlink(missing_ok=True)

    def apply(self, record: Record) -> Record:
        tags = self.tags.get(record.key)
        return record if tags is None else replace(record, tags=tags)

    def set_tags(self, original: Record, tags: Mapping[str, Iterable[str]]) -> Tags:
        tags = normalize_tags(tags)
        if tags == normalize_tags(original.tags):
            self.tags.pop(original.key, None)
        else:
            self.tags[original.key] = tags
        return tags

    def confirmed(self, pairs: Iterable[Pair]) -> bool:
        return all(pair in self.reviewed for pair in pairs)
