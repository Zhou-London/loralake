from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..core.dataset import Record
from .curation import Curation
from .dedup import DedupResult, deduplicate
from .tags import all_tags


@dataclass
class Selection:
    records: list[Record] = field(default_factory=list)
    excluded_count: int = 0
    hits_by_tag: dict[str, int] = field(default_factory=dict)
    dropped_count: int = 0
    edited_count: int = 0
    dedup: DedupResult | None = None


def select_records(records: Iterable[Record], exclude_images: frozenset[str]) -> Selection:
    selection = Selection()
    for record in records:
        if hits := all_tags(record) & exclude_images:
            selection.excluded_count += 1
            for tag in hits:
                selection.hits_by_tag[tag] = selection.hits_by_tag.get(tag, 0) + 1
        else:
            selection.records.append(record)
    return selection


def select_and_dedup(dataset_dir: Path, records: Iterable[Record], exclude_images: frozenset[str],
                     dedup: int | None) -> Selection:
    """先应用 llk --review 的整理结果（改过的 tag），再按 tag 排除、去重，最后去掉人工丢弃的图片。

    dedup: 去重的 pHash 距离阈值，None 表示不去重。
    """
    curation = Curation.load(dataset_dir)
    selection = select_records((curation.apply(record) for record in records), exclude_images)
    if dedup is not None and selection.records:
        selection.dedup = deduplicate(dataset_dir, selection.records, curation.reviewed, curation.dropped, dedup)
        selection.records = selection.dedup.records
    kept = [record for record in selection.records if record.key not in curation.dropped]
    selection.dropped_count = len(selection.records) - len(kept)
    selection.edited_count = sum(record.key in curation.tags for record in kept)
    selection.records = kept
    return selection
