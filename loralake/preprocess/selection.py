from collections.abc import Iterable
from dataclasses import dataclass, field

from ..core.dataset import Record
from .tags import all_tags


@dataclass
class Selection:
    records: list[Record] = field(default_factory=list)
    excluded_count: int = 0
    hits_by_tag: dict[str, int] = field(default_factory=dict)


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
