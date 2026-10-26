import re
from collections.abc import Iterable

from ..core.dataset import Record

KAOMOJI = frozenset({
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9",
    ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
})
COUNT_TAG = re.compile(r"^(?:\d+\+?(?:girls?|boys?|others?)|multiple_(?:girls|boys|others)|no_humans)$")


def canonical_tag(tag: str) -> str:
    return "_".join(tag.strip().lower().split())


def canonical_tags(tags: Iterable[str]) -> frozenset[str]:
    return frozenset(canonical_tag(tag) for tag in tags if tag.strip())


def readable_tag(tag: str) -> str:
    return tag if tag in KAOMOJI else tag.replace("_", " ")


def is_count_tag(tag: str) -> bool:
    return COUNT_TAG.match(tag) is not None


def all_tags(record: Record) -> set[str]:
    return {tag for tags in record.tags.values() for tag in tags}
