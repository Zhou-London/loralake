from dataclasses import dataclass

from ..core.dataset import Record
from .tags import is_count_tag, readable_tag

ANIMA_RATINGS = {"general": "safe", "sensitive": "sensitive", "questionable": "nsfw", "explicit": "explicit"}
ANIMA_META_TAGS = frozenset({
    "highres", "absurdres", "incredibly_absurdres", "lowres", "official_art", "anime_screenshot",
    "game_cg", "scan", "traditional_media", "novel_illustration", "jpeg_artifacts",
})


@dataclass(frozen=True)
class AnimaCaptionOptions:
    trigger: str = ""
    prefix_tags: tuple[str, ...] = ()
    remove_tags: frozenset[str] = frozenset()
    meta_tags: frozenset[str] = ANIMA_META_TAGS
    include_rating: bool = True
    include_year: bool = True
    include_artist: bool = True


def anima_caption(record: Record, options: AnimaCaptionOptions = AnimaCaptionOptions()) -> str:
    def pick(category: str) -> list[str]:
        return [tag for tag in record.tags.get(category, []) if tag not in options.remove_tags]

    general = pick("general")
    leading = list(options.prefix_tags)
    leading += [readable_tag(tag) for tag in pick("meta") if tag in options.meta_tags]
    if options.include_year and record.year:
        leading.append(f"year {record.year}")
    if options.include_rating and record.rating in ANIMA_RATINGS:
        leading.append(ANIMA_RATINGS[record.rating])

    ordered = [
        *([options.trigger] if options.trigger else []),
        *leading,
        *(readable_tag(tag) for tag in general if is_count_tag(tag)),
        *(readable_tag(tag) for tag in pick("character")),
        *(readable_tag(tag) for tag in pick("copyright")),
        *(f"@{readable_tag(tag)}" for tag in pick("artist") if options.include_artist),
        *(readable_tag(tag) for tag in general if not is_count_tag(tag)),
    ]
    return ", ".join(dict.fromkeys(tag for tag in ordered if tag))
