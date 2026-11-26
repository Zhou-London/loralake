from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .core.config import ROOT
from .preprocess.anima import AnimaCaptionOptions
from .preprocess.tags import canonical_tags
from .train.settings import TrainSettings

PRESET_DIR = ROOT / "presets"
DEFAULT_PRESET = "default"
PRESET_SUFFIXES = (".yaml", ".yml")
TAG_FIELDS = frozenset({"remove_tags", "meta_tags"})
SECTIONS = ("preprocess", "train")

Options = TypeVar("Options", AnimaCaptionOptions, TrainSettings)


class PresetError(Exception):
    pass


@dataclass(frozen=True)
class Preset:
    source: Path | None = None
    exclude_images: frozenset[str] = frozenset()
    dedup: bool = True
    caption: AnimaCaptionOptions = field(default_factory=AnimaCaptionOptions)
    train: TrainSettings = field(default_factory=TrainSettings)


def resolve_preset_path(name_or_path: str | None) -> Path | None:
    if name_or_path is None:
        default = PRESET_DIR / f"{DEFAULT_PRESET}.yaml"
        return default if default.is_file() else None
    path = Path(name_or_path)
    candidates = [path, PRESET_DIR / path, *(PRESET_DIR / f"{name_or_path}{suffix}" for suffix in PRESET_SUFFIXES)]
    if found := next((candidate for candidate in candidates if candidate.is_file()), None):
        return found
    available = ", ".join(sorted(p.stem for p in PRESET_DIR.glob("*.y*ml"))) or "无"
    raise PresetError(f"找不到预设 {name_or_path}（{PRESET_DIR} 中可用: {available}）")


def _as_tag_list(value: Any, where: str) -> list[str]:
    if isinstance(value, str):
        return [part for part in value.split(",") if part.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise PresetError(f"{where} 应为字符串列表")


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    origin = get_origin(annotation)
    if annotation is bool:
        if isinstance(value, bool):
            return value
    elif annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif annotation is str:
        if isinstance(value, str):
            return value
    elif origin in (tuple, frozenset) and get_args(annotation)[0] is str:
        items = _as_tag_list(value, where)
        return tuple(items) if origin is tuple else frozenset(items)
    else:
        raise PresetError(f"{where} 的类型 {annotation} 不支持从预设读取")
    raise PresetError(f"{where} 的值 {value!r} 类型错误，应为 {getattr(annotation, '__name__', annotation)}")


def apply_options(options: Options, values: Mapping[str, Any], section: str) -> Options:
    hints = get_type_hints(type(options))
    names = {f.name for f in fields(options)}
    if unknown := sorted(set(values) - names):
        raise PresetError(f"{section} 中有未知的键 {', '.join(unknown)}（可用: {', '.join(sorted(names))}）")
    changes = {key: _coerce(value, hints[key], f"{section}.{key}") for key, value in values.items()}
    changes = {key: canonical_tags(value) if key in TAG_FIELDS else value for key, value in changes.items()}
    return replace(options, **changes)


def _section(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = document.get(name) or {}
    if not isinstance(section, dict):
        raise PresetError(f"{name} 应为键值映射")
    return section


def load_preset(name_or_path: str | None) -> Preset:
    path = resolve_preset_path(name_or_path)
    if path is None:
        return Preset()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise PresetError(f"预设 {path} 不是合法的 YAML: {e}") from e
    if not isinstance(document, dict):
        raise PresetError(f"预设 {path} 的顶层应为键值映射")
    if unknown := sorted(set(document) - set(SECTIONS)):
        raise PresetError(f"预设 {path} 中有未知的节 {', '.join(unknown)}（可用: {', '.join(SECTIONS)}）")

    preprocess = _section(document, "preprocess")
    exclude_images = canonical_tags(_as_tag_list(preprocess.pop("exclude_images", []), "preprocess.exclude_images"))
    try:
        dedup = _coerce(preprocess.pop("dedup", True), bool, "preprocess.dedup")
        return Preset(
            source=path,
            exclude_images=exclude_images,
            dedup=dedup,
            caption=apply_options(AnimaCaptionOptions(), preprocess, "preprocess"),
            train=apply_options(TrainSettings(), _section(document, "train"), "train"),
        )
    except PresetError as e:
        raise PresetError(f"预设 {path}: {e}") from e
