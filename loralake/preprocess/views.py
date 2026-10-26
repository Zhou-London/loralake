import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..core.dataset import Record

CAPTION_SUFFIX = ".txt"
OWNED_FILE = re.compile(r"^([a-z0-9]+_\d+)[._]")


@dataclass
class ViewSummary:
    path: Path
    images: int = 0
    linked: int = 0
    copied: int = 0
    captions_written: int = 0
    removed: int = 0


def _same_content(src: Path, dst: Path) -> bool:
    try:
        return os.path.samefile(src, dst) or src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def _link_or_copy(src: Path, dst: Path) -> bool:
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _owner_stem(file: Path) -> str | None:
    match = OWNED_FILE.match(file.name)
    return match.group(1) if match else None


def _remove_derived_caches(view_dir: Path, stem: str) -> None:
    for cache in view_dir.glob(f"{stem}_*.npz"):
        cache.unlink()


def materialize_view(dataset_dir: Path, records: Iterable[Record], view_dir: Path,
                     caption: Callable[[Record], str]) -> ViewSummary:
    view_dir.mkdir(parents=True, exist_ok=True)
    summary = ViewSummary(view_dir)
    expected_stems: set[str] = set()
    for record in records:
        source = dataset_dir / record.file
        image = view_dir / record.file
        expected_stems.add(image.stem)
        if not _same_content(source, image):
            _remove_derived_caches(view_dir, image.stem)
            if _link_or_copy(source, image):
                summary.linked += 1
            else:
                summary.copied += 1
        summary.captions_written += _write_if_changed(image.with_suffix(CAPTION_SUFFIX), caption(record))
        summary.images += 1
    for file in view_dir.iterdir():
        stem = _owner_stem(file)
        if file.is_file() and stem is not None and stem not in expected_stems:
            file.unlink()
            summary.removed += 1
    return summary
