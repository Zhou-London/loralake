"""查重：自动去掉像素级相同的图片，其余疑似重复交给人工确认（llk --review）。

1. 64 位感知哈希（pHash）距离不超过阈值（预设的 preprocess.dedup_threshold，默认 DEFAULT_THRESHOLD）的两张图是候选。
2. 宽高比相同、并且缩到 THUMB_SIZE 后逐块比较的最大差异不超过 SAME_MAX_BLOCK_DIFF 的，
   视为同一张图（缩放、重新编码、轻微调色），自动只保留一张。
3. 其余候选（去字版、加 logo、表情差分、去背景等）按连通关系分组，等待人工确认；
   确认结果保存在数据集目录的 review.json 中（见 curation.py），没确认之前全部保留。
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core.dataset import Record
from .tags import all_tags

CACHE_FILE = ".dedup_cache.json"
# 只决定哪些图交给人工确认，自动去除另有逐块比较把关，所以放宽一些。
# 中位数二值化的哈希恰好一半位为 1，距离总是偶数；实测 12～14 大多是真的差分，16 约三分之一，18 起基本是无关的图
DEFAULT_THRESHOLD = 16
# 引入可调阈值之前的固定值，run.json 里 "dedup": true 的旧训练按它续训
LEGACY_THRESHOLD = 10
MAX_THRESHOLD = 64
SAME_ASPECT_TOLERANCE = 0.02
SAME_MAX_BLOCK_DIFF = 10.0
THUMB_SIZE = 128
BLOCK_SIZE = 8
HASH_INPUT = 32
HASH_SIDE = 8
DEGRADED_TAGS = frozenset({"md5_mismatch", "downscaled", "resized", "jpeg_artifacts"})
CACHE_VERSION = 1


def _dct_matrix(size: int, rows: int) -> np.ndarray:
    k = np.arange(rows)[:, None]
    n = np.arange(size)[None, :]
    return np.cos(np.pi * (2 * n + 1) * k / (2 * size))


DCT = _dct_matrix(HASH_INPUT, HASH_SIDE)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def _fingerprint(path: Path) -> tuple[int, float]:
    image = _open_rgb(path)
    gray = np.asarray(image.convert("L").resize((HASH_INPUT, HASH_INPUT), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (DCT @ gray @ DCT.T).flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view(">u8")[0]), image.width / image.height


def _thumbnail(path: Path) -> np.ndarray:
    image = _open_rgb(path).resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.BOX)
    return np.asarray(image, dtype=np.float32)


def max_block_diff(a: np.ndarray, b: np.ndarray) -> float:
    blocks = THUMB_SIZE // BLOCK_SIZE
    diff = np.abs(a - b).mean(axis=-1)
    return float(diff.reshape(blocks, BLOCK_SIZE, blocks, BLOCK_SIZE).mean(axis=(1, 3)).max())


def _signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


@dataclass(frozen=True)
class Candidate:
    a: str
    b: str
    distance: int
    block_diff: float | None
    same: bool


@dataclass
class ReviewGroup:
    keys: list[str]
    pairs: list[tuple[str, str]]
    reviewed: bool


@dataclass
class DedupResult:
    records: list[Record]
    auto_dropped: dict[str, str]
    groups: list[ReviewGroup]

    @property
    def pending(self) -> list[ReviewGroup]:
        return [group for group in self.groups if not group.reviewed]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for item in self.parent:
            groups.setdefault(self.find(item), []).append(item)
        return list(groups.values())


def _workers(jobs: int) -> int:
    return max(1, min(jobs, os.cpu_count() or 1, 16))


def _map(function: Any, paths: list[Path]) -> list[Any]:
    if len(paths) < 16:
        return [function(path) for path in paths]
    with ProcessPoolExecutor(_workers(len(paths))) as pool:
        return list(pool.map(function, paths, chunksize=8))


def _load_cache(dataset_dir: Path) -> dict[str, Any]:
    try:
        cache = json.loads((dataset_dir / CACHE_FILE).read_text(encoding="utf-8"))
        if cache.get("version") == CACHE_VERSION:
            return cache
    except (OSError, ValueError):
        pass
    return {"version": CACHE_VERSION, "images": {}, "pairs": {}}


def _save_cache(dataset_dir: Path, cache: dict[str, Any]) -> None:
    path = dataset_dir / CACHE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def find_candidates(dataset_dir: Path, records: list[Record], threshold: int) -> list[Candidate]:
    cache = _load_cache(dataset_dir)
    paths = {record.key: dataset_dir / record.file for record in records}
    signatures = {key: _signature(path) for key, path in paths.items()}
    cached = cache["images"]
    images = {key: cached[key] for key in paths
              if key in cached and cached[key]["sig"] == signatures[key]}
    if missing := [key for key in paths if key not in images]:
        for key, (phash, aspect) in zip(missing, _map(_fingerprint, [paths[key] for key in missing])):
            images[key] = {"sig": signatures[key], "phash": f"{phash:016x}", "aspect": aspect}

    keys = sorted(paths)
    hashes = np.array([int(images[key]["phash"], 16) for key in keys], dtype=np.uint64)
    near: list[tuple[str, str, int]] = []
    for i in range(len(keys) - 1):
        distances = np.bitwise_count(hashes[i] ^ hashes[i + 1:])
        for j in np.nonzero(distances <= threshold)[0]:
            near.append((keys[i], keys[i + 1 + j], int(distances[j])))

    pair_cache = cache["pairs"]
    pairs: dict[str, float | None] = {}
    to_compare: list[tuple[str, str, str]] = []
    for a, b, _ in near:
        pair_id = f"{a}:{signatures[a]}|{b}:{signatures[b]}"
        if abs(np.log(images[a]["aspect"] / images[b]["aspect"])) > SAME_ASPECT_TOLERANCE:
            pairs[pair_id] = None
        elif pair_id in pair_cache:
            pairs[pair_id] = pair_cache[pair_id]
        else:
            to_compare.append((a, b, pair_id))
    if to_compare:
        needed = sorted({key for a, b, _ in to_compare for key in (a, b)})
        thumbs = dict(zip(needed, _map(_thumbnail, [paths[key] for key in needed])))
        for a, b, pair_id in to_compare:
            pairs[pair_id] = round(max_block_diff(thumbs[a], thumbs[b]), 2)

    _save_cache(dataset_dir, {"version": CACHE_VERSION, "images": {**cached, **images},
                              "pairs": {**pair_cache, **{k: v for k, v in pairs.items() if v is not None}}})
    candidates = []
    for a, b, distance in near:
        block_diff = pairs[f"{a}:{signatures[a]}|{b}:{signatures[b]}"]
        candidates.append(Candidate(a, b, distance, block_diff,
                                    block_diff is not None and block_diff <= SAME_MAX_BLOCK_DIFF))
    return candidates


def _keeper_rank(record: Record) -> tuple[int, int, int]:
    return (-record.width * record.height, bool(all_tags(record) & DEGRADED_TAGS), record.id)


def deduplicate(dataset_dir: Path, records: list[Record], reviewed: set[tuple[str, str]],
                dropped: set[str], threshold: int) -> DedupResult:
    """reviewed: 已确认的图片对；dropped: 已丢弃的图片；threshold: 候选的 pHash 距离上限。至多只剩一张没丢弃的组也算已确认。"""
    by_key = {record.key: record for record in records}
    candidates = find_candidates(dataset_dir, records, threshold)

    same = _UnionFind()
    for candidate in candidates:
        if candidate.same:
            same.union(candidate.a, candidate.b)
    keeper: dict[str, str] = {}
    auto_dropped: dict[str, str] = {}
    for members in same.groups():
        best = min(members, key=lambda key: _keeper_rank(by_key[key]))
        for key in members:
            keeper[key] = best
            if key != best:
                auto_dropped[key] = best

    review = _UnionFind()
    review_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        if candidate.same:
            continue
        a, b = sorted((keeper.get(candidate.a, candidate.a), keeper.get(candidate.b, candidate.b)))
        if a != b:
            review.union(a, b)
            review_pairs.add((a, b))
    pairs_by_root: dict[str, list[tuple[str, str]]] = {}
    for pair in sorted(review_pairs):
        pairs_by_root.setdefault(review.find(pair[0]), []).append(pair)
    groups = []
    for pairs in pairs_by_root.values():
        keys = sorted({key for pair in pairs for key in pair}, key=lambda key: by_key[key].id)
        done = all(pair in reviewed for pair in pairs) or sum(key not in dropped for key in keys) <= 1
        groups.append(ReviewGroup(keys=keys, pairs=pairs, reviewed=done))
    groups.sort(key=lambda group: by_key[group.keys[0]].id)

    kept = [record for record in records if record.key not in auto_dropped]
    return DedupResult(kept, auto_dropped, groups)
