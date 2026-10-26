import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import httpx

from ..core.net import request
from .booru import Post

REPO = "deepghs/danbooru2024"
SNAPSHOT_MAX_ID = 8_361_000
SHARD_COUNT = 1000
SIGNED_URL_TTL = 50 * 60
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


class MirrorError(Exception):
    pass


class TarEntry(NamedTuple):
    offset: int
    size: int


class Mirror:
    def __init__(self, client: httpx.AsyncClient, token: str, cache_dir: Path) -> None:
        self.client = client
        self.token = token
        self.index_dir = cache_dir / REPO.replace("/", "__")
        self._indexes: dict[int, dict[int, TarEntry]] = {}
        self._signed_urls: dict[int, tuple[str, float]] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def may_contain(post: Post) -> bool:
        return post.id <= SNAPSHOT_MAX_ID

    @staticmethod
    def shard_of(post: Post) -> int:
        return post.id % SHARD_COUNT

    def _resolve_url(self, path: str) -> str:
        return f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def verify_access(self) -> None:
        response = await request(self.client, "HEAD", self._resolve_url("README.md"), headers=self._auth_headers())
        if response.status_code in (401, 403):
            raise MirrorError(f"无法访问 {REPO} (HTTP {response.status_code})，请检查 HF token 并确认已在数据集页面同意访问条款")
        if response.status_code >= 400:
            raise MirrorError(f"无法访问 {REPO} (HTTP {response.status_code})")

    async def find(self, post: Post) -> TarEntry | None:
        index = await self._load_index(self.shard_of(post))
        return index.get(post.id)

    async def signed_shard_url(self, shard: int, refresh: bool = False) -> str:
        async with self._locks[f"url:{shard}"]:
            cached = self._signed_urls.get(shard)
            if cached and not refresh and time.monotonic() - cached[1] < SIGNED_URL_TTL:
                return cached[0]
            response = await request(
                self.client, "GET", self._resolve_url(f"images/{shard:04d}.tar"),
                headers={**self._auth_headers(), "Range": "bytes=0-0"}, follow_redirects=False,
            )
            location = response.headers.get("location")
            if response.status_code not in REDIRECT_STATUS or not location:
                raise MirrorError(f"镜像分片 {shard:04d} 地址获取失败: HTTP {response.status_code}")
            self._signed_urls[shard] = (location, time.monotonic())
            return str(location)

    async def _load_index(self, shard: int) -> dict[int, TarEntry]:
        if shard in self._indexes:
            return self._indexes[shard]
        async with self._locks[f"index:{shard}"]:
            if shard not in self._indexes:
                cache_file = self.index_dir / f"{shard:04d}.json"
                if not cache_file.exists():
                    await self._download_index(shard, cache_file)
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                self._indexes[shard] = {int(post_id): TarEntry(*entry) for post_id, entry in raw.items()}
            return self._indexes[shard]

    async def _download_index(self, shard: int, cache_file: Path) -> None:
        response = await request(self.client, "GET", self._resolve_url(f"images/{shard:04d}.json"), headers=self._auth_headers())
        if response.status_code != 200:
            raise MirrorError(f"镜像索引 {shard:04d} 下载失败: HTTP {response.status_code}")
        compact = {name.split(".")[0]: [info["offset"], info["size"]] for name, info in response.json()["files"].items()}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(compact), encoding="utf-8")
        os.replace(tmp, cache_file)
