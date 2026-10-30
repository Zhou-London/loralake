import asyncio
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import httpx
import numpy as np

from ..core.net import request
from .booru import Post

REPO = "nyanko-devs/danbooru2026"
INDEX_REPO = "petersunde/danbooru2026-index"
INDEX_FILE = "tar-index.parquet"
INDEX_ARRAYS = ("id", "offset_bytes", "length_bytes")
SHARD_COUNT = 1000
SIGNED_URL_TTL = 50 * 60
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


class MirrorError(Exception):
    pass


class TarEntry(NamedTuple):
    offset: int
    size: int


def convert_index(parquet: Path, out_dir: Path) -> None:
    try:
        import polars as pl
    except ImportError:
        raise MirrorError("转换镜像索引需要 polars，请先 pip install polars")
    table = pl.read_parquet(parquet, columns=list(INDEX_ARRAYS)).sort("id")
    for column in INDEX_ARRAYS:
        tmp = out_dir / f"{column}.tmp.npy"
        np.save(tmp, table[column].to_numpy().astype(np.int64))
        os.replace(tmp, out_dir / f"{column}.npy")


class Mirror:
    def __init__(self, client: httpx.AsyncClient, token: str, cache_dir: Path) -> None:
        self.client = client
        self.token = token
        self.index_dir = cache_dir / REPO.replace("/", "__")
        self._ids: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._lengths: np.ndarray | None = None
        self._signed_urls: dict[int, tuple[str, float]] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def max_id(self) -> int:
        assert self._ids is not None
        return int(self._ids[-1])

    @staticmethod
    def shard_of(post: Post) -> int:
        return post.id % SHARD_COUNT

    def _resolve_url(self, path: str, repo: str = REPO) -> str:
        return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def verify_access(self) -> None:
        response = await request(self.client, "HEAD", self._resolve_url("README.md"), headers=self._auth_headers())
        if response.status_code in (401, 403):
            raise MirrorError(f"无法访问 {REPO} (HTTP {response.status_code})，请检查 HF token 并确认已在数据集页面同意访问条款")
        if response.status_code >= 400:
            raise MirrorError(f"无法访问 {REPO} (HTTP {response.status_code})")

    async def load_index(self) -> None:
        if self._ids is not None:
            return
        if not all((self.index_dir / f"{column}.npy").exists() for column in INDEX_ARRAYS):
            await self._download_index()
        self._ids, self._offsets, self._lengths = (
            np.load(self.index_dir / f"{column}.npy", mmap_mode="r") for column in INDEX_ARRAYS
        )

    async def _download_index(self) -> None:
        response = await request(self.client, "GET", self._resolve_url(INDEX_FILE, INDEX_REPO), headers=self._auth_headers())
        if response.status_code != 200:
            raise MirrorError(f"镜像索引下载失败: HTTP {response.status_code}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        parquet = self.index_dir / INDEX_FILE
        parquet.write_bytes(response.content)
        await asyncio.to_thread(convert_index, parquet, self.index_dir)
        parquet.unlink()

    def find(self, post: Post) -> TarEntry | None:
        assert self._ids is not None and self._offsets is not None and self._lengths is not None
        i = int(np.searchsorted(self._ids, post.id))
        if i >= len(self._ids) or self._ids[i] != post.id:
            return None
        return TarEntry(int(self._offsets[i]), int(self._lengths[i]))

    async def signed_shard_url(self, shard: int, refresh: bool = False) -> str:
        async with self._locks[f"url:{shard}"]:
            cached = self._signed_urls.get(shard)
            if cached and not refresh and time.monotonic() - cached[1] < SIGNED_URL_TTL:
                return cached[0]
            response = await request(
                self.client, "GET", self._resolve_url(f"original/data-{shard:04d}.tar"),
                headers={**self._auth_headers(), "Range": "bytes=0-0"}, follow_redirects=False,
            )
            location = response.headers.get("location")
            if response.status_code not in REDIRECT_STATUS or not location:
                raise MirrorError(f"镜像分片 {shard:04d} 地址获取失败: HTTP {response.status_code}")
            self._signed_urls[shard] = (location, time.monotonic())
            return str(location)
