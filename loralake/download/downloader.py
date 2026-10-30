import asyncio
import hashlib
import json
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, Self

import httpx
from PIL import Image
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ..core.dataset import METADATA_FILE, available_records
from ..core.formatting import human_size
from ..core.net import RETRY_STATUS, gate_for, paused_sites
from .booru import Booru, Post
from .mirror import Mirror

CDN_QUEUE_SIZE = 5000
CDN_BURST_MAX = 150
CDN_PACED_RATE = 2.5
MAX_ATTEMPTS = 5
MAX_RATE_LIMIT_RETRIES = 60
MIRROR_ATTEMPTS = 3
ASPECT_TOLERANCE = 0.02
WEBP_QUALITY = 95
WEBP_MAX_SIDE = 16383
SPEED_WINDOW = 10.0
REFRESH_INTERVAL = 0.5
CHUNK_SIZE = 1 << 16

Via = Literal["mirror", "cdn"]


def load_downloaded_keys(out_dir: Path) -> set[str]:
    return {record.key for record in available_records(out_dir)}


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def same_aspect(size: tuple[int, int], post: Post) -> bool:
    if not post.width or not post.height:
        return True
    return abs(size[0] / size[1] / (post.width / post.height) - 1) <= ASPECT_TOLERANCE


def convert_to_webp(src: Path, dst: Path) -> tuple[int, int]:
    encoded = dst.with_name(dst.name + ".encoding")
    with Image.open(src) as image:
        image.load()
        size = image.size
        reencode = image.format != "WEBP"
        if reencode:
            has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
            converted = image.convert("RGBA" if has_alpha else "RGB")
            converted.thumbnail((WEBP_MAX_SIDE, WEBP_MAX_SIDE))
            converted.save(encoded, "WEBP", quality=WEBP_QUALITY, method=4)
            size = converted.size
    os.replace(encoded if reencode else src, dst)
    if reencode:
        src.unlink()
    return size


class RetryableError(Exception):
    pass


@dataclass
class DownloadResult:
    downloaded: int = 0
    from_mirror: int = 0
    skipped: int = 0
    total_bytes: int = 0
    failed: list[tuple[Post, str]] = field(default_factory=list)

    @property
    def from_cdn(self) -> int:
        return self.downloaded - self.from_mirror


class DownloadProgress:
    def __init__(self, console: Console, total: int, show_sources: bool) -> None:
        self._show_sources = show_sources
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[size]}"),
            TextColumn("{task.fields[speed]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
        )
        self._task = self._progress.add_task("下载", total=max(total, 1), size="0 B", speed="", status="")
        self._byte_samples: deque[tuple[float, int]] = deque([(time.monotonic(), 0)])
        self.total = total

    def __enter__(self) -> Self:
        self._progress.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._progress.__exit__(*exc_info)

    def set_total(self, total: int) -> None:
        self.total = total
        self._progress.update(self._task, total=max(total, 1))

    def advance(self) -> None:
        self._progress.advance(self._task)

    def refresh(self, result: DownloadResult, streamed_bytes: int) -> None:
        now = time.monotonic()
        self._byte_samples.append((now, streamed_bytes))
        while len(self._byte_samples) > 2 and now - self._byte_samples[0][0] > SPEED_WINDOW:
            self._byte_samples.popleft()
        since, bytes_then = self._byte_samples[0]
        speed = (streamed_bytes - bytes_then) / max(now - since, 1e-6)
        status = [f"[dim]镜像 {result.from_mirror} · CDN {result.from_cdn}[/]"] if self._show_sources else []
        status += [f"[yellow]{site} 限流中，{seconds:.0f}s 后继续[/]" for site, seconds in paused_sites()]
        self._progress.update(self._task, size=human_size(result.total_bytes),
                              speed=f"{human_size(speed)}/s", status=" ".join(status))


class Downloader:
    def __init__(self, client: httpx.AsyncClient, out_dir: Path, console: Console, cdn_workers: int,
                 mirror: Mirror | None = None, mirror_workers: int = 0) -> None:
        self.client = client
        self.out_dir = out_dir
        self.console = console
        self.cdn_workers = cdn_workers
        self.mirror = mirror
        self.mirror_workers = mirror_workers if mirror else 0
        self.result = DownloadResult()
        self._streamed_bytes = 0
        self._queued = 0
        self._limit: int | None = None
        self._cdn_expected = 0
        self._cdn_started = 0
        self._next_cdn_slot = 0.0
        self._downloaded_keys: set[str] = set()
        self._cdn_queue: asyncio.Queue[Post | None] = asyncio.Queue(maxsize=CDN_QUEUE_SIZE)
        self._mirror_queue: asyncio.Queue[Post | None] = asyncio.Queue(maxsize=max(self.mirror_workers, 1) * 4)

    async def run(self, booru: Booru, tags: str, expected_images: int, limit: int | None,
                  mirror_share: float = 0.5, cdn_expected: int = 0) -> DownloadResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._downloaded_keys = load_downloaded_keys(self.out_dir)
        self._limit = limit
        self._cdn_expected = cdn_expected
        planned = min(expected_images, limit) if limit else expected_images

        with (DownloadProgress(self.console, planned, show_sources=self.mirror is not None) as progress,
              (self.out_dir / METADATA_FILE).open("a", encoding="utf-8") as metadata):
            refresher = asyncio.create_task(self._refresh_periodically(progress))
            mirror_tasks = [asyncio.create_task(self._mirror_worker(progress, metadata)) for _ in range(self.mirror_workers)]
            cdn_tasks = [asyncio.create_task(self._cdn_worker(progress, metadata)) for _ in range(self.cdn_workers)]
            producer = asyncio.create_task(self._produce(booru, tags, progress, mirror_share))
            try:
                await producer
                await asyncio.gather(*mirror_tasks)
                for _ in range(self.cdn_workers):
                    await self._cdn_queue.put(None)
                await asyncio.gather(*cdn_tasks)
                progress.refresh(self.result, self._streamed_bytes)
            finally:
                for task in (refresher, producer, *mirror_tasks, *cdn_tasks):
                    task.cancel()
        return self.result

    async def _refresh_periodically(self, progress: DownloadProgress) -> None:
        while True:
            progress.refresh(self.result, self._streamed_bytes)
            await asyncio.sleep(REFRESH_INTERVAL)

    def _limit_reached(self) -> bool:
        return self._limit is not None and self._queued >= self._limit

    async def _enqueue(self, posts: AsyncIterator[Post], queue: asyncio.Queue[Post | None],
                       progress: DownloadProgress, quota: int | None = None) -> bool:
        if (quota is not None and quota <= 0) or self._limit_reached():
            return False
        enqueued = 0
        async for post in posts:
            if post.key in self._downloaded_keys:
                self.result.skipped += 1
                continue
            await queue.put(post)
            self._queued += 1
            enqueued += 1
            if self._queued > progress.total:
                progress.set_total(self._queued)
            if (quota is not None and enqueued >= quota) or self._limit_reached():
                return False
        return True

    async def _produce(self, booru: Booru, tags: str, progress: DownloadProgress, mirror_share: float) -> None:
        try:
            if self.mirror is None:
                await self._enqueue(booru.iter_posts(tags), self._cdn_queue, progress)
                return
            recent_posts = booru.iter_posts(tags, id_above=self.mirror.max_id)
            archived_posts = booru.iter_posts(tags, id_below=self.mirror.max_id + 1)
            mirror_quota = round(self._limit * mirror_share) if self._limit else None
            cdn_quota = self._limit - mirror_quota if self._limit and mirror_quota is not None else None
            recent_exhausted, archived_exhausted = await asyncio.gather(
                self._enqueue(recent_posts, self._cdn_queue, progress, cdn_quota),
                self._enqueue(archived_posts, self._mirror_queue, progress, mirror_quota),
            )
            if self._limit and not recent_exhausted:
                await self._enqueue(recent_posts, self._cdn_queue, progress)
            if self._limit and not archived_exhausted:
                await self._enqueue(archived_posts, self._mirror_queue, progress)
        finally:
            progress.set_total(self._queued)
            for _ in range(self.mirror_workers):
                await self._mirror_queue.put(None)

    def _record(self, post: Post, via: Via, size: tuple[int, int], metadata: IO[str], progress: DownloadProgress) -> None:
        (self.out_dir / f"{post.key}.txt").write_text(post.caption(), encoding="utf-8")
        record = {**post.to_record(), "file": self._target(post).name, "width": size[0], "height": size[1], "via": via}
        metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
        metadata.flush()
        self.result.downloaded += 1
        progress.advance()

    async def _mirror_worker(self, progress: DownloadProgress, metadata: IO[str]) -> None:
        while (post := await self._mirror_queue.get()) is not None:
            try:
                size = await self._download_from_mirror(post)
            except Exception:
                size = None
            if size:
                self.result.from_mirror += 1
                self._record(post, "mirror", size, metadata, progress)
            else:
                await self._cdn_queue.put(post)

    async def _cdn_worker(self, progress: DownloadProgress, metadata: IO[str]) -> None:
        while (post := await self._cdn_queue.get()) is not None:
            self._cdn_started += 1
            try:
                size = await self._download_from_cdn(post)
                self._record(post, "cdn", size, metadata, progress)
            except Exception as e:
                self.result.failed.append((post, str(e) or type(e).__name__))
                progress.advance()

    async def _write_stream(self, response: httpx.Response, path: Path) -> str:
        digest = hashlib.md5()
        with path.open("wb") as f:
            async for chunk in response.aiter_bytes(CHUNK_SIZE):
                f.write(chunk)
                digest.update(chunk)
                self._streamed_bytes += len(chunk)
        return digest.hexdigest()

    def _target(self, post: Post) -> Path:
        return self.out_dir / f"{post.key}.webp"

    async def _cdn_turn(self) -> None:
        if self._cdn_expected < CDN_BURST_MAX and self._cdn_started < CDN_BURST_MAX:
            return
        now = time.monotonic()
        slot = max(self._next_cdn_slot, now)
        self._next_cdn_slot = slot + 1 / CDN_PACED_RATE
        await asyncio.sleep(slot - now)

    async def _download_from_mirror(self, post: Post) -> tuple[int, int] | None:
        assert self.mirror is not None
        entry = self.mirror.find(post)
        if entry is None:
            return None
        target = self._target(post)
        part = target.with_name(target.name + ".part")
        byte_range = {"Range": f"bytes={entry.offset}-{entry.offset + entry.size - 1}"}
        shard = self.mirror.shard_of(post)
        for attempt in range(MIRROR_ATTEMPTS):
            url = await self.mirror.signed_shard_url(shard, refresh=attempt > 0)
            try:
                async with self.client.stream("GET", url, headers=byte_range) as response:
                    if response.status_code != 206:
                        continue
                    await self._write_stream(response, part)
            except httpx.TransportError:
                await asyncio.sleep(2**attempt)
                continue
            size = image_size(part) if part.stat().st_size == entry.size else None
            if size and same_aspect(size, post):
                os.replace(part, target)
                self.result.total_bytes += entry.size
                return size
            break
        part.unlink(missing_ok=True)
        return None

    async def _download_from_cdn(self, post: Post) -> tuple[int, int]:
        target = self._target(post)
        part = self.out_dir / f"{post.filename}.part"
        gate = gate_for(post.file_url)
        error = ""
        attempt = rate_limited = 0
        while attempt < MAX_ATTEMPTS:
            await gate.wait()
            await self._cdn_turn()
            try:
                async with self.client.stream("GET", post.file_url) as response:
                    if response.status_code == 429 and rate_limited < MAX_RATE_LIMIT_RETRIES:
                        rate_limited += 1
                        gate.pause(response)
                        continue
                    if response.status_code in RETRY_STATUS:
                        raise RetryableError(f"HTTP {response.status_code}")
                    if response.status_code >= 400:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    if response.headers.get("Content-Type", "").startswith("text/"):
                        raise RuntimeError(f"返回了网页而不是图片: {response.url}")
                    file_md5 = await self._write_stream(response, part)
                if post.md5 and file_md5 != post.md5:
                    raise RetryableError("md5 校验失败")
                size = await asyncio.to_thread(convert_to_webp, part, target)
                self.result.total_bytes += target.stat().st_size
                return size
            except (httpx.TransportError, RetryableError) as e:
                error = str(e) or type(e).__name__
                attempt += 1
                await asyncio.sleep(min(30, 2**attempt))
        part.unlink(missing_ok=True)
        raise RuntimeError(error)
