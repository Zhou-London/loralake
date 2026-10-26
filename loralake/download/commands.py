import asyncio
from pathlib import Path

import httpx
from rich.table import Table
from rich.text import Text

from ..core.config import Config
from ..core.console import confirm, console
from ..core.dataset import dataset_name
from ..core.formatting import human_size
from ..core.net import RATE_LIMIT_PAUSE, RequestError, make_client, total_pauses
from .booru import BooruError, Danbooru, Estimate, TagInfo
from .downloader import DownloadResult, Downloader, load_downloaded_keys
from .mirror import REPO, SNAPSHOT_MAX_ID, Mirror, MirrorError

SEARCH_CONCURRENCY = 4
MIRROR_MIN_IMAGES = 200
FAILURES_SHOWN = 20
CATEGORY_STYLES = {
    "artist": ("作者", "red"),
    "character": ("角色", "green"),
    "copyright": ("作品", "magenta"),
    "general": ("通用", "blue"),
    "meta": ("meta", "yellow"),
}


def estimate_cells(estimate: Estimate) -> list[str]:
    approx = "" if estimate.exact else "≈"
    return [f"{estimate.post_count:,}", f"{approx}{estimate.image_count:,}", f"{approx}{human_size(estimate.total_bytes)}"]


def print_search_results(query: str, tags: list[TagInfo], estimates: dict[str, Estimate | None]) -> None:
    table = Table(title=Text(query, style="bold"))
    table.add_column("#", justify="right", style="dim")
    table.add_column("tag", overflow="fold")
    table.add_column("类别")
    table.add_column("帖子数", justify="right")
    table.add_column("可下载图片", justify="right")
    table.add_column("预估大小", justify="right")
    for rank, tag in enumerate(tags, 1):
        label, style = CATEGORY_STYLES.get(tag.category, (tag.category, "white"))
        estimate = estimates.get(tag.name)
        cells = estimate_cells(estimate) if estimate else [f"{tag.post_count:,}", "-", "-"]
        table.add_row(str(rank), Text(tag.name, style=style), Text(label, style=style), *cells)
    console.print(table)
    console.print("[dim]下载: llk --download <tag>")


async def search(query: str, limit: int, config: Config) -> int:
    query = "_".join(query.split())
    async with make_client(config) as client:
        booru = Danbooru(client, config)
        with console.status("正在搜索 ..."):
            try:
                tags = await booru.search_tags(query, limit)
            except (BooruError, RequestError) as e:
                console.print(f"[red]{e}")
                return 1
        if not tags:
            console.print(f"[yellow]没有找到与 {query} 匹配的 tag")
            return 1

        semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

        async def estimate_tag(name: str) -> tuple[str, Estimate | None]:
            async with semaphore:
                try:
                    return name, await booru.quick_estimate(name)
                except (BooruError, RequestError):
                    return name, None

        with console.status(f"正在估算 {len(tags)} 个 tag 的数据量 ..."):
            estimates = dict(await asyncio.gather(*(estimate_tag(tag.name) for tag in tags)))

    print_search_results(query, tags, estimates)
    return 0


async def prepare_mirror(client: httpx.AsyncClient, booru: Danbooru, tags: str, estimate: Estimate,
                         planned: int, use_mirror: bool | None, config: Config) -> tuple[Mirror, float] | None:
    if use_mirror is False:
        return None
    if not config.hf_token:
        if use_mirror:
            console.print("[red]使用镜像需要在 config 中填写 HF_TOKEN")
        return None
    if use_mirror is None and planned < MIRROR_MIN_IMAGES:
        return None

    mirror = Mirror(client, config.hf_token, config.cache_dir)
    with console.status("正在检查 HF 镜像 ..."):
        try:
            await mirror.verify_access()
        except (MirrorError, RequestError) as e:
            console.print(f"[yellow]{e}，本次只从 Danbooru 下载")
            return None
        try:
            covered_posts = await booru.count(f"{tags} id:<={SNAPSHOT_MAX_ID}")
        except (BooruError, RequestError):
            console.print(f"HF 镜像 ({REPO}) 已启用，镜像中有的图片优先从镜像下载")
            return mirror, 0.5

    share = covered_posts / estimate.post_count if estimate.post_count else 0
    if share == 0:
        console.print("[dim]HF 镜像中没有该 tag 的图片，全部从 Danbooru 下载")
        return None
    console.print(f"HF 镜像 ({REPO}) 约覆盖 {round(estimate.image_count * share):,} 张 ({share:.0%})，这部分优先从镜像下载")
    return mirror, share


def print_estimate(tags: str, estimate: Estimate) -> None:
    table = Table(title=Text(tags, style="bold"))
    table.add_column("帖子数", justify="right")
    table.add_column("可下载图片", justify="right")
    table.add_column("预估大小", justify="right")
    table.add_row(*estimate_cells(estimate))
    console.print(table)


def print_download_result(result: DownloadResult) -> None:
    sources = f"，镜像 {result.from_mirror:,} / CDN {result.from_cdn:,}" if result.from_mirror else ""
    console.print(f"[green]完成: 新下载 {result.downloaded:,} 张 ({human_size(result.total_bytes)}{sources})，跳过 {result.skipped:,} 张")
    if pauses := total_pauses():
        console.print(f"[yellow]期间被站点限流 {pauses} 次，每次暂停 {RATE_LIMIT_PAUSE:.0f} 秒")
    if result.failed:
        console.print(f"[red]失败 {len(result.failed)} 张:")
        for post, reason in result.failed[:FAILURES_SHOWN]:
            console.print(f"  {post.key} {post.file_url} - {reason}")
        if len(result.failed) > FAILURES_SHOWN:
            console.print(f"  ... 另有 {len(result.failed) - FAILURES_SHOWN} 张")
        console.print("重新运行同一命令即可重试失败项")


async def download(tags: str, output: Path | None, limit: int | None, workers: int | None,
                   yes: bool, use_mirror: bool | None, config: Config) -> int:
    out_dir = output or config.dataset_dir / dataset_name(tags)
    async with make_client(config) as client:
        booru = Danbooru(client, config)
        with console.status("正在查询 ..."):
            try:
                estimate = await booru.estimate(tags)
            except (BooruError, RequestError) as e:
                console.print(f"[red]{e}")
                return 1
        print_estimate(tags, estimate)
        if estimate.image_count == 0:
            console.print("[yellow]没有可下载的图片")
            return 0

        downloaded_count = len(load_downloaded_keys(out_dir))
        console.print(f"保存目录: {out_dir}")
        if downloaded_count:
            console.print(f"本地已有 {downloaded_count:,} 张，将跳过")
        if limit:
            console.print(f"本次最多下载 {limit:,} 张")
        expected_images = max(estimate.image_count - downloaded_count, 0)
        planned = min(expected_images, limit) if limit else expected_images
        mirror_plan = await prepare_mirror(client, booru, tags, estimate, planned, use_mirror, config)

        if not yes and not confirm("是否开始下载?"):
            console.print("已取消")
            return 0

        mirror, mirror_share = mirror_plan or (None, 0.0)
        downloader = Downloader(client, out_dir, console, workers or config.cdn_workers,
                                mirror=mirror, mirror_workers=config.mirror_workers)
        try:
            result = await downloader.run(booru, tags, expected_images, limit, mirror_share)
        except (BooruError, RequestError) as e:
            console.print(f"[red]{e}")
            result = downloader.result

    print_download_result(result)
    return 2 if result.failed else 0
