"""llk -d <Google Drive 分享链接>：把 Drive 上的单个文件下载到 datasets/，.tar.zst 压缩包下载后自动解压。

只支持设为“知道链接的任何人可查看”的文件，不需要登录。中断后重新运行同一命令会从 .part 文件续传。
"""

import html
import os
import re
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from ..archive import ARCHIVE_SUFFIX, unpack
from ..core.config import Config
from ..core.console import confirm, console, transfer_progress
from ..core.net import USER_AGENT, backoff_delay

DRIVE_HOSTS = frozenset({"drive.google.com", "drive.usercontent.google.com", "docs.google.com"})
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"
CHUNK_SIZE = 1 << 20
ATTEMPTS = 6


class DriveError(Exception):
    pass


def is_drive_url(text: str) -> bool:
    return (urlsplit(text.strip()).hostname or "") in DRIVE_HOSTS


def drive_file_id(url: str) -> str:
    parts = urlsplit(url.strip())
    if "/folders/" in parts.path:
        raise DriveError("不支持文件夹链接，请分享单个文件（比如 llk --pack 生成的压缩包）")
    if match := re.search(r"/d/([\w-]{10,})", parts.path):
        return match.group(1)
    if ids := parse_qs(parts.query).get("id"):
        return ids[0]
    raise DriveError(f"认不出链接里的文件 ID：{url}")


def _make_client(config: Config) -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(60.0, connect=15.0),
                        follow_redirects=True, proxy=config.proxy or None)


def _confirm_form(page: str) -> tuple[str, dict[str, str]] | None:
    """大文件会先返回“无法扫描病毒”的确认页，取出页面里下载表单的地址和参数。"""
    form = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"[^>]*>(.*?)</form>', page, re.S)
    if not form:
        return None
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', form.group(2)))
    return html.unescape(form.group(1)), {key: html.unescape(value) for key, value in fields.items()}


def _explain(status: int, page: str) -> str:
    if status == 404:
        return "文件不存在，检查链接是否正确"
    if re.search(r"quota|too many users", page, re.I):
        return "这个文件近期下载次数太多，Google 暂时限制了下载，过一段时间再试"
    return "无法下载：确认文件已在 Drive 里设为“知道链接的任何人可查看”"


def _open(client: httpx.Client, file_id: str, offset: int) -> httpx.Response:
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = client.build_request("GET", DOWNLOAD_URL, params={"id": file_id, "export": "download", "confirm": "t"},
                                   headers=headers)
    for _ in range(2):
        response = client.send(request, stream=True)
        if response.status_code == 416:  # .part 已经是完整文件或者比文件大，从头下载
            response.close()
            return _open(client, file_id, 0)
        if not response.headers.get("content-type", "").startswith("text/html"):
            if response.status_code >= 400:
                response.close()
                raise DriveError(_explain(response.status_code, ""))
            return response
        page = response.read().decode(errors="replace")
        response.close()
        form = _confirm_form(page)
        if form is None:
            raise DriveError(_explain(response.status_code, page))
        action, fields = form
        request = client.build_request("GET", action, params=fields, headers=headers)
    raise DriveError(_explain(200, ""))


def _filename(response: httpx.Response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    if match := re.search(r"filename\*\s*=\s*[\w-]+'[^']*'([^;]+)", disposition):
        name = unquote(match.group(1).strip())
    elif match := re.search(r'filename\s*=\s*"([^"]*)"|filename\s*=\s*([^;]+)', disposition):
        name = (match.group(1) or match.group(2)).strip()
    else:
        name = ""
    name = Path(name.replace("\\", "/")).name
    return fallback if name in ("", ".", "..") or name.startswith(".") else name


def _total_size(response: httpx.Response, offset: int) -> int | None:
    if match := re.search(r"/(\d+)$", response.headers.get("content-range", "")):
        return int(match.group(1))
    length = response.headers.get("content-length")
    return offset + int(length) if length and length.isdigit() else None


def _fetch(client: httpx.Client, file_id: str, part: Path, dataset_dir: Path, yes: bool) -> Path | None:
    """下载到 part，返回最终文件路径；用户拒绝覆盖时返回 None。网络错误时保留 part 以便续传。"""
    target: Path | None = None
    for attempt in range(1, ATTEMPTS + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with closing(_open(client, file_id, offset)) as response:
                if response.status_code != 206:
                    offset = 0
                if target is None:
                    target = dataset_dir / _filename(response, file_id)
                    if target.is_dir():
                        raise DriveError(f"{target} 是一个目录，无法保存同名文件")
                    if target.exists() and not yes and not confirm(f"{target} 已存在，覆盖？"):
                        return None
                    if offset:
                        console.print(f"[dim]从 {offset / 2**20:.1f} MB 处继续下载")
                total = _total_size(response, offset)
                with transfer_progress() as progress, part.open("ab" if offset else "wb") as f:
                    task = progress.add_task(target.name, total=total, completed=offset)
                    for chunk in response.iter_bytes(CHUNK_SIZE):
                        f.write(chunk)
                        progress.advance(task, len(chunk))
            if total is not None and part.stat().st_size != total:
                raise httpx.ReadError(f"只收到 {part.stat().st_size}/{total} 字节")
            return target
        except httpx.TransportError as e:
            if attempt == ATTEMPTS:
                raise DriveError(f"网络错误：{e!r}。重新运行同一命令可继续下载") from e
            console.print(f"[yellow]网络错误，{attempt}/{ATTEMPTS - 1} 次重试：{e!r}")
            time.sleep(backoff_delay(attempt, cap=30.0))
    raise AssertionError("unreachable")


def download_drive(url: str, yes: bool, config: Config) -> int:
    try:
        file_id = drive_file_id(url)
        config.dataset_dir.mkdir(parents=True, exist_ok=True)
        part = config.dataset_dir / f".gdrive-{file_id}.part"
        with _make_client(config) as client:
            target = _fetch(client, file_id, part, config.dataset_dir, yes)
    except DriveError as e:
        console.print(f"[red]{e}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断，重新运行同一命令可继续下载")
        return 130
    if target is None:
        return 1
    os.replace(part, target)
    console.print(f"[green]已下载 → {target}")
    if target.name.endswith(ARCHIVE_SUFFIX):
        return unpack(str(target), yes, config)
    return 0
