"""llk -d <Google Drive 链接>：用 rclone 把 Drive 上的单个文件下载到 datasets/，.tar.zst 压缩包下载后自动解压。

按链接里的文件 ID 下载（rclone backend copyid），rclone 授权的账号能访问的文件都可以：自己的文件不需要公开分享。
"""

import os
import re
import shutil
from urllib.parse import parse_qs, urlsplit

from ..archive import ARCHIVE_SUFFIX, unpack
from ..core import rclone
from ..core.config import Config
from ..core.console import confirm, console

DRIVE_HOSTS = frozenset({"drive.google.com", "drive.usercontent.google.com", "docs.google.com"})


class DriveError(Exception):
    pass


def is_drive_url(text: str) -> bool:
    return (urlsplit(text.strip()).hostname or "") in DRIVE_HOSTS


def drive_file_id(url: str) -> str:
    parts = urlsplit(url.strip())
    if "/folders/" in parts.path:
        raise DriveError("不支持文件夹链接，请给单个文件的链接（比如 llk --pack 生成的压缩包）")
    if match := re.search(r"/d/([\w-]{10,})", parts.path):
        return match.group(1)
    if ids := parse_qs(parts.query).get("id"):
        return ids[0]
    raise DriveError(f"认不出链接里的文件 ID：{url}")


def file_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def download_drive(url: str, yes: bool, config: Config) -> int:
    try:
        file_id = drive_file_id(url)
    except DriveError as e:
        console.print(f"[red]{e}")
        return 1
    if error := rclone.check(config):
        console.print(f"[red]{error}")
        return 1

    staging = config.dataset_dir / f".gdrive-{file_id}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        # 目标以 / 结尾时 rclone 沿用 Drive 上的文件名
        if rclone.run(config, "backend", "copyid", rclone.remote_name(config), file_id, f"{staging}/") != 0:
            console.print(f"[red]下载失败：检查链接是否正确，以及 {rclone.remote_name(config)} 授权的账号能否访问这个文件")
            return 1
        downloaded = [path for path in staging.iterdir() if path.is_file()]
        if len(downloaded) != 1:
            console.print(f"[red]下载失败：{url} 不是单个文件")
            return 1
        target = config.dataset_dir / downloaded[0].name
        if target.is_dir():
            console.print(f"[red]{target} 是一个目录，无法保存同名文件")
            return 1
        if target.exists() and not yes and not confirm(f"{target} 已存在，覆盖？"):
            return 1
        os.replace(downloaded[0], target)
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断")
        return 130
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    console.print(f"[green]已下载 → {target}")
    if target.name.endswith(ARCHIVE_SUFFIX):
        return unpack(str(target), yes, config)
    return 0
