"""llk --upload：用 rclone 把训练输出或数据集传到 Google Drive（比如在计算中心训练完之后）。

- 训练输出 output/<名字>/ → <RCLONE_REMOTE>/output/<名字>/，整个目录上传（中断的训练带着训练状态，可以在别处续训）
- 数据集 → 先 --pack 成 .tar.zst，上传到 <RCLONE_REMOTE>/datasets/，打印文件链接，别的机器用 llk -d <链接> 取回

RCLONE_REMOTE 默认 gdrive:loralake，对应 Colab 笔记本使用的 MyDrive/loralake/。
"""

import json
from pathlib import Path

from .archive import ARCHIVE_SUFFIX, pack
from .core import rclone
from .core.config import Config
from .core.console import confirm, console
from .core.dataset import resolve_dataset
from .core.formatting import human_size
from .download.gdrive import file_url


def _upload_run(run_dir: Path, yes: bool, config: Config) -> int:
    dest = rclone.remote_path(config.rclone_remote, "output", run_dir.name)
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if not yes and not confirm(f"上传训练输出 {run_dir}（{human_size(size)}）到 {dest}？"):
        return 1
    if rclone.run(config, "copy", str(run_dir), dest) != 0:
        console.print("[red]上传失败，重新运行同一命令会跳过已上传的文件")
        return 1
    console.print(f"[green]已上传 → {dest}")
    return 0


def _file_id(config: Config, path: str) -> str | None:
    listed = rclone.capture(config, "lsjson", path)
    try:
        entries = json.loads(listed.stdout) if listed.returncode == 0 else []
    except ValueError:
        return None
    return entries[0].get("ID") if len(entries) == 1 else None


def _upload_dataset(dataset_dir: Path, yes: bool, config: Config) -> int:
    archive = config.dataset_dir / f"{dataset_dir.name}{ARCHIVE_SUFFIX}"
    dest = rclone.remote_path(config.rclone_remote, "datasets", archive.name)
    if not yes and not confirm(f"把数据集 {dataset_dir.name} 打包上传到 {dest}？"):
        return 1
    if pack(str(dataset_dir), True, config) != 0:
        return 1
    if rclone.run(config, "copyto", str(archive), dest) != 0:
        console.print("[red]上传失败，重新运行同一命令即可")
        return 1
    console.print(f"[green]已上传 → {dest}")
    if file_id := _file_id(config, dest):
        console.print(f'在其他机器上取回：[bold]llk -d "{file_url(file_id)}"')
    else:
        console.print("[yellow]没取到文件 ID，可以在 Drive 网页上复制这个文件的链接，再用 llk -d <链接> 取回")
    return 0


def upload(name: str, yes: bool, config: Config) -> int:
    if error := rclone.check(config):
        console.print(f"[red]{error}")
        return 1
    run_dir = config.output_dir / name
    try:
        if name and run_dir.is_dir():
            return _upload_run(run_dir, yes, config)
        if dataset_dir := resolve_dataset(name, config.dataset_dir):
            return _upload_dataset(dataset_dir, yes, config)
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断，重新运行同一命令可继续")
        return 130
    console.print(f"[red]找不到训练输出 {run_dir} 或数据集 {name}")
    return 1
