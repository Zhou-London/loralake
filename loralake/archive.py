"""llk --pack / --unpack：把数据集打包成 datasets/<名字>.tar.zst 带到别的机器，或者解压回来。

压缩用 zstd（多线程，带校验和）。图片本身是 WebP，再压缩基本不会变小，打包主要是把几千个小文件合成一个，
方便拷贝和上传，解压时还能发现传输损坏。
"""

import os
import shutil
import sys
from pathlib import Path

from .core.config import Config
from .core.console import confirm, console, transfer_progress
from .core.dataset import METADATA_FILE, resolve_dataset
from .core.formatting import human_size

if sys.version_info >= (3, 14):
    import tarfile
    from compression import zstd
else:
    from backports import zstd
    from backports.zstd import tarfile

ARCHIVE_SUFFIX = ".tar.zst"
COMPRESSION_LEVEL = 10
# 下载或写入到一半的临时文件
SKIP_SUFFIXES = (".part", ".tmp")


class ArchiveError(Exception):
    pass


def _compression_options() -> dict[int, int]:
    param = zstd.CompressionParameter
    options = {param.compression_level: COMPRESSION_LEVEL, param.checksum_flag: 1}
    if param.nb_workers.bounds()[1] > 0:
        options[param.nb_workers] = os.cpu_count() or 1
    return options


def _portable(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def pack(name_or_path: str, yes: bool, config: Config) -> int:
    dataset_dir = resolve_dataset(name_or_path, config.dataset_dir)
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {name_or_path}（需要包含 {METADATA_FILE}）")
        return 1
    archive = config.dataset_dir / f"{dataset_dir.name}{ARCHIVE_SUFFIX}"
    if archive.exists() and not yes and not confirm(f"{archive} 已存在，覆盖？"):
        return 1

    files = sorted(path for path in dataset_dir.rglob("*")
                   if path.is_file() and not path.name.endswith(SKIP_SUFFIXES))
    total = sum(path.stat().st_size for path in files)
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(f"{archive.name}.tmp")
    try:
        with transfer_progress() as progress, tarfile.open(tmp, "w:zst", options=_compression_options()) as tar:
            task = progress.add_task("打包", total=total)
            for path in files:
                arcname = f"{dataset_dir.name}/{path.relative_to(dataset_dir).as_posix()}"
                tar.add(path, arcname=arcname, recursive=False, filter=_portable)
                progress.advance(task, path.stat().st_size)
        os.replace(tmp, archive)
    finally:
        tmp.unlink(missing_ok=True)

    packed = archive.stat().st_size
    console.print(f"[green]已打包 {len(files)} 个文件 → {archive}")
    console.print(f"[dim]{human_size(total)} → {human_size(packed)}（{packed / max(total, 1):.0%}）")
    return 0


def _resolve_archive(name_or_path: str, dataset_dir: Path) -> Path | None:
    candidates = [Path(name_or_path), dataset_dir / name_or_path, dataset_dir / f"{name_or_path}{ARCHIVE_SUFFIX}"]
    return next((path for path in candidates if path.is_file()), None)


def _top_level(archive: Path) -> str:
    """只读开头一小段，取压缩包里的数据集目录名。"""
    with tarfile.open(archive, "r|zst") as tar:
        first = tar.next()
    top = first.name.split("/", 1)[0] if first else ""
    if top in ("", ".", ".."):
        raise ArchiveError(f"{archive} 不是 llk --pack 生成的数据集压缩包")
    return top


def _extract(archive: Path, top: str, target: Path) -> None:
    def member_filter(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
        if not member.name.startswith(f"{top}/"):
            raise ArchiveError(f"{archive} 里有数据集目录之外的文件：{member.name}")
        return tarfile.data_filter(member, dest)

    with transfer_progress() as progress, progress.wrap_file(archive.open("rb"), total=archive.stat().st_size,
                                                     description="解压") as raw:
        with tarfile.open(fileobj=raw, mode="r|zst") as tar:
            tar.extractall(target, filter=member_filter)


def unpack(name_or_path: str, yes: bool, config: Config) -> int:
    archive = _resolve_archive(name_or_path, config.dataset_dir)
    if archive is None:
        console.print(f"[red]找不到压缩包 {name_or_path}")
        return 1
    try:
        name = _top_level(archive)
        target = config.dataset_dir / name
        if target.exists() and not yes and not confirm(f"{target} 已存在，用压缩包里的内容替换？"):
            return 1

        config.dataset_dir.mkdir(parents=True, exist_ok=True)
        staging = config.dataset_dir / f".{name}.unpacking"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            _extract(archive, name, staging)
            extracted = staging / name
            if not (extracted / METADATA_FILE).exists():
                raise ArchiveError(f"{archive} 里没有 {METADATA_FILE}，不是数据集压缩包")
            if not target.exists():
                extracted.rename(target)
            else:
                previous = staging / ".previous"  # 压缩包里只有 <name>/，不会重名
                target.rename(previous)
                try:
                    extracted.rename(target)
                except BaseException:
                    previous.rename(target)
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    except (ArchiveError, tarfile.TarError, zstd.ZstdError, EOFError) as e:
        console.print(f"[red]解压失败：{e}")
        return 1

    count = sum(1 for path in target.rglob("*") if path.is_file())
    console.print(f"[green]已解压 {count} 个文件 → {target}")
    return 0
